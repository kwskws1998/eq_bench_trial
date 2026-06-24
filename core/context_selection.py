from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .compression import ContextChunk, combine_chunk_scores, predicted_trt_chunk_scores
from .encoder import encode_passage_tokens, encode_query, load_e5_encoder
from .gaze_attention import (
    chunk_token_indices,
    compute_gaze_distribution,
    gaze_query_attention,
    normalize_vector,
    pool_chunk,
    token_trt_from_predicted_words,
)
from .predicted_trt import load_trt_predictor


BASELINE = "baseline"
TEXT_CONTEXT = "text_context"
EMOTION_ET_CONTEXT = "emotion_et_context"
TEXT_PLUS_EMOTION_ET_CONTEXT = "text_plus_emotion_et_context"
GAZE_QUERY_ATTENTION_ET = "gaze_query_attention_et"
SUPPORTED_CONDITIONS = {
    BASELINE,
    TEXT_CONTEXT,
    EMOTION_ET_CONTEXT,
    TEXT_PLUS_EMOTION_ET_CONTEXT,
    GAZE_QUERY_ATTENTION_ET,
}


@dataclass(frozen=True)
class SelectionResult:
    condition: str
    original_context: str
    selected_context: str
    selected_chunks: list[ContextChunk]
    selected_scores: list[float]
    original_word_count: int
    selected_word_count: int

    @property
    def compression_ratio(self) -> float:
        if self.original_word_count <= 0:
            return 1.0
        return self.selected_word_count / self.original_word_count


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_text_chunks(text: str, max_words: int = 80) -> list[ContextChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    chunks: list[ContextChunk] = []
    for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        local_trim = len(match.group(0)) - len(match.group(0).lstrip())
        start = match.start() + local_trim
        words = list(re.finditer(r"\S+", sentence))
        if len(words) <= max_words:
            chunks.append(ContextChunk(f"chunk={len(chunks)}", sentence, start, start + len(sentence), len(chunks)))
            continue
        for word_start in range(0, len(words), max_words):
            group = words[word_start : word_start + max_words]
            chunk_text = sentence[group[0].start() : group[-1].end()]
            chunk_start = start + group[0].start()
            chunk_end = start + group[-1].end()
            chunks.append(ContextChunk(f"chunk={len(chunks)}", chunk_text, chunk_start, chunk_end, len(chunks)))
    if chunks:
        return chunks
    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot split empty context.")
    start = text.index(stripped)
    return [ContextChunk("chunk=0", stripped, start, start + len(stripped), 0)]


class ContextSelector:
    def __init__(
        self,
        condition: str,
        context_budget_words: int = 220,
        chunk_max_words: int = 80,
        alpha: float = 1.0,
        beta: float = 0.25,
        tau_q: float = 0.05,
        tau_g: float = 0.1,
        encoder_name: str = "intfloat/e5-large-v2",
        encoder_device: str = "auto",
        predictor_backend: str = "skboy",
        predictor_model_name: str = "skboy/emotion_et_2nd_model",
        predictor_weights: str = "et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors",
        predictor_subfolder: str | None = "hf_emotion_et_aug_lr2e-5_len256_seed123",
        predictor_local_files_only: bool = False,
        cache_dir: str | None = None,
        encoder_max_length: int = 256,
    ) -> None:
        if condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
        self.condition = condition
        self.context_budget_words = context_budget_words
        self.chunk_max_words = chunk_max_words
        self.alpha = alpha
        self.beta = beta
        self.tau_q = tau_q
        self.tau_g = tau_g
        self.encoder_max_length = encoder_max_length
        self.encoder = None
        self.predictor = None
        if condition in {TEXT_CONTEXT, TEXT_PLUS_EMOTION_ET_CONTEXT, GAZE_QUERY_ATTENTION_ET}:
            self.encoder = load_e5_encoder(encoder_name, encoder_device, cache_dir)
        if condition in {EMOTION_ET_CONTEXT, TEXT_PLUS_EMOTION_ET_CONTEXT, GAZE_QUERY_ATTENTION_ET}:
            self.predictor = load_trt_predictor(
                backend=predictor_backend,
                repo_id=predictor_model_name,
                weights_filename=predictor_weights,
                subfolder=predictor_subfolder,
                cache_dir=cache_dir,
                local_files_only=predictor_local_files_only,
            )

    def select(self, context: str, query: str) -> SelectionResult:
        context = context.strip()
        if not context:
            raise ValueError("context must not be empty.")
        chunks = split_text_chunks(context, self.chunk_max_words)
        original_words = count_words(context)
        if self.condition == BASELINE or original_words <= self.context_budget_words:
            return SelectionResult(
                condition=self.condition,
                original_context=context,
                selected_context=context,
                selected_chunks=chunks,
                selected_scores=[1.0 for _ in chunks],
                original_word_count=original_words,
                selected_word_count=original_words,
            )
        scores = self._score(context, query, chunks)
        selected, selected_scores = self._select_by_budget(chunks, scores)
        selected_context = "\n\n".join(chunk.text for chunk in selected)
        return SelectionResult(
            condition=self.condition,
            original_context=context,
            selected_context=selected_context,
            selected_chunks=selected,
            selected_scores=selected_scores,
            original_word_count=original_words,
            selected_word_count=count_words(selected_context),
        )

    def _score(self, context: str, query: str, chunks: list[ContextChunk]) -> np.ndarray:
        if self.condition == TEXT_CONTEXT:
            return combine_chunk_scores(
                "query_only_compression",
                query_scores=self._query_scores(query, chunks),
                tau_q=self.tau_q,
            )
        if self.condition == EMOTION_ET_CONTEXT:
            return combine_chunk_scores(
                "predicted_et_only_compression",
                trt_scores=self._trt_scores(chunks),
                tau_g=self.tau_g,
            )
        if self.condition == TEXT_PLUS_EMOTION_ET_CONTEXT:
            return combine_chunk_scores(
                "query_x_predicted_et_compression",
                query_scores=self._query_scores(query, chunks),
                trt_scores=self._trt_scores(chunks),
                alpha=self.alpha,
                beta=self.beta,
                tau_q=self.tau_q,
                tau_g=self.tau_g,
            )
        if self.condition == GAZE_QUERY_ATTENTION_ET:
            return self._gaze_query_attention_scores(context, query, chunks)
        raise ValueError(f"Unsupported scoring condition: {self.condition}")

    def _query_scores(self, query: str, chunks: list[ContextChunk]) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("query scoring requires an encoder.")
        query_vector = encode_query(self.encoder, query, max_length=self.encoder_max_length)
        scores: list[float] = []
        for chunk in chunks:
            encoding = encode_passage_tokens(self.encoder, chunk.text, max_length=self.encoder_max_length)
            mask = encoding.attention_mask.astype(bool)
            hidden = encoding.hidden_states[mask]
            if hidden.size == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(query_vector, normalize_vector(hidden.mean(axis=0)))))
        return np.asarray(scores, dtype=np.float64)

    def _trt_scores(self, chunks: list[ContextChunk]) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("TRT scoring requires a predictor.")
        return predicted_trt_chunk_scores(self.predictor, chunks)

    def _gaze_query_attention_scores(self, context: str, query: str, chunks: list[ContextChunk]) -> np.ndarray:
        if self.encoder is None or self.predictor is None:
            raise RuntimeError("gaze-query attention scoring requires encoder and predictor.")
        encoding = encode_passage_tokens(self.encoder, context, max_length=max(512, self.encoder_max_length))
        if encoding.offset_mapping is None:
            raise RuntimeError("gaze-query attention requires tokenizer offset mappings.")
        predicted_words = self.predictor.predict_words(context)
        token_trt = token_trt_from_predicted_words(context, predicted_words, encoding.offset_mapping)
        gaze = compute_gaze_distribution(token_trt)
        attended = gaze_query_attention(encoding.hidden_states, gaze, mask=encoding.attention_mask.astype(bool))
        query_vector = encode_query(self.encoder, query, max_length=self.encoder_max_length)
        scores: list[float] = []
        for chunk in chunks:
            indices = chunk_token_indices(chunk.char_start, chunk.char_end, encoding.offset_mapping)
            chunk_vector = pool_chunk(attended, indices)
            scores.append(float(np.dot(query_vector, chunk_vector)))
        return combine_chunk_scores("query_only_compression", query_scores=np.asarray(scores), tau_q=self.tau_q)

    def _select_by_budget(self, chunks: list[ContextChunk], scores: np.ndarray) -> tuple[list[ContextChunk], list[float]]:
        word_counts = [max(1, count_words(chunk.text)) for chunk in chunks]
        ranked = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), chunks[index].index))
        selected_indices: list[int] = []
        total_words = 0
        for index in ranked:
            next_total = total_words + word_counts[index]
            if next_total <= self.context_budget_words or not selected_indices:
                selected_indices.append(index)
                total_words = next_total
        selected_indices.sort(key=lambda index: chunks[index].index)
        return [chunks[index] for index in selected_indices], [float(scores[index]) for index in selected_indices]


def maybe_selector(condition: str, **kwargs) -> ContextSelector | None:
    if condition == BASELINE:
        return None
    return ContextSelector(condition=condition, **kwargs)


def token_count_fn_from_tokenizer(tokenizer) -> Callable[[str], int]:
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
