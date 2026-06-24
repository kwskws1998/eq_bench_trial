from __future__ import annotations

import math
import re

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def compute_gaze_distribution(token_trt: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    trt = np.asarray(token_trt, dtype=np.float64)
    trt = np.nan_to_num(trt, nan=0.0, posinf=0.0, neginf=0.0)
    trt = np.maximum(trt, 0.0)
    weights = np.log1p(trt) + eps
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(trt.shape, 1.0 / len(trt), dtype=np.float64)
    return weights / total


def gaze_query_attention(hidden: np.ndarray, gaze: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hidden_array = np.asarray(hidden, dtype=np.float64)
    if hidden_array.ndim != 2 or hidden_array.shape[0] == 0:
        raise ValueError("hidden must have shape (tokens, dimensions).")
    gaze_array = np.asarray(gaze, dtype=np.float64)
    if gaze_array.shape != (hidden_array.shape[0],):
        raise ValueError("gaze must have shape (tokens,).")
    query = gaze_array[:, None] * hidden_array
    scores = query @ hidden_array.T / math.sqrt(hidden_array.shape[1])
    valid_mask = None
    if mask is not None:
        valid_mask = np.asarray(mask).astype(bool)
        if valid_mask.shape != (hidden_array.shape[0],):
            raise ValueError("mask must have shape (tokens,).")
        scores[:, ~valid_mask] = -1e30
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    if valid_mask is not None:
        exp_scores[:, ~valid_mask] = 0.0
    denom = exp_scores.sum(axis=1, keepdims=True)
    probs = np.divide(exp_scores, denom, out=np.zeros_like(exp_scores), where=denom > 0.0)
    attended = probs @ hidden_array
    if valid_mask is not None:
        attended[~valid_mask] = 0.0
    return attended


def token_trt_from_predicted_words(text: str, predicted_words, token_offsets: list[tuple[int, int]]) -> np.ndarray:
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    trt_values = [max(0.0, float(row.trt)) for row in predicted_words]
    token_trt = np.zeros(len(token_offsets), dtype=np.float64)
    for (word_start, word_end), trt in zip(spans, trt_values):
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_start >= 0 and token_end > word_start and token_start < word_end
        ]
        if not token_indices:
            continue
        share = trt / len(token_indices)
        for token_index in token_indices:
            token_trt[token_index] += share
    return token_trt


def chunk_token_indices(chunk_start: int, chunk_end: int, token_offsets: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for index, (token_start, token_end) in enumerate(token_offsets):
        if token_start >= 0 and token_end > chunk_start and token_start < chunk_end:
            indices.append(index)
    return indices


def pool_chunk(hidden: np.ndarray, token_indices: list[int]) -> np.ndarray:
    if not token_indices:
        return np.zeros(hidden.shape[1], dtype=np.float64)
    return normalize_vector(hidden[np.asarray(token_indices, dtype=np.int64)].mean(axis=0))
