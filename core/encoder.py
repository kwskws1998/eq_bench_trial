from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@dataclass(frozen=True)
class EncoderBundle:
    tokenizer: object
    model: object
    device: str
    hidden_size: int


@dataclass(frozen=True)
class TokenEncoding:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    tokens: list[str]
    offset_mapping: list[tuple[int, int]] | None
    hidden_states: np.ndarray


def resolve_torch_device(device: str):
    import torch

    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            try:
                torch.empty(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def load_e5_encoder(
    model_name: str = "intfloat/e5-large-v2",
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> EncoderBundle:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)
    hidden_size = int(getattr(model.config, "hidden_size"))
    return EncoderBundle(
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
        hidden_size=hidden_size,
    )


def encode_query(bundle: EncoderBundle, question: str, max_length: int = 128) -> np.ndarray:
    import torch

    encoded = bundle.tokenizer(
        QUERY_PREFIX + question,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = bundle.model(**encoded)
    pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return normalize_vector(pooled[0].detach().cpu().numpy())


def encode_passage_tokens(bundle: EncoderBundle, text: str, max_length: int = 512) -> TokenEncoding:
    import torch

    prefixed = PASSAGE_PREFIX + text
    try:
        encoded = bundle.tokenizer(
            prefixed,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        raw_offsets = encoded.pop("offset_mapping")[0].tolist()
        offsets = adjust_offsets(
            [(int(start), int(end)) for start, end in raw_offsets],
            len(PASSAGE_PREFIX),
            len(text),
        )
    except (NotImplementedError, TypeError):
        encoded = bundle.tokenizer(
            prefixed,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        offsets = None

    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = bundle.model(**encoded)
    hidden = output.last_hidden_state[0].detach().cpu().numpy()
    input_ids = encoded["input_ids"][0].detach().cpu().numpy()
    attention_mask = encoded["attention_mask"][0].detach().cpu().numpy()
    tokens = bundle.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return TokenEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokens=tokens,
        offset_mapping=offsets,
        hidden_states=hidden,
    )


def adjust_offsets(
    offsets: list[tuple[int, int]],
    prefix_len: int,
    original_text_len: int,
) -> list[tuple[int, int]]:
    adjusted: list[tuple[int, int]] = []
    for start, end in offsets:
        if start == end or end <= prefix_len:
            adjusted.append((-1, -1))
            continue
        shifted_start = max(0, start - prefix_len)
        shifted_end = min(original_text_len, end - prefix_len)
        if shifted_start >= shifted_end:
            adjusted.append((-1, -1))
        else:
            adjusted.append((shifted_start, shifted_end))
    return adjusted
