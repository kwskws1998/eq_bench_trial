from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    index: int


def predicted_trt_chunk_scores(predictor, chunks: list[ContextChunk]) -> np.ndarray:
    scores: list[float] = []
    for chunk in chunks:
        predicted_words = predictor.predict_words(chunk.text)
        trt = np.asarray([max(0.0, row.trt) for row in predicted_words], dtype=np.float64)
        if trt.size == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.mean(np.log1p(trt))))
    return np.asarray(scores, dtype=np.float64)


def combine_chunk_scores(
    condition: str,
    query_scores: np.ndarray | None = None,
    trt_scores: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.25,
    tau_q: float = 0.05,
    tau_g: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    if condition == "query_only_compression":
        if query_scores is None:
            raise ValueError("query_scores are required for query-only compression.")
        return softmax(query_scores, tau_q)
    if condition == "predicted_et_only_compression":
        if trt_scores is None:
            raise ValueError("trt_scores are required for predicted-ET compression.")
        return softmax(trt_scores, tau_g)
    if condition == "query_x_predicted_et_compression":
        if query_scores is None or trt_scores is None:
            raise ValueError("query_scores and trt_scores are required for query x predicted-ET compression.")
        query_distribution = softmax(query_scores, tau_q)
        trt_distribution = softmax(trt_scores, tau_g)
        combined = np.power(query_distribution + eps, alpha) * np.power(trt_distribution + eps, beta)
        total = float(np.sum(combined))
        if total <= 0.0 or not np.isfinite(total):
            return np.full_like(combined, 1.0 / len(combined), dtype=np.float64)
        return combined / total
    raise ValueError(f"Unsupported compression condition: {condition}")


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty 1D array.")
    safe = np.where(np.isfinite(array), array, -np.inf)
    if not np.any(np.isfinite(safe)):
        return np.full(len(array), 1.0 / len(array), dtype=np.float64)
    scaled = safe / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(array), 1.0 / len(array), dtype=np.float64)
    return weights / total
