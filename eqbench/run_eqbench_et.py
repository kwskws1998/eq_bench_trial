#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.context_selection import BASELINE, ContextSelector
from core.io_utils import write_csv, write_json, write_jsonl
from core.llm_scoring import generate_text, load_generator


DEFAULT_CONDITIONS = [
    BASELINE,
    "text_context",
    "emotion_et_context",
    "text_plus_emotion_et_context",
    "gaze_query_attention_et",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=ROOT / "vendor/repos/EQ-Bench-main_v2_4")
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/eqbench_emotion_et")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--context-budget-words", type=int, default=180)
    parser.add_argument("--chunk-max-words", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--tau-q", type=float, default=0.05)
    parser.add_argument("--tau-g", type=float, default=0.1)
    parser.add_argument("--encoder-name", default="intfloat/e5-large-v2")
    parser.add_argument("--encoder-device", default=None)
    parser.add_argument("--predictor-backend", choices=["skboy", "heuristic"], default="skboy")
    parser.add_argument("--predictor-model-name", default="skboy/emotion_et_2nd_model")
    parser.add_argument("--predictor-weights", default="et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors")
    parser.add_argument("--predictor-subfolder", default="hf_emotion_et_aug_lr2e-5_len256_seed123")
    parser.add_argument("--predictor-local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.questions_path or args.repo_dir / "data" / "eq_bench_v2_questions_171.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    return rows


def build_selector(condition: str, args: argparse.Namespace) -> ContextSelector | None:
    if condition == BASELINE:
        return None
    return ContextSelector(
        condition=condition,
        context_budget_words=args.context_budget_words,
        chunk_max_words=args.chunk_max_words,
        alpha=args.alpha,
        beta=args.beta,
        tau_q=args.tau_q,
        tau_g=args.tau_g,
        encoder_name=args.encoder_name,
        encoder_device=args.encoder_device or args.device,
        predictor_backend=args.predictor_backend,
        predictor_model_name=args.predictor_model_name,
        predictor_weights=args.predictor_weights,
        predictor_subfolder=args.predictor_subfolder,
        predictor_local_files_only=args.predictor_local_files_only,
        cache_dir=args.cache_dir,
    )


def extract_dialogue(prompt: str) -> tuple[str, str, str] | None:
    match = re.search(r"(Your task is to predict.*?dialogue:\s*\n\n)(.*?)(\n\[End dialogue\].*)", prompt, re.S)
    if not match:
        return None
    return match.group(1), match.group(2).strip(), match.group(3)


def build_prompt_variant(prompt: str, selector: ContextSelector | None) -> tuple[str, dict[str, Any]]:
    extracted = extract_dialogue(prompt)
    if extracted is None or selector is None:
        return prompt, {
            "original_word_count": len(prompt.split()),
            "selected_word_count": len(prompt.split()),
            "compression_ratio": 1.0,
            "selected_chunk_count": None,
            "selected_chunks": [],
        }
    prefix, dialogue, suffix = extracted
    selection = selector.select(dialogue, suffix)
    prompt_variant = prefix + selection.selected_context + suffix
    return prompt_variant, {
        "original_word_count": selection.original_word_count,
        "selected_word_count": selection.selected_word_count,
        "compression_ratio": selection.compression_ratio,
        "selected_chunk_count": len(selection.selected_chunks),
        "selected_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "score": score,
                "text": chunk.text,
            }
            for chunk, score in zip(selection.selected_chunks, selection.selected_scores)
        ],
    }


def reference_scores(row: dict[str, Any]) -> dict[str, float]:
    ref = row.get("reference_answer_fullscale") or row.get("reference_answer")
    scores: dict[str, float] = {}
    for index in range(1, 5):
        name = str(ref[f"emotion{index}"])
        scores[name] = float(ref[f"emotion{index}_score"])
    return scores


def parse_scores(output: str, emotions: list[str]) -> dict[str, float]:
    section = output
    match = re.search(r"Revised scores:\s*(.*)", output, re.S | re.I)
    if match:
        section = match.group(1)
    parsed: dict[str, float] = {}
    for emotion in emotions:
        pattern = re.escape(emotion) + r"\s*:\s*(-?\d+(?:\.\d+)?)"
        value_match = re.search(pattern, section, re.I)
        if value_match:
            parsed[emotion] = float(value_match.group(1))
    return parsed


def score_prediction(predicted: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    missing = [emotion for emotion in reference if emotion not in predicted]
    if missing:
        return {
            "parse_ok": False,
            "missing_emotions": missing,
            "mae": None,
            "rmse": None,
            "correct": False,
        }
    diffs = np.asarray([predicted[emotion] - reference[emotion] for emotion in reference], dtype=np.float64)
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs * diffs)))
    return {
        "parse_ok": True,
        "missing_emotions": [],
        "mae": mae,
        "rmse": rmse,
        "correct": mae <= 2.0,
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        parsed = [row for row in group if row["parse_ok"]]
        by_condition.append(
            {
                "condition": condition,
                "count": len(group),
                "parse_rate": len(parsed) / len(group) if group else 0.0,
                "mae": float(np.mean([row["mae"] for row in parsed])) if parsed else None,
                "rmse": float(np.mean([row["rmse"] for row in parsed])) if parsed else None,
                "accuracy_mae_le_2": float(np.mean([1.0 if row["correct"] else 0.0 for row in group])),
                "mean_compression_ratio": float(np.mean([float(row["compression_ratio"]) for row in group])),
            }
        )
    return {"prediction_records": len(rows), "by_condition": by_condition}, by_condition


def main() -> None:
    args = parse_args()
    prediction_path = args.artifacts_dir / "predictions" / "eqbench_predictions.jsonl"
    if args.overwrite and prediction_path.exists():
        prediction_path.unlink()
    generator = load_generator(args.model_name, args.device, args.dtype, args.cache_dir)
    questions = load_questions(args)
    selectors = {condition: build_selector(condition, args) for condition in args.conditions}
    results: list[dict[str, Any]] = []
    for condition in args.conditions:
        selector = selectors[condition]
        for index, row in enumerate(questions):
            prompt, metadata = build_prompt_variant(str(row["prompt"]), selector)
            raw = generate_text(
                generator,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_length=args.max_length,
            )
            ref = reference_scores(row)
            pred = parse_scores(raw, list(ref.keys()))
            scored = score_prediction(pred, ref)
            out = {
                "id": f"eqbench:{index}::{condition}",
                "condition": condition,
                "row_index": index,
                "raw_output": raw,
                "reference_scores": ref,
                "predicted_scores": pred,
                **metadata,
                **scored,
            }
            results.append(out)
            if len(results) % args.flush_every == 0:
                write_jsonl(prediction_path, results)
    write_jsonl(prediction_path, results)
    summary, by_condition = summarize(results)
    write_json(args.artifacts_dir / "results" / "summary.json", summary)
    write_csv(args.artifacts_dir / "results" / "by_condition.csv", by_condition)
    print(summary)


if __name__ == "__main__":
    main()
