#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.context_selection import BASELINE, ContextSelector
from core.io_utils import read_jsonl, write_csv, write_json, write_jsonl
from core.llm_scoring import load_generator, score_answer_options


LETTERS = string.ascii_uppercase
DEFAULT_CONDITIONS = [
    BASELINE,
    "text_context",
    "emotion_et_context",
    "text_plus_emotion_et_context",
    "gaze_query_attention_et",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/emotionqueen_emotion_et")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--context-budget-words", type=int, default=160)
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
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        rows = obj if isinstance(obj, list) else list(obj.values())
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise ValueError(f"Could not infer any of these fields: {names}")


def infer_example(row: dict[str, Any]) -> tuple[str, str, list[str], str]:
    context = str(first_present(row, ["context", "scenario", "statement", "text", "passage"]))
    question = str(first_present(row, ["question", "query", "prompt", "task"]))
    choices = first_present(row, ["choices", "options", "answers", "candidates"])
    if isinstance(choices, dict):
        choices = list(choices.values())
    choices = [str(choice) for choice in choices]
    answer = first_present(row, ["answer", "label", "gold", "target", "correct_answer"])
    if isinstance(answer, int):
        gold = choices[answer]
    else:
        answer_str = str(answer)
        if len(answer_str) == 1 and answer_str.upper() in LETTERS[: len(choices)]:
            gold = choices[LETTERS.index(answer_str.upper())]
        else:
            gold = answer_str
    return context, question, choices, gold


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


def format_prompt(context: str, question: str, choices: list[str]) -> str:
    choice_block = "\n".join(f"{LETTERS[index]}) {choice}" for index, choice in enumerate(choices))
    return (
        "Read the emotional scenario and answer the multiple-choice question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"{choice_block}\n\n"
        "Answer with the selected option text.\nAnswer: "
    )


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "count": len(group),
                "accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in group])),
                "mean_compression_ratio": float(np.mean([float(row["compression_ratio"]) for row in group])),
            }
        )
    return {"prediction_records": len(rows), "by_condition": by_condition}, by_condition


def main() -> None:
    args = parse_args()
    prediction_path = args.artifacts_dir / "predictions" / "emotionqueen_predictions.jsonl"
    if args.overwrite and prediction_path.exists():
        prediction_path.unlink()
    rows = load_rows(args.data_path, args.max_examples)
    generator = load_generator(args.model_name, args.device, args.dtype, args.cache_dir)
    selectors = {condition: build_selector(condition, args) for condition in args.conditions}
    results: list[dict[str, Any]] = []
    for condition in args.conditions:
        selector = selectors[condition]
        for index, row in enumerate(rows):
            context, question, choices, gold = infer_example(row)
            if selector is None:
                selection = ContextSelector(BASELINE).select(context, question)
            else:
                selection = selector.select(context, question)
            scores = score_answer_options(
                generator,
                format_prompt(selection.selected_context, question, choices),
                choices,
                max_length=args.max_length,
                normalization=args.score_normalization,
            )
            pred_index = int(np.argmax(scores))
            pred = choices[pred_index]
            results.append(
                {
                    "id": f"emotionqueen:{index}::{condition}",
                    "condition": condition,
                    "row_index": index,
                    "pred": pred,
                    "gold": gold,
                    "correct": pred == gold,
                    "scores": scores.tolist(),
                    "original_word_count": selection.original_word_count,
                    "selected_word_count": selection.selected_word_count,
                    "compression_ratio": selection.compression_ratio,
                    "selected_chunk_count": len(selection.selected_chunks),
                }
            )
    write_jsonl(prediction_path, results)
    summary, by_condition = summarize(results)
    write_json(args.artifacts_dir / "results" / "summary.json", summary)
    write_csv(args.artifacts_dir / "results" / "by_condition.csv", by_condition)
    print(summary)


if __name__ == "__main__":
    main()
