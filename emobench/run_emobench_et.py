#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser.add_argument("--repo-dir", type=Path, default=ROOT / "vendor/repos/EmoBench-master")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/emobench_emotion_et")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tasks", nargs="+", default=["EU", "EA"], choices=["EU", "EA"])
    parser.add_argument("--lang", default="en", choices=["en", "zh", "all"])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--context-budget-words", type=int, default=120)
    parser.add_argument("--chunk-max-words", type=int, default=45)
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
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(repo_dir: Path, task: str, lang: str, max_examples: int | None) -> list[dict[str, Any]]:
    path = repo_dir / "data" / f"{task}.jsonl"
    rows = [row for row in read_jsonl(path) if lang == "all" or row.get("language") == lang]
    if max_examples is not None:
        rows = rows[:max_examples]
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


def rank_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[index]}) {choice}" for index, choice in enumerate(choices))


def format_eu_prompt(scenario: str, subject: str, emotion_choices: list[str], cause_choices: list[str]) -> str:
    return (
        "Read the scenario and answer both multiple-choice questions.\n\n"
        f"Scenario:\n{scenario}\n\n"
        f"Subject: {subject}\n\n"
        "Question 1: Which emotion is the subject most likely feeling?\n"
        f"{rank_choices(emotion_choices)}\n\n"
        "Question 2: Which cause best explains the emotion?\n"
        f"{rank_choices(cause_choices)}\n\n"
        "Answer with the selected option text."
    )


def format_ea_prompt(scenario: str, subject: str, q_type: str, choices: list[str]) -> str:
    return (
        "Read the scenario and choose the best emotional intelligence response.\n\n"
        f"Scenario:\n{scenario}\n\n"
        f"Subject: {subject}\n"
        f"Question type: {q_type}\n\n"
        f"{rank_choices(choices)}\n\n"
        "Answer with the selected option text."
    )


def score_eu(bundle, row: dict[str, Any], scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = format_eu_prompt(
        scenario,
        str(row["subject"]),
        list(row["emotion_choices"]),
        list(row["cause_choices"]),
    )
    emotion_scores = score_answer_options(
        bundle,
        prompt + "\nQuestion 1 answer: ",
        [str(choice) for choice in row["emotion_choices"]],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    cause_scores = score_answer_options(
        bundle,
        prompt + "\nQuestion 2 answer: ",
        [str(choice) for choice in row["cause_choices"]],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    emotion_pred = int(np.argmax(emotion_scores))
    cause_pred = int(np.argmax(cause_scores))
    emotion_gold = list(row["emotion_choices"]).index(row["emotion_label"])
    cause_gold = list(row["cause_choices"]).index(row["cause_label"])
    return {
        "emotion_pred": LETTERS[emotion_pred],
        "emotion_gold": LETTERS[emotion_gold],
        "cause_pred": LETTERS[cause_pred],
        "cause_gold": LETTERS[cause_gold],
        "correct": emotion_pred == emotion_gold and cause_pred == cause_gold,
        "emotion_scores": emotion_scores.tolist(),
        "cause_scores": cause_scores.tolist(),
    }


def score_ea(bundle, row: dict[str, Any], scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = format_ea_prompt(
        scenario,
        str(row["subject"]),
        str(row["question type"]),
        [str(choice) for choice in row["choices"]],
    )
    scores = score_answer_options(
        bundle,
        prompt + "\nAnswer: ",
        [str(choice) for choice in row["choices"]],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    pred = int(np.argmax(scores))
    gold = list(row["choices"]).index(row["label"])
    return {
        "pred": LETTERS[pred],
        "gold": LETTERS[gold],
        "correct": pred == gold,
        "scores": scores.tolist(),
    }


def result_metadata(selection) -> dict[str, Any]:
    return {
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


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["task"]), str(row["language"]))
        by_key.setdefault(key, []).append(row)
    by_condition: list[dict[str, Any]] = []
    for (condition, task, language), group in sorted(by_key.items()):
        by_condition.append(
            {
                "condition": condition,
                "task": task,
                "language": language,
                "count": len(group),
                "accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in group])),
                "mean_compression_ratio": float(np.mean([float(row["compression_ratio"]) for row in group])),
            }
        )
    summary = {
        "prediction_records": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "by_condition": by_condition,
    }
    return summary, by_condition


def main() -> None:
    args = parse_args()
    prediction_path = args.artifacts_dir / "predictions" / "emobench_predictions.jsonl"
    if args.overwrite and prediction_path.exists():
        prediction_path.unlink()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    generator = load_generator(args.model_name, args.device, args.dtype, args.cache_dir)
    selectors = {condition: build_selector(condition, args) for condition in args.conditions}

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for condition in args.conditions:
        selector = selectors[condition]
        for task in args.tasks:
            rows = load_rows(args.repo_dir, task, args.lang, args.max_examples)
            for row in rows:
                scenario = str(row["scenario"])
                query = scenario
                if task == "EA":
                    query = f"{row.get('question type', '')} {row.get('subject', '')}"
                selection = selector.select(scenario, query) if selector is not None else ContextSelector(BASELINE).select(scenario, query)
                if task == "EU":
                    scored = score_eu(generator, row, selection.selected_context, args)
                else:
                    scored = score_ea(generator, row, selection.selected_context, args)
                out = {
                    "id": f"{task}:{row['language']}:{row['qid']}::{condition}",
                    "condition": condition,
                    "task": task,
                    "language": row["language"],
                    "qid": row["qid"],
                    "category": row.get("coarse_category") or row.get("category"),
                    **result_metadata(selection),
                    **scored,
                }
                results.append(out)
                pending.append(out)
                if len(pending) >= args.flush_every:
                    write_jsonl(prediction_path, results)
                    pending.clear()
    write_jsonl(prediction_path, results)
    summary, by_condition = summarize(results)
    write_json(args.artifacts_dir / "results" / "summary.json", summary)
    write_csv(args.artifacts_dir / "results" / "by_condition.csv", by_condition)
    print(summary)


if __name__ == "__main__":
    main()
