#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.context_selection import BASELINE, ContextSelector
from core.io_utils import read_jsonl, write_csv, write_json, write_jsonl
from core.llm_scoring import generate_text, load_generator, score_answer_options


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
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scoring-mode", choices=["generate", "loglik"], default="generate")
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--disable-progress", action="store_true")
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


def official_response_format(task: str) -> str:
    if task == "EA":
        conditions = '"answer": "<Respond with the corresponding letter numbering>"'
    elif task == "EU":
        conditions = (
            '"answer_q1": "<Respond to the Question 1 with the corresponding letter numbering>",\n'
            '    "answer_q2": "<Respond to the Question 2 with the corresponding letter numbering>"'
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return (
        "Provide only one single correct answer to this question. "
        "Do not provide any additional information or explanations.\n"
        "The response should be in the following JSON format:\n"
        "```json\n"
        "    {\n"
        f"    {conditions}\n"
        "    }\n"
        "```"
    )


def official_system_prompt(task: str) -> str:
    return (
        "# Instructions\n\n"
        "In this task, you are presented with a scenario, a question, and multiple choices.\n"
        "Carefully analyze the scenario and take the perspective of the individual involved.\n"
        "Then, select the option that best reflects their perspective or emotional response.\n\n"
        "# Output\n"
        f"{official_response_format(task)}"
    )


def format_eu_prompt(scenario: str, subject: str, emotion_choices: list[str], cause_choices: list[str]) -> str:
    return (
        f"{official_system_prompt('EU')}\n\n"
        "## Scenario\n"
        f"{scenario}\n\n"
        "## Question 1\n"
        f"What emotion(s) would {subject} ultimately feel in this situation?\n\n"
        "## Choices for Question 1\n"
        f"{rank_choices(emotion_choices)}\n\n"
        "## Question 2\n"
        f"Why would {subject} feel these emotions in this situation?\n\n"
        "## Choices for Question 2\n"
        f"{rank_choices(cause_choices)}\n"
    )


def format_ea_prompt(scenario: str, subject: str, q_type: str, choices: list[str]) -> str:
    return (
        f"{official_system_prompt('EA')}\n\n"
        "## Scenario\n"
        f"{scenario}\n\n"
        "## Question\n"
        f"In this scenario, what is the most effective {q_type} for {subject}?\n\n"
        "## Choices\n"
        f"{rank_choices(choices)}\n"
    )


def parse_json_response(raw: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_letter(value: Any, num_choices: int) -> str:
    text = str(value).strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    if not match:
        return ""
    letter = match.group(1)
    return letter if letter in LETTERS[:num_choices] else ""


def score_eu(bundle, row: dict[str, Any], scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = format_eu_prompt(
        scenario,
        str(row["subject"]),
        list(row["emotion_choices"]),
        list(row["cause_choices"]),
    )
    emotion_gold = list(row["emotion_choices"]).index(row["emotion_label"])
    cause_gold = list(row["cause_choices"]).index(row["cause_label"])
    if args.scoring_mode == "generate":
        raw = generate_text(
            bundle,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_length=args.max_length,
        )
        parsed = parse_json_response(raw) or {}
        emotion_answer = normalize_letter(parsed.get("answer_q1", ""), len(row["emotion_choices"]))
        cause_answer = normalize_letter(parsed.get("answer_q2", ""), len(row["cause_choices"]))
        return {
            "emotion_pred": emotion_answer,
            "emotion_gold": LETTERS[emotion_gold],
            "cause_pred": cause_answer,
            "cause_gold": LETTERS[cause_gold],
            "correct": emotion_answer == LETTERS[emotion_gold] and cause_answer == LETTERS[cause_gold],
            "raw_output": raw,
            "parse_ok": bool(emotion_answer and cause_answer),
            "scoring_mode": args.scoring_mode,
        }
    emotion_scores = score_answer_options(
        bundle,
        prompt + "\nQuestion 1 answer: ",
        [LETTERS[index] for index in range(len(row["emotion_choices"]))],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    cause_scores = score_answer_options(
        bundle,
        prompt + "\nQuestion 2 answer: ",
        [LETTERS[index] for index in range(len(row["cause_choices"]))],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    emotion_pred = int(np.argmax(emotion_scores))
    cause_pred = int(np.argmax(cause_scores))
    return {
        "emotion_pred": LETTERS[emotion_pred],
        "emotion_gold": LETTERS[emotion_gold],
        "cause_pred": LETTERS[cause_pred],
        "cause_gold": LETTERS[cause_gold],
        "correct": emotion_pred == emotion_gold and cause_pred == cause_gold,
        "emotion_scores": emotion_scores.tolist(),
        "cause_scores": cause_scores.tolist(),
        "parse_ok": True,
        "scoring_mode": args.scoring_mode,
    }


def score_ea(bundle, row: dict[str, Any], scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = format_ea_prompt(
        scenario,
        str(row["subject"]),
        str(row["question type"]),
        [str(choice) for choice in row["choices"]],
    )
    gold = list(row["choices"]).index(row["label"])
    if args.scoring_mode == "generate":
        raw = generate_text(
            bundle,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_length=args.max_length,
        )
        parsed = parse_json_response(raw) or {}
        answer = normalize_letter(parsed.get("answer", ""), len(row["choices"]))
        return {
            "pred": answer,
            "gold": LETTERS[gold],
            "correct": answer == LETTERS[gold],
            "raw_output": raw,
            "parse_ok": bool(answer),
            "scoring_mode": args.scoring_mode,
        }
    scores = score_answer_options(
        bundle,
        prompt + "\nAnswer: ",
        [LETTERS[index] for index in range(len(row["choices"]))],
        max_length=args.max_length,
        normalization=args.score_normalization,
    )
    pred = int(np.argmax(scores))
    return {
        "pred": LETTERS[pred],
        "gold": LETTERS[gold],
        "correct": pred == gold,
        "scores": scores.tolist(),
        "parse_ok": True,
        "scoring_mode": args.scoring_mode,
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
                "official_accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in group])),
                "parse_rate": float(np.mean([1.0 if row.get("parse_ok", True) else 0.0 for row in group])),
                "mean_compression_ratio": float(np.mean([float(row["compression_ratio"]) for row in group])),
            }
        )
    summary = {
        "prediction_records": len(rows),
        "metric": "official_accuracy",
        "scoring_protocol": rows[0].get("scoring_mode", "unknown") if rows else None,
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
    rows_by_task = {task: load_rows(args.repo_dir, task, args.lang, args.max_examples) for task in args.tasks}
    total_rows = sum(len(rows) for rows in rows_by_task.values())
    total_records = len(args.conditions) * total_rows
    start_time = time.time()
    print(
        json.dumps(
            {
                "benchmark": "emobench",
                "tasks": args.tasks,
                "rows_by_task": {task: len(rows) for task, rows in rows_by_task.items()},
                "conditions": args.conditions,
                "total_records": total_records,
                "artifacts_dir": str(args.artifacts_dir),
                "predictions_path": str(prediction_path),
                "flush_every": args.flush_every,
                "metric": "official_accuracy",
                "scoring_mode": args.scoring_mode,
            },
            indent=2,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    progress = tqdm(
        total=total_records,
        desc="emobench",
        unit="record",
        disable=args.disable_progress,
    )
    try:
        for condition in args.conditions:
            selector = selectors[condition]
            progress.write(f"[emobench] condition start: {condition} ({total_rows} rows)")
            condition_start = time.time()
            condition_records = 0
            for task in args.tasks:
                rows = rows_by_task[task]
                progress.write(f"[emobench] task start: {task} ({len(rows)} rows)")
                for row in rows:
                    scenario = str(row["scenario"])
                    query = scenario
                    if task == "EA":
                        query = f"{row.get('question type', '')} {row.get('subject', '')}"
                    selection = (
                        selector.select(scenario, query)
                        if selector is not None
                        else ContextSelector(BASELINE).select(scenario, query)
                    )
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
                    condition_records += 1
                    progress.update(1)
                    progress.set_postfix(condition=condition, task=task)
                    if len(pending) >= args.flush_every:
                        write_jsonl(prediction_path, results)
                        pending.clear()
                        progress.write(
                            f"[emobench] flushed {len(results)}/{total_records} records -> {prediction_path}"
                        )
                    elif args.log_every > 0 and condition_records % args.log_every == 0:
                        progress.write(
                            f"[emobench] progress condition={condition} "
                            f"{condition_records}/{total_rows}; total={len(results)}/{total_records}"
                        )
            elapsed = time.time() - condition_start
            progress.write(f"[emobench] condition done: {condition}; elapsed={elapsed:.1f}s")
    finally:
        progress.close()
    write_jsonl(prediction_path, results)
    summary, by_condition = summarize(results)
    summary["elapsed_seconds"] = time.time() - start_time
    write_json(args.artifacts_dir / "results" / "summary.json", summary)
    write_csv(args.artifacts_dir / "results" / "by_condition.csv", by_condition)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
