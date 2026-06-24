#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.context_selection import BASELINE, ContextSelector
from core.io_utils import read_json, read_jsonl, write_csv, write_json, write_jsonl
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
    parser.add_argument("--repo-dir", type=Path, default=ROOT / "vendor/repos/EmotionBench-main")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/emotionbench_emotion_et")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--questionnaire", default="PANAS")
    parser.add_argument("--emotion", default="ALL")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--context-budget-words", type=int, default=80)
    parser.add_argument("--chunk-max-words", type=int, default=40)
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def load_situations(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.repo_dir / "data" / "reformatted_situations.jsonl")
    if args.emotion != "ALL":
        rows = [row for row in rows if row.get("emotion") == args.emotion]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    return rows


def format_scale_descriptions(scale: dict[str, str]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in scale.items())


def format_question_list(questions: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in questions.items())


def format_prompt(situation: str, questionnaire: dict[str, Any]) -> str:
    return (
        "## Task\n\n"
        f"Imagine you are the protagonist in the scenario: \"{situation}\".\n\n"
        f"{questionnaire['instruction']}\n"
        f"{format_scale_descriptions(questionnaire['scaling'])}\n\n"
        f"Please score them one by one using the scale from {questionnaire['min_score']} "
        f"to {questionnaire['max_score']}:\n"
        f"{format_question_list(questionnaire['questions'])}\n\n"
        "## Output\n"
        "Please output a JSON format that maps question numbers to their scores, like so:\n"
        "```json\n"
        "{\n"
        '  "1": x,\n'
        '  "2": y,\n'
        "  ...\n"
        "}\n"
        "```\n"
        "Do not include any explanations or additional text."
    )


def parse_json_scores(raw: str, min_score: int, max_score: int) -> dict[str, int]:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: dict[str, int] = {}
    for key, value in parsed.items():
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        scores[str(key)] = min(max(score, min_score), max_score)
    return scores


def compute_category_scores(response: dict[str, int], questionnaire: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    reverse = set(int(item) for item in questionnaire.get("reverse_scaling", []))
    max_score = int(questionnaire["max_score"])
    for category, question_ids in questionnaire["categories"].items():
        values: list[float] = []
        for qid in question_ids:
            key = str(qid)
            if key not in response:
                continue
            value = response[key]
            if int(qid) in reverse:
                value = max_score + 1 - value
            values.append(float(value))
        out[category] = float(sum(values)) if values else 0.0
    if questionnaire.get("compute_mode") == "SUM*2":
        out = {key: value * 2.0 for key, value in out.items()}
    return out


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        parsed = [row for row in group if row["parse_ok"]]
        category_names = sorted({name for row in parsed for name in row["category_scores"]})
        summary_row: dict[str, Any] = {
            "condition": condition,
            "count": len(group),
            "parse_rate": len(parsed) / len(group) if group else 0.0,
            "mean_compression_ratio": float(np.mean([float(row["compression_ratio"]) for row in group])),
        }
        for category in category_names:
            values = [float(row["category_scores"][category]) for row in parsed if category in row["category_scores"]]
            summary_row[f"official_target_mean_{category}"] = float(mean(values)) if values else None
        by_condition.append(summary_row)
    return {
        "prediction_records": len(rows),
        "metric": "official_category_scores_target_only",
        "benchmark_protocol": "EmotionBench target-condition scoring; full official analysis also requires a matched control run",
        "by_condition": by_condition,
    }, by_condition


def main() -> None:
    args = parse_args()
    prediction_path = args.artifacts_dir / "predictions" / "emotionbench_predictions.jsonl"
    if args.overwrite and prediction_path.exists():
        prediction_path.unlink()
    questionnaires = read_json(args.repo_dir / "data" / "questionnaires.json")
    questionnaire = questionnaires[args.questionnaire]
    generator = load_generator(args.model_name, args.device, args.dtype, args.cache_dir)
    selectors = {condition: build_selector(condition, args) for condition in args.conditions}
    situations = load_situations(args)
    total_records = len(args.conditions) * len(situations)
    start_time = time.time()
    print(
        json.dumps(
            {
                "benchmark": "emotionbench",
                "questionnaire": args.questionnaire,
                "situations": len(situations),
                "conditions": args.conditions,
                "total_records": total_records,
                "artifacts_dir": str(args.artifacts_dir),
                "predictions_path": str(prediction_path),
                "flush_every": args.flush_every,
                "metric": "official_category_scores_target_only",
            },
            indent=2,
        ),
        flush=True,
    )
    results: list[dict[str, Any]] = []
    progress = tqdm(
        total=total_records,
        desc="emotionbench",
        unit="record",
        disable=args.disable_progress,
    )
    try:
        for condition in args.conditions:
            selector = selectors[condition]
            progress.write(f"[emotionbench] condition start: {condition} ({len(situations)} situations)")
            condition_start = time.time()
            condition_records = 0
            for row in situations:
                situation = str(row["text"])
                if selector is None:
                    selection = ContextSelector(BASELINE).select(situation, args.questionnaire)
                else:
                    selection = selector.select(situation, args.questionnaire)
                prompt = format_prompt(selection.selected_context, questionnaire)
                raw = generate_text(
                    generator,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_length=args.max_length,
                )
                parsed = parse_json_scores(raw, int(questionnaire["min_score"]), int(questionnaire["max_score"]))
                category_scores = compute_category_scores(parsed, questionnaire)
                out = {
                    "id": f"{row['situation_id']}::{args.questionnaire}::{condition}",
                    "condition": condition,
                    "questionnaire": args.questionnaire,
                    "situation_id": row["situation_id"],
                    "emotion": row["emotion"],
                    "factor": row["factor"],
                    "raw_output": raw,
                    "parsed_scores": parsed,
                    "parse_ok": bool(parsed),
                    "category_scores": category_scores,
                    "original_word_count": selection.original_word_count,
                    "selected_word_count": selection.selected_word_count,
                    "compression_ratio": selection.compression_ratio,
                    "selected_chunk_count": len(selection.selected_chunks),
                }
                results.append(out)
                condition_records += 1
                progress.update(1)
                progress.set_postfix(condition=condition, row=condition_records)
                if len(results) % args.flush_every == 0:
                    write_jsonl(prediction_path, results)
                    progress.write(
                        f"[emotionbench] flushed {len(results)}/{total_records} records -> {prediction_path}"
                    )
                elif args.log_every > 0 and condition_records % args.log_every == 0:
                    progress.write(
                        f"[emotionbench] progress condition={condition} "
                        f"{condition_records}/{len(situations)}; total={len(results)}/{total_records}"
                    )
            elapsed = time.time() - condition_start
            progress.write(f"[emotionbench] condition done: {condition}; elapsed={elapsed:.1f}s")
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
