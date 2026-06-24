#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python emobench/run_emobench_et.py \
  --repo-dir vendor/repos/EmoBench-master \
  --artifacts-dir "${ARTIFACTS_DIR:-artifacts/emobench_llama3}" \
  --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float16}" \
  --tasks EU EA \
  --lang en \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --scoring-mode generate \
  --overwrite
