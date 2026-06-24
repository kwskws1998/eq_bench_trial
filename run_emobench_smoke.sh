#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python emobench/run_emobench_et.py \
  --repo-dir vendor/repos/EmoBench-master \
  --artifacts-dir artifacts/emobench_smoke \
  --model-name "${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M-Instruct}" \
  --device "${DEVICE:-auto}" \
  --dtype "${DTYPE:-auto}" \
  --tasks EA \
  --lang en \
  --max-examples 5 \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend "${PREDICTOR_BACKEND:-heuristic}" \
  --scoring-mode generate \
  --overwrite
