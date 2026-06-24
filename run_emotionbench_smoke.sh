#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python emotionbench/run_emotionbench_et.py \
  --repo-dir vendor/repos/EmotionBench-main \
  --artifacts-dir artifacts/emotionbench_smoke \
  --model-name "${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M-Instruct}" \
  --device "${DEVICE:-auto}" \
  --dtype "${DTYPE:-auto}" \
  --questionnaire PANAS \
  --emotion Anger \
  --max-examples 3 \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend "${PREDICTOR_BACKEND:-heuristic}" \
  --overwrite
