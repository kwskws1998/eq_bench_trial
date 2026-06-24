#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

bash run_emobench_smoke.sh
bash run_eqbench_smoke.sh
bash run_emotionbench_smoke.sh

if [ -n "${EMOTIONQUEEN_DATA_PATH:-}" ]; then
  python emotionqueen/run_emotionqueen_et.py \
    --data-path "$EMOTIONQUEEN_DATA_PATH" \
    --artifacts-dir artifacts/emotionqueen_smoke \
    --model-name "${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M-Instruct}" \
    --device "${DEVICE:-auto}" \
    --dtype "${DTYPE:-auto}" \
    --max-examples 5 \
    --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
    --predictor-backend "${PREDICTOR_BACKEND:-heuristic}" \
    --overwrite
else
  echo "Skipping EmotionQueen smoke because EMOTIONQUEEN_DATA_PATH is not set."
fi
