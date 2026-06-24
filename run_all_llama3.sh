#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-float16}"

python emobench/run_emobench_et.py \
  --repo-dir vendor/repos/EmoBench-master \
  --artifacts-dir artifacts/emobench_llama3 \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --tasks EU EA \
  --lang en \
  --context-budget-words "${EMOBENCH_BUDGET_WORDS:-80}" \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite

python eqbench/run_eqbench_et.py \
  --repo-dir vendor/repos/EQ-Bench-main_v2_4 \
  --artifacts-dir artifacts/eqbench_llama3 \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --context-budget-words "${EQBENCH_BUDGET_WORDS:-180}" \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite

python emotionbench/run_emotionbench_et.py \
  --repo-dir vendor/repos/EmotionBench-main \
  --artifacts-dir artifacts/emotionbench_llama3 \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --questionnaire PANAS \
  --emotion ALL \
  --context-budget-words "${EMOTIONBENCH_BUDGET_WORDS:-60}" \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite

if [ -n "${EMOTIONQUEEN_DATA_PATH:-}" ]; then
  python emotionqueen/run_emotionqueen_et.py \
    --data-path "$EMOTIONQUEEN_DATA_PATH" \
    --artifacts-dir artifacts/emotionqueen_llama3 \
    --model-name "$MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --context-budget-words "${EMOTIONQUEEN_BUDGET_WORDS:-160}" \
    --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
    --predictor-backend skboy \
    --overwrite
else
  echo "Skipping EmotionQueen because EMOTIONQUEEN_DATA_PATH is not set."
fi
