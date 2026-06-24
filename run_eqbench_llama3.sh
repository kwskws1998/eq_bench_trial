#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python eqbench/run_eqbench_et.py \
  --repo-dir vendor/repos/EQ-Bench-main_v2_4 \
  --artifacts-dir "${ARTIFACTS_DIR:-artifacts/eqbench_llama3_budget128}" \
  --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float16}" \
  --context-budget-words "${EQBENCH_BUDGET_WORDS:-128}" \
  --chunk-max-words "${EQBENCH_CHUNK_MAX_WORDS:-60}" \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --predictor-model-name skboy/emotion_et_2nd_model \
  --predictor-weights et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors \
  --predictor-subfolder hf_emotion_et_aug_lr2e-5_len256_seed123 \
  --max-length "${MAX_LENGTH:-4096}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-320}" \
  --temperature 0.0 \
  --overwrite
