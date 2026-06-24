# Emotion ET Benchmarks

Standalone harness for testing whether an emotion-specific ET predictor helps emotion benchmarks through context selection or predicted-gaze attention.

This directory is intentionally independent from `src/gaze_query_rag`. Core code was copied into `emotion_et_benchmarks/core` so the folder can be moved to a GPU machine as a self-contained experiment.

## Included Official Repos

The setup script downloads these official repositories as zip archives into `vendor/zips` and extracts them into `vendor/repos`.

- EmoBench: `https://github.com/Sahandfer/EmoBench`
- EQ-Bench: `https://github.com/EQ-bench/EQ-Bench`
- EmotionBench: `https://github.com/CUHK-ARISE/EmotionBench`

EmotionQueen does not have a pinned official repo in this harness yet. Use `emotionqueen/run_emotionqueen_et.py --data-path ...` with a local JSON or JSONL dataset.

## Conditions

- `baseline`: original context.
- `text_context`: E5 query-context relevance selects context chunks.
- `emotion_et_context`: `skboy/emotion_et_2nd_model` predicted TRT salience selects context chunks.
- `text_plus_emotion_et_context`: E5 relevance and predicted TRT salience are combined.
- `gaze_query_attention_et`: predicted TRT is converted into a token gaze distribution and used in the gaze-query attention equation before chunk scoring.

## Setup

```bash
cd /workspace/gaze_query_rag_project/emotion_et_benchmarks

export HF_TOKEN="your_hf_token"
huggingface-cli login --token "$HF_TOKEN"

DOWNLOAD_REPOS=1 DOWNLOAD_HF=1 RUN_SMOKE=1 bash setup_emotion_benchmarks.sh
```

For a local quick setup without downloading HF assets:

```bash
DOWNLOAD_REPOS=1 DOWNLOAD_HF=0 RUN_SMOKE=1 bash setup_emotion_benchmarks.sh
```

## Smoke Runs

Fast CPU/GPU smoke with heuristic ET predictor:

```bash
bash run_all_smoke.sh
```

Or run each benchmark separately:

```bash
bash run_emobench_smoke.sh
bash run_eqbench_smoke.sh
bash run_emotionbench_smoke.sh
```

EmoBench with Llama 3 and the real emotion ET predictor:

```bash
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export DEVICE=cuda
export DTYPE=float16
bash run_emobench_llama3.sh
```

Run all available Llama 3 adapters:

```bash
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export DEVICE=cuda
export DTYPE=float16
bash run_all_llama3.sh
```

Set `EMOTIONQUEEN_DATA_PATH=/path/to/data.jsonl` before `run_all_smoke.sh` or `run_all_llama3.sh` if you want the generic EmotionQueen adapter included.

## Direct Commands

EmoBench:

```bash
python emobench/run_emobench_et.py \
  --repo-dir vendor/repos/EmoBench-master \
  --artifacts-dir artifacts/emobench_llama3 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --dtype float16 \
  --tasks EU EA \
  --lang en \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite
```

EQ-Bench:

```bash
python eqbench/run_eqbench_et.py \
  --repo-dir vendor/repos/EQ-Bench-main_v2_4 \
  --artifacts-dir artifacts/eqbench_llama3 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --dtype float16 \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite
```

EmotionBench:

```bash
python emotionbench/run_emotionbench_et.py \
  --repo-dir vendor/repos/EmotionBench-main \
  --artifacts-dir artifacts/emotionbench_llama3 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --dtype float16 \
  --questionnaire PANAS \
  --emotion ALL \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite
```

EmotionQueen-style JSON/JSONL:

```bash
python emotionqueen/run_emotionqueen_et.py \
  --data-path /path/to/emotionqueen.jsonl \
  --artifacts-dir artifacts/emotionqueen_llama3 \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --dtype float16 \
  --conditions baseline text_context emotion_et_context text_plus_emotion_et_context gaze_query_attention_et \
  --predictor-backend skboy \
  --overwrite
```

## Outputs

Each runner writes:

- `artifacts/<run>/predictions/*.jsonl`
- `artifacts/<run>/results/summary.json`
- `artifacts/<run>/results/by_condition.csv`

Bundle outputs for download:

```bash
bash zip_outputs.sh emotion_et_outputs.zip
```
