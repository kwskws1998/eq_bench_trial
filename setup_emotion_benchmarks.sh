#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [ "${DOWNLOAD_REPOS:-1}" = "1" ]; then
  bash download_official_repos.sh
fi

if [ "${DOWNLOAD_HF:-1}" = "1" ]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="skboy/emotion_et_2nd_model",
    allow_patterns=[
        "model.py",
        "hf_emotion_et_aug_lr2e-5_len256_seed123/*",
    ],
)
hf_hub_download("intfloat/e5-large-v2", "config.json")
print("HF assets cached")
PY
fi

if [ "${RUN_SMOKE:-1}" = "1" ]; then
  python -m py_compile \
    core/*.py \
    emobench/*.py \
    eqbench/*.py \
    emotionbench/*.py \
    emotionqueen/*.py
fi
