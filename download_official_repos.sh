#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_DIR="$ROOT/vendor/zips"
REPO_DIR="$ROOT/vendor/repos"

mkdir -p "$ZIP_DIR" "$REPO_DIR"

download_zip() {
  local name="$1"
  local url="$2"
  local out="$ZIP_DIR/${name}.zip"
  echo "Downloading $name"
  curl -L -s -o "$out.tmp" "$url"
  python -m zipfile -t "$out.tmp" >/dev/null
  mv "$out.tmp" "$out"
  python -m zipfile -e "$out" "$REPO_DIR"
}

download_zip "EmoBench" "https://github.com/Sahandfer/EmoBench/archive/refs/heads/master.zip"
download_zip "EQ-Bench" "https://github.com/EQ-bench/EQ-Bench/archive/refs/heads/master.zip"
download_zip "EmotionBench" "https://github.com/CUHK-ARISE/EmotionBench/archive/refs/heads/main.zip"

echo "EmotionQueen official code repo was not pinned here; use emotionqueen/run_emotionqueen_et.py with --data-path."
find "$REPO_DIR" -maxdepth 1 -type d -print | sort
