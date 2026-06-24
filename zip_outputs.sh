#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT="${1:-emotion_et_benchmarks_outputs.zip}"
python -m zipfile -c "$OUT" artifacts
realpath "$OUT"
