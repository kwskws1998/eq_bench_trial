#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-emotion_et_benchmarks_standalone.zip}"
cd "$(dirname "$ROOT")"

python -m zipfile -c "$OUT" emotion_et_benchmarks
realpath "$OUT"
