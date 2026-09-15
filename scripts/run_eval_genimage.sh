#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${1:?Usage: bash scripts/run_eval_genimage.sh /path/best.pt [options]}"
CKPT="$1"; shift
python tools/eval.py --config "${CONFIG:-configs/default.yaml}" --checkpoint "$CKPT" "$@"
