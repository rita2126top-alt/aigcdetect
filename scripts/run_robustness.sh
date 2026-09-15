#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:-outputs/genimage_full/best.pt}"
python tools/eval_robustness.py --config configs/default.yaml --checkpoint "$CKPT"
