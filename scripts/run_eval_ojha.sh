#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:-outputs/ojha_full/best.pt}"
python tools/eval.py --config configs/ojha.yaml --checkpoint "$CKPT"
