#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/default.yaml "$@"
