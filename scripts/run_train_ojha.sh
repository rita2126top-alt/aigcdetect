#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python tools/train.py --config "${CONFIG:-configs/ojha.yaml}" "$@"
