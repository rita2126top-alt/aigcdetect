#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python tools/download_ojha.py --config "${CONFIG:-configs/ojha.yaml}" "$@"
