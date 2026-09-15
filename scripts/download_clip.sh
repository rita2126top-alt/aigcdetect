#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python tools/download_assets.py --kind clip --config "${CONFIG:-configs/default.yaml}" "$@"
