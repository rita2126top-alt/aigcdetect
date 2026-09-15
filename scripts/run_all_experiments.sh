#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python tools/run_suite.py --config "${CONFIG:-configs/default.yaml}" --suite all "$@"
