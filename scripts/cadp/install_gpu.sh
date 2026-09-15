#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VARIANT="${1:-cu126}"
case "$VARIANT" in cu126|cu128|cu130|cpu) ;; *) echo 'Use cu126, cu128, cu130 or cpu' >&2; exit 2;; esac
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/${VARIANT}"
python -m pip install -r requirements-cadp.txt
python -m pip install --no-deps -e .
python -m pip check
python -m cadp.cli doctor
