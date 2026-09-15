#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?Usage: full.sh /absolute/config.yaml /absolute/output [main|all|ablations|robustness|mc|efficiency|cross-source]}"
OUTPUT="${2:?An experiment output directory is required}"
GROUP="${3:-main}"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
python -m cadp.cli verify-upstream
python -m cadp.cli audit --config "$CONFIG" --rehash
python -m cadp.cli preflight --config "$CONFIG"
python -m cadp.cli suite --config "$CONFIG" --group "$GROUP" --output "$OUTPUT" --execute --resume
