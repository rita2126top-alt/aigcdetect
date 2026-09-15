#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?Usage: baseline.sh /absolute/config.yaml /absolute/output}"
OUTPUT="${2:?Output directory required}"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
for SEED in 42 43 44; do
  RUN="$OUTPUT/runs/ppm_baseline/seed$SEED"
  RESUME=()
  if [[ -f "$RUN/last.pt" ]]; then RESUME=(--set "train.resume=$RUN/last.pt"); fi
  python -m cadp.cli train --config "$CONFIG" --set model.method=ppm_baseline --set "seed=$SEED" --set "paths.output_dir=$RUN" "${RESUME[@]}"
  python -m cadp.cli evaluate --config "$RUN/config.yaml" --checkpoint "$RUN/best.pt" --output "$RUN/eval_clean"
done
python -m cadp.cli aggregate --root "$OUTPUT" --output "$OUTPUT/aggregate"
