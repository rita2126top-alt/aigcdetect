#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?Usage: cpu_smoke.sh /new/absolute/output-directory}"
[[ ! -e "$OUT" ]] || { echo "Choose a fresh directory: $OUT" >&2; exit 2; }
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
python -m cadp.cli toy-data --output "$OUT"
python -m cadp.cli preflight --config "$OUT/toy.yaml"
python -m cadp.cli train --config "$OUT/toy.yaml" --stop-after-epoch 1
python -m cadp.cli train --config "$OUT/toy.yaml" --set "train.resume=$OUT/run/last.pt"
python -m cadp.cli evaluate --config "$OUT/toy.yaml" --checkpoint "$OUT/run/best.pt"
python -m cadp.cli predict --config "$OUT/toy.yaml" --checkpoint "$OUT/run/best.pt" --input "$OUT/images/test" --output "$OUT/predictions.csv" --attention
python -m cadp.cli calibrate --config "$OUT/toy.yaml" --checkpoint "$OUT/run/best.pt" --output "$OUT/threshold.json"
python -m cadp.cli benchmark --config "$OUT/toy.yaml" --checkpoint "$OUT/run/best.pt" --output "$OUT/benchmark.json" --warmup 1 --repeats 2
python -m cadp.cli verify-upstream
