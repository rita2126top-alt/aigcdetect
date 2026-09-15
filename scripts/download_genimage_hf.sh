#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/download_genimage_hf.sh [output_dir] [test|paper]
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/datasets/GenImage_arrow}"
MODE="${2:-test}"
mkdir -p "$OUT"
GENERATORS=(ADM BigGAN glide Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 VQDM wukong)
for G in "${GENERATORS[@]}"; do
  hf download nebula/GenImage-arrow --repo-type dataset --include "data/test/$G/*" --local-dir "$OUT"
done
if [[ "$MODE" == "paper" ]]; then
  hf download nebula/GenImage-arrow --repo-type dataset --include "data/train/stable_diffusion_v_1_4/*" --local-dir "$OUT"
elif [[ "$MODE" != "test" ]]; then
  echo "Unknown mode: $MODE (expected test or paper)" >&2; exit 2
fi
cat <<EOF2
Downloaded GenImage Arrow data to: $OUT
Export test: python tools/export_genimage_arrow.py --snapshot "$OUT" --output "$ROOT/datasets/GenImage" --split test
For SD1.4 training (paper mode):
  python tools/export_genimage_arrow.py --snapshot "$OUT" --output "$ROOT/datasets/Genimages_SD-V1.4/train" --split train
  python tools/export_genimage_arrow.py --snapshot "$OUT" --output "$ROOT/datasets/Genimages_SD-V1.4/val" --split test --generator stable_diffusion_v_1_4
EOF2
