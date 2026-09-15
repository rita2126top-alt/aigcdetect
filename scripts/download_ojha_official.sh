#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/third_party/UniversalFakeDetect"
mkdir -p "$ROOT/third_party" "$ROOT/datasets"
if [[ ! -d "$THIRD/.git" ]]; then git clone --depth 1 https://github.com/WisconsinAIVision/UniversalFakeDetect.git "$THIRD"; fi
cat <<'MSG'
UniversalFakeDetect helper repository is ready.
Large images are hosted outside GitHub by the paper authors. Follow the official README data links, then edit configs/ojha.yaml or arrange:
  datasets/ForenSynths_4classtrain_val_test/train
  datasets/ForenSynths_4classtrain_val_test/val
  datasets/UniversalFakeDetect_test/{dalle,glide_100_10,glide_100_27,glide_50_27,guided,ldm_100,ldm_200,ldm_200_cfg}
Official instructions: https://github.com/WisconsinAIVision/UniversalFakeDetect#data
MSG
