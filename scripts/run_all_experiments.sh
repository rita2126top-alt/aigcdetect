#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_train_genimage.sh
bash scripts/run_eval_genimage.sh outputs/genimage_full/best.pt
bash scripts/run_robustness.sh outputs/genimage_full/best.pt
bash scripts/run_ablations.sh
# Optional Ojha suite after local data are prepared:
# bash scripts/run_train_ojha.sh
# bash scripts/run_eval_ojha.sh outputs/ojha_full/best.pt
