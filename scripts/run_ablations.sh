#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/default.yaml --set paths.output_dir=./outputs/abl_no_cross --set ablation.disable_cross_attention=true
python tools/eval.py --config configs/default.yaml --checkpoint outputs/abl_no_cross/best.pt --set paths.output_dir=./outputs/abl_no_cross --set ablation.disable_cross_attention=true
python tools/train.py --config configs/default.yaml --set paths.output_dir=./outputs/abl_fixed_length --set ablation.disable_dynamic_length=true
python tools/eval.py --config configs/default.yaml --checkpoint outputs/abl_fixed_length/best.pt --set paths.output_dir=./outputs/abl_fixed_length --set ablation.disable_dynamic_length=true
python tools/train.py --config configs/default.yaml --set paths.output_dir=./outputs/abl_no_prob --set ablation.disable_probabilistic_prompt=true
python tools/eval.py --config configs/default.yaml --checkpoint outputs/abl_no_prob/best.pt --set paths.output_dir=./outputs/abl_no_prob --set ablation.disable_probabilistic_prompt=true
python tools/train.py --config configs/default.yaml --set paths.output_dir=./outputs/abl_no_dct --set ablation.disable_dct_loss=true
python tools/eval.py --config configs/default.yaml --checkpoint outputs/abl_no_dct/best.pt --set paths.output_dir=./outputs/abl_no_dct --set ablation.disable_dct_loss=true
