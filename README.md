# AIGCDetect: Class-Anchored Dynamic Probabilistic PPM-CLIP

This repository preserves the official **PPM-CLIP** code unchanged as a pinned Git submodule and adds a new AI-generated image detector around it. The new implementation follows the supplied method specification: two learnable Real/Fake class anchors, shared + image-conditioned probabilistic prompt tokens, image-dependent prompt length, Planar Flow, context-to-patch cross-attention, interactive text prototypes, vision LoRA, and DCT patch contrastive regularization.

## Repository layout

```text
PPM_CLIP/                  # untouched upstream baseline, pinned git submodule
aigcdetect/                # new method implementation
  model/                   # prompt flow, dynamic length, cross-attention, detector
  data/                    # local-folder dataset loading and robustness transforms
  engine/                  # train/evaluate/metrics/checkpoint utilities
configs/                   # all local paths + experimental hyperparameters
scripts/                   # downloads + full train/eval/ablation/robustness commands
tools/                     # Python entry points
tests/                     # CPU/static component tests
docs/RUN_GUIDE_ZH.md       # detailed Chinese from-zero server guide
```

The baseline submodule is pinned to upstream commit `09d05b9fc4be6a2b079356bbf337a05e193db0be`; no upstream PPM-CLIP source file is modified.

## Fast start

```bash
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect
conda env create -f environment.yml
conda activate aigcdetect
pip install -e .
bash scripts/download_clip.sh
python tools/check_env.py --config configs/default.yaml
```

Edit the local paths in `configs/default.yaml`, then run the complete GenImage experiment:

```bash
bash scripts/run_train_genimage.sh
bash scripts/run_eval_genimage.sh outputs/genimage_full/best.pt
bash scripts/run_robustness.sh outputs/genimage_full/best.pt
bash scripts/run_ablations.sh
```

Single-image inference:

```bash
python tools/infer.py /path/to/image.jpg \
  --config configs/default.yaml \
  --checkpoint outputs/genimage_full/best.pt
```

For the exact installation, dataset layouts, all commands, baseline reproduction, YAML overrides, checkpoint/resume behavior, ablations and troubleshooting, see **[docs/RUN_GUIDE_ZH.md](docs/RUN_GUIDE_ZH.md)**.

## Validation performed before delivery

The added code is checked with Python `compileall`, CPU component/shape tests, YAML parsing, shell `bash -n`, and CLI import/help checks. Full CUDA training is intentionally not executed here because the final GPU/CUDA runtime and the local datasets/checkpoint live on the user's remote server.
