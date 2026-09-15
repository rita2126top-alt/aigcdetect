# AIGCDetect 完整运行指南（Conda + 远程 GPU 服务器）

本文档从一台“只有 Git、Conda 和 NVIDIA 驱动”的 Linux 服务器开始，说明如何完整运行：原始 PPM-CLIP baseline、新方法训练、验证、8 个 GenImage 域测试、Ojha/UniversalFakeDetect 测试、鲁棒性测试、4 项消融实验以及单图推理。

> **重要设计原则**：`PPM_CLIP/` 是官方仓库的固定 commit 子模块，保持原代码不变；所有新方法代码都位于 `aigcdetect/`、`tools/`、`configs/` 和 `scripts/`。

## 1. 克隆完整仓库

```bash
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect
```

若已经普通 clone，补执行：

```bash
git submodule update --init --recursive
git -C PPM_CLIP rev-parse HEAD
```

PPM-CLIP 应固定在 `09d05b9fc4be6a2b079356bbf337a05e193db0be`。

## 2. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate aigcdetect
pip install -e .
```

分别用于创建 Python 3.10 + PyTorch 2.5.1 + torchvision 0.20.1 + CUDA 12.1 环境、激活环境、editable 安装项目。检查 GPU：

```bash
python - <<'PY'
import torch
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
print('cuda version =', torch.version.cuda)
if torch.cuda.is_available(): print('gpu =', torch.cuda.get_device_name(0))
PY
```

若服务器需要其它 CUDA 版本，只修改 `environment.yml` 的 `pytorch-cuda` 即可。

## 3. 下载本地 CLIP ViT-L/14

新方法只从本地目录加载模型：

```bash
bash scripts/download_clip.sh
```

默认保存 `weights/clip/ViT-L-14.pt` 并校验官方 SHA256。已有权重时直接修改：

```yaml
paths:
  clip_model: /your/local/path/ViT-L-14.pt
```

也可以指定下载位置：

```bash
bash scripts/download_clip.sh /data/models/ViT-L-14.pt
```

## 4. 数据目录约定

加载器递归扫描图片，以下目录名会自动得到标签：

```text
0_real/ 或 real/ 或 nature/ -> label 0
1_fake/ 或 fake/ 或 ai/    -> label 1
```

因此兼容 GenImage 的 `nature/ai` 与 ForenSynths/UniversalFakeDetect 的 `0_real/1_fake`。

### 4.1 GenImage 主实验

`configs/default.yaml` 默认目录：

```text
datasets/
├── Genimages_SD-V1.4/
│   ├── train/
│   └── val/
└── GenImage/
    ├── Midjourney/
    ├── stable_diffusion_v_1_4/
    ├── stable_diffusion_v_1_5/
    ├── ADM/
    ├── glide/
    ├── wukong/
    ├── VQDM/
    └── BigGAN/
```

论文式设置：只用 Stable Diffusion v1.4 训练/验证，然后跨 8 个 generator 测试泛化。

下载 8 个测试域的 Hugging Face Arrow 镜像：

```bash
bash scripts/download_genimage_hf.sh datasets/GenImage_arrow test
python tools/export_genimage_arrow.py \
  --snapshot datasets/GenImage_arrow \
  --output datasets/GenImage \
  --split test
```

论文式 SD1.4 训练：

```bash
bash scripts/download_genimage_hf.sh datasets/GenImage_arrow paper
python tools/export_genimage_arrow.py \
  --snapshot datasets/GenImage_arrow \
  --output datasets/Genimages_SD-V1.4/train \
  --split train
```

该镜像的 validation 与官方 test 是同一物理数据而非独立第三 split。只导出 SD1.4 作为验证集：

```bash
python tools/export_genimage_arrow.py \
  --snapshot datasets/GenImage_arrow \
  --output datasets/Genimages_SD-V1.4/val \
  --split test \
  --generator stable_diffusion_v_1_4
```

若使用 GenImage 官方下载的原始图片，直接在 YAML 修改本地路径即可，无需转换。

### 4.2 Ojha / UniversalFakeDetect

```bash
bash scripts/download_ojha_official.sh
```

脚本拉取官方 UniversalFakeDetect 辅助仓库并给出作者数据入口。大数据本身由论文作者托管在外部存储，不在本仓库重托管。默认 `configs/ojha.yaml` 假设：

```text
datasets/ForenSynths_4classtrain_val_test/train
datasets/ForenSynths_4classtrain_val_test/val
datasets/UniversalFakeDetect_test/
  dalle/
  glide_100_10/
  glide_100_27/
  glide_50_27/
  guided/
  ldm_100/
  ldm_200/
  ldm_200_cfg/
```

也可以完全使用自己的目录，只改 `configs/ojha.yaml` 的 `train_root`、`val_root`、`test_root`。

## 5. 修改本地路径与 GPU

```bash
vim configs/default.yaml
```

最常改：

```yaml
paths:
  ppmclip_root: ./PPM_CLIP
  clip_model: /data/models/ViT-L-14.pt
  output_dir: /data/experiments/aigcdetect/genimage_full
runtime:
  device: cuda:0
data:
  train_root: /data/GenImage/SD14/train
  val_root: /data/GenImage/SD14/val
  test_root: /data/GenImage/test
```

也能用 CLI 临时覆盖而不改 YAML：

```bash
python tools/train.py --config configs/default.yaml \
  --set runtime.device=cuda:1 \
  --set data.train_root=/data/my_sd14/train \
  --set paths.output_dir=/data/exp/run1
```

## 6. 运行前检查

```bash
python tools/check_env.py --config configs/default.yaml
```

应重点看到：`cuda_available=True`、`ppmclip_root=True`、`clip_model=True`、三个 data root 都为 `True`。有 `False` 时先修路径。

## 7. 新方法完整训练

GenImage：

```bash
bash scripts/run_train_genimage.sh
```

等价于：

```bash
python tools/train.py --config configs/default.yaml
```

默认固定为：ViT-L/14；vision LoRA half-up、qkv、rank=4、alpha=0.5；K=2；shared=3；private max=7；private candidates `{1,3,5,7}`；10 层 Planar Flow；cross-attention dim=256、heads=4；S_train=1、S_test=10；batch=48；epochs=100；AdamW；新模块 LR=1e-4；LoRA LR=1e-5；weight decay=1e-4。

输出：

```text
outputs/genimage_full/
├── best.pt
├── last.pt
└── history.jsonl
```

显存不足时优先只降 batch size，不改变方法：

```bash
python tools/train.py --config configs/default.yaml --set training.batch_size=16
```

从 checkpoint 继续：

```bash
python tools/train.py --config configs/default.yaml \
  --set training.resume=./outputs/genimage_full/last.pt
```

Ojha 训练：

```bash
bash scripts/run_train_ojha.sh
```

## 8. 完整评测

GenImage 8-domain：

```bash
bash scripts/run_eval_genimage.sh outputs/genimage_full/best.pt
```

分别评测 Midjourney、SD1.4、SD1.5、ADM、GLIDE、Wukong、VQDM、BigGAN 和 YAML validation。结果写入：

```text
outputs/genimage_full/eval/<domain>_metrics.json
outputs/genimage_full/eval/<domain>_predictions.csv
```

指标包含 Accuracy、AP、ROC-AUC、F1、real accuracy、fake accuracy、mean private length。

Ojha：

```bash
bash scripts/run_eval_ojha.sh outputs/ojha_full/best.pt
```

单图推理：

```bash
python tools/infer.py /data/example.jpg \
  --config configs/default.yaml \
  --checkpoint outputs/genimage_full/best.pt
```

输出 Real/Fake 概率、类别和 router 选择的 private token 数量。

## 9. 鲁棒性实验

```bash
bash scripts/run_robustness.sh outputs/genimage_full/best.pt
```

每个测试域评测 clean、JPEG 95/75/50、Gaussian blur radius 1/2，结果写入 `output_dir/eval/`。

## 10. 消融实验

```bash
bash scripts/run_ablations.sh
```

脚本会逐个完成**训练 + 8-domain 评测**：

```text
outputs/abl_no_cross      # 去掉 context -> patch Cross-Attention
outputs/abl_fixed_length  # 固定 private length=7
outputs/abl_no_prob       # 去掉 probabilistic prompt bias / flow loss
outputs/abl_no_dct        # 去掉 DCT patch contrastive loss
```

单项也能手动运行：

```bash
python tools/train.py --config configs/default.yaml \
  --set ablation.disable_cross_attention=true \
  --set paths.output_dir=./outputs/abl_no_cross
```

## 11. 一键全量 GenImage 实验

```bash
bash scripts/run_all_experiments.sh
```

顺序：完整方法训练 → 8-domain 评测 → 鲁棒性 → 四个消融各自训练和评测。Ojha 因为大型数据是可选下载，默认注释；准备好后执行对应 train/eval 脚本即可。

## 12. 原始 PPM-CLIP baseline

原始源代码位于 `PPM_CLIP/`，没有修改：

```bash
bash scripts/run_baseline_ppmclip.sh
```

这个 wrapper 只做两件不改源码的辅助工作：把本地 CLIP 权重软链接到 upstream 默认缓存位置；从 `PPM_CLIP/_runtime` 启动，使 upstream 硬编码的 `../../datasets/...` 解析到本仓库 `datasets/`。

Ojha baseline：

```bash
bash scripts/run_baseline_ppmclip.sh --dataset ojha
```

## 13. 方法与代码对应

```text
Vision LoRA + CLS/Patch tokens       aigcdetect/model/detector.py
DCT patch contrastive loss           detector.py + upstream DCT_score.py
Shared/private Gaussian + PlanarFlow aigcdetect/model/prompt_flow.py
Token-specific prompt generator      aigcdetect/model/prompt_flow.py
Length router + hard Gumbel          aigcdetect/model/components.py
Real/Fake class anchor residuals     aigcdetect/model/detector.py
Dynamic prompt construction          aigcdetect/model/detector.py
Frozen CLIP text encoder             aigcdetect/model/detector.py
Context-to-patch Cross-Attention     aigcdetect/model/components.py
Interactive prototype fusion         aigcdetect/model/components.py
Training objective                   aigcdetect/engine/train.py + detector.py
Pair-softmax then K×S probability avg detector.py
Metrics / CSV / JSON                 aigcdetect/engine/
```

训练时标签只进入 classification loss，不进入 prompt 构造。Real/Fake 对同一 image/repository/sample/length 共用相同 context，只由 class anchor 区分类别。测试时先对每个 Real/Fake pair 做 softmax，再对 `K=2 × S_test=10` 的概率平均，绝不直接平均 logits。

## 14. 静态检查

```bash
python -m compileall -q aigcdetect tools tests
pytest -q tests/test_components.py
for f in scripts/*.sh; do bash -n "$f"; done
python tools/train.py --help
python tools/eval.py --help
python tools/infer.py --help
python tools/eval_robustness.py --help
python tools/export_genimage_arrow.py --help
```

这些不需要数据，也不启动完整 GPU 训练。CUDA end-to-end 运行需要服务器上的本地 CLIP 权重和数据。

## 15. 常见问题

`Cannot import upstream PPM_CLIP`：

```bash
git submodule update --init --recursive
```

`Local CLIP checkpoint not found`：

```bash
bash scripts/download_clip.sh
```

`No labeled images found`：确认路径里存在 `0_real/1_fake` 或 `nature/ai`。

CUDA OOM：先 `--set training.batch_size=16`，仍不足再降 8/4；不要先改 prompt 数量或模型维度。

多 GPU：当前是单进程单 GPU，`runtime.device=cuda:N` 选择卡；后续可在不改模型定义情况下增加 DDP launcher。

## 16. 最小完整命令清单

```bash
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect
conda env create -f environment.yml
conda activate aigcdetect
pip install -e .
bash scripts/download_clip.sh
vim configs/default.yaml
python tools/check_env.py --config configs/default.yaml
bash scripts/run_train_genimage.sh
bash scripts/run_eval_genimage.sh outputs/genimage_full/best.pt
bash scripts/run_robustness.sh outputs/genimage_full/best.pt
bash scripts/run_ablations.sh
python tools/infer.py /path/to/image.jpg --config configs/default.yaml --checkpoint outputs/genimage_full/best.pt
```
