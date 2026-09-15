# AIGCDetect — Class-Anchored Dynamic Probabilistic PPM-CLIP

基于用户方法设计的AI生成图像检测项目：**两个类别锚点 + 动态概率提示 + context-to-patch Cross-Attention + 交互文本原型**，保留vision LoRA和DCT约束。

**完整中文教程：[docs/RUN_GUIDE_ZH.md](docs/RUN_GUIDE_ZH.md)**  
**验证证据与限制：[docs/VALIDATION_ZH.md](docs/VALIDATION_ZH.md)**

## 原始代码与新增代码分离

`PPM_CLIP/` 固定上游提交 `09d05b9fc4be6a2b079356bbf337a05e193db0be`，原始文件不修改；根目录亦保留上游副本。新方法在 `aigcdetect/`，配置在 `configs/`，命令入口在 `tools/`，全量实验脚本在 `scripts/`。用 `python tools/verify_upstream.py` 检查原版文件哈希。

旧 `requirements.txt` 属于原版；新项目使用 **environment.yml + requirements-runtime.txt**。

## 安装与本地资源

```bash
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect
conda env create -f environment.yml
conda activate aigcdetect
python -m pip install -e . --no-deps
mkdir -p "$HOME/aigcdetect_configs"
cp configs/default.yaml "$HOME/aigcdetect_configs/genimage.yaml"
export CONFIG="$HOME/aigcdetect_configs/genimage.yaml"
# 编辑CONFIG里的模型、下载、train/val/test与输出本地路径。
bash scripts/download_clip.sh
bash scripts/download_genimage_hf.sh --mode paper
python tools/prepare_genimage.py --config "$CONFIG"
python tools/audit_data.py --config "$CONFIG" --hash --decode
python tools/check_env.py --config "$CONFIG" --stage all --forward --backward
```

GenImage以原始图片字节导出，不二次JPEG编码；从训练源按内容分组划分独立验证集，不使用镜像的test/validation别名选模。默认数据协议是本项目的显式工程配置，不冒充上游论文原划分。

## 全量训练与实验

```bash
# 默认100轮，不是smoke脚本。
bash scripts/run_train_genimage.sh
# 将checkpoint路径改为CONFIG中实际output_dir下的best.pt。
bash scripts/run_eval_genimage.sh outputs/genimage_full/best.pt
bash scripts/run_robustness.sh outputs/genimage_full/best.pt
# 7个方法配置×3个随机种子，各自训练与评测。
bash scripts/run_ablations.sh --seeds 0,1,2
# 同样21个任务，并为每个checkpoint增加所有配置鲁棒性条件。
bash scripts/run_all_experiments.sh --seeds 0,1,2
```

模型/数据只从本地路径读取；显式下载入口才访问网络。评测自动恢复checkpoint对应的模型与消融配置，默认只读取测试域。支持epoch边界续训、优化器/调度器/scaler/RNG保存、分块文本计算和梯度累积、逐图CSV/域间macro/种子均值标准差。另提供Ojha8域和19域配置与官方下载工具。

## 检查与交付边界

```bash
python tools/validate_delivery.py --output validation
```

本地35项CPU集成测试通过，包含真正的上游小型随机CLIP前后向、六种消融、训练/保存/恢复/评测/单图推理、Arrow原字节导出。**这不等于正式预训练模型GPU全量实验已经跑完。** 详细记录在 `provenance/local_validation.json`，GitHub复验见Actions及发布时生成的CI记录。没有编造实验成绩。

原版启动入口只保留原算法行为：其固定提交用测试分数驱动早停，不能不加说明地作为无泄漏公平比较；详见教程第13节。
