# AIGCDetect：从零开始的远程 GPU 运行指南

## 0. 先明确交付内容与实验边界

本项目在原始 PPM-CLIP 之上新增独立实现，不修改 `PPM_CLIP/` 中的原始代码。上游固定为 `bandaidssssss/PPM_CLIP@09d05b9fc4be6a2b079356bbf337a05e193db0be`；根目录也保留其原始代码副本。`requirements.txt` 恢复为上游原文件；**新项目使用 `environment.yml` 和 `requirements-runtime.txt`，不要混用旧依赖文件**。

新方法按照用户提供的设计实现：两个 Real/Fake 类别锚点、共享/图像私有概率提示、动态长度路由、10 层 Planar Flow、context→patch Cross-Attention、交互式文本原型、视觉 LoRA 和六项训练损失。默认参数保持 ViT-L/14、K=2、共享3/private最多7、长度候选1/3/5/7、训练采样1/测试采样10、batch48、100 epochs。没有用简化分类器替代新方法。

需要区分三件事：**代码功能测试通过，不代表已经完成真实数据集全量训练，更不代表达到论文性能。** 交付前在本地 CPU 上运行了实际上游 CLIP 小型随机权重的训练、反向、断点恢复、推理和数据管线测试；它保留24层视觉网络，但宽度/输入分辨率缩小，不是正式预训练 ViT-L/14。具体证据见 `VALIDATION_ZH.md` 和 `provenance/local_validation.json`。真实 GPU、驱动、显存、正式权重和数据集的最终联合验证要在你的服务器执行本指南的环境检查命令。本项目没有伪造任何 ACC/AP/AUC 结果。

用户的方法文件没有指定完整的数据集划分协议；这里提供的 **GenImage SD1.4 训练源、按内容分组的10%训练内验证集、8生成器测试** 是明确的工程实验配置，不声称与上游论文原划分完全一致。Ojha 8域和19域配置也分别命名，不能混用平均指标。

## 1. 登录服务器与安装 Conda

下面假设服务器为 **Linux x86_64 + NVIDIA GPU**。`username`、`server-address`、`/data/rita` 是你需要替换的示例，不是已连接的服务器。Windows 本机只负责 SSH；训练命令在服务器上执行。

```bash
# 在本机终端登录远程服务器；替换账户和地址。
ssh username@server-address

# 在服务器确认系统和架构。不是 x86_64 时不要用下方 x86_64 安装包。
uname -s
uname -m

# 查看 GPU、驱动和显存。此命令失败时先找服务器管理员处理驱动/设备分配。
nvidia-smi

# 已安装 Conda 时，执行 conda --version 后直接跳到下一节。
conda --version
```

未安装 Conda 才执行以下命令。安装脚本来自 Anaconda 官方分发目录；在执行脚本前，按官方目录给出的 SHA256 校验下载文件，并阅读适用许可。不要覆盖已有的 `$HOME/miniconda3` 安装。

```bash
# 下载 Linux x86_64 Miniconda 安装脚本到用户主目录。
cd "$HOME"
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh

# 显示 SHA256，与官方分发目录发布的对应文件校验值核对。
sha256sum miniconda.sh

# 安装到个人目录，无需 sudo；需要先完成上面的校验。
bash miniconda.sh -b -p "$HOME/miniconda3"

# 在当前终端加载 Conda，之后才能使用 conda activate。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

官方参考：<https://repo.anaconda.com/miniconda/>、<https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>。

## 2. 拉取项目；旧副本不要强制覆盖

```bash
# 创建代码父目录，不会删除已有目录。
mkdir -p "$HOME/projects"
cd "$HOME/projects"

# 首次下载：同时拉取固定版本的 PPM-CLIP 子模块。
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect

# 检查当前提交及子模块状态，便于记录实验来源。
git log -1 --oneline
git submodule status
```

已经有旧副本时，不要再次 clone 到同名目录，也不要 `git reset --hard`。先 `git status`，把自己的配置复制到仓库之外；已有代码改动需先提交或自行合并，再执行：

```bash
# 只允许快进更新，避免意外合并或覆盖本地提交。
git pull --ff-only

# 补齐并更新到仓库指定的原版子模块提交。
git submodule update --init --recursive
```

## 3. 创建项目环境

默认环境采用固定的 Python3.10、PyTorch2.5.1、torchvision0.20.1、CUDA12.1运行时。这是兼容基线，不是“最新版本”，也不是对所有新 GPU 的通用承诺。PyTorch 官方版本组合参考：<https://pytorch.org/get-started/previous-versions/>。驱动或 GPU 架构不兼容时，根据官方支持矩阵另外建立环境；不要仅替换 torchvision 而保留不匹配的 torch。

```bash
# 必须在仓库根目录运行：按 YAML 创建名为 aigcdetect 的 Conda 环境。
conda env create -f environment.yml

# 激活后，python/pip 指向该环境。
conda activate aigcdetect

# 安装当前工程。依赖已由环境安装，--no-deps 避免重新替换 torch。
python -m pip install -e . --no-deps

# 显示解释器位置，并检查包依赖关系。
which python
python -m pip check

# 检查 CUDA 是否真的可用；只看 nvidia-smi 不能代替这个检查。
python -c "import torch,torchvision; print('torch',torch.__version__,'vision',torchvision.__version__,'CUDA runtime',torch.version.cuda,'available',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

每次重新 SSH 登录，需要重新 `source .../conda.sh`（若 shell 尚未初始化）并 `conda activate aigcdetect`。已有同名环境时，先确认其中没有其他项目依赖，不要盲目删除；可用 `conda env create -n aigcdetect_v2 -f environment.yml` 创建独立环境。

仅做 CPU 开发检查时也可使用 Conda 管理解释器，再装 CPU torch：

```bash
# 创建独立的 CPU 验证环境，不修改正式 GPU 环境。
conda create -n aigcdetect_cpu python=3.10 pip -y
conda activate aigcdetect_cpu
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
```

## 4. 把所有本地路径集中到自己的 YAML

建议将个人配置放在仓库之外，拉取更新时不会冲突。默认相对路径统一相对于**仓库根目录**，不是当前终端，也不是 YAML 所在目录；`extends` 文件路径例外，按子 YAML 所在位置解析。路径支持 `~` 和环境变量展开。

```bash
# 将完整默认配置复制到个人配置目录。
mkdir -p "$HOME/aigcdetect_configs"
cp configs/default.yaml "$HOME/aigcdetect_configs/genimage.yaml"

# 设置当前 shell 使用的配置文件；本项目所有 shell 包装脚本都支持 CONFIG。
export CONFIG="$HOME/aigcdetect_configs/genimage.yaml"

# 创建示例数据、权重和结果目录；按服务器实际存储位置修改 /data/rita。
mkdir -p /data/rita/aigcdetect/{weights,datasets,outputs}

# 编辑 YAML；不会自动下载任何数据。
nano "$CONFIG"
```

主要字段可改为：

```yaml
paths:
  ppmclip_root: ./PPM_CLIP
  clip_model: /data/rita/aigcdetect/weights/ViT-L-14.pt
  output_dir: /data/rita/aigcdetect/outputs/genimage_full
  genimage_snapshot: /data/rita/aigcdetect/datasets/GenImage_arrow
  genimage_export: /data/rita/aigcdetect/datasets/GenImage_raw
  download_dir: /data/rita/aigcdetect/datasets/downloads
  suite_output: /data/rita/aigcdetect/outputs/genimage_suite
runtime:
  device: cuda:0
# data 下其他字段保持原配置，只修改下面三个根目录。
data:
  train_root: /data/rita/aigcdetect/datasets/GenImage_SD14/train
  val_root: /data/rita/aigcdetect/datasets/GenImage_SD14/val
  test_root: /data/rita/aigcdetect/datasets/GenImage
```

**这是局部示例，不要用它覆盖整个文件**，否则会丢失 model/training 等必要配置。默认配置中的所有字段及超参数都可调整。单次命令也能用 `--set`：

```bash
# 仅对本次命令选择 GPU1；不会写回 YAML。
python tools/check_env.py --config "$CONFIG" --stage infer --set runtime.device=cuda:1
```

`--set` 支持已有的点分隔键，拼错键名会报错，不会悄悄忽略。列表覆盖示例为 `--set 'data.test_sets=[ADM,BigGAN]'`。单卡运行使用一个 `cuda:N`；当前训练器不实现 DDP，不要用 `torchrun` 冒充多卡支持。可在不同 GPU 上启动不同种子的独立进程，必须使用不同输出目录。

## 5. 下载正式 CLIP 到本地

```bash
# 查看目标文件、正式 OpenAI 下载地址和预期 SHA256，不进行下载。
bash scripts/download_clip.sh --dry-run

# 下载到 paths.clip_model；支持 .partial 断点文件，完成后校验官方 URL 中的 SHA256。
bash scripts/download_clip.sh
```

训练、评测和推理只读 `paths.clip_model`，不会联网自动补模型。既可读正式 OpenAI TorchScript checkpoint，也可读受支持的本地 CLIP state_dict。推理必须使用与训练相同的骨干权重，adapter 检查点会记录并检查其 SHA256。测试生成的随机 tiny 权重不可用作正式实验权重。

若权重已手动放在服务器，直接设置该路径即可；不必重复下载。下载器遇到不同 SHA256 的已有正式目标文件会拒绝覆盖，需先查清文件来源后手动另存。

## 6. 下载 GenImage 并正确划分 train/val/test

数据镜像：<https://huggingface.co/datasets/nebula/GenImage-arrow>。其每个 `data/<split>/<generator>/` 是独立 Arrow/save_to_disk 数据集。**镜像的 validation 是 test 的别名，不能拿它做模型选择的验证集。** 本项目从 SD1.4 的训练源另划验证集；测试图片不参加训练或模型选择。

镜像总量很大。默认 `--mode paper` 只是历史命名的下载组合：**SD1.4 训练源 + 八个生成器测试集**，不是整站全部训练数据，也不表示复现某篇论文的原始协议。`--mode test` 只下载八域测试，`--mode all-train` 下载所有训练源与测试源；后者不会自动执行8×8交叉生成器训练。下载前检查镜像页面的体积、你的磁盘配额及适用许可；源码仓库不附带数据集授权。

```bash
# 查看当前空闲空间。Arrow快照与导出图片会同时占用空间。
df -h /data/rita/aigcdetect

# 查询镜像的实际 revision 和匹配文件模式，打印计划，不下载图片。
bash scripts/download_genimage_hf.sh --mode paper --dry-run

# 实际下载默认训练源和八域测试；将解析出的 commit revision 写入 DOWNLOAD_MANIFEST.json。
bash scripts/download_genimage_hf.sh --mode paper

# 原字节导出训练/测试图片，并从训练源生成独立 train/val 目录。
python tools/prepare_genimage.py --config "$CONFIG" --stage all
```

导出器支持真实嵌入式 bytes 列、Hugging Face Image 字段，以及 Arrow IPC 流/文件。它校验标签、可用的源 MD5 和图片编码；直接保存原始字节，**不会统一重新编码成 JPEG**。生成 `export_manifest.csv`，记录源路径、标签、SHA256和目标相对路径。相同来源可以重跑；源身份不同或发现残留图片会拒绝混入旧导出目录，应选择新的输出目录。

训练划分按 SHA256 对重复图片分组，再按类别、固定种子选择验证组，同一原图的副本不会跨 train/val。默认约10%的唯一内容组用于验证；因为组大小不同，图片数量比例不一定恰好10%。默认以符号链接建立 train/val，节省重复存储；不要删除或移动 `genimage_export`。不便使用符号链接时，从新的空目录用 `--copy` 准备一份独立副本。

最终目录：

```text
GenImage_arrow/data/train/stable_diffusion_v_1_4/  # 原始 Arrow
GenImage_raw/train/stable_diffusion_v_1_4/
  0_real/*.jpg|png|...
  1_fake/*.jpg|png|...
  export_manifest.csv
GenImage_SD14/
  train/0_real/...     train/1_fake/...
  val/0_real/...       val/1_fake/...
GenImage/
  ADM/0_real/...       ADM/1_fake/...
  BigGAN/...
  Midjourney/...
  stable_diffusion_v_1_4/...
  stable_diffusion_v_1_5/...
  glide/...   wukong/...   VQDM/...
```

数据加载器会递归读取图片，支持 `0_real/1_fake`、`real/fake`、`nature/ai` 标签目录；坏图片会明确报错，不会伪造一张全零图片继续训练。

## 7. 数据完整性与运行检查

```bash
# 比对原版源码/资源 SHA256；确认没有改动上游实现。
python tools/verify_upstream.py

# 快速检查目录和解析后的路径交集；同一图片文件不可同时出现在 train/val。
python tools/audit_data.py --config "$CONFIG" --output /data/rita/aigcdetect/outputs/data_audit_paths.json

# 全量读图和内容哈希检查：可发现不同路径下相同字节的训练/测试图片。
# 这一步有完整磁盘扫描成本，不只是检查若干样本。
python tools/audit_data.py --config "$CONFIG" --hash --decode --output /data/rita/aigcdetect/outputs/data_audit_sha256.json

# 用正式本地权重、服务器指定 GPU 检查一次真实前向与 FP32 反向。
python tools/check_env.py --config "$CONFIG" --stage all --forward --backward

# 独立运行本项目的CPU回归测试；测试自己生成小模型和小数据，不使用你的训练集。
python tools/validate_delivery.py --output /data/rita/aigcdetect/outputs/code_validation
```

内容审计失败时应先定位、记录并修复数据来源/划分问题，不要忽略报错直接汇报无泄漏结果。不同测试域可能共享真实图片，工具不将 test-vs-test 共享自动判为 train/test 泄漏；你的论文仍应注明测试集重叠及宏平均的含义。哈希检查只能识别字节完全相同的副本，不能识别所有重新压缩或裁剪后的近重复。

只训练时 `check_env --stage train` 不要求下载测试集；只推理时 `--stage infer` 不要求任何数据集目录。正式 eval 默认仅要求测试域，不再为了测试去读取训练集。

## 8. 先做服务器端小规模启动，再运行全量实验

下列启动命令是真实模型/真实本地数据的1轮训练，不是合成样本测试。但它只是启动检查；不能把结果当成100轮全量结果。单独输出目录，避免污染正式实验。

```bash
# 使用小 micro-batch 跑一轮；accumulate_steps=12使满组有效 batch=4×12=48。
python tools/train.py --config "$CONFIG" \
  --set training.epochs=1 \
  --set training.batch_size=4 \
  --set training.accumulate_steps=12 \
  --set paths.output_dir=/data/rita/aigcdetect/outputs/server_startup
```

该命令仍遍历完整训练集一轮；不是只取一个 batch。服务器前后向能否工作可先用上一节 `check_env` 判断。1轮启动结果不能直接作为“同一100轮计划”的精确续训起点，因为其学习率调度周期不同。完整实验从新输出目录开始，默认命令如下：

```bash
# 全量正式训练：读取 YAML 中100轮、batch48及所有方法设置。
bash scripts/run_train_genimage.sh

# 用验证集选出的 best.pt 完整评测八个生成器。
# 将下面路径替换成你 YAML 中 paths.output_dir 对应的 best.pt。
bash scripts/run_eval_genimage.sh /data/rita/aigcdetect/outputs/genimage_full/best.pt

# 同一个已训练模型做 clean + JPEG95/75/50 + GaussianBlur1/2，默认共6个条件×8域。
bash scripts/run_robustness.sh /data/rita/aigcdetect/outputs/genimage_full/best.pt
```

正式训练可按显存将 `training.batch_size=4`、`training.accumulate_steps=12` 写入 YAML，保持满组有效 batch48；这与单次batch48不保证逐位等价，因为共享噪声、随机增强及随机路由的分组改变了。显存进一步不足时减小 `model.prompt.image_chunk_size` 或 `text_chunk_size`；`sample_chunk_size=1` 和激活重计算默认开启，**不会把测试10个样本偷换成1个样本**。训练时仍计算4个长度分支以获得 straight-through 路由梯度，不能将“前向选择一个长度”误当成只需计算一条训练分支。

训练入口不偷偷使用测试域选最佳权重。默认只根据独立验证集 AP 保存 `best.pt`；`last.pt` 保存完整续训状态。训练100轮是配置的计划，不保证每个硬件都能用默认batch48成功，GPU启动检查必须实际执行。

## 9. 断点续训、日志及 checkpoint

```bash
# 在原输出目录、相同模型和训练预算下，从最后完成的 epoch 恢复。
python tools/train.py --config "$CONFIG" \
  --set training.resume=/data/rita/aigcdetect/outputs/genimage_full/last.pt

# 查看最近的训练记录；每行JSON对应一个完成的epoch。
tail -n 5 /data/rita/aigcdetect/outputs/genimage_full/history.jsonl

# 实时观察显存/进程信息，不参与训练逻辑。
nvidia-smi
```

不要修改 epochs、学习率、weight_decay、batch_size、accumulate_steps、model 或 ablation 后还声称“精确续训”；程序对关键训练参数做一致性检查。恢复只支持 epoch 边界，进程在 epoch 中间被打断时，重跑该未完成轮。学习率调度器、优化器、AMP scaler、Python/NumPy/Torch随机状态均保存；CPU回归测试验证了同一配置下“连续2轮”与“1轮后重启续训到2轮”的全部模型状态逐项相等。跨 GPU型号、驱动、torch版本或非确定性CUDA算子不承诺位级一致。

`best.pt/last.pt` 为新模块+LoRA的 adapter 文件，不重复保存整个冻结 CLIP。必须同时保留原 CLIP 文件和其来源。程序使用受限 `weights_only=True` 加载本项目检查点，不接受任意陌生 pickle 对象。旧版缺少 scheduler/RNG 的检查点不能获得相同续训保证。续训请在原输出目录进行，避免把旧 best.pt 与新目录分离。

```text
paths.output_dir/
  resolved_config.json           # 实际生效的训练配置
  history.jsonl                 # 逐轮损失、验证指标、下一轮LR和best值
  last.pt                       # 最近完成轮的参数及优化状态
  best.pt                       # 由验证指标选出的权重
  eval/
    ADM_metrics.json            # 指标取值0~1，不是0~100百分数
    ADM_predictions.csv         # 路径、标签、fake概率、private长度
    summary.json                # 各域结果 + 域间等权macro
    ADM__jpeg75_metrics.json    # 鲁棒性条件结果
    robustness_summary.json
```

若需要断开SSH继续训练，请使用服务器已提供的终端会话管理器，例如 tmux；这是你在服务器执行的进程，不是聊天助手在后台替你运行。保存命令输出时可使用 `bash scripts/run_train_genimage.sh 2>&1 | tee train_console.log`；shell应开启 `set -o pipefail`，避免只看到 tee 的成功状态而遗漏训练失败。

## 10. 全量消融、三个种子和结果汇总

消融包括 full、去Cross-Attention、固定private长度7、去概率提示、去DCT损失、固定类别锚点、去LoRA，共7个配置。去概率提示会同时关闭其KL/重构项；其余损失设置不暗改。每个配置独立训练并自动从自身checkpoint恢复正确模型结构做评测，避免拿 full 配置误测消融权重。

```bash
# 仅打印完整计划，不写结果、不训练：7配置×3种子=21个训练任务。
bash scripts/run_ablations.sh --seeds 0,1,2 --dry-run

# 真正执行21次完整训练及八域clean评测；默认每次100轮。
bash scripts/run_ablations.sh --seeds 0,1,2

# 完整方法单独做3个种子；独立输出目录避免覆盖前面单种子实验。
python tools/run_suite.py --config "$CONFIG" --suite full --seeds 0,1,2 \
  --output /data/rita/aigcdetect/outputs/full_three_seeds

# 最完整组合：21次训练 + 每次八域clean与六条件鲁棒性评测。
# 不要在已有同名结果的情况下不带 --resume 重复启动。
bash scripts/run_all_experiments.sh --seeds 0,1,2

# 中断后恢复同一个完整实验计划；已训练完的任务不会再训练新轮。
bash scripts/run_all_experiments.sh --seeds 0,1,2 --resume

# 只重跑已有消融checkpoint的评测，不再训练。
bash scripts/run_ablations.sh --seeds 0,1,2 --stage eval

# 对实际存在的各seed结果汇总均值和样本标准差。
python tools/collect_results.py /data/rita/aigcdetect/outputs/genimage_suite
```

每个运行保存 `run.yaml`，目录形如 `full_seed0/`、`no_cross_attention_seed0/`。汇总为 `aggregate.json`：先取各测试域的等权宏平均，再跨完成的种子求均值/样本标准差；只完成一个种子时 std=null。缺失结果不会填0，也不会冒充已完成3个种子。`collect_results` 仅汇总 clean summary；鲁棒性结果保留在每个运行的 `robustness_summary.json` 中，不伪造跨种子的鲁棒性统计。

上述 `all` 指本指南定义的“7配置×指定种子×配置测试域×鲁棒性条件”，不含尚未实现的多机训练、所有公开检测方法或任意8×8训练源矩阵。用户方案没有提供这些额外实验的确定协议。

## 11. 单图推理、指定测试域、调整采样数

```bash
# 用正式训练的 best.pt 对本地一张图片判断，输出 real/fake概率及动态private长度。
python tools/infer.py /data/rita/example.jpg --config "$CONFIG" \
  --checkpoint /data/rita/aigcdetect/outputs/genimage_full/best.pt

# 只测试ADM和BigGAN。其余本地域无需存在；训练/验证目录也无需存在。
python tools/eval.py --config "$CONFIG" \
  --checkpoint /data/rita/aigcdetect/outputs/genimage_full/best.pt \
  --set 'data.test_sets=[ADM,BigGAN]' \
  --set paths.output_dir=/data/rita/aigcdetect/outputs/two_domains

# 采样数消融示例：另存结果，明确它不是默认S_test=10的主结果。
python tools/eval.py --config "$CONFIG" \
  --checkpoint /data/rita/aigcdetect/outputs/genimage_full/best.pt \
  --set model.prompt.test_samples=1 \
  --set paths.output_dir=/data/rita/aigcdetect/outputs/test_samples_1
```

默认噪声由 `model.prompt.eval_seed=0` 产生，在构造模型时固定并在每张图像复用基础噪声；private分布仍由各图像条件化，因此并未变成相同提示。先对每个Real/Fake pair做softmax，再平均K×S个概率。阈值0.5；单类别测试子集的AP/AUC为null而不是0。结果不是经过概率校准的真实性证明。

## 12. Ojha / UniversalFakeDetect 实验

`configs/ojha.yaml` 为常用的8个扩散域，`configs/ojha_19.yaml` 为11个GAN域+8个扩散域。默认训练/验证只选 ProGAN 的 `car,cat,chair,horse` 四类；要求目录中有这些类名。此处四类是本项目显式选择的实验设置，原官方数据包包含更多类别。公开扩散测试包与原论文报告的抽样规模不完全一致，报告结果时必须标明实际包版本与各域数量。

官方下载来源可核对：<https://github.com/WisconsinAIVision/UniversalFakeDetect>、<https://github.com/PeterWang512/CNNDetection>。Google Drive 配额、权限和链接有效性不由本仓库控制；脚本遇到失败会报错，不会生成假的下载成功标记。`downloads.ojha_archives` 中各ID可更改为官方后续发布的替代地址ID。

```bash
# 生成一份已合并extends的个人配置，方便单独改本地路径。
python -c "from aigcdetect.config import load_config; import yaml; from pathlib import Path; Path.home().joinpath('aigcdetect_configs/ojha.yaml').write_text(yaml.safe_dump(load_config('configs/ojha.yaml'),sort_keys=False))"
export CONFIG="$HOME/aigcdetect_configs/ojha.yaml"
nano "$CONFIG"

# 下载计划：不进行网络传输。
bash scripts/download_ojha_official.sh --subset all --dry-run

# 实际下载训练、验证、GAN测试和扩散测试压缩包到 paths.download_dir/ojha。
bash scripts/download_ojha_official.sh --subset all
```

压缩包不自动解压，以免不同发布版本的顶层目录互相覆盖。假设 `paths.download_dir=/data/rita/aigcdetect/datasets/downloads`，先安装或使用服务器已有的7z，然后执行：

```bash
# Debian/Ubuntu且有sudo权限时才执行；其他系统请按管理员提供的方式安装7z。
sudo apt-get install p7zip-full

# 先查看包内结构，再分别解压。文件后缀不决定真实压缩格式，7z按内容识别。
7z l /data/rita/aigcdetect/datasets/downloads/ojha/train.zip
7z x /data/rita/aigcdetect/datasets/downloads/ojha/train.zip -o/data/rita/aigcdetect/datasets/ojha_unpack/train
7z x /data/rita/aigcdetect/datasets/downloads/ojha/val.zip -o/data/rita/aigcdetect/datasets/ojha_unpack/val
7z x /data/rita/aigcdetect/datasets/downloads/ojha/cnn_test.zip -o/data/rita/aigcdetect/datasets/ojha_unpack/cnn_test
7z x /data/rita/aigcdetect/datasets/downloads/ojha/diffusion.zip -o/data/rita/aigcdetect/datasets/ojha_unpack/diffusion

# 列出标签目录，找到压缩包真实的顶层层级，不猜测嵌套路径。
find /data/rita/aigcdetect/datasets/ojha_unpack -type d -name 0_real | head -n 30
```

将 `data.train_root` 和 `val_root` 指向实际包含 car/cat/chair/horse 的父目录，允许前面还有 progan 层。测试域若分布在多个压缩包中，可令 `data.test_root` 指向共同父目录，并在 **YAML内**使用“域名→相对真实路径”的映射：

```yaml
data:
  test_root: /data/rita/aigcdetect/datasets/ojha_unpack
  test_sets:
    dalle: diffusion/实际顶层/dalle
    glide_100_10: diffusion/实际顶层/glide_100_10
    # 其余域按 find 的真实结果补齐；不要保留“实际顶层”字样。
    progan: cnn_test/实际顶层/progan
```

这只是映射写法示例，不是可以直接执行的完整19域配置。实际目录一致时可直接使用 `configs/ojha_19.yaml` 中的19个名称列表；否则把该列表逐域改为正确映射。不能把缺失域当成0分或称为19域评测。

```bash
# 对准备好的本地路径做完整检查。
python tools/check_env.py --config "$CONFIG" --stage all --forward
python tools/audit_data.py --config "$CONFIG" --hash --decode

# 按当前CONFIG训练、评测8域或19域；取决于该YAML实际的test_sets。
bash scripts/run_train_ojha.sh
bash scripts/run_eval_ojha.sh /data/rita/aigcdetect/outputs/ojha_full/best.pt

# 同一实验协议下的消融，输出路径必须与GenImage分开。
bash scripts/run_ablations.sh --seeds 0,1,2 --output /data/rita/aigcdetect/outputs/ojha_suite
```

## 13. 原版 PPM-CLIP 的保留入口与局限

`tools/run_baseline.py` 仅在当前进程里把原版数据根目录和 `clip.load` 指向 YAML 的本地资源，输出放在 `paths.output_dir/original_baseline/`。没有编辑上游文件。它还去掉不同torch版本中废弃的 scheduler `verbose` 展示参数；其余原版训练逻辑保留。

**重要：固定提交中的原版 `main.py` 在 val_acc≥0.90 后，用测试集均值驱动 early stopping；这不是无测试泄漏的模型选择协议。原版还使用其自身数据增强、随机采样、梯度累积和保存规则。** 本项目不掩盖这些差异，也不将原版输出当成公平对照结果。要做论文级公平比较，需要在单独的受控复现分支统一验证集选模等规则并披露改动；这不属于“原代码保持不变”的启动入口。原版完整GPU训练未在本地验证。

```bash
# 先恢复GenImage个人配置；检查原版启动参数和限制，不运行训练。
export CONFIG="$HOME/aigcdetect_configs/genimage.yaml"
bash scripts/run_baseline_ppmclip.sh --action train --dataset genimage --dry-run

# 启动固定提交的原始训练逻辑；注意上面明确说明的协议局限。
bash scripts/run_baseline_ppmclip.sh --action train --dataset genimage

# 使用原版自身保存的state_dict评测，不能传入新方法的adapter best.pt。
bash scripts/run_baseline_ppmclip.sh --action eval --dataset genimage \
  --checkpoint /data/rita/aigcdetect/outputs/genimage_full/original_baseline/checkpoints/原版实际文件名.pt
```

原版要求 num_workers≥1。Ojha原版不会按 `train_classes` 过滤，因此需要先准备只含四类的独立目录，再把 `data.train_classes=[]`；启动器会阻止不一致配置，而不是偷偷使用20类。新方法的训练器不受这些原版限制。

## 14. 常见错误与处理

| 情况 | 应对方式 |
|---|---|
| `Local CLIP checkpoint not found` | 下载正式权重或修改 `paths.clip_model`，不要把目录名当成文件路径。 |
| `No module named clip` / 上游导入冲突 | `git submodule update --init --recursive`；使用独立Python进程，不先导入另一个同名clip包。 |
| CUDA不可用 | 检查实际激活的解释器、torch构建、驱动、作业GPU分配；程序不会静默改用CPU训练。 |
| CUDA OOM | 减小micro-batch并增加accumulate_steps；减小image/text chunk；不要擅自把主实验测试采样数改成1。 |
| torchvision算子缺失 | torch/torchvision需按官方配对版本重建环境；不要混装CPU torchvision和CUDA torch。 |
| Arrow导出为空 | 检查 `data/train或test/生成器/` 的真实结构和下载模式；validation别名不是独立split。 |
| 导出源身份/残留文件冲突 | 为新revision设置新的导出目录；不要把旧JPEG重编码版本混在原字节版本里。 |
| 数据泄漏审计失败 | 根据审计JSON定位重复来源，重新制定并记录划分；不能靠忽略错误宣称无泄漏。 |
| `Existing run` | 使用原last.pt续训，或选择新的output_dir；不会静默覆盖已有实验。 |
| adapter SHA256不匹配 | 找回训练时使用的骨干文件；不要删掉校验逻辑绕过。 |
| 单域AUC=null | 该域只有一个标签类别；检查0_real/1_fake目录，不将null当作0。 |
| Google Drive下载失败 | 配额/权限/链接可能变化，查官方源更新ID；手动获取后仍需检查压缩包与目录。 |
| GitHub/服务器显示的日期不同 | Git提交和Actions通常显示UTC，服务器/浏览器可能用本地时区；记录提交SHA比口头日期可靠。 |

## 15. 方法与文件对应

| 设计部分 | 代码位置 |
|---|---|
| shared/private Gaussian、条件Planar Flow、KL/重构 | `aigcdetect/model/prompt_flow.py` |
| ST Gumbel路由、Cross-Attention、门控融合 | `aigcdetect/model/components.py` |
| 类别锚点、CLIP/LoRA、DCT、动态EOS、六损失、固定采样 | `aigcdetect/model/detector.py` |
| 本地数据、增广、分阶段加载 | `aigcdetect/data/folder_dataset.py` |
| 原字节Arrow导出、SHA分组划分、泄漏审计 | `aigcdetect/data/preparation.py` |
| AdamW参数组、梯度累积、调度器、完整续训 | `aigcdetect/engine/train.py` |
| 指标、逐图预测、域间macro | `aigcdetect/engine/metrics.py`、`evaluate.py` |
| 命令入口、全量实验编排、下载、检查 | `aigcdetect/cli.py`、`tools/`、`scripts/` |

每个 `tools/*.py` 主入口支持 `--help`（校验原版源码的简单工具除外），新增入口在导入时不会启动训练。新实验代码和用户数据始终与保留的上游源码分离。
