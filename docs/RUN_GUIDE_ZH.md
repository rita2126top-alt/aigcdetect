# 从零开始运行：PPM-CLIP 原版与 CADP-CLIP 新方法

## 0. 先区分三个层次

本指南默认服务器是 **Linux x86_64 + NVIDIA GPU**，你有自己的账号及可写数据盘。Windows 电脑只用来 SSH 登录；训练命令在服务器终端里执行。服务器是 ARM、非 NVIDIA 或已有集群模块环境时，需要调整环境安装部分，不能照搬 x86_64 安装包。

`CPU 单元测试`检查程序和算法契约；`真实 GPU preflight`检查你的权重、数据与硬件能否完成一次真实前向、反向及优化器更新；`正式训练与评测`才能产生有研究意义的指标。三者不能互相冒充。交付验证记录见 `VALIDATION_REPORT.md`，没有预先填入论文成绩。

主程序不会偷偷联网下载模型或数据。只有显式执行 `download-model` / `download-data --execute` 才联网。下载的图片、预训练模型、训练 checkpoint 都保存在仓库外的本地目录，不上传 GitHub。

本次源码的正式入口是 `cadp/`，不是仓库中前次交付遗留的 `aigcdetect/`、`tools/` 和 `scripts/run_*.sh`。旧文件保留供追溯；本指南只指向已经验收的 CADP 入口。`PPM_CLIP/` 必须存在，导入与 SHA256 校验都指向这个固定版本的目录。

本地完整 ZIP 已包含该目录；ZIP 不包含 Git 历史，不能把它当成已经推送的 Git 提交。发布状态见 `docs/DELIVERY_STATUS.md`。

## 1. 登录服务器与安装 conda

先在自己电脑终端执行下面一行，替换两个占位符：

```bash
ssh YOUR_USER@YOUR_SERVER  # 登录你的服务器；认证方式由你的服务器决定。
```

登录后执行：

```bash
uname -m                 # 确认 CPU 架构；下方安装包只适用于 x86_64。
nvidia-smi               # 检查 NVIDIA 驱动能否看到 GPU；记下显存和驱动信息。
conda --version          # 检查是否已有 conda；已有可用 conda 时跳过下面安装块。
```

没有 conda 时，以下是用户目录安装方式，不需要 sudo。先核对官方安装页面及安装包校验值，再运行安装器。下载页及许可证要求以官方为准。

```bash
mkdir -p "$HOME/installers"  # 创建安装包保存目录。
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/installers/miniconda.sh"  # 下载官方 Linux x86_64 安装器。
sha256sum "$HOME/installers/miniconda.sh"  # 与官方 Miniconda 下载索引中的 SHA256 对照，不要忽略校验不一致。
bash "$HOME/installers/miniconda.sh" -b -p "$HOME/miniconda3"  # 阅读并接受相应许可后，安装到自己的 HOME。
source "$HOME/miniconda3/etc/profile.d/conda.sh"  # 在当前 shell 中启用 conda activate。
```

安装参考：<https://www.anaconda.com/docs/getting-started/installation>；校验值索引：<https://repo.anaconda.com/miniconda/>。已有环境不要重复向同一目录安装。

## 2. 克隆代码、创建隔离环境

**完整 ZIP 用户**：解压后先 `cd` 到其中的 `aigcdetect/`；跳过下面的 `git clone`、`cd aigcdetect` 和 `git submodule update`，直接从 `conda env create` 开始。ZIP 已携带完整上游目录，但没有 `.git`。**GitHub 克隆用户**：先按 `DELIVERY_STATUS.md` 确认这次补丁已经推送，再执行下面整个命令块。

```bash
mkdir -p "$HOME/projects"                         # 为源码创建目录。
cd "$HOME/projects"                              # 进入源码目录。
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git  # 获取原 PPM 快照和新增完整实现。
cd aigcdetect                                    # 后续相对路径命令均从仓库根目录执行。
git submodule update --init --recursive           # 已克隆过仓库时补齐固定版本的 PPM_CLIP；不更新到上游最新提交。
conda env create -f environment-cadp.yml          # 创建名为 cadp、Python 3.11 的隔离环境并安装普通依赖。
conda activate cadp                              # 让 python 和 pip 都使用该环境。
bash scripts/cadp/install_gpu.sh cu126            # 安装匹配的 torch 2.10.0 / torchvision 0.25.0 CUDA 12.6 wheel，并检查依赖。
export PYTHONDONTWRITEBYTECODE=1                  # 不改写上游已经跟踪的 .pyc 文件。
export OMP_NUM_THREADS=4                          # 限制每个进程的 CPU 线程，避免数据加载/线性代数过度抢占。
python -m cadp.cli doctor                         # 打印 Python、PyTorch、CUDA 和关键模块导入状态。
python -m cadp.cli verify-upstream                # 校验原始 47 个文件是否逐字节保持不变。
```

这里固定的是一个经过 CPU 代码测试的版本组合，不宣称是当前最新版本。PyTorch 官方提供此组合的 `cu126`、`cu128`、`cu130` 和 `cpu` 安装选项：<https://pytorch.org/get-started/previous-versions/>。`install_gpu.sh` 接受这四种参数。

选择 CUDA wheel 必须匹配实际 GPU 架构和驱动。`nvidia-smi` 的 CUDA 字样不等于当前 Python 已安装了 GPU 版 torch。若 `doctor` 显示 `cuda_available: false`，应修复环境/驱动或联系服务器管理员，而不是把正式配置改成 CPU 来掩盖问题。新架构 GPU 可能需要不同 wheel；不要因为本文示例用了 cu126 就假定所有 GPU 都能用它。

原版依赖请以 `PPM_CLIP/requirements.txt` 为准；根目录另有前次交付遗留的依赖文件。**不要在新环境里再执行 `pip install -r requirements.txt` 混装两套 torch。** 本交付使用 `requirements-cadp.txt`；子模块中的原始依赖文件仍然原封不动保留。新环境创建完成后可保存安装记录：

```bash
python -m pip freeze > "$HOME/cadp-pip-freeze.txt"  # 保存实际安装版本供复现，而不是只记录宽泛的依赖区间。
conda env export > "$HOME/cadp-environment-export.yml"  # 保存 conda 层面的环境记录。
```

## 3. 设置本地目录与 YAML

示例把数据盘设为 `/data/aigcdetect`。没有该目录权限时，改成自己的可写目录，例如 `/home/你的用户名/aigcdetect_data`。不要强行给公共数据盘修改权限。

```bash
mkdir -p /data/aigcdetect/{models,datasets,manifests,runs,experiments}  # 分开存放模型、数据、索引、训练输出和实验矩阵。
cp configs/cadp/default.yaml configs/cadp/server.yaml               # 创建自己的配置，不改默认模板。
nano configs/cadp/server.yaml                                     # 编辑路径和硬件相关设置；也可使用熟悉的编辑器。
```

至少检查以下字段：

```yaml
paths:
  clip_checkpoint: /data/aigcdetect/models/ViT-L-14.pt
  data_root: /data/aigcdetect/datasets
  manifest_dir: /data/aigcdetect/manifests
  output_dir: /data/aigcdetect/runs/main_seed42
model:
  method: cadp
  tiny: false
  checkpoint_blocks: true
  text_chunk_size: 8
train:
  device: cuda
  amp: bf16
  epochs: 100
  batch_size: 2
  accumulation_steps: 24
```

**YAML 中的 `batch_size` 是每次送入网络的 microbatch。** 默认 `2 × 24 = 48` 是单次优化更新的有效 batch size。附件要求的 48 不会被误当成默认每次只训练 2 张。尾部不满一个累积窗口时按真实样本数归一化，不丢弃尾部梯度。

梯度累积与真正一次送入 48 张并不在所有随机细节上完全相同：每个 microbatch 会重新采样 shared latent。若要一次物理 batch=48，可设置 `batch_size: 48, accumulation_steps: 1`，但显存能否容纳必须实测，交付时未测得任何显存容量保证。

显存不足时先把 microbatch 改为 1、累积改为 48，并保持 `checkpoint_blocks: true`；还可把 `text_chunk_size` 改成 4 或 2。减小 `eval.batch_size` 可降低测试显存。不要为了节省显存偷偷把 `model.tiny` 改成 true 用于论文实验。

不支持 bf16 的卡改 `train.amp: fp16` 或 `off`。FP16 使用动态 GradScaler，`train.grad_scaler_init_scale` 默认 256；遇到溢出会跳过该次优化更新并降低 scale，日志记录 `skipped_amp_steps`。一整轮都未成功更新时立即报错，不把空训练当成功。`off` 是精度模式字符串，本项目正确处理它，不会将其误解成 False。

每条命令都可以临时覆盖配置，多个 `--set` 必须分别写：

```bash
python -m cadp.cli doctor --config configs/cadp/server.yaml --set train.device=cuda:0  # 仅为本次命令覆盖设备，不改 YAML。
```

## 4. 下载并校验本地 CLIP

```bash
python -m cadp.cli download-model --config configs/cadp/server.yaml  # 下载官方 ViT-L/14 到 paths.clip_checkpoint，并校验完整 SHA256。
sha256sum /data/aigcdetect/models/ViT-L-14.pt                        # 独立复核文件哈希。
```

期望 SHA256 为：

```text
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

下载器使用 `.part` 文件断点续传，校验通过后才重命名为正式文件。正式文件存在但哈希不符时拒绝覆盖；先检查是不是下载损坏或拿错了模型，再由你明确移动错误文件。中断留下的 `.part` 可以重试；校验错误的 `.part` 应移走后重新下载。

没有外网的服务器可以在能联网的机器执行下载命令，再使用 `scp`/已有文件传输方式把**同一文件**放到服务器 YAML 指定的位置。训练和推理传给 `clip.load` 的是实际本地文件路径，不是会触发联网的模型名称。CADP 正式配置校验的是 224 分辨率版本 ViT-L/14，不接受把 L/14@336 当成同一个配置。

## 5. 下载真实数据，而不是运行 toy 数据冒充实验

本交付默认研究协议使用 GenImage。附件规定了方法与超参数，没有给出完整实验数据集清单，因此这里明确增加了 GenImage 主协议和通用 CSV 数据接口，不声称自动复现所有检测论文的数据集/成绩。

GenImage 官方说明及目录来自：<https://github.com/GenImage-Dataset/GenImage/blob/main/Readme.md>。真实图像涉及 ImageNet 来源，使用前应自行确认相应数据条款和访问资格。

```bash
python -m cadp.cli download-data --config configs/cadp/server.yaml  # 只打印官方下载来源和本地目标，不开始大规模下载。
python -m cadp.cli download-data --config configs/cadp/server.yaml --execute  # 从配置的官方 Drive 文件夹下载归档，可能占用大量磁盘。
find /data/aigcdetect/datasets/downloads/GenImage -maxdepth 3 -type f | head -30  # 检查实际下载了什么文件。
```

Google Drive 的额度、权限、大目录限制和发布者文件变更是外部条件；脚本不会绕过限制，也不会把没有文件的下载宣称成功。默认是百万级数据集的下载入口，不是“小型快速数据包”。交付时没有下载整个 GenImage。下载受限时可按官方 README 的其他渠道获得归档，再放到本地；不是让训练代码去读网盘 URL。

普通 ZIP/TAR 可使用安全解包命令，**将输入替换为你实际下载到的完整归档文件**：

```bash
python -m cadp.cli extract --input /实际下载路径/dataset.zip --destination /data/aigcdetect/datasets/GenImage  # 解包，不允许目录穿越、符号链接或覆盖已有文件。
```

多卷压缩文件如 `.z01/.zip` 或 `.7z.001` 不假定普通 ZIP 读取器能处理。先收齐所有分卷，按发布者说明使用兼容解压程序；不要把一个分卷当成完整数据集。解包后检查目录，再编辑 `configs/cadp/genimage.yaml` 的 `domains`。发布版本的目录名可能与示例不同，**修改映射即可，不必移动整个数据集**。

期望的逻辑结构是：

```text
/data/aigcdetect/datasets/GenImage/
  Stable Diffusion V1.4/
    train/nature/...图片...
    train/ai/...图片...
    val/nature/...图片...
    val/ai/...图片...
  Stable Diffusion V1.5/...
  Midjourney/...
  ADM/...
  GLIDE/...
  Wukong/...
  VQDM/...
  BigGAN/...
```

`nature/real/0_real` 都映射为 0；`ai/fake/1_fake` 都映射为 1。目录标签不明确时程序报错，不猜测标签。自定义 HTTP 资产必须在下载 YAML 填入发布方提供的 SHA256；Hugging Face 下载器支持固定 revision 的 snapshot，但不同数据集的 Parquet/image schema 不能凭空通用转换，必须先转换为下一节的本地图像和 CSV。

### 5.1 已保存到本地的 Hugging Face Arrow 数据

`export-arrow` 支持 Hugging Face `Dataset.save_to_disk` 的本地目录，要求列名为 `image`、`label`，标签为二值 0/1。支持直接数据集目录，或 `data/test/生成器名` / `test/生成器名` 等分层目录。它不是通用 Parquet 转换器；其他 schema 需要明确适配，程序不会猜测标签。

```bash
python -m cadp.cli export-arrow --snapshot /data/aigcdetect/datasets/downloads/my_snapshot --output /data/aigcdetect/datasets/arrow_images --split test --fake-label 1  # 原标签 1 是 fake；保持原图编码字节，不重新压缩 JPEG。
python -m cadp.cli scan --root /data/aigcdetect/datasets/arrow_images --source local_arrow --split test --output /data/aigcdetect/manifests/arrow_test.csv  # 转成通用图像 CSV 清单，再登记到 YAML 的 data.tests。
```

发布者用 0 表示 fake 时，必须明确改为 `--fake-label 0`。导出时逐图检查格式、原始字节 SHA256 和已有文件冲突；已导出且内容一致的文件可复用。下载器 `huggingface` 模式需填写发布方真实 `repo_id`、固定 `revision` 及所需文件范围；示例中不虚构一个特定数据集的 schema 或授权。

## 6. 构建训练、验证、测试清单

### 6.1 默认 GenImage：SD1.4 训练，八域测试

先编辑 `configs/cadp/genimage.yaml` 的根目录和八个域的实际文件夹名，再执行：

```bash
python -m cadp.cli prepare-protocol --config configs/cadp/server.yaml --protocol configs/cadp/genimage.yaml --source sd14  # 扫描真实图像、计算哈希、从源 train 划出验证集、登记八个测试域。
python -m cadp.cli audit --config /data/aigcdetect/manifests/sd14.yaml --rehash  # 重新读取图片字节，检查标签、损坏文件及 train/val/test 泄漏。
```

输出包含 `source_sd14/train.csv`、`source_sd14/val.csv`、`tests/<域名>.csv`、`sd14.yaml` 和 `protocol.json`。后续正式训练使用这个生成的 **`sd14.yaml`**，不是仍然没有测试集列表的初始 server.yaml。

这里做了一个重要区分：**GenImage 官方目录名 `val` 在本协议中用作最终测试集**。模型选择验证集从 SD1.4 的官方 `train` 中按组划出 10%；split seed 固定为 42，并不随训练 seed 42/43/44 改变。不要用最终测试集既调参又汇报成绩。

### 6.2 自己的数据或额外 OOD 测试集

CSV 至少包括 `path,label`；本项目生成和严格审核使用完整字段：

```csv
path,label,source,split,group,sha256
/data/.../real_001.jpg,0,cameraA,train,content_group_001,实际64位哈希
/data/.../fake_001.png,1,generatorA,train,content_group_002,实际64位哈希
```

不要照抄上面占位哈希。用扫描命令自动生成：

```bash
python -m cadp.cli scan --root /你的训练图片根目录 --source my_source --split train --output /data/aigcdetect/manifests/all_train.csv  # 从明确的 real/fake 子目录识别标签并计算内容哈希。
python -m cadp.cli split --config configs/cadp/server.yaml --input /data/aigcdetect/manifests/all_train.csv --train-output /data/aigcdetect/manifests/train.csv --val-output /data/aigcdetect/manifests/val.csv --fraction 0.1  # 按组划分训练与验证。
python -m cadp.cli scan --root /你的独立测试图片根目录 --source unseen_generator --split test --output /data/aigcdetect/manifests/unseen.csv  # 登记完全独立的测试集。
```

然后在自己的配置中写：

```yaml
data:
  train: train.csv
  val: val.csv
  tests:
    - name: unseen_generator
      manifest: unseen.csv
```

相对 manifest 文件名相对于 `paths.manifest_dir`；CSV 内的相对图片路径相对于 `paths.data_root`，不是 CSV 所在目录。可以在扫描时使用 `--relative-to /共同的数据根目录` 创建可迁移的相对路径清单。

扫描默认用文件内容 SHA256 作为 group，能够阻止同一文件内容跨拆分泄漏，**不能识别所有近重复图、语义配对或同一视频相邻帧**。这类数据必须在划分前设置共同 `group`，例如“同一原始内容的所有增强版本/视频帧/生成配对”。每个分层至少需要两个独立 group。相同字节被标为不同真假标签时直接拒绝。

独立 `train` 与 `preflight` 只读取 train/val，不要求先安装测试集；`evaluate` 和完整实验套件会检查全部训练/验证/测试清单，防止泄漏。

## 7. 在真正的 GPU 上验收一次

```bash
export CUDA_VISIBLE_DEVICES=0  # 本进程只看物理 GPU 0；程序中的 cuda:0 就是这张卡。
python -m cadp.cli doctor --config /data/aigcdetect/manifests/sd14.yaml  # 确认当前激活环境及本地权重路径。
python -m cadp.cli preflight --config /data/aigcdetect/manifests/sd14.yaml  # 使用真实权重和真实训练图片完成前向、六项损失、反向及有限梯度检查。
```

结果写入该配置 `paths.output_dir/preflight.json`，包含 `device`、`tiny_random_backbone`、CLIP 哈希、各项损失和梯度范数。正式验收必须看到 `tiny_random_backbone: false`、实际 CUDA 设备及通过状态。此命令不修改模型权重，不代表 100 epoch 已跑完，也不证明准确率达到目标。

从 CPU 小网络测试不能推出任何 GPU 峰值显存保证。OOM 时按第 3 节降低 microbatch / text chunk，检查其他进程占用显存，然后重新 preflight。默认 DCT loss 对正负 patch 距离做“求和”而不是 patch 数量均值，因此数值可能远大于分类 loss；程序按附件保留该定义，日志会分别记录，不能擅自改权重后仍当成原设定。

## 8. 正式完整训练与断点恢复

### 8.1 单个完整训练任务

```bash
python -m cadp.cli train --config /data/aigcdetect/manifests/sd14.yaml  # 使用全部训练清单，完整训练配置中的 100 个 epoch。
```

没有 `--stop-after-epoch`，没有 `model.tiny=true`，没有图片数上限。AdamW 使用新模块 lr=1e-4、视觉 LoRA lr=1e-5、weight decay=1e-4；默认启用可关闭的 cosine 调度和梯度裁剪，见方法映射中的工程约定。

每轮完整结束后原子写入 `last.pt`；只有源验证集 AUROC 严格提升时更新 `best.pt`。测试集从不参与选择。训练文件如下：

```text
runs/sd14_seed42/
  config.yaml
  data_audit.json
  parameters.json
  history.jsonl
  last.pt
  best.pt
  best_validation.predictions.csv
  best_validation.metrics.json
```

`history.jsonl` 每行是一轮的全部损失、验证指标、耗时和学习率。CADP checkpoint 只保存新增模块、LoRA 和必需的缓冲区，不重复存整套冻结 CLIP；另外保存优化器、调度器、scaler、训练轮数、模型配置、基础模型哈希和清单哈希。

```bash
tail -n 2 /data/aigcdetect/runs/sd14_seed42/history.jsonl  # 查看最近两轮实际记录。
python -m cadp.cli train --config /data/aigcdetect/manifests/sd14.yaml --set train.resume=/data/aigcdetect/runs/sd14_seed42/last.pt  # 从最后一个完整 epoch 恢复，不重新开始。
```

恢复必须指向相同 run 目录，并保持训练协议一致。移动机器时移动整个 run 目录及数据/CLIP，再改本地路径；不要只复制 last.pt 到一个空输出目录却丢失 best.pt。中途断电最多重跑尚未写入 checkpoint 的那一轮，不声称支持 batch 中途逐步恢复。完成 100 轮后再次恢复不会凭空多训练 100 轮。

已有 checkpoint 的目录不允许无 `resume` 重开实验；改随机种子、消融配置或训练长度时使用新目录。**本版本每个实验用一个设备，不提供 DDP/torchrun**；多 GPU 可运行独立 seed 或不同训练源任务，并为每个任务分配不同输出目录。

### 8.2 三随机种子主实验

```bash
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group main --output /data/aigcdetect/experiments/sd14  # 只生成完整实验计划与每个任务配置，不训练。
bash scripts/cadp/full.sh /data/aigcdetect/manifests/sd14.yaml /data/aigcdetect/experiments/sd14 main  # 审核、GPU preflight、三个 100-epoch 训练及各自八域测试。
```

三个训练 seed 是 42、43、44，每个训练都使用同一训练/验证划分。每个 run 自己的 `best.pt` 用于自己的全部测试。已经完成且配置、数据和依赖 checkpoint 未变的任务可跳过；失败任务写独立日志并停止后续任务，不吞掉异常。

## 9. 使用训练结果评测与推理

以下使用单 run 的示例路径；批量 suite 的 checkpoint 路径是 `experiments/sd14/runs/full/seed42/best.pt`，配置也在对应 run 下。必须配对使用，不能把固定长度消融的 checkpoint 交给完整模型配置。

```bash
python -m cadp.cli evaluate --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --output /data/aigcdetect/runs/sd14_seed42/evaluation  # 对 YAML 中所有独立测试域评测，默认 S=10、阈值 0.5。
python -m cadp.cli predict --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --input /你的图片.jpg --output /data/aigcdetect/prediction.csv  # 单图推理，不要求真假标签。
python -m cadp.cli predict --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --input /你的图片文件夹 --output /data/aigcdetect/folder_predictions.csv  # 递归逐图推理支持的图像格式。
```

每个域输出逐图 CSV 和指标 JSON；包含 accuracy、balanced accuracy、真实/生成图准确率、AUROC、AP、F1、TPR@FPR、ECE、Brier、NLL、混淆矩阵、动态长度直方图。指标数值为 0–1，不是已经乘 100 的百分数。只有一个类别时 AUROC/AP 写 `null`，不是伪造 0 或 1。

最终 `summary.json` 给出八域分别的结果和域间宏平均，避免把在多个域重复出现的真实图片全部混在一起而误导微平均。图片级 bootstrap 可以设置 `--set eval.bootstrap=1000`；它不替代训练随机种子标准差，也不是组级置信区间。

### 阈值校准（单独报告，不替换默认 0.5 主结果）

```bash
python -m cadp.cli calibrate --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --output /data/aigcdetect/threshold.json  # 只读取配置中的验证集，用 Youden J 选择阈值。
THRESHOLD=$(python -c "import json; print(json.load(open('/data/aigcdetect/threshold.json'))['threshold'])")  # 从实际校准文件读取数字。
python -m cadp.cli evaluate --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --set "eval.threshold=$THRESHOLD" --output /data/aigcdetect/runs/sd14_seed42/eval_calibrated  # 对测试集应用事先选择好的阈值，另存结果。
```

`threshold.json` 记录 checkpoint 和验证清单哈希。使用者必须确保把它用于同一个 checkpoint、相同采样数和退化设置，不能对着测试集改阈值追求更高分。

### 注意力与效率

```bash
python -m cadp.cli predict --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --input /你的图片.jpg --output /data/aigcdetect/explain/prediction.csv --attention  # 额外导出每次概率采样、repository、类别、head、context、patch 的注意力 NPZ。
python -m cadp.cli benchmark --config /data/aigcdetect/runs/sd14_seed42/config.yaml --checkpoint /data/aigcdetect/runs/sd14_seed42/best.pt --output /data/aigcdetect/benchmark.json --warmup 3 --repeats 20  # 同步 CUDA 后测量完整 S=10 前向吞吐、P50/P95 批延迟与峰值 allocated 显存。
```

注意力数组形状为 `[B_group,S,K,2,heads,M,Npatch]`，ViT-L/14 的 `Npatch=256`。这是相关性可视化数据，不是因果证据。本入口不自动生成可能误导的“伪造区域真值”。效率测量包含视觉编码、动态长度、全部 MC 文本推理及 Cross-Attention，不包含磁盘读取、预处理、CPU→GPU 拷贝和权重加载；报告中明确区分。

## 10. 完整消融、鲁棒性、MC 与跨源矩阵

```bash
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group all --output /data/aigcdetect/experiments/sd14  # 查看全部任务数量及配置，再决定占用哪些计算资源。
bash scripts/cadp/full.sh /data/aigcdetect/manifests/sd14.yaml /data/aigcdetect/experiments/sd14 all  # 完整执行：45 个训练任务、81 个评测/效率任务，共 126 个任务。
```

这里的 45 个训练任务是 **完整方法 + 14 个消融，共 15 个方法变体 × 3 个 seed**。不是 45 个 smoke。默认每个训练 100 epochs，真实运行量可能很大，计划文件不会把它隐藏起来。

主实验完成后，也可按组单独补跑，`--resume` 会检查已完成任务：

```bash
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group ablations --output /data/aigcdetect/experiments/sd14 --execute --resume  # 逐个重新训练消融，不把推理开关当成训练消融。
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group robustness --output /data/aigcdetect/experiments/sd14 --execute --resume  # 完整方法在 JPEG、模糊、缩放、噪声下测试，不重新训练。
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group mc --output /data/aigcdetect/experiments/sd14 --execute --resume  # 同一 checkpoint 使用 1/5/10/20 次样本，检验性能与采样成本。
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group efficiency --output /data/aigcdetect/experiments/sd14 --execute --resume  # 每个 seed 测完整推理效率。
python -m cadp.cli aggregate --root /data/aigcdetect/experiments/sd14 --output /data/aigcdetect/experiments/sd14/aggregate  # 汇总实际完成的结果；没有结果时明确报错。
```

`aggregate/benchmarks.csv` 保存实际完成的效率结果；只有 benchmark 时仅输出效率结果，不生成虚假准确率。`aggregate/per_seed.csv` 保存每个 seed；`mean_std.csv` 保存样本标准差 `ddof=1`。只有一个 seed 时标准差为 null，不当成 0。不同退化、阈值、MC 数量分别分组，不能混成同一主表。

可选八训练源 × 八测试域矩阵，需要下载八个训练域，而不是只有 SD1.4 train：

```bash
python -m cadp.cli prepare-protocol --config configs/cadp/server.yaml --protocol configs/cadp/genimage.yaml --all-sources  # 分别构建八个训练源自己的 train/val，以及共享八域测试清单。
nano configs/cadp/suite.yaml  # 将 cross_source_configs 填为上一条输出的八个真实 YAML 绝对路径。
python -m cadp.cli suite --config /data/aigcdetect/manifests/sd14.yaml --group cross-source --output /data/aigcdetect/experiments/cross_source  # 预览 8 源 × 3 seed 的训练及其全部测试任务。
bash scripts/cadp/full.sh /data/aigcdetect/manifests/sd14.yaml /data/aigcdetect/experiments/cross_source cross-source  # 实际执行完整跨源矩阵。
```

`all` **不自动包含**这项额外的八源矩阵，避免不知情启动更多大规模训练。矩阵结果中按训练源和测试域读取 `mean_std.csv` 即可整理 8×8 表。详情与限制见 `EXPERIMENTS_ZH.md`。

## 11. 运行原 PPM-CLIP 对照模型

原始 `main.py/test.py` 保留原有行为，包括其硬编码数据路径、旧依赖和训练协议，不作为新方法的正式入口。新增 `cadp/legacy.py` 直接使用原始 `PPM_clip`、原始 prompt/PFL/LoRA 架构，仅做进程内适配，不写原文件。

```bash
bash scripts/cadp/baseline.sh /data/aigcdetect/manifests/sd14.yaml /data/aigcdetect/experiments/baseline  # 原始 PPM 架构，三个 seed，使用同一 CSV 划分、完整训练和八域评测。
```

该对照称为 **matched-harness PPM baseline**：使用和新方法一致的数据划分与验证集选模、相同优化器组策略；FP32 运行原始 flow，保留原始随机挑一个 repository 的训练方式及 orthogonal loss。原版常量固定 K=2、shared=3、private=7、class=10、flows=10、rank=4、alpha=.5。不要称它“逐项复现原论文成绩”，原训练器还有基于测试成绩早停等不同设置。

适配修复了原全局 flow 硬编码 `.cuda()`、LoRA 模式切换的权重合并副作用，并强制从本地加载 CLIP；原 PPM 推理按单图固定随机噪声，防止缓存跨图片/批次影响。基线 checkpoint 为完整原网络 state，体积明显大于新方法的增量 checkpoint；两个方法的 checkpoint 不能混用。原模型没有新增 Cross-Attention，因此不支持 `--attention`，也不伪造概率样本方差。

## 12. 只检查程序的 CPU 测试方式

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1            # 避免环境中不相关 pytest 插件联网/挂起或改写测试行为。
python -m pytest -c pytest-cadp.ini -p no:cacheprovider -q  # 运行隔离的自动化测试，避免把上游 test.py 当 pytest 文件。
python scripts/cadp/check_sources.py               # 解析全部原始/新增 Python 和 shell 脚本，且不改上游 pyc。
bash scripts/cadp/cpu_smoke.sh /tmp/cadp_acceptance_new  # 在一个新目录生成随机 fixture，跑训练、恢复、评测和推理闭环。
```

`cpu_smoke.sh` 是开发验收，不是第 8–10 节的真实实验替代品。随机图片的真假标签只是接口测试标签，不具有数据集含义。目录必须全新，避免覆盖已有结果。

## 13. 常见失败与正确处理

| 报错/现象 | 检查与处理 |
|---|---|
| `No module named cadp` | 确认在仓库根目录、正确 conda 环境运行 `python -m cadp.cli`。 |
| `Local CLIP checkpoint missing` | YAML 路径要指向真实 `.pt` 文件；先执行显式下载或传输。不要填文件夹或 Hugging Face repo 名。 |
| `SHA256 mismatch` | 不继续训练；核对版本与下载完整性，明确移走错误文件后重下。 |
| 数据目录找不到 | 修改 GenImage YAML 的域名→实际目录映射；确认是否多套了一层解压目录。 |
| `Data leakage` | 调整真实数据拆分/分组，不删除审核来追求跑通；检查同一真实图片是否混入 train 与 test。 |
| `Cannot decode/preprocess image` | 修复或明确清理损坏图像、重新生成清单；程序不会静默返回黑图。 |
| CUDA OOM | microbatch 降至 1、累积相应增大、启用块 checkpoint、减 text_chunk/eval batch，关闭其他占卡任务。 |
| bf16 不支持 | 根据 GPU 能力改成 fp16 或 off，然后重新 preflight；保持实验记录可追溯。 |
| 非有限 loss/梯度 | 停止检查精度、学习率、数据与损失尺度；不能在日志里替换 NaN 后继续当正常结果。 |
| `Checkpoint model configuration differs` | 使用该 run 保存的 config.yaml，只改本地路径及明确允许的评测/吞吐字段。 |
| `Training source code changed` | 在新的实验目录重跑，避免复用旧代码训练出的结果却当成新代码实验。 |
| pytest 显示进度完成后不退出 | 使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，排除系统预装的无关插件。 |
| 某个指标为 null | 单类别 AUROC/AP、单 seed 标准差本来没有定义，需补齐相应数据/seed。 |

训练前完整哈希和图像审核可能耗时，尤其在大型远程文件系统。首次审核通过并锁定数据后，可明确设 `data.hash_check: false`、`data.verify_images: false` 减少重复逐文件读取；仍要求清单哈希存在，但这样不再重新验证当前图片字节，必须理解这个取舍。

## 14. 交付后你实际需要完成的事情

你需要在服务器准备合法获得的真实数据和本地 CLIP，确认 YAML 路径与驱动，完成真实 GPU preflight，再执行完整训练/实验矩阵。需要论文结果时，还要检查真实/生成图内容泄漏、数据来源偏差、跨生成器泛化和统计显著性，不能把软件测试通过写成方法有效性证据。

本交付不会声称已在你的服务器跑过，不提供虚构准确率、显存占用或训练用时保证。所有正式结果由上述命令在你真实环境中产生，并保留配置、输入哈希、checkpoint 和逐图预测以供核验。
