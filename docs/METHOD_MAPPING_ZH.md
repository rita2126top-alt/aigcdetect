# 附件方案到实现的逐项映射

本项目以用户给出的《方法设计思路报告》为方法规范。`CADP-CLIP` 仅为实现工作名称。默认模型不是普通 CLIP 线性分类头，也不是简单在原 PPM 推理后增加一个无梯度的注意力可视化模块。

## 1. 主数据流

```text
image → 原 CLIP visual + 上半层 qkv LoRA
      → raw projected CLS g + LN 后全部 patch tokens V
      → detached normalized g → length router
      → raw g → private Gaussian + conditional planar flow
      + global shared Gaussian + global planar flow
      → token-specific shared/private offsets + 2 个 context repositories
      → 同时构建 real 与 fake 两个假设，不读取标签
      → 冻结、但允许对输入反传的 CLIP text encoder
      → EOS feature + class-conditioned context hidden states
      → context 作 Q、全部视觉 patches 作 K/V 的 cross-attention
      → 分别归一化 EOS/local feature，再有界残差融合
      → normalized CLS 与两个原型的 scaled cosine
      → 每个 real/fake pair 内 softmax
      → 训练按 repository/sample 分别 NLL；推理平均 pair 概率
```

## 2. 公式对应表

| 附件部分 | 实现位置 | 关键约束 |
|---|---|---|
| 式 16–27：视觉特征 | `cadp/backbone.py:Backbone.encode_image` | 最终所有 token 都过 visual LN；CLS 过 CLIP visual projection，patch 不被压成单个 CLS。 |
| Vision LoRA | `QKVLoRA` | ViT-L/14 的 12–23 层，q/k/v、rank=4，原投影冻结；不做破坏性合并。 |
| 式 28–32：class anchor | `Detector.class_residual/word_embeddings` | real/fake 的真实 BPE embedding + 零初始化 residual；token 是否为单 BPE 在构造时校验。 |
| 式 33–38：banks/长度 | `Detector.prompt_shared/prompt_private` | K=2，shared=3、private 最大 7，候选 1/3/5/7。 |
| 式 39–45：路由 | `Detector.forward` | 只对 detached normalized CLS 路由；训练 ST hard Gumbel，测试 argmax。 |
| 式 46–55：概率基分布 | `ProbabilisticContext` | shared 全局均值/对数方差；private 条件于 raw CLS；latent dim = CLIP joint dim。 |
| 式 56–62：可逆 Flow | `planar_step` / `_flow` | 10 步，u_hat 可逆重参数化；shared 全局参数、private 参数由 g 线性生成。 |
| 式 63–74：token-specific bias | shared/private generator | 两层 GELU MLP 输出整个 token 集合，每个 token 有独立坐标，rho=.1；同一 pair 完全共享 context。 |
| 式 75–85：文本 | `build_prompts` / `encode_text_embeddings` | `[SOS,a,class,photo,of,contexts,EOS,pad]`，EOS 明确在 5+M，不用 token ID 最大值猜 learned prompt 的 EOS。 |
| 式 86：冻结 | `Backbone.__init__` 与 `_block` | text 参数 requires_grad=False，但训练没有用 no_grad 切断 prompt 梯度。 |
| 式 87–99：跨模态交互 | `TokenCrossAttention` | 一组 Q/K/V 投影，256 维、4 heads、Pre-LN、残差、4倍 FFN、dropout=.1，两个类别共享模块参数。 |
| 式 100–107：融合 | `Detector._branch` / alpha | mean pool 后投影；EOS 与 local 分别 L2 normalize，再 Norm(t + alpha r)；alpha=.5 sigmoid，初始 .1。 |
| 式 108–112：预测 | `Detector._branch/forward` | 温度初始 exp(log(1/.07))；pair softmax；长度按 hard ST 权重选择。 |
| 式 113：分类损失 | `detection_loss` | 对 B×S×K 个正确类概率分别 log，再求均值；不是先把概率平均再做 NLL。 |
| 式 114–120：KL | `ProbabilisticContext.forward` | q0 density − flow logdet − standard Gaussian prior；shared+private 除 latent dim。 |
| 式 121–123：重构 | private decoder | 重构 raw g，target.detach；MSE 自动对 B×S×d 求均值。 |
| 式 124–128：DCT | 原 `DCTPatches` + `Backbone.patch_loss` | 直接复用六频段加权打分与 top/complement 选择；正样本距离求和，负样本 hinge，margin=1。 |
| 式 129–132：正则/总目标 | `Detector.forward` / `detection_loss` | anchor 相对原词向量范数；length 为期望 private/7；权重完整保持 .5/.001/.5/.001/.001。 |
| 式 137–141：测试 | eval 分支 | 固定 10 组 Gaussian 基噪声、2 repositories，每个 pair 先 softmax，再对 S×K 平均，fake≥.5。 |

## 3. 需要明确说明的实现选择

**详细公式优先。** 附件总览的融合式与详细式 107、后面论文简写式并非完全一样。本实现遵循详细式 103–107：先各自 normalize EOS 和局部投影，再融合和 normalize，而不是把未归一化的局部向量直接加进去。

**LoRA alpha 的含义。** 原仓库 `loralib/layers.py` 使用 `alpha/sqrt(rank)`。因此 .5 与 rank=4 的实际缩放是 .25；不能不说明就改成常见的 alpha/r=.125。本项目的 nonmerging qkv adapter 保持原缩放，单元测试校验该数值及原权重不被 train/eval 切换改写。

**冻结不等于截断梯度。** 原 visual/text 权重冻结；LoRA、新 tokens、class residual、flows、generators、router、CA、fusion、temperature 和 decoder 学习。文本 transformer 在训练时保留 autograd；块级 activation checkpoint 使用非 reentrant 路径。梯度测试分别检查 router 的纯分类梯度及 text 输入上游模块梯度。

**硬路由不等于训练只计算一条分支。** 要得到附件式 112 的 straight-through 任务梯度，训练需要四种长度的概率。前向输出只由一个 one-hot 长度贡献；反向利用软选择的梯度更新 router。若先 `.argmax().item()` 选完一个长度再只算那条分支，router 得不到这个分类梯度。本实现在测试时才按选择长度分组，避免四倍无意义测试文本计算。

**固定 MC 基噪声。** checkpoint 保存 shared/private 两套标准正态 noise banks，默认容量 50。测试取前 10 项，不在每个 batch 重新抽噪声。private 使用 common random numbers：同一组基础噪声作用于不同图片的各自条件分布；不是让不同图片拥有相同 latent。这个选择使更换 batch size/顺序不会改变噪声估计样本，数值结果仍可能有浮点末位差别。训练 shared 每 microbatch 一次，private 每图片独立重参数化采样。

**未规定的 MLP 隐层。** 附件规定了两层 MLP 但未指定 hidden width；本实现统一默认 256，可配置并写入 checkpoint。Cross-Attention 按一组 Q/K/V projection + 分头 attention + 输出投影实现，没有在它们后面再隐式叠加第二组未说明的 QKV 线性层。

**数值稳定与训练工程默认值。** Gaussian logvar clamp 到 [-12,8]；flow/density 用 FP32；向量归一化、logdet 和 log probability 做最小数值保护。单样本 KL 估计可能为负，不擅自 clamp 到零。AdamW 外额外提供默认 cosine 调度和 norm=1 的梯度裁剪，这两项不是附件新增损失：要用恒定学习率和无裁剪，设 `train.scheduler=none, train.grad_clip=0` 并另记实验配置。训练 epoch 参数取 1…E，Gumbel 温度按附件 max(.1,1−.9e/E)，最后一轮达到 .1。

**有效 batch 与显存策略。** 默认 microbatch2、累积24，优化器每48图片更新一次。与物理48 batch 的随机 shared sample 组织不同，不声称逐步数值等价。需要严格物理48时将配置改成48/1并先实测显存。

**图像预处理。** 保留上游“小图先 resize 到方形256，再 crop224”的基本策略，训练 random crop/hflip，测试 center crop；新增 EXIF 朝向规范化和损坏图像 fail-fast。没有改为 CLIP 官方默认 resize-shorter-side 然后 center crop 却不说明。DCT 在与原实现一致的归一化图像张量上计算。

## 4. checkpoint 与数据契约

CADP checkpoint 省略冻结 CLIP 参数，但包含其文件 SHA256；加载时校验模型结构配置、全部预期增量 state keys 和基础模型指纹。`strict=False` 仅用于省略已校验的冻结 backbone keys，并不允许缺失新模块参数静默通过。自己的 checkpoint 用 `weights_only=True` 读取。

图像 forward 没有 label 参数。标签只在 `detection_loss` 和指标模块使用。推理可以不提供清单和标签，直接输入本地图片。训练/评测清单角色明确；验证选模、验证阈值校准与最终测试隔离；结果同时记录 checkpoint、原始 CLIP 与 manifest 哈希。

## 5. 原 PPM baseline 与新方法不是同一个结构

`model.method=ppm_baseline` 调用原始 `PPM_clip`，仍有原 real/fake class prompt 和对应 PFL，不是把 CADP 的 CA 关闭后冒充 PPM。进程内适配只解决本地路径、设备兼容、LoRA 合并副作用及可复现的单图测试；训练与验证基础设施使用同一套 manifest/选模策略。

该 matched baseline 改进了原训练流程的测试集选模问题，因此必须如实标记，不声称是上游训练脚本逐步重放。原文件包括这个历史训练行为全部保留，可以通过固定 commit 和文件哈希审计。
