# 实验协议与完整矩阵

## 1. 数据协议不是隐藏假设

附件给出了方法、100 epochs 和优化器等设置，没有逐一列出训练/测试数据及全部论文表格。本项目新增默认 GenImage 协议：SD1.4 官方 train 按内容 group、标签分层切出10%源验证集；八个域的官方 val 作为最终测试集。数据划分 seed=42，与训练 seed 分离；所有方法使用同一 CSV。

这不是宣称已经获得 GenImage/PPM-CLIP 论文中的同一训练子集，更不是声称已复现其数值。若要对齐某篇论文，应根据该论文实际使用的训练类目、样本数、拆分、预处理、验证策略另建清单，并明确版本。添加其他数据集只需准备同结构本地图像及 CSV，不能把另一个目录的标签含义默认等同于 GenImage。

源内与跨源结果分别报告。SD1.4 是见过的生成源；另外七个域可以在该单源训练协议下作为跨生成源测试域。跨源八训练源矩阵进一步检查训练源选择是否决定优势。训练期间不读取最终测试标签做优化、早停或阈值选择。

## 2. 主方法 + 14 个重新训练的消融

| 实验 ID | 对完整方法的唯一主要改动 | 解答的问题 |
|---|---|---|
| full | 完整附件方法 | 主结果 |
| no_cross_attention | 去掉 CA 和局部融合，只用 EOS 原型 | 局部跨模态交互是否有效？ |
| fixed_private_1 | 每图固定 private=1 | 很短提示是否足够？ |
| fixed_private_3 | 固定 private=3 | 动态长度是否优于中短长度？ |
| fixed_private_5 | 固定 private=5 | 动态长度是否优于中长长度？ |
| fixed_private_7 | 固定 private=7 | 性能是否只来自总是使用最长提示？ |
| no_flow | 0层 Flow，保留 Gaussian 和 token generator | 可逆分布变换是否有增益？ |
| no_patch_loss | DCT contrastive loss 权重0 | 局部视觉正则是否必要？ |
| fixed_class_anchors | class residual 冻结为0，并去掉无作用的 anchor penalty | 类别锚定残差学习的作用？ |
| no_length_penalty | length penalty 权重0 | 路由是否总选择长提示？ |
| one_repository | K=1 | 多 repository 的作用？ |
| no_vision_lora | 不插视觉 LoRA，原视觉 backbone 完全冻结 | 参数高效视觉适配的作用？ |
| deterministic_latent | train/test epsilon=0，其他结构不变 | 随机采样的作用？这不是“删除整个概率模块”。 |
| no_reconstruction | rec 权重0 | private latent 重构约束的作用？ |
| no_kl | KL 权重0 | 分布正则的作用？ |

默认15变体×3训练seed=45个训练任务，每个100 epochs、全量清单。每个变体都有单独配置、优化器、best checkpoint 和测试输出。消融不是加载 full checkpoint 后在推理时随手关掉模块；那样无法回答这些训练机制的作用。

新的固定长度 checkpoint 与动态长度 checkpoint 在结构配置校验上不同，不互相冒用；其他消融同理。默认长度候选始终为1/3/5/7，固定分支会停止训练不使用的路由参数。

## 3. 鲁棒性

对 full 的每个 seed checkpoint 进行七种额外退化，不再训练：JPEG quality65/30、PIL GaussianBlur radius3/5、224裁剪后降采样到112/64再上采样、像素[0,1]域 Gaussian noise std=.01。噪声由图片内容哈希和退化配置确定，在相同图片上可复现。

**退化是在基准 crop 后施加，且 blur 使用 Pillow 的 radius 实现。** 这是清楚定义的项目鲁棒性协议，不冒称与所有已有论文的退化顺序/实现完全相同。真实与生成图使用相同退化，防止标签相关处理泄漏。需要不同强度，编辑 suite.yaml 并另存结果，不能混入 clean 主结果。

每个评测输出 accuracy/AUROC/AP 等、每图得分和长度分布。应同时观察分数校准、真实图误报率和生成图漏报率，而不仅看总体 accuracy。

## 4. Monte Carlo 与效率

同一个 full checkpoint 使用 S=1/5/10/20，K仍为2。取相同固定 noise bank 的前 S 项，比较差异时不额外引入完全不同的随机噪声集合。对应报告性能、ECE/Brier、长度统计和推理成本。默认 full 主结果 S=10，不把最有利的 S 在测试集上选完后还叫固定设置。

默认每个 seed 测一次完整 S=10 端到端 GPU forward 的吞吐及峰值 allocated 显存，包含文字端多次采样和 CA，不能只测一次视觉编码。输入预处理/读盘不计入此 forward 基准，结果 JSON 明确写出范围。可对 MC 不同 S 另外执行 benchmark 命令，保存各自文件。

NPZ attention 可用于展示两种类别 query 是否关注不同 patch，但不能把 attention 大小当成真实伪造位置标签或因果证明。项目不凭空构造区域真值。

## 5. 原 PPM 架构对照

运行 `scripts/cadp/baseline.sh`，得到额外3个完整 PPM baseline 训练和八域测试。该脚本不包含在 CADP 的45个消融训练里。原 PPM 架构直接来自上游，不用普通CLIP线性头替代它；数据清单、验证选模与指标基础设施对齐，工程适配和FP32设置见运行指南。

这个对照只能支持“在本项目明确协议下”的比较。原论文报告值可以作为带来源的参考行，但不能在没实际运行相同协议时冒充本项目重现行。性能提升是否稳定，要看三个 seed 的均值、标准差和各个测试域，不用单一最佳seed替代主表。

## 6. 八源 × 八域扩展

`prepare-protocol --all-sources` 产生8个独立训练源配置，每个训练源有自己的训练/源验证集。cross-source组默认训练8×3=24个模型；每个模型测试8个域，生成192份域级结果，聚合为每个指标的一张8×8均值/标准差表。

八源矩阵是单独的可选运行组，不在 `all` 里暗中启动。官方各测试域之间可能复用真实图，允许 test/test 重叠；不允许任何 train/val/test角色之间重叠。宏平均与图像级 bootstrap不能消除这些域间相关性，统计检验时应继续考虑内容group和配对预测。

## 7. 可直接用于论文的产物

主表可从 `aggregate/mean_std.csv` 按 variant=full、corruption=clean、samples=10、threshold=.5 提取八域 AUROC/AP/accuracy；消融表固定协议比较15变体；鲁棒性表分开不同退化；MC表比较S；效率表来自真实GPU benchmark JSON；长度直方图与attention来自实际预测输出。

标准差用样本标准差ddof=1，一份seed结果不报虚假的0方差。测试结果不会因为某个域只有一个类别就伪造AUROC=0。所有训练/评测保存输入清单及基础模型指纹；汇总器只处理完成的 `cadp-evaluation-v1` 输出，不创建占位结果。

论文仍需要真实训练和运行上述实验、对照更多合适方法、错误案例分析、计算代价与局限讨论，以及对数据重复、JPEG来源、语义类别等潜在捷径的审查。**有一套能运行的研究代码，不等于已经证明方法具有泛化优势或可以保证论文录用。**
