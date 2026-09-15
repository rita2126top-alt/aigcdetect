# 交付验证记录（不是论文实验结果）

## 结论

新增 CADP 实现、运行基础设施和文档已完成，并成功上传至 `rita2126top-alt/aigcdetect` 的 `main`。实现提交为 `d137f74a5c432797f9a6515601ac327d10cd1514`；GitHub Actions 发布验证 run `34974737395` 结论为 `success`。该远程运行再次得到 **55 tests、0 failures、0 errors、0 skipped**，并核对 PPM-CLIP 47/47 文件。当前发布状态见 `DELIVERY_STATUS.md`。

## 实际执行的检查

| 检查 | 结果 | 记录 |
|---|---|---|
| CADP 自动化测试 | **55 passed，0 failed，2 warnings** | `validation/pytest.log`、`validation/pytest.xml` |
| GitHub Actions 远程复核 | **55 tests，0 failures/errors/skipped；source check passed** | run `34974737395`、artifact `cadp-v02-validation-evidence` |
| Python 源码解析/编译（不生成 pyc） | 73 文件通过 | `validation/source_checks.json` |
| Shell `bash -n` | 16 脚本通过 | 同上 |
| YAML 解析 | 13 文件通过 | 同上 |
| CADP 模块导入 | 14 模块通过 | 同上 |
| CLI 帮助/参数入口 | 20 子命令通过 | 同上 |
| PPM_CLIP 子模块 SHA256 | 47/47 文件一致 | 同上、`provenance/upstream.json` |
| 根目录既有 Python 源码 | 与恢复快照逐字节一致 | `validation/root_source_preservation.json` |
| 完整 Shell 开发验收流程 | 退出码0 | `validation/shell_complete.log` |
| 原 PPM 架构 CPU BF16 训练 | 一轮训练、验证、checkpoint 完成 | `validation/legacy_bf16.log` |
| 实验调度器实际运行 | 两种变体、退化/MC/效率，共9个小规模任务完成 | `validation/suite_resume.log` |
| 调度重复执行 | 9/9 已完成任务被正确跳过 | `validation/suite_skip.log` |
| Editable 安装及仓库外 CLI 导入 | 成功 | `validation/editable_install.log`、`outside_cwd_help.log` |

测试环境为 Linux、Python 3.13.5、PyTorch 2.10.0+cpu、torchvision 0.25.0+cpu，CUDA 不可用。逐项包版本见 `validation/environment_and_results.json`。推荐的服务器 Conda 环境为 Python 3.11 + 匹配 CUDA wheel；不能把本次 CPU 环境说成已经验收了该 GPU 环境。

测试使用原始 CLIP 模块构建的小尺寸随机 Transformer；执行真实张量运算、autograd 和 optimizer.step，不以固定概率或 mock 模型输出替代检测器。开发图片是随机生成的接口 fixture。日志中的 accuracy/AUROC 等数字仅验证计算与文件流，不具有检测研究意义。

## 自动化测试覆盖

严格 YAML 参数与错误输入、Planar Flow logdet 对自动微分 Jacobian、LoRA 零初始化与原权重不变、双类别共享输入 context 与因果编码差异、动态 EOS 索引、纯分类损失向路由器回传、全部14个消融的真实前后向、冻结 backbone、固定 MC 噪声和 batch 划分一致性、pair-wise softmax 后平均、checkpoint key/模型指纹校验、断点恢复和不中断训练增量参数逐项一致、BF16、CPU FP16 内核和 GradScaler 溢出恢复、指标边界、泄漏检查、GenImage 验证集必须来自源 train、退化可复现、解压路径穿越拒绝、本地 Arrow 原始 JPEG 字节保留、HTTP 下载断点/校验/原子落盘的模拟网络边界、仅有效率结果时不伪造准确率表。

HTTP 测试 mock 了网络响应；不能把它写成“已经从发布者下载完整权重或数据”。Arrow 测试实际调用 datasets 的 save_to_disk/load_from_disk，而不是伪造 Arrow 返回值。

Shell 流程实际执行 toy-data → preflight（真实前向、反向、优化器更新）→ 训练一轮 → 恢复到第二轮 → 测试集评测 → 文件夹推理与注意力 NPZ → 验证阈值校准 → 完整推理计时 → 上游哈希核对。

实验调度首次检查遇到本会话命令执行时间上限而中断；再次 --resume 跳过已有7个成功任务，完成剩余2个；最后一次执行9个全部跳过。相关日志一并保留，没有把初次中断日志删除后声称从未中断。

## 警告

两个 pytest 警告来自**未修改的上游** `PPM_CLIP/networks/PFL.py` 第156、240行文档字符串中的 `\s` 转义。Python 3.13 产生 SyntaxWarning，不是 SyntaxError，测试仍全部通过。静态检查也列出根目录同一旧源码的对应警告。本次没有为了消除警告改写原始文件。

## 没有执行的内容

未下载正式 ViT-L/14 预训练文件及全量 GenImage；未连接用户远程 GPU 服务器；未执行真实 CUDA FP16/BF16、GPU 峰值显存测试或100-epoch正式训练；未完成45训练/126任务的真实数据实验；未证明泛化性能优于 PPM-CLIP。GitHub Actions 已完成 CPU 复核，但这不替代 GPU 实验。项目明确只支持每个实验单设备，不宣称支持 DDP。

原 PPM 对照保留其模型结构，但采用本项目的清单、验证选模和设备适配，必须标为 matched-harness baseline，不是原论文数值复现。旧 `aigcdetect/` 与旧 `tools/` 保留供追溯，本报告的运行测试针对新 `cadp/` 入口；旧入口仅做了源码语法检查。

默认微批量2、累积24达到有效batch48，但 shared latent 每个 microbatch 重新抽样，不声称与物理batch48逐步等价。需要严格对应物理batch时配置48/1并重新做 GPU preflight。

因此交付结论是“已通过记录范围内的代码、数学契约与 CPU 端到端验证”，不是“保证任意服务器、显存、驱动、数据集和下载链接永不失败”。真实服务器验收请执行运行指南第7节，再启动正式实验。
