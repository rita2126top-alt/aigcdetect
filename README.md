# aigcdetect — PPM-CLIP 原版快照 + CADP-CLIP 新方法

本仓库保存未经修改的 PPM-CLIP 源码快照，并在独立 `cadp/` 目录实现用户提供的类别锚定、动态图像条件概率提示、文本 context → 视觉 patch cross-attention 方法。

**CADP-CLIP 是本实现的工作名称，不是声称已有论文名称或已获验证的性能结论。**

## 从这里开始

- [完整中文运行指南：从零安装到全量实验](docs/RUN_GUIDE_ZH.md)
- [公式到代码映射与实现约定](docs/METHOD_MAPPING_ZH.md)
- [实验协议与全部实验矩阵](docs/EXPERIMENTS_ZH.md)
- [实际测试范围及限制](docs/VALIDATION_REPORT.md)
- [第三方代码与数据来源](docs/THIRD_PARTY_NOTICE.md)

**交付/上传状态与补丁发布方式见 [docs/DELIVERY_STATUS.md](docs/DELIVERY_STATUS.md)。**

正式入口是 `python -m cadp.cli`。默认配置 `configs/cadp/default.yaml` 使用**本地 ViT-L/14 权重、100 epochs、有效 batch size 48、测试 10 个概率样本**，不是 smoke 配置。

```bash
# 在 conda 环境中安装 GPU 版依赖后，使用你已经准备好的真实数据配置。
python -m cadp.cli preflight --config /data/aigcdetect/manifests/sd14.yaml
bash scripts/cadp/full.sh /data/aigcdetect/manifests/sd14.yaml /data/aigcdetect/experiments main
```

`full.sh ... all` 执行主方法、14 个消融、鲁棒性、MC 样本数及效率实验；八训练源 × 八测试源矩阵用 `cross-source` 单独执行。每个训练任务使用全部 manifest 数据和配置中的全部 epoch；没有隐藏的图片数截断。

## 原始文件保持不变

原始来源：`bandaidssssss/PPM_CLIP`，固定 commit `09d05b9fc4be6a2b079356bbf337a05e193db0be`。

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m cadp.cli verify-upstream
```

`provenance/upstream.json` 保存 47 个上游文件的 SHA256，包括原仓库已经跟踪的二进制文件。校验目标是 `PPM_CLIP/` 固定子模块；根目录的原始 Python 文件也未改动。README、指南与包配置按新入口更新，旧 `aigcdetect/` 实验代码仅保留供追溯。导入的是该 commit 的文件快照，不是复制整个上游 Git 历史。

## 功能边界

已实现本地数据/权重、训练、恢复、验证选模、多域测试、单图/文件夹推理、阈值校准、注意力导出、多个随机种子、消融/退化/采样数实验和结果汇总。提供同一数据清单上的原始 PPM 架构对照入口。

CPU 自动化测试使用真实上游 CLIP 模块构建的缩小随机网络，而非伪造网络输出。**这不等于真实 ViT-L/14 预训练权重的 GPU 全量训练验收，不代表检测准确率得到验证。** 请按指南在真实服务器运行 `preflight`，再启动正式实验。
