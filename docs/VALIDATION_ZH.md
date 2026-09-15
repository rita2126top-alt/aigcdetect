# 交付验证与已知边界

## 已实际完成的本地检查

记录位于 `provenance/local_validation.json`。Linux x86_64，Python3.13.5，PyTorch2.10.0+cpu，未检测到CUDA。

- 35项pytest测试全部通过。不是只跑语法检查。
- 使用真正的上游CLIP、tokenizer、LoRA和DCT实现，创建随机小型CLIP state_dict，从磁盘经正式加载路径实例化新方法。
- 测试模型视觉24层、宽度64、输入28×28、文本2层宽度64，词表49408、文本长度77；保留两类、两个prompt repositories、四个动态长度和10层flow。
- 实际前向/六项损失/反向、prompt/anchor/router/flow/Cross-Attention/LoRA的非零有限梯度；优化一步后原始冻结参数不变。
- 六种模块消融均执行前向与反向；共享context、标签不进入预测构造、固定噪声推理和不同batch/chunk的输出一致性；Planar Flow解析logdet对自动微分Jacobian。
- 真正的小数据训练、保存best/last、重载、验证/测试、JPEG/blur鲁棒性、单图推理、正式doctor前后向。
- 同一CPU环境连续两轮训练，与一轮后保存/重建模型/恢复再训练一轮，模型state_dict逐项完全相同。测试含梯度累积。
- 使用真实datasets3.6.0和pyarrow20.0.0，覆盖bytes列、HF Image列和IPC；导出字节与输入逐字节一致，分组划分无交叉内容。
- 配置继承、错误键、循环继承、数据加载阶段分离、坏图错误、单类别指标null、21任务规划、已完成结果汇总、13个CLI帮助入口和shell语法。
- Python AST语法解析、YAML解析和39项原版非缓存文件/资源副本哈希校验通过。

本地警告来自PyTorch2.10对torch.jit.load的弃用提示；为兼容正式OpenAI CLIP TorchScript文件保留该读取路径。这不是运行失败。

## 复查命令

```bash
python tools/validate_delivery.py --output validation
```

生成 `summary.json`、`pytest.xml`、`pytest.log`。测试只创建临时随机模型和图片，不下载预训练权重或大数据集，不输出可用作科学结论的检测成绩。

## GitHub验证

仓库提供CPU CI，使用Python3.10/PyTorch2.5.1+cpu进行同一套检查。是否成功应以对应提交的Actions日志及 `provenance/ci_validation.json`（若已生成）为准，不以README文字代替运行证据。主发布流程只有测试通过后才会提交解包后的功能文件。

## 未执行、不能声称已验证的事项

没有在用户服务器运行，没有执行真实预训练ViT-L/14的CUDA全量训练，没有下载并遍历完整GenImage或Ojha数据集，没有实测目标GPU显存/吞吐/100轮收敛，没有证明取得任何ACC/AP/AUC或论文性能。网络下载脚本的参数、解析、离线数据处理已有检查，但没有在本地完整下载数百GB数据；外部配额和链接可用性仍受提供方影响。

原版PPM-CLIP完整训练未测试，原版的测试集早停、随机验证增强等逻辑被明确保留而非掩盖，不能视为无泄漏公平对照。新方法的主训练器使用独立验证集选模。当前新训练器支持单卡/CPU及梯度累积，不支持DDP或精确batch中途恢复。哈希去重不是近重复语义去重。
