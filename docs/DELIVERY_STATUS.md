# 本次交付与 GitHub 发布状态

**CADP v0.2 完整实现已经成功上传到 `rita2126top-alt/aigcdetect` 的 `main` 分支。**

## 当前发布记录

- 实现提交：`d137f74a5c432797f9a6515601ac327d10cd1514`
- 提交说明：`Publish validated CADP v0.2 implementation, experiments and guide`
- GitHub Actions 发布验证：run `34974737395`，结论 `success`
- 验证证据 artifact：`cadp-v02-validation-evidence`，artifact id `10397844379`
- 证据 ZIP digest：`sha256:20d070b00a77b890e644bdb5ebf165b46cc26bc448d038e24a13c8ea7a7e9797`
- PPM-CLIP 固定子模块：`09d05b9fc4be6a2b079356bbf337a05e193db0be`

发布工作流先核对 65,192 字节交付归档的 SHA256、解压后 tar 的 SHA256、54 个唯一普通文件、文件权限与路径安全，再安装固定 CPU 验证依赖，执行源码检查和 pytest。最终结果为 **55 tests、0 failures、0 errors、0 skipped**；上游 PPM-CLIP 47/47 文件校验通过。代码以普通 fast-forward push 提交，没有 force push。

## 服务器获取方式

```bash
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git
cd aigcdetect
git submodule update --init --recursive
```

然后从 `docs/RUN_GUIDE_ZH.md` 开始，修改 `configs/cadp/default.yaml` 或生成自己的 manifest 配置。正式入口为 `python -m cadp.cli`。

## 验证边界

GitHub Actions 和打包环境完成的是 CPU 代码正确性、张量前后向、训练/恢复/评测流程及静态检查。未下载正式 ViT-L/14 权重与全量真实数据，未执行远程服务器 CUDA 训练、GPU 显存测试或 100-epoch 正式实验，因此不能把本状态文件理解为论文性能复现或任意 GPU 环境保证。

`VALIDATION_REPORT.md` 与 `validation/environment_and_results.json` 中的本地环境记录保留打包时测试上下文；当前远程发布事实以本文件和上述 GitHub commit/run 为准。
