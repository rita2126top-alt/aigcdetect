# 本次交付与 GitHub 发布状态

**本次新增 CADP 完整文件已生成并在本地验证，但尚未推送到远程仓库。**

目标：`rita2126top-alt/aigcdetect`，分支 `main`。核对到的现有远程提交：`4e4eac9152ebc1a87afe858d7bac98d50109f32d`，该提交仍是前次依赖工作流修改，不是这次完成后的 CADP 提交。

当前会话可用 GitHub 接口只暴露读取操作；命令行连接 GitHub 返回 `Could not resolve host: github.com`。本报告没有虚构新远程 commit，也没有把本地补丁当成已上传代码。错误原文见 `validation/git_connectivity.log`。

## 两种使用方式

完整 ZIP 已包含固定 `PPM_CLIP/` 源码，可以直接解压，在根目录按 `RUN_GUIDE_ZH.md` 创建 Conda 环境并运行。ZIP 不含模型和数据；这些由显式下载命令准备。ZIP 不含 Git 历史。

需要将完成文件发布到你的 GitHub 时，使用随交付提供的 `aigcdetect_completion.patch`，在一台能连接 GitHub、且你已配置好该仓库写权限的机器执行：

```bash
mkdir -p "$HOME/projects"  # 创建项目目录。
cd "$HOME/projects"  # 进入该目录。
git clone --recurse-submodules https://github.com/rita2126top-alt/aigcdetect.git aigcdetect-publish  # 新克隆保留正确 Git 历史及固定子模块。
bash /你的下载目录/apply_and_push.sh /你的下载目录/aigcdetect_completion.patch "$HOME/projects/aigcdetect-publish" --check  # 只检查远程目标、干净工作区和补丁可应用性，不修改源码。
bash /你的下载目录/apply_and_push.sh /你的下载目录/aigcdetect_completion.patch "$HOME/projects/aigcdetect-publish" --push  # 检查作者配置，拉取main并仅快进，应用补丁，核对47个上游文件，创建提交，再普通push。
```

脚本不含任何 token，不索取聊天中的凭据，不使用 force push，不覆盖不干净的工作区。如果远程在此之后有冲突修改，脚本停止，不擅自替换对方工作。运行前需要本机已有 Git 作者身份及 GitHub HTTPS/SSH 认证；认证失败是本机账号连接问题，不应把口令写进源码。

也可不使用脚本，在干净克隆内执行：

```bash
git apply --check --index /你的下载目录/aigcdetect_completion.patch  # 校验补丁与当前内容匹配。
git apply --index /你的下载目录/aigcdetect_completion.patch  # 应用并暂存本次文件，不移动上游子模块指针。
git diff --cached --stat  # 查看实际要提交的文件。
git commit -m "Complete CADP-CLIP implementation and validated experiment pipeline"  # 创建本地提交。
git push origin HEAD:main  # 上传；只有此命令成功后才算 GitHub 发布完成。
```

本报告反映打包时状态。后续实际发布成功后，可按真实 GitHub commit 更新本文件，不要保留一个与事实不一致的状态。
