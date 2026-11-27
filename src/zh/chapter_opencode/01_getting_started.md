# 16A.1 安装、连接模型与首次会话

## 确认是 V2

以下命令适用于 OpenCode V2。用 npm 安装时包名是 `@opencode/cli`：

```bash
npm install -g @opencode/cli
opencode --version
```

在已有 V1 的机器上，先用 `command -v opencode`（macOS/Linux）或 `Get-Command opencode`（PowerShell）检查终端实际启动的文件。两个版本使用同一个命令名。其他安装方式和 Windows 二进制下载见[官方安装页](https://opencode.ai/docs/)。已有会话、插件或自定义接口时，先看[迁移说明](https://opencode.ai/docs/migrate-v1/)。

## 打开项目，选定一个模型

把终端切到你想操作的仓库根目录，运行 `opencode` 打开终端界面（TUI）。先输入 `/connect`，选择提供商并完成身份验证，再输入 `/models` 选择本次工作的模型。用 `/models` 能看到当前服务与项目可用的选项。团队配置里的 `provider/model` 应取自实际列表。密钥通过提供商的认证流程或环境变量输入，项目配置可提交给团队复用。[模型连接与认证](https://opencode.ai/docs/cli/providers/) · [模型配置](https://opencode.ai/docs/models/)

## 做一次低风险的交互

1. 用 `/agents` 选择 **Plan**。请它找出项目入口、测试命令和两个关键文件。观察它实际读取了哪些文件。
2. 用 `@` 选中一个文件，再问一个明确的问题。例如要求它指出某个函数处理空输入的位置。
3. 如果需要改动，用 `/agents` 切到 **Build**，把目标、可改文件和验收条件写清楚。完成后亲自检查 `git diff` 并运行验收命令。

Plan 和 Build 是不同的内置 Agent。Plan 默认禁止修改普通项目文件，仍有自己的命令权限和计划文件规则。Build 默认有更广的工具权限。[内置 Agents](https://opencode.ai/docs/agents/) · [终端交互](https://opencode.ai/docs/cli/tui/)

| 操作 | 用法 |
|---|---|
| 切换 Agent | `/agents`，或 V2 默认的 `Shift+Tab` |
| 新开会话、找回旧会话 | `/new`、`/sessions` |
| 撤回刚才一轮 | `/undo` 暂存撤回。继续提交提示会接受撤回，`/redo` 可在此之前取消 |
| 查看可用操作 | 输入 `/`，或用 `Ctrl+P` 打开命令面板 |

撤回依赖 OpenCode 的快照范围。提交、外部服务状态和其他副作用应有各自的恢复手段。[快照与撤回](https://opencode.ai/docs/snapshots/)
