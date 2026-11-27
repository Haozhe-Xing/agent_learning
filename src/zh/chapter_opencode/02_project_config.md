# 16A.2 项目规则、权限与 V1 迁移

## 把项目约定写进仓库

在 `AGENTS.md` 里写能帮下一位开发者开工的内容：入口、测试命令、哪些文件是生成物、验收时看什么。把 `opencode.json` 或 `opencode.jsonc` 用于模型、工具权限和集成配置。两者都可以版本化。终端显示、快捷键等界面设置属于全局 `cli.json`。[项目指令](https://opencode.ai/docs/instructions/) · [配置位置与优先级](https://opencode.ai/docs/config/) · [终端设置](https://opencode.ai/docs/cli/config/)

## 用权限限制练习的修改范围

配套练习的配置设置了默认 Plan，禁止访问工作区以外的目录，并要求审批对 `ranking.py` 的修改和 shell 命令。Plan 另外拒绝所有写入和 shell，切到 Build 后才会出现练习所需的审批。这里的文件名和规则应放在练习目录中使用。

```json
{{#include ../../../examples/opencode-workshop/opencode.json}}
```

V2 的每条规则由 `action`、`resource`、`effect` 组成。普通权限使用**最后一条匹配规则**，所以文件级例外写在 `*` 之后。项目配置也可能叠加用户、Agent 和管理员规则，排查时要一起检查。Shell 仍以当前系统用户执行，权限匹配依赖 OpenCode 能识别的命令和路径。接手不熟悉的仓库时，把试验放在单独工作目录中。[权限匹配与目录规则](https://opencode.ai/docs/permissions/) · [策略](https://opencode.ai/docs/policies/)

## 如何识别旧教程

| 旧教程里的 V1 写法 | 对应的 V2 写法 |
|---|---|
| 安装 `opencode-ai` | 安装 `@opencode/cli` |
| 用 `permission` 配置 `bash`、`task` | 在 `permissions` 数组中使用 `shell`、`subagent` |
| MCP 服务直接挂在 `mcp` 下 | 把服务放在 `mcp.servers`，停用字段为 `disabled` |
| 修改项目里的 `tui.json` | 改全局 `cli.json` |
| 直接接入 V1 插件或服务器 SDK | 按新版 API 迁移，分别核对插件和客户端 |

维护 V1 项目时可以继续查[官方 V1 文档](https://opencode.ai/v1/docs/)。升级已有项目时，先备份配置与重要会话，按[官方迁移表](https://opencode.ai/docs/migrate-v1/)核对实际用到的功能。
