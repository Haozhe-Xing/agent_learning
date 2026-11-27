# 第 16A 章 OpenCode V2：开源编程 Agent 实战

> 命令和配置以 OpenCode V2 为准，核验日期为 2026-11-27。

OpenCode 把编程 Agent 做成可以直接使用的工具。它支持终端、桌面和 Web。V2 把界面和执行服务分开，服务负责会话、模型连接、权限检查和工具执行。这个分工让你可以用终端做交互，也可以用 CLI 在同一个项目上执行受约束的任务。[官方入门](https://opencode.ai/docs/) · [CLI 架构与用法](https://opencode.ai/docs/cli/) · [OpenCode 源码与许可](https://github.com/anomalyco/opencode)

本章沿用 Damien Sileo 的实操课顺序：先读代码并计划，再小范围修改和验收，最后把审阅步骤做成可复用的 Agent。示例换成了一个只依赖 Python 标准库的排名练习，配置按 OpenCode V2 更新。上游教程采用 MIT 许可，配套示例保留了版权和许可文件。[上游引导版](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/GUIDED.md) · [MIT 许可](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/LICENSE)

| 小节 | 练习目标 |
|---|---|
| [16A.1 安装、连接模型与首次会话](./01_getting_started.md) | 装好 V2，选模型，用 Plan 和 Build 完成一次交互 |
| [16A.2 项目规则、权限与 V1 迁移](./02_project_config.md) | 配置 `AGENTS.md` 和工具权限，识别旧教程写法 |
| [16A.3 自定义 Agents、Skills 与 MCP](./03_extensions.md) | 给审阅员设权限，把常用流程做成 Skill，接入有用的外部工具 |
| [16A.4 实战：修复一个排名 Bug](./04_workshop.md) | 用测试复现问题，检查 diff，再用 CLI 跑一次只读审阅 |

开始前准备 Git、Python 3.10 或以上版本，以及一个可用的模型接入。OpenCode 可以在终端操作项目，模型服务的账户、额度和数据处理取决于所选提供商。[模型接入文档](https://opencode.ai/docs/cli/providers/)

**也可跟着读**：[OpenCode 中文社区实战课](https://www.opencodechina.com/)适合按场景逐节练习，[OpenCode 中文文档的 V2 入门](https://www.opencodecn.com/docs/v2/install)适合快速查中文命令。使用旧教程时，可结合[官方 V1 迁移说明](https://opencode.ai/docs/migrate-v1/)核对。
