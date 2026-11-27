# 12.7 LangChain 生态 2026

> 本节介绍 LangChain v1、Deep Agents 和托管运行的职责，以及如何迁移旧版工具调用代码。

---

## LangChain 生态全景

LangChain 已经从一个单一框架发展为一个完整的生态系统。截至 2026 年，LangChain 生态的核心成员如下：

| 工具 | 定位 | 核心价值 |
|------|------|---------|
| **LangChain** | 核心编排框架 | 组件抽象 + LCEL 表达式语言 |
| **LangGraph** | 有状态 Agent 框架 | 图结构编排、循环、人机协作 |
| **Deep Agents** | 可复用的 Agent Harness | 文件系统、上下文管理、子 Agent 与可选的规划和 Skills |
| **Managed Deep Agents** | Harness 的托管交付方式 | 声明 Agent 行为，由 LangSmith 托管运行 |
| **LangSmith Agent Server** | 托管运行服务 | 部署、扩缩容、持久化 |
| **LangSmith** | Agent 工程平台 | 追踪、评估、Prompt 管理与部署运行 |

> 💡 **演进逻辑**：LangChain 提供组件，LangGraph 支撑有状态编排与运行，Deep Agents 组合现成的 Harness 能力。部署时可以自行运行，也可以选择 Managed Deep Agents 等托管路径；LangSmith 则覆盖追踪、评估与部署。理解这些职责，比记住产品名称更重要。

---

## Deep Agents 与 Managed Deep Agents：从 Harness 到托管运行

前面第 7–9 章分别讨论了上下文工程、Harness 和 Skills。以研究助手为例，除了搜索工具，它还可能需要保存资料、管理不断增长的上下文，并将部分工作交给子 Agent。**Deep Agents 将这些能力组合为可复用的 Harness**：它基于 LangChain 的 Agent 组件，并使用 LangGraph 运行时支持持久执行、流式输出和人机协作。[5]

### Managed 增加了什么？

**Managed Deep Agents（MDA）复用开源 Deep Agents Harness**，由 LangSmith Agent Server 托管运行。开发者定义模型、指令、工具和业务行为，平台负责组装 Harness 与运行基础设施。这里的变化在于工程责任的划分：Agent 行为由开发者定义，运行生命周期交给平台管理。[6]

| 关注点 | Deep Agents：自行运行 | Managed Deep Agents：托管运行 |
|--------|----------------------|-----------------------------|
| Agent 定义 | `create_deep_agent` 返回编译后的 Agent | `define_deep_agent` 返回定义，由平台编译 |
| 业务行为 | 配置模型、指令、工具和可选能力 | 在项目文件中声明模型、指令、工具和可选能力 |
| 运行基础设施 | 自行配置 backend、store、checkpointer 和服务 | 由托管运行时提供相关组件和生命周期管理 |
| 选择依据 | 需要自行运行或自定义底层组件 | 希望沿用现成 Harness，减少基础设施工作 |

### 将 Agent 行为组织为项目文件

研究助手可以从一个小目录开始，只添加任务需要的能力：

```text
research-assistant/
├── agent.py          # 模型与核心配置
├── instructions.md   # 行为指令
├── tools/            # 搜索等业务工具
└── skills/           # 按需加载的研究流程
```

`agent.py` 声明 Agent 定义，工具实现在 `tools/` 中。指令说明如何完成研究任务，Skills 封装可复用的操作流程。上传项目后，托管运行时负责将定义组装成可运行的 Agent。[6]

这与前文的能力划分一致：**Tool 提供动作，Skill 描述流程，Harness 管理执行，托管运行时负责运行基础设施**。掌握这一划分后，可以按任务需要逐步添加能力；完整配置和部署步骤见官方文档。[6]

> 💡 **选型提示**：先判断是否需要 Deep Agents 的现成能力，再决定是否托管。简单工具调用可以继续使用 LangChain Agent；需要明确控制流程时可以编写 LangGraph 工作流。选择 MDA 后，业务工具的正确性、权限边界和效果评估仍需要开发者设计。部署选型详见 [22.1 Agent 应用的部署架构](../chapter_22_deployment/01_deployment_architecture.md)。

> **版本说明**：截至 2026 年 10 月 7 日，官方文档将 MDA 标为 Beta。本文侧重职责与组织方式，具体 API 和部署要求以官方文档为准。[5][6]

---

## 部署与 MCP 集成

LangSmith Agent Server 提供持久执行、线程状态和流式接口，可部署到 LangSmith Cloud、自有云或自托管环境。选择前明确谁管理执行环境、检查点、故障恢复和身份校验。维护 LangServe 项目时参考其仓库文档，部署有状态 Agent 时参考 [LangSmith Agent Server 部署指南](https://docs.langchain.com/langsmith/deploy-overview)及[第 22 章](../chapter_22_deployment/README.md)。

本地开发时，用 `langgraph.json` 注册图的 Python 导出对象，执行 `langgraph dev` 测试线程、调用和流式返回。生产前可用 `langgraph up` 验证带 PostgreSQL 的运行环境。参见[本地开发与测试](https://docs.langchain.com/langsmith/local-dev-testing)。

LangChain 的 MCP 集成负责发现远程工具并包装成可传给 `create_agent` 的工具。官方文档中的 `MCPAdapter` 需要 `langchain[mcp]>=1.4.0`，该接口为 Beta。运行前检查适配器和服务端的协议版本，并让有连接状态的工具在连接范围内执行。参见 [LangChain MCP 集成](https://docs.langchain.com/oss/python/langchain/mcp)和 [19.5 SDK 示例](../chapter_19_protocol/05_practice_mcp_integration.md)。

## 从旧版迁移到 v1

| 旧写法 | v1 写法 |
|---|---|
| `AgentExecutor` + `create_openai_tools_agent` | `create_agent(model, tools, system_prompt=...)` |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `input`、`chat_history`、`agent_scratchpad` | `messages` 和可选 checkpointer；用 `thread_id` 区分会话 |
| 手工插入前置或后置 hook | Agent middleware |
| `LLMChain` 等旧组件 | 固定流程用 LCEL，兼容需求用 `langchain-classic` |

迁移后比较工具参数、返回消息和失败行为。标准工具调用可从[第 12.3 节](./03_langchain_agents.md)开始。需要直接控制图的状态和执行顺序时，编写 [LangGraph 状态图](../chapter_langgraph/README.md)。详见 [LangChain v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)。

## 参考资料

[5] LangChain Team. [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview). 查阅日期：2026-10-07。

[6] LangChain Team. [Managed Deep Agents Overview](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview). 查阅日期：2026-10-07。
