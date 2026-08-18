# 19.6 框架补充：AutoGen（多 Agent 对话框架）

> 💬 *"让 Agent 之间用自然语言讨论，比让它们按预定节点流转更接近真实人类工作方式。"*

---

AutoGen 是微软开发的多 Agent 框架，核心创新是**通过 Agent 之间的"对话"来推进任务**，而非传统的调用链。与 LangChain/LangGraph 的"节点—边"模型不同，AutoGen 把每个 Agent 看作一个"会议参与者"，通过自然语言你来我往地讨论。

> ⚠️ **版本说明**：2024 年底 AutoGen 团队分叉为两个项目——微软官方 **AutoGen 0.4**（完全重写，事件驱动架构）与社区维护的 **AG2**（延续 0.2 API）。本节以 0.4 为准。新项目推荐 0.4（`autogen-agentchat`），老项目可用 AG2。
>
> ⚠️ **最新动态（2026）**：微软已将重心转向 **Microsoft Agent Framework**——一个原生支持 A2A 与 MCP 协议的新编排框架，官方把 AutoGen 定位为"维护模式"，并提供了 AutoGen → Agent Framework 的迁移路径。新项目若选微软生态，优先评估 Agent Framework；本节保留 AutoGen 是因为它"代码执行沙箱 + 群组对话"的设计思路仍是理解多 Agent 对话式协作的经典样本。

AutoGen 0.4 的几条设计主线：

- **异步消息传递 + 事件驱动**：Agent 间通过异步消息通信，基于事件循环执行
- **可插拔运行时**：支持单进程与分布式运行时
- **类型安全**：基于 Pydantic 的消息类型
- **杀手级特性：代码自动执行**——AI 生成代码 → 沙箱执行 → 结果反馈 → AI 修正，形成"生成-执行-修正"闭环

![AutoGen 0.4 事件驱动多Agent架构](../svg/chapter_frameworks_03_autogen.svg)

## 杀手级特性：代码执行沙箱

这是 AutoGen 区别于其他框架的根本优势。它不是"只会聊天"，而是**真正能跑代码、看报错、自己改**的框架。0.4 提供两类执行器：

| 执行器 | 隔离性 | 用途 |
|--------|--------|------|
| `DockerCommandLineCodeExecutor` | Docker 沙箱，安全可控 | 生产环境首选 |
| `LocalCommandLineCodeExecutor` | 直接在宿主机运行 | 仅开发/调试（无隔离，慎用） |

一个最小可说明的闭环——"程序员写代码 → 执行器在 Docker 里跑 → 返回结果 → 程序员修正"：

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

code_executor = DockerCommandLineCodeExecutor(image="python:3.12-slim", timeout=60)
executor_agent = CodeExecutorAgent("代码执行器", code_executor=code_executor)
coder = AssistantAgent("程序员", system_message="写 Python，出错就改，完成说 TERMINATE",
                       model_client=model_client)

async with code_executor:
    team = RoundRobinGroupChat([coder, executor_agent],
                               termination_condition=TextMentionTermination("TERMINATE"),
                               max_turns=10)
    await team.run(task="下载并分析 iris 数据集，绘制特征分布图")
```

> 💡 **核心理解**：代码执行让 AutoGen 从"对话系统"变成"能交付结果的编程 Agent"。它在自动化软件开发、数据分析、自动化测试场景里几乎是唯一选择——因为"让 Agent 真正运行代码"是它独有的能力。

## 多 Agent 群组对话

AutoGen 0.4 提供多种群组模式，决定"下一个该谁发言"：

| 模式 | 发言调度 | 适用 |
|------|---------|------|
| `RoundRobinGroupChat` | 轮流发言（A→B→A→B…） | 固定角色顺序的讨论 |
| `SelectorGroupChat` | **LLM 根据上下文动态选下一个发言者** | 需求/开发/测试角色随阶段切换 |

给 Agent 注册工具也比 LangChain 简洁——直接传 Python 函数即可，无需 `@tool` 装饰器。

## AutoGen vs CrewAI

| 维度 | AutoGen | CrewAI |
|------|---------|--------|
| **核心理念** | Agent 间自由对话 | 角色扮演 + 任务流程 |
| **代码执行** | ✅ 内置沙箱（杀手级） | ❌ 不支持 |
| **灵活性** | 高，对话自由流转 | 中，按预定流程执行 |
| **成本** | 较高（多轮对话） | 较低（流程可控） |
| **适合场景** | 代码生成/调试、数据分析 | 内容创作、流水线任务 |

**选择建议**：需要生成并执行代码 → AutoGen（别无选择）；角色分工明确的流水线 → CrewAI；灵活多 Agent 讨论 → AutoGen。

---

## 小结

AutoGen 的核心价值在于**代码自动生成与执行**能力，以及**基于对话的多 Agent 协作**。0.4 的事件驱动架构让它在生产环境更可靠。对于需要代码自动化和多 Agent 讨论的场景，它是非常强大的选择。

---

*上一节：[19.5 实战：基于 MCP 的工具集成](../chapter_19_protocol/05_practice_mcp_integration.md)*  
*下一节：[第19章 Agent 通信协议 章节首页](../chapter_19_protocol/README.md)*
