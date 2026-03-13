# AutoGen：微软的多 Agent 对话框架

AutoGen 是微软开发的多 Agent 对话框架，其核心创新是：通过 **Agent 之间的对话** 来完成任务，而非传统的调用链。

> ⚠️ **重要更新**：2024 年底，AutoGen 经历了重大变化。原 AutoGen 团队分叉为两个项目：微软官方的 **AutoGen 0.4**（完全重写，基于事件驱动架构）和社区维护的 **AG2**（延续 0.2 版本的 API）。本节以最新的 AutoGen 0.4 为主进行介绍。

与 LangChain/LangGraph 的"节点-边"模型不同，AutoGen 把每个 Agent 看作一个"对话参与者"。Agent 之间通过自然语言交流——就像一个虚拟团队在开会讨论一样。这种设计让多 Agent 系统的构建变得非常直观。

AutoGen 最突出的特性是**自动代码执行**：AI 生成代码后，可以在沙箱中直接执行代码并将结果反馈给 AI，形成"生成-执行-修正"的自动化循环。

## AutoGen 0.4：全新事件驱动架构

AutoGen 0.4 相比旧版做了完全重写，引入了以下核心概念：
- **异步消息传递**：Agent 通过异步消息进行通信
- **事件驱动**：基于事件循环的执行模型
- **可插拔运行时**：支持单进程和分布式运行时
- **类型安全**：基于 Pydantic 的消息类型系统

```python
# pip install autogen-agentchat autogen-ext

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# AutoGen 0.4 配置
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# ============================
# 基础：创建 Agent
# ============================

assistant = AssistantAgent(
    name="助手",
    system_message="""你是一个有帮助的 AI 助手。
    你能够编写代码来解决问题。
    当任务完成时，回复 TERMINATE。""",
    model_client=model_client,
)

# ============================
# 多 Agent 群组对话
# ============================

coder = AssistantAgent(
    name="程序员",
    system_message="你是一名 Python 专家，负责编写代码。",
    model_client=model_client,
)

reviewer = AssistantAgent(
    name="代码审查员",
    system_message="""你是代码审查专家，负责：
    1. 检查代码的正确性
    2. 指出潜在的 bug
    3. 建议性能优化
    """,
    model_client=model_client,
)

# 创建群组对话（轮流发言模式）
termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat(
    [coder, reviewer],
    termination_condition=termination,
    max_turns=10,
)

# 运行对话
async def main():
    result = await team.run(
        task="请开发一个安全的用户登录验证函数，使用 bcrypt 进行密码哈希"
    )
    print(result)

asyncio.run(main())
```

## AutoGen vs CrewAI 对比

```
AutoGen 特点：
✅ 自动代码执行（杀手级特性）
✅ Agent 间自由对话
✅ 强大的代码生成能力
❌ 代码执行安全风险（需要沙箱）
❌ 对话可能偏离主题
❌ 成本较高（多轮对话）

适合场景：
- 需要生成并验证代码的任务
- 代码重构、优化、调试
- 数据分析（生成并执行分析脚本）
- 自动化测试
```

---

## 小结

AutoGen 的核心价值在于**代码的自动生成和执行**能力，以及**基于对话的多 Agent 协作**模式。AutoGen 0.4 的全新事件驱动架构使其在生产环境中更加可靠。对于需要代码自动化和多 Agent 讨论的场景，AutoGen 是非常强大的选择。

> 💡 **版本选择建议**：新项目推荐使用 AutoGen 0.4（`autogen-agentchat`），老项目如果使用 0.2 版本，可以考虑迁移到社区维护的 AG2（`ag2`）。

---

*下一节：[10.4 Dify / Coze 等低代码 Agent 平台](./04_low_code_platforms.md)*
