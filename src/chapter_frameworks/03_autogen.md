# AutoGen：微软的多 Agent 对话框架

AutoGen 是微软开发的多 Agent 对话框架，其核心创新是：通过 **Agent 之间的对话** 来完成任务，而非传统的调用链。

> ⚠️ **重要更新**：2024 年底，AutoGen 经历了重大变化。原 AutoGen 团队分叉为两个项目：微软官方的 **AutoGen 0.4**（完全重写，基于事件驱动架构）和社区维护的 **AG2**（延续 0.2 版本的 API）。本节以最新的 AutoGen 0.4 为主进行介绍。

与 LangChain/LangGraph 的"节点-边"模型不同，AutoGen 把每个 Agent 看作一个"对话参与者"。Agent 之间通过自然语言交流——就像一个虚拟团队在开会讨论一样。这种设计让多 Agent 系统的构建变得非常直观。

AutoGen 最突出的特性是**自动代码执行**：AI 生成代码后，可以在沙箱中直接执行代码并将结果反馈给 AI，形成"生成-执行-修正"的自动化循环。

![AutoGen 0.4 事件驱动多Agent架构](../svg/chapter_frameworks_03_autogen.svg)

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

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# ============================
# 基础：创建单个 Agent
# ============================
assistant = AssistantAgent(
    name="助手",
    system_message="""你是一个有帮助的 AI 助手。
    你能够编写代码来解决问题。
    当任务完成时，回复 TERMINATE。""",
    model_client=model_client,
)
```

## 核心特性：自动代码执行沙箱

这是 AutoGen 区别于其他框架的**杀手级特性**。AI 生成代码 → 沙箱执行 → 将执行结果反馈 → AI 根据结果修正，形成自动化编程循环。

AutoGen 0.4 提供了两种代码执行器：

### Docker 沙箱执行器（推荐用于生产环境）

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import asyncio

# Docker 沙箱 —— 代码在隔离容器中执行，安全可控
code_executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",          # 使用的 Docker 镜像
    work_dir="coding_output",          # 工作目录
    timeout=60,                        # 单次执行超时（秒）
)

# 代码执行 Agent —— 负责运行代码并返回结果
executor_agent = CodeExecutorAgent(
    name="代码执行器",
    code_executor=code_executor,
)

# 编程 Agent —— 负责生成代码
coder_agent = AssistantAgent(
    name="程序员",
    system_message="""你是一名 Python 专家。请按以下规则工作：
    1. 编写完整的、可直接运行的 Python 代码
    2. 所有代码放在 ```python 代码块中
    3. 分析执行结果，如有错误则修正代码
    4. 任务完成后回复 TERMINATE""",
    model_client=model_client,
)

async def coding_task():
    # 启动 Docker 容器
    async with code_executor:
        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat(
            [coder_agent, executor_agent],
            termination_condition=termination,
            max_turns=10,
        )
        # 执行流程：程序员写代码 → 执行器运行 → 返回结果 → 程序员修正
        result = await team.run(
            task="请编写代码：下载并分析 iris 数据集，绘制各特征的分布图，保存为 iris_analysis.png"
        )
        print(result)

asyncio.run(coding_task())
```

### 本地进程执行器（适合开发调试）

```python
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

# 本地执行 —— 直接在宿主机上运行代码（注意安全风险！）
local_executor = LocalCommandLineCodeExecutor(
    work_dir="local_output",
    timeout=30,
    virtual_env_context=None,   # 可指定虚拟环境
)

# ⚠️ 安全警告：本地执行器没有沙箱隔离！
# 恶意代码可以访问文件系统、网络等宿主机资源
# 仅在开发/调试环境使用，生产环境务必使用 Docker 执行器
```

> 💡 **核心理解**：代码执行能力让 AutoGen 不仅仅是"聊天"，而是**真正能完成编程任务**的框架。AI 写错了代码？没关系，执行器返回报错信息，AI 看到错误后自动修正 —— 这个循环是全自动的。

## 多 Agent 群组对话

AutoGen 0.4 提供了多种群组对话模式：

### RoundRobinGroupChat（轮流发言）

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

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
    审查通过后回复 TERMINATE。""",
    model_client=model_client,
)

# 轮流发言：程序员 → 审查员 → 程序员 → 审查员 → ...
termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat(
    [coder, reviewer],
    termination_condition=termination,
    max_turns=10,
)

async def main():
    result = await team.run(
        task="请开发一个安全的用户登录验证函数，使用 bcrypt 进行密码哈希"
    )
    print(result)

asyncio.run(main())
```

### SelectorGroupChat（智能选择下一个发言者）

这是 AutoGen 0.4 引入的高级模式——由 LLM 根据对话上下文**动态决定**下一个该哪个 Agent 发言：

```python
from autogen_agentchat.teams import SelectorGroupChat

# 多角色团队
planner = AssistantAgent(
    name="项目经理",
    system_message="你是项目经理，负责任务拆解和进度跟踪。",
    model_client=model_client,
)

coder = AssistantAgent(
    name="开发工程师",
    system_message="你是全栈开发工程师，负责实现功能。",
    model_client=model_client,
)

tester = AssistantAgent(
    name="测试工程师",
    system_message="你是测试工程师，负责编写测试用例并验证功能。任务完成后回复 TERMINATE。",
    model_client=model_client,
)

# LLM 根据上下文自动选择下一个发言者
# 例如：需求分析阶段 → 项目经理；需要写代码 → 开发工程师；代码完成 → 测试工程师
team = SelectorGroupChat(
    [planner, coder, tester],
    model_client=model_client,   # 用于选择下一个发言者的 LLM
    termination_condition=TextMentionTermination("TERMINATE"),
    max_turns=15,
)

async def dev_task():
    result = await team.run(
        task="开发一个命令行 TODO 应用，支持添加、删除、标记完成、列表展示"
    )
    print(result)

asyncio.run(dev_task())
```

## Agent 使用工具

AutoGen 0.4 支持给 Agent 注册自定义工具：

```python
from autogen_agentchat.agents import AssistantAgent

# 定义工具函数（普通 Python 函数即可）
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    # 实际项目中接入搜索 API
    return f"搜索结果：关于 '{query}' 的最新信息..."

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 注意：生产环境应使用安全的表达式解析器
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# 创建带工具的 Agent
agent_with_tools = AssistantAgent(
    name="研究助手",
    system_message="你是一个研究助手，可以搜索网络和进行计算。",
    model_client=model_client,
    tools=[search_web, calculate],   # 直接传入 Python 函数列表
)
```

> 💡 **对比 LangChain**：在 LangChain 中注册工具需要使用 `@tool` 装饰器或 `StructuredTool`，而 AutoGen 直接接受普通 Python 函数，更加简洁。

## AutoGen vs CrewAI 对比

| 维度 | AutoGen | CrewAI |
|------|---------|--------|
| **核心理念** | Agent 间自由对话 | 角色扮演 + 任务流程 |
| **代码执行** | ✅ 内置沙箱（杀手级特性） | ❌ 不支持 |
| **灵活性** | 高，对话可自由流转 | 中，任务按预定流程执行 |
| **成本控制** | 较高（多轮对话） | 较低（流程可控） |
| **上手难度** | 中等 | 简单 |
| **适合场景** | 代码生成/调试、数据分析、自动化测试 | 内容创作、研究报告、流水线任务 |

**选择建议**：
- 需要**生成并执行代码** → AutoGen（别无选择，这是它的杀手级特性）
- 需要**角色分工明确的工作流** → CrewAI
- 需要**灵活的多 Agent 讨论** → AutoGen
- 需要**与 LangChain 生态集成** → 都可以，但 CrewAI 更原生

---

## 小结

AutoGen 的核心价值在于**代码的自动生成和执行**能力，以及**基于对话的多 Agent 协作**模式。AutoGen 0.4 的全新事件驱动架构使其在生产环境中更加可靠。对于需要代码自动化和多 Agent 讨论的场景，AutoGen 是非常强大的选择。

> 💡 **版本选择建议**：新项目推荐使用 AutoGen 0.4（`autogen-agentchat`），老项目如果使用 0.2 版本，可以考虑迁移到社区维护的 AG2（`ag2`）。

---

*下一节：[13.4 Dify / Coze 等低代码 Agent 平台](./04_low_code_platforms.md)*
