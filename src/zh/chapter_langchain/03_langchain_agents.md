# 12.3 使用 LangChain 构建 Agent

前面几章我们都是"手工"构建 Agent——自己写工具 Schema、自己管理消息循环、自己处理工具调用。这虽然有助于理解底层原理，但在实际项目中太繁琐了。LangChain 提供了标准化的工具接口和 AgentExecutor，把那些"样板代码"封装成可复用组件。

> 📌 **前置提醒**：LangChain 的 `AgentExecutor` 属于 legacy Agent 方案。官方推荐新项目使用 **LangGraph** 构建 Agent（见第 13 章）。本节保留它，是为了让你用最少的认知负担理解"Agent 循环的封装"这一核心概念——理解了它，LangGraph 只是换了一种更灵活的表达方式。

## LangChain 到底帮你做了什么

LangChain 把你在前几章手写的样板，收敛成三层可复用组件：

| 你手搓时要做的事 | LangChain 的封装 |
|----------------|------------------|
| 手写 JSON Schema 描述工具 | `@tool` 装饰器，从函数签名 + docstring 自动提取 |
| 自己写"循环调模型→解析工具调用→执行→回填" | `AgentExecutor` 自动驱动整个循环 |
| 自己管理 scratchpad（推理空间） | `MessagesPlaceholder("agent_scratchpad")` 自动维护 |

![LangChain Agent三层架构](../svg/chapter_langchain_03_agent_architecture.svg)

## 工具定义：两种方式怎么选

LangChain 提供两种定义工具的方式，选择标准很清晰：

| 方式 | 适用场景 | 关键成本 |
|------|---------|---------|
| `@tool` 装饰器 | 大多数情况，逻辑简单 | 几乎零样板，docstring 即工具描述 |
| 继承 `BaseTool` | 需要内部状态、异步执行、自定义参数校验 | 多写一个 Pydantic `args_schema` |

> ⚠️ **关键细节**：`@tool` 的函数 **docstring 会直接成为工具描述**，喂给模型看。描述写不清，模型就调不对——这是新手最容易踩的坑。把"这个工具做什么、参数含义、何时用"写清楚，比换更强的模型更管用。

```python
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 方式1：@tool —— 最简单，docstring 即描述
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 calculate("sqrt(144) + 2 * 3")。"""
    try:
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        return str(eval(expression, {"__builtins__": {}}, safe_env))
    except Exception as e:
        return f"计算错误：{e}"

# 方式2：BaseTool —— 需要自定义校验/状态时
class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "获取指定城市的当前天气"
    # args_schema 用 Pydantic 定义结构化入参
```

## Agent 与执行循环

工具准备好后，用 `create_openai_tools_agent` 创建 Agent，再用 `AgentExecutor` 驱动循环：

```python
tools = [calculate, WeatherTool()]
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能助手，遇到需要工具的问题先调用工具。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # Agent 的"推理空间"
])
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,
                               max_iterations=5, return_intermediate_steps=True)
```

上面几个概念值得理解其"为什么"：

- **`MessagesPlaceholder("agent_scratchpad")`**：Agent 的"推理空间"。LangChain 把工具调用的中间步骤（调了什么工具、得到什么结果）填到这里，让模型能"看到"自己的推理过程，从而决定下一步。
- **`AgentExecutor`**：整个 Agent 循环的"驱动器"。它反复执行"模型决策 → 执行工具 → 回填结果"，直到模型给出最终回答或触发停止条件。
- **`return_intermediate_steps=True`**：让你在最终结果里看到 Agent 经历了哪些中间步骤——调试 Agent 时 invaluable。

> 💡 **直觉理解**：`AgentExecutor` 本质上就是一个**固定结构的 ReAct 循环器**——它的能力边界，就是"观察→思考→行动"这一固定顺序的重复。一旦你需要"先审批再执行""循环次数动态""并行多步"，它就力不从心了，必须上 LangGraph。这也解释了为什么官方把它列为 legacy。

## 生产环境的执行控制

```python
agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=10,          # 防无限循环（模型反复调同一工具）
    max_execution_time=30,      # 总超时，保护用户体验
    handle_parsing_errors=True,  # 模型输出格式异常时自动恢复
    early_stopping_method="generate",  # 超限时让模型基于现有信息给尽量好的回答
)
```

| 参数 | 解决的问题 |
|------|-----------|
| `max_iterations` | 模型陷入死循环烧 token |
| `max_execution_time` | 单次请求总超时 |
| `handle_parsing_errors` | 模型输出格式错乱时不直接崩 |
| `early_stopping_method` | 超限时优雅收尾而非报错 |

---

## 小结

LangChain Agent 的关键组件：
- `@tool` 装饰器：最快的工具定义方式（docstring 即描述）
- `BaseTool`：需要复杂逻辑/状态时使用
- `create_openai_tools_agent`：创建使用 Function Calling 的 Agent
- `AgentExecutor`：驱动"决策→执行→回填"循环

> 📌 记住 `AgentExecutor` 的能力边界（固定 ReAct 循环），它是你理解第 13 章 LangGraph 为何必要的认知锚点。

---

*下一节：[12.4 LCEL：LangChain 表达式语言](./04_lcel.md)*
