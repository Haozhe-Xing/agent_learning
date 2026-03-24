# 使用 LangChain 构建 Agent

前面几章我们都是"手工"构建 Agent——自己写工具 Schema、自己管理消息循环、自己处理工具调用。这虽然有助于理解底层原理，但在实际项目中太繁琐了。LangChain 提供了标准化的工具接口和 AgentExecutor，大大简化了 Agent 开发。

LangChain 做的事情，本质上是把我们之前手写的那些"样板代码"封装成了可复用的组件：

- **工具定义**：不用再手写 JSON Schema，用 `@tool` 装饰器或继承 `BaseTool` 即可
- **Agent 创建**：不用再自己写执行循环，`create_openai_tools_agent` 一行搞定
- **执行管理**：`AgentExecutor` 自动处理工具调用循环、超时控制、错误恢复

![LangChain Agent三层架构](../svg/chapter_langchain_03_agent_architecture.svg)

## LangChain 工具定义

LangChain 提供了两种定义工具的方式：

**方式一：`@tool` 装饰器**——最简单直接。你只需要写一个普通的 Python 函数，加上 `@tool` 装饰器，LangChain 会自动从函数签名和 docstring 中提取参数信息、生成 Schema。这个 docstring 会成为工具的描述——所以一定要写清楚。

**方式二：继承 `BaseTool` 类**——当你需要更复杂的逻辑时使用，比如工具需要维护内部状态、需要异步执行、或者需要自定义参数验证。

```python
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
# ⚠️ 注意：AgentExecutor 是 LangChain 的 legacy Agent 方案。
# LangChain 官方推荐新项目使用 LangGraph 构建 Agent（见第12章）。
# 此处使用 AgentExecutor 是为了快速理解 Agent 概念。
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import math

# ============================
# 方式1：@tool 装饰器（最简单）
# ============================

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。
    支持基本运算和 math 模块函数（sqrt, sin, cos, log 等）。
    示例：calculate("sqrt(144) + 2 * 3")
    """
    try:
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def search_knowledge(query: str) -> str:
    """
    在知识库中搜索相关信息。
    适合查询产品信息、政策文档、FAQ等内部资料。
    """
    # 模拟知识库查询
    knowledge_base = {
        "退款": "退款政策：购买7天内可以申请退款，需要保留原始包装",
        "发货": "发货时间：工作日1-3天内发货，节假日顺延",
        "保修": "保修政策：硬件产品享有1年保修期",
    }
    
    for keyword, info in knowledge_base.items():
        if keyword in query:
            return info
    
    return "未在知识库中找到相关信息"

@tool  
def get_order_status(order_id: str) -> str:
    """
    查询订单状态。
    输入：订单编号（格式：ORD-XXXXXXXX）
    返回：订单的当前状态和物流信息
    """
    # 模拟订单查询
    mock_orders = {
        "ORD-12345678": "已发货，预计明天到达，快递单号：SF1234567890",
        "ORD-87654321": "处理中，预计明天发货",
    }
    
    return mock_orders.get(order_id, f"订单 {order_id} 不存在，请检查订单号是否正确")

# ============================
# 方式2：继承 BaseTool（更灵活）
# ============================

from typing import Type
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="城市名称")
    unit: str = Field(default="celsius", description="温度单位：celsius 或 fahrenheit")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "获取指定城市的当前天气信息"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, unit: str = "celsius") -> str:
        """同步执行"""
        # 模拟天气数据
        weather = {"北京": 15, "上海": 20, "广州": 28}
        temp = weather.get(city, 18)
        
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
            unit_str = "°F"
        else:
            unit_str = "°C"
        
        return f"{city}：{temp}{unit_str}，晴"
    
    async def _arun(self, city: str, unit: str = "celsius") -> str:
        """异步执行（可以调用异步API）"""
        return self._run(city, unit)  # 这里简单地调用同步版本

# ============================
# 创建 Agent
# ============================

tools = [calculate, search_knowledge, get_order_status, WeatherTool()]

# 系统提示
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能客服助手。
    
你可以使用以下工具帮助用户：
- calculate：数学计算
- search_knowledge：查询产品政策和FAQ
- get_order_status：查询订单状态
- get_weather：获取天气

遇到需要工具的问题先使用工具，再给出回答。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent 推理空间
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建 OpenAI Tools Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# AgentExecutor：负责运行循环
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印推理过程
    max_iterations=5,
    return_intermediate_steps=True  # 返回中间步骤
)

# ============================
# 使用 Agent
# ============================

# 单次对话
result = agent_executor.invoke({
    "input": "帮我查询订单 ORD-12345678 的状态，另外北京今天天气怎么样？",
    "chat_history": []
})
print(result["output"])

# 带历史的多轮对话
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def chat_with_agent(user_message: str) -> str:
    result = agent_executor.invoke({
        "input": user_message,
        "chat_history": chat_history
    })
    
    # 更新历史
    chat_history.extend([
        HumanMessage(content=user_message),
        AIMessage(content=result["output"])
    ])
    
    return result["output"]

# 多轮对话测试
print(chat_with_agent("我想了解退款政策"))
print(chat_with_agent("如果我昨天买的，还能退吗？"))  # 应该记住上下文
```

上面代码中有几个关键概念值得理解：

- **`MessagesPlaceholder("agent_scratchpad")`**：这是 Agent 的"推理空间"——LangChain 会把工具调用的中间步骤（调用了什么工具、得到什么结果）填充到这里，让模型能看到之前的推理过程
- **`AgentExecutor`**：它是整个 Agent 循环的"驱动器"。设置 `verbose=True` 可以在终端看到完整的推理过程，方便调试
- **`return_intermediate_steps=True`**：让我们能在最终结果中看到 Agent 经历了哪些中间步骤——调用了哪些工具、传了什么参数、得到什么结果

## 自定义 Agent 执行策略

在生产环境中，你需要对 Agent 的执行行为做更精细的控制。以下参数是最常用的：

- `max_iterations`：防止 Agent 陷入无限循环（比如模型反复调用同一个工具）
- `max_execution_time`：设置总超时时间，保护用户体验
- `handle_parsing_errors`：当模型输出格式异常时自动恢复，而不是直接报错
- `early_stopping_method`：超限时的停止策略——`"generate"` 会让模型根据当前信息生成一个尽可能好的回答

```python
# 控制 Agent 的执行行为
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,          # 最大循环次数
    max_execution_time=30,      # 最大执行时间（秒）
    handle_parsing_errors=True, # 自动处理解析错误
    early_stopping_method="generate",  # 超限时如何停止
    verbose=True
)
```

---

## 小结

LangChain Agent 的关键组件：
- `@tool` 装饰器：最快的工具定义方式
- `BaseTool`：需要复杂逻辑时使用
- `create_openai_tools_agent`：创建使用 OpenAI Function Calling 的 Agent
- `AgentExecutor`：负责运行 Agent 循环，处理工具调用

---

*下一节：[11.4 LCEL：LangChain 表达式语言](./04_lcel.md)*
