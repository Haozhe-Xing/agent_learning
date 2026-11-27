# 12.3 使用 LangChain 构建 Agent

用 LangChain v1 的 `create_agent` 组合模型、工具和系统提示。模型提出工具调用，运行时执行工具并把结果加回消息，直到模型返回答案。运行示例前请设置 `OPENAI_API_KEY`。

```bash
python -m pip install "langchain>=1,<2" "langchain-openai>=1,<2"
```

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = create_agent(
    ChatOpenAI(model="gpt-4.1-mini"),
    tools=[add],
    system_prompt="Use the add tool to answer addition questions.",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 12 plus 8?"}]},
    config={"recursion_limit": 12},
)
print(result["messages"][-1].content)
```

工具函数的类型注解决定参数 Schema，docstring 帮助模型选择工具。Agent 的输入和结果都包含 `messages`。这个示例的 `recursion_limit` 限制图执行步数。需要跨调用保留会话时，见 [12.5 客服实战](./05_practice_customer_service.md)。需要自定义状态和路由时，见 [第 13 章](../chapter_langgraph/README.md)。
