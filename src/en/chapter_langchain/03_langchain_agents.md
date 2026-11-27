# 12.3 Building Agents with LangChain

Use LangChain v1’s `create_agent` to combine a model, tools, and a system prompt. The model proposes tool calls, and the runtime executes them and appends the results until the model returns an answer. Set `OPENAI_API_KEY` before running the example.

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

Type annotations define the tool schema, and the docstring helps the model select a tool. Both agent input and output contain `messages`. The `recursion_limit` caps graph execution steps. See the [customer service example in 12.5](./05_practice_customer_service.md) to keep conversation state across calls and [Chapter 13](../chapter_langgraph/README.md) for custom state and routing.
