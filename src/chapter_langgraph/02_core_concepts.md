# LangGraph 核心概念：节点、边、状态

## 状态（State）

State 是 LangGraph 的数据中心，所有节点都从 State 读取数据并写入数据。

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph import MessagesState
import operator

# 方式1：使用内置的 MessagesState（推荐）
# 已内置 messages: Annotated[list, add_messages]
class MyState(MessagesState):
    # 在基础上添加额外字段
    user_name: Optional[str]
    task_complete: bool

# 方式2：完全自定义
from langchain_core.messages import BaseMessage

class CustomState(TypedDict):
    # 使用 Annotated + operator.add 表示"追加"语义
    messages: Annotated[list[BaseMessage], operator.add]
    
    # 普通字段（每次更新会覆盖）
    current_step: str
    error_count: int
    result: Optional[str]
```

## 节点（Node）

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START, MessagesState

llm = ChatOpenAI(model="gpt-4o")

# 节点就是普通的 Python 函数
def agent_node(state: MessagesState) -> dict:
    """
    Agent 节点：调用 LLM 处理消息
    
    接收：当前 State
    返回：State 的更新部分（字典）
    """
    messages = state["messages"]
    
    # 调用 LLM
    response = llm.invoke(messages)
    
    # 返回更新（只返回变化的部分）
    return {"messages": [response]}

def tool_node(state: MessagesState) -> dict:
    """工具节点：执行工具调用"""
    import json
    
    last_message = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        # 执行工具（这里模拟）
        result = f"工具 {tool_call['name']} 的结果"
        
        from langchain_core.messages import ToolMessage
        tool_results.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": tool_results}
```

## 边（Edge）

```python
# 普通边：固定指向
graph.add_edge("node_a", "node_b")  # A 总是指向 B

# 条件边：动态决定
def route_after_agent(state: MessagesState) -> str:
    """根据 Agent 输出决定下一步"""
    last_message = state["messages"][-1]
    
    # 如果有工具调用 → 执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    
    # 否则 → 结束
    return END

graph.add_conditional_edges(
    "agent_node",
    route_after_agent,
    {
        "tool_node": "tool_node",
        END: END
    }
)
```

## 完整的 ReAct Graph

```python
from langchain_core.tools import tool
import math

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, 
                     {k: getattr(math, k) for k in dir(math)})
        return str(result)
    except Exception as e:
        return f"错误：{e}"

tools = [calculate]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_executor(state: MessagesState) -> dict:
    from langchain_core.messages import ToolMessage
    import json
    
    last_msg = state["messages"][-1]
    results = []
    
    for tool_call in last_msg.tool_calls:
        if tool_call["name"] == "calculate":
            result = calculate.invoke(tool_call["args"])
        else:
            result = "未知工具"
        
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": results}

def should_use_tools(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_executor)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools)
graph.add_edge("tools", "agent")  # 工具执行后回到 agent

app = graph.compile()

# 运行
result = app.invoke({"messages": [HumanMessage(content="计算 sqrt(2) * pi")]})
print(result["messages"][-1].content)
```

---

## 小结

LangGraph 三要素：
- **State**：共享数据容器，TypedDict 定义结构
- **Node**：处理函数，接收 State，返回部分更新
- **Edge**：连接关系，普通边或条件边

---

*下一节：[9.3 构建你的第一个 Graph Agent](./03_first_graph_agent.md)*
