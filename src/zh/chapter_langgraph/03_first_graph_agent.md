# 13.3 构建你的第一个 Graph Agent

本节用 LangGraph 构建一个完整的、带工具调用与循环推理的 Agent。与前面用原生 API 或 `AgentExecutor` 不同，LangGraph 用**图**描述 Agent 行为——执行流程**可视化、可控、可持久化**：你能清楚看到 Agent 走了哪些步骤，能在任意节点暂停/恢复，也能通过加边精确控制走向。

构建一个 Graph Agent 分四步：①定义工具 → ②创建节点（接收 State、返回更新）→ ③构建图（加节点与边）→ ④`compile()` 运行。下面用 LangGraph 内置的 `MessagesState`、`ToolNode`、`tools_condition` 来搭——这些是常见 Agent 模式的快捷组件。

![Graph Agent执行流程](../svg/chapter_langgraph_03_graph_agent.svg)

```python
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。"""
    # 生产环境用 AST 沙箱求值，绝不用 eval()（即使清空 __builtins__ 仍可被对象链逃逸）
    return safe_calculate_expr(expression)   # 见下方"工具安全"说明

tools = [calculate]
graph = StateGraph(MessagesState)
graph.add_node("agent", lambda s: {"messages": [llm.bind_tools(tools).invoke(s["messages"])]})
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # 有工具调用→tools，否则→END
graph.add_edge("tools", "agent")                       # 工具执行后回到 agent → 形成循环
app = graph.compile()
```

> 💡 **直觉理解**：图的拓扑本身就是一个"循环"——`agent → tools → agent → … → END`。每轮 `agent` 调 LLM 推理，`tools_condition` 检查返回是否含工具调用：有则去 `tools`，没有（模型认为可直接回答）则去 `END`。这正是 ReAct 循环在图里的落地。

**工具安全（重要）**：上面的 `calculate` 用 AST 沙箱而非 `eval()` 求值。即使把 `__builtins__` 清空，攻击者仍可用 `().__class__.__bases__[0].__subclasses__()` 这类对象链逃逸，所以**只允许"算术"这一种语义**——把字符串 `ast.parse` 成抽象语法树，在白名单（`+ - * / % **`、`sqrt/sin/log/pi`…）里逐项求值。这是把"工具执行"当作不可信输入处理的基本纪律。

## 追踪执行过程

调试 Agent 时，`app.stream()` 能逐节点流式输出状态，让你看到每个节点的决策、工具调用与结果：

```python
for event in app.stream({"messages": [HumanMessage(content="地球直径多少公里？换算成英里？")]}):
    for node, state in event.items():
        if node == "__end__":
            continue
        last = state.get("messages", [])[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            print(f"[{node}] 调用：{[tc['name'] for tc in last.tool_calls]}")
```

## 与 AgentExecutor 的本质差异

| 维度 | AgentExecutor | LangGraph |
|------|--------------|-----------|
| 透明度 | 黑盒循环，内部不透明 | 每个节点/边都可见 |
| 控制力 | 有限配置项 | 完全自定义拓扑与终止条件 |
| 持久化 | 不支持 | Checkpoint 原生支持 |
| 人在回路 | 需额外实现 | 内置 `interrupt` / resume |
| 适用 | 简单工具调用 | 复杂多步、有状态工作流 |

> 💡 **迁移建议**：若只需"LLM + 工具"的简单循环，`AgentExecutor` 已够用；一旦需要条件分支、多阶段流程、人工审批、中间状态持久化，就该上 LangGraph。

## 常见陷阱（用 prose 记牢，不必背代码）

1. **空消息列表**：取最后一条消息前先做非空校验，避免 `IndexError`。
2. **工具抛异常拖垮整个 Agent**：工具内部 `try/except`，返回友好错误信息而非上抛——让 Agent 见到错误后能自我纠正。
3. **无限循环无安全出口**：显式设 `recursion_limit`（如 `app.invoke(input, {"recursion_limit": 15})`），作为防失控兜底。

---

## 小结

第一个 Graph Agent 展示了：`StateGraph` + `MessagesState` 快速搭建；`ToolNode` + `tools_condition` 处理工具；`app.stream()` 追踪过程。下一节我们把"循环"变成可控、可路由的核心能力。

---

*下一节：[13.4 条件路由与循环控制](./04_conditional_routing.md)*
