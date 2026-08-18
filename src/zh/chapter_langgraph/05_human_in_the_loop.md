# 13.5 Human-in-the-Loop：人机协作

生产环境中让 Agent 完全自主运行是有风险的——删除文件、发送邮件、执行支付一旦出错往往不可撤回。**Human-in-the-Loop（人在回路）** 允许在 Agent 执行危险操作前暂停，等人类确认后再继续。LangGraph 通过 **Checkpointer** 实现这一能力：它在图执行的每一步保存状态快照，使图能在任意节点暂停并恢复。

## 核心思路

1. **工具分级**：把工具分为"安全"（如搜索）与"危险"（如发邮件、删文件）两类
2. **条件路由**：检测到危险工具调用时，停止图执行
3. **状态保存**：用 Checkpointer 保存当前状态
4. **人工审批**：在外部获取用户确认
5. **恢复执行**：通过后用 `app.invoke(None, config)` 从暂停点继续

![Human-in-the-Loop人机协作流程](../svg/chapter_langgraph_05_hitl.svg)

```python
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

DANGEROUS_TOOLS = {"send_email", "delete_file"}
tools = [send_email, delete_file, safe_search]

def check_needs_approval(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if any(tc["name"] in DANGEROUS_TOOLS for tc in last.tool_calls):
            return "needs_approval"     # 暂停！等人工审批
        return "tools"                  # 安全工具直接执行
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", check_needs_approval,
                           {"needs_approval": END, "tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["tools"])
```

运行时：第一次 `invoke` 在危险工具前暂停 → 把待执行操作呈现给人工 → 批准后用 `app.invoke(None, config)` 从断点恢复。`thread_id`（通过 `config` 传入）标识一个会话，Checkpointer 据此恢复状态。

> 💡 **关键提示**：`MemorySaver` 仅适合开发/测试。正式部署用 `PostgresSaver` 或 `RedisSaver`——否则服务重启后状态丢失，Agent 无法续跑。

## 三种审批模式

| 模式 | 含义 | 适用 |
|------|------|------|
| **Gate（批准/拒绝）** | 人类决定是否执行 | 不可逆操作（删除、发送、支付） |
| **Edit（编辑后执行）** | 人类可修改 Agent 的工具参数再执行 | 需要人工纠偏的写操作 |
| **Tiered（分级审批）** | 按风险等级走不同审批强度（低自动 / 中确认 / 高二次确认+审计） | 风险异构的工具集 |

实现上，三者都是"暂停 → 读 `tool_calls` → 人工决策 → 用 `update_state` 改参数或直接 resume"的变体。

## 生产环境的落地形态

生产里"人工确认"通常不是命令行 `input()`，而是异步通知链路：

```
Agent 遇需审批操作 → 暂停并保存 Checkpoint（PostgreSQL/Redis）
→ 发审批通知（Slack/企业微信/邮件）→ 人类在界面决策
→ Webhook 调恢复 API → Agent 从 Checkpoint 续跑
```

> 📌 这与第 15 章 Claude Code 的"6 阶段权限决策流水线"是同一思想的工业级落地——把"是否放行一个危险动作"建模成一个可审计、可暂停、可恢复的状态机节点。

---

## 小结

Human-in-the-Loop 实现要点：
- **Checkpointer**：保存执行状态、支持恢复
- **`interrupt_before/after`**：在指定节点前后中断
- **`thread_id`**：标识会话，用于状态恢复
- **安全策略**：对危险操作（删除、发邮件、支付）必须确认，并按风险分级

---

*下一节：[13.6 实战：工作流自动化 Agent](./06_practice_workflow_agent.md)*
