# Human-in-the-Loop：人机协作

在生产环境中，让 Agent 完全自主运行是有风险的——删除文件、发送邮件、执行支付等操作一旦出错就无法撤回。**Human-in-the-Loop（人机协作）** 机制允许在 Agent 执行危险操作前暂停，等待人类确认后再继续。

LangGraph 通过 **Checkpointer**（检查点）机制实现这一能力。Checkpointer 会在图执行的每一步保存状态快照，使得图可以在任意节点暂停并恢复。

## 使用 Checkpointer 实现暂停恢复

### 核心思路

1. **分类工具的危险等级**：将工具分为"安全"和"危险"两类
2. **条件路由**：检测到危险工具调用时，停止图执行
3. **状态保存**：通过 `MemorySaver` 保存图的当前状态
4. **人工审批**：在外部获取用户确认
5. **恢复执行**：审批通过后，用 `app.invoke(None, config)` 从上次暂停的位置继续

![Human-in-the-Loop人机协作流程](../svg/chapter_langgraph_05_hitl.svg)

```python
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ============================
# 定义需要人工确认的敏感工具
# ============================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（危险操作，需要人工确认）"""
    print(f"\n[模拟] 发送邮件给 {to}")
    return f"邮件已发送给 {to}"

@tool
def delete_file(path: str) -> str:
    """删除文件（危险操作，需要人工确认）"""
    print(f"\n[模拟] 删除文件：{path}")
    return f"文件 {path} 已删除"

@tool
def safe_search(query: str) -> str:
    """安全搜索（不需要确认）"""
    return f"搜索'{query}'的结果：[相关信息...]"

DANGEROUS_TOOLS = {"send_email", "delete_file"}
tools = [send_email, delete_file, safe_search]

# ============================
# 节点定义
# ============================

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def check_needs_approval(state: MessagesState) -> str:
    """检查是否有危险工具调用需要审批"""
    last_msg = state["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            if tc["name"] in DANGEROUS_TOOLS:
                return "needs_approval"
        return "tools"  # 安全工具直接执行
    
    return END

# ============================
# 构建带 Checkpointer 的图
# ============================

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", check_needs_approval, {
    "needs_approval": END,  # 暂停！等待人工审批
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "agent")

# MemorySaver 允许恢复执行
memory = MemorySaver()

# 关于中断机制的说明：
# 这里同时使用了两种互补的安全策略：
# 1. check_needs_approval（条件路由）：对危险工具返回 END 直接停止图执行
# 2. interrupt_before（编译参数）：作为额外安全网，在所有工具节点前暂停
#
# 实际效果：
# - 安全工具（如 safe_search）：通过 check_needs_approval 路由到 "tools" 节点，
#   但 interrupt_before 会在执行前暂停（可根据需要移除此行使安全工具自动执行）
# - 危险工具（如 send_email）：直接在 check_needs_approval 中被路由到 END
#
# 如果只需要对危险工具中断，可以去掉 interrupt_before，
# 完全依赖 check_needs_approval 的路由逻辑。

app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # 额外安全网：在所有工具执行前暂停
)

# ============================
# 运行并处理人工确认
# ============================

def run_with_human_approval(task: str):
    """执行任务，危险操作需要人工确认"""
    
    thread_id = "human_approval_demo"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n任务：{task}")
    
    # 第一次运行
    state = app.invoke(
        {"messages": [HumanMessage(content=task)]},
        config=config
    )
    
    # 检查是否需要审批
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        dangerous_calls = [
            tc for tc in last_msg.tool_calls
            if tc["name"] in DANGEROUS_TOOLS
        ]
        
        if dangerous_calls:
            print("\n⚠️  检测到危险操作，需要人工确认：")
            for tc in dangerous_calls:
                print(f"  工具：{tc['name']}")
                print(f"  参数：{tc['args']}")
            
            approval = input("\n是否批准执行？(y/n)：").strip().lower()
            
            if approval == 'y':
                print("✅ 已批准，继续执行...")
                # 恢复执行
                final_state = app.invoke(None, config=config)
                return final_state["messages"][-1].content
            else:
                print("❌ 已拒绝，操作取消")
                # 注入拒绝消息
                app.invoke(
                    {"messages": [HumanMessage(content="用户已拒绝此操作，请告知用户")]},
                    config=config
                )
                return "操作已被用户取消"
    
    return last_msg.content if hasattr(last_msg, 'content') else "任务完成"

# 测试
result = run_with_human_approval("请帮我发邮件给 boss@company.com，主题是测试邮件")
print(f"\n最终结果：{result}")
```

## 三种审批模式

在实际应用中，人机协作不止"批准/拒绝"这一种模式。根据业务场景的不同，可以设计三种审批策略：

**模式 1：简单批准/拒绝（Gate Pattern）**

这就是上面的例子——Agent 提出操作请求，人类决定是否执行。

```python
# 适用场景：不可逆操作（删除、发送、支付）
# 流程：Agent → 暂停 → 人工决策 → 执行或取消
```

**模式 2：编辑后执行（Edit Pattern）**

人类不仅可以批准/拒绝，还可以**修改** Agent 的决策内容后再执行。

```python
def edit_and_resume(app, config, state):
    """允许人工编辑 Agent 的工具调用参数"""
    last_msg = state["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tc = last_msg.tool_calls[0]
        print(f"Agent 想要执行：{tc['name']}({tc['args']})")
        
        # 允许人工修改参数
        edited_args = input("修改参数（JSON格式，直接回车保持原样）：").strip()
        
        if edited_args:
            import json
            tc['args'] = json.loads(edited_args)
            # 更新状态中的消息
            app.update_state(config, {"messages": [last_msg]})
        
        # 继续执行（使用原始或修改后的参数）
        return app.invoke(None, config=config)
```

**模式 3：分级审批（Tiered Pattern）**

不同风险级别的操作采用不同的审批策略：

```python
# 风险分级定义
TOOL_RISK_LEVELS = {
    "search": "low",        # 低风险：自动执行
    "read_file": "low",
    "send_email": "medium",  # 中风险：需要确认
    "create_order": "medium",
    "delete_data": "high",   # 高风险：需要二次确认
    "transfer_money": "high",
}

def tiered_approval(tool_name: str) -> str:
    """根据风险等级决定审批策略"""
    risk = TOOL_RISK_LEVELS.get(tool_name, "medium")
    
    if risk == "low":
        return "auto_approve"  # 自动执行
    elif risk == "medium":
        return "single_approve"  # 单次确认
    else:
        return "double_approve"  # 二次确认 + 审计日志
```

## 生产环境中的 Human-in-the-Loop

在生产环境中，"人工确认"通常不是命令行中的 `input()` 调用，而是通过异步通知系统实现的：

```python
# 生产环境的典型架构
"""
1. Agent 遇到需要审批的操作 → 暂停执行
2. 后端保存 Checkpoint 到持久化存储（PostgreSQL/Redis）
3. 发送审批通知（Slack/邮件/企业微信/钉钉）
4. 人类在审批界面做出决策
5. 前端/Webhook 调用恢复 API
6. Agent 从 Checkpoint 恢复，继续执行
"""

# 使用 PostgreSQL 持久化 Checkpoint 的示例
from langgraph.checkpoint.postgres import PostgresSaver

# 连接数据库
DB_URI = "postgresql://user:pass@localhost:5432/agent_db"
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# 编译图，使用持久化检查点
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_tools"]
)

# FastAPI 端点：恢复执行
# @app.post("/approve/{thread_id}")
# async def approve_action(thread_id: str, approved: bool):
#     config = {"configurable": {"thread_id": thread_id}}
#     if approved:
#         result = agent_app.invoke(None, config=config)
#         return {"status": "completed", "result": result}
#     else:
#         return {"status": "rejected"}
```

> 💡 **关键提示**：生产环境中，`MemorySaver` 仅适合开发和测试。正式部署应使用 `PostgresSaver` 或 `RedisSaver` 来确保 Agent 状态在服务重启后仍可恢复。

---

## 小结

Human-in-the-Loop 的实现要点：
- **MemorySaver**：保存图的执行状态，支持恢复
- **interrupt_before/after**：在特定节点前后中断
- **thread_id**：标识一个会话，用于状态恢复
- **安全策略**：对危险操作（删除、发邮件、支付）必须要求确认

---

*下一节：[12.6 实战：工作流自动化 Agent](./06_practice_workflow_agent.md)*
