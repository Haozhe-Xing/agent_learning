# 12.5 实战：多功能客服 Agent

把前几节的能力综合起来，构建一个能查 FAQ、查订单、提交投诉、做推荐的多功能客服 Agent。这一节的重点不是"把代码抄会",而是理解**把一个真实业务场景拆解成 Agent 架构**的套路。

![多功能客服 Agent 系统架构](../svg/chapter_langchain_05_customer_service.svg)

## 设计思路：把业务拆成"工具 + 编排"

客服场景看似复杂，拆解后只有两层：

1. **工具层（Tool Layer）**：每个业务能力封装成一个 `@tool` 函数——`search_faq`（查知识库）、`check_order`（查订单）、`submit_complaint`（提交工单）、`recommend_products`（推荐）。工具内部是确定性的业务代码（查数据库、调 API），**不依赖 LLM**。
2. **编排层（Agent Layer）**：一个 Agent 根据对话，自主决定"此刻该调哪个工具"。模型只负责"决策"，不负责"执行业务逻辑"。

> 💡 **直觉理解**：好的 Agent 架构，是让**模型做它擅长的（理解意图、决定下一步），让代码做它擅长的（精确、可审计的业务逻辑）**。工具函数就是"模型的手脚"——模型决定伸手去拿哪个工具，但真正执行动作的是代码。这就是为什么工具的描述（docstring）如此重要：模型靠它来"认手"。

## 关键片段：Agent 的构建

整个系统的"胶水"只有这一段——工具列表 + 系统提示 + 执行器：

```python
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def search_faq(query: str) -> str:
    """搜索常见问题解答库。适合回答产品使用、政策、流程等问题。"""
    # ... 内部查知识库（确定性逻辑）...

@tool
def check_order(order_id: str) -> str:
    """查询订单状态和物流信息。输入订单编号如 ORD-12345678。"""
    # ... 内部查订单系统 ...

tools = [search_faq, check_order, submit_complaint, recommend_products]

system_message = """你是"小慧"，一位热心、专业的客服助手。
## 服务准则
1. 先理解用户需求，再给帮助
2. 用工具前先想清楚哪个工具最合适
3. 无法解决时礼貌转人工（400-123-4567）
## 权限限制
- 不能修改订单金额；不能直接执行退款，只能提交申请
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
agent_executor = AgentExecutor(
    agent=create_openai_tools_agent(llm, tools, prompt),
    tools=tools, max_iterations=5, handle_parsing_errors=True,
)
```

> 📌 完整可运行代码（含交互式 CLI、会话历史管理）见本书配套仓库 `examples/` 目录。本节重在架构套路，交互细节留给读者按上面的模式自行补全。

## 设计要点（可迁移的经验）

| 要点 | 说明 |
|------|------|
| **工具粒度适中** | 一个工具做一件事；太粗难复用，太细模型决策成本高 |
| **描述即接口** | docstring 写清"何时用、参数含义"，比换模型更影响效果 |
| **权限边界前置** | 在 system prompt 显式声明"不能做什么"，降低越权风险 |
| **失败优雅降级** | 工具内部捕获异常，返回错误信息而非崩溃，让 Agent 自我纠正 |
| **有状态会话** | 用 `RunnableWithMessageHistory` 把历史注入 `chat_history`，但生产环境建议换成 LangGraph 的 checkpointer（第 13 章） |

> ⚠️ **生产化提醒**：`AgentExecutor` 的会话历史在多轮、长程、需中断恢复时力不从心。正式客服系统应迁移到 LangGraph（有状态、可 interrupt/resume、可审计），详见第 13 章与 12.8 节。

---

## 小结

本章通过多功能客服 Agent，综合运用了 LangChain 的核心能力。系统的核心套路是：**把业务拆成独立工具函数，再让 LLM Agent 按用户意图自主选工具响应**——这种"工具驱动"架构在实际项目中极其常见，也是 LangChain Agent 最典型的使用方式。

| 技能 | 要点 |
|------|------|
| 工具定义 | `@tool` 装饰器，docstring 即描述 |
| LCEL 链 | `\|` 管道，可读性强 |
| Agent | `create_openai_tools_agent` + `AgentExecutor` |
| 会话历史 | `RunnableWithMessageHistory` |

---

*下一章：[第13章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md)*
