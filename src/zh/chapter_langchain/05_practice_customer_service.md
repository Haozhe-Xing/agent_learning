# 10.5 实战：多功能客服 Agent

## 本次迭代与验收

用框架承接工具编排，保持原有验收与权限约束。

- **成功标准**：用固定订单、FAQ 和投诉样例检查选工具与最终回复；工具后端用测试数据，区分查询结果与写入结果。
- **失败用例**：测试不存在的订单、其他用户订单与重复提交；失败时不能声称操作已完成。
- **安全边界**：业务后端验证身份、订单归属和写入权限；system prompt 的限制不能代替服务端鉴权。

先保留输入、预期和实际结果，再继续扩展。评估方法见[第18章](../chapter_20_evaluation/README.md)，信任与权限设计见[第19章](../chapter_21_security/README.md)。

把前几节的能力综合起来，构建一个能查 FAQ、查订单、提交投诉的多功能客服 Agent。这一节的重点不是"把代码抄会",而是理解**把一个真实业务场景拆解成 Agent 架构**的套路。

![多功能客服 Agent 系统架构](../svg/chapter_langchain_05_customer_service.svg)

## 设计思路：把业务拆成"工具 + 编排"

客服场景看似复杂，拆解后只有两层：

1. **工具层（Tool Layer）**：每个业务能力封装成一个 `@tool` 函数——`search_faq`（查知识库）、`check_order`（查订单）、`submit_complaint`（提交工单）。工具内部是确定性的业务代码（查数据库、调 API），**不依赖 LLM**。
2. **编排层（Agent Layer）**：一个 Agent 根据对话，自主决定"此刻该调哪个工具"。模型只负责"决策"，不负责"执行业务逻辑"。

> 💡 **直觉理解**：好的 Agent 架构，是让**模型做它擅长的（理解意图、决定下一步），让代码做它擅长的（精确、可审计的业务逻辑）**。工具函数就是"模型的手脚"——模型决定伸手去拿哪个工具，但真正执行动作的是代码。这就是为什么工具的描述（docstring）如此重要：模型靠它来"认手"。

## 关键片段：Agent 的构建

整个系统的"胶水"只有这一段——工具列表 + 系统提示 + 执行器：

```python
from examples.chapter12.customer_service import build_agent

# Fixture identity is set by the host. Real services must authenticate it.
agent = build_agent(owner="alice", allow_writes=False)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "check ORD-1"}]},
    config={"recursion_limit": 8},
)
print(result["messages"][-1].content)  # shipped
```

完整入口为 `examples/chapter12/customer_service.py`，使用 LangChain v1 的 `create_agent`。默认模型是确定性的测试替身，但工具编排、工具消息和权限判断均由真实框架执行。示例包含订单查询、FAQ 与幂等投诉提交；所有订单均为虚构数据。写入开关由宿主传入，提示词不能修改它。

```bash
python examples/chapter12/customer_service.py --query "check ORD-1"
python examples/chapter12/customer_service.py --query "check ORD-2"
python examples/chapter12/customer_service.py --query "complaint ORD-1"
python examples/chapter12/customer_service.py --query "complaint ORD-1" --allow-writes
```

依次应得到 `shipped`、拒绝访问、拒绝写入和 `TICKET-1`。同一 Agent 实例重复提交相同投诉返回相同编号。需要真实模型时加 `--real --model MODEL`，并配置对应 API Key。

本节不再重复旧版 `AgentExecutor` 的组装代码。迁移接口依据 [LangChain 官方快速入门](https://docs.langchain.com/oss/python/langchain/quickstart)。

## 设计要点（可迁移的经验）

| 要点 | 说明 |
|------|------|
| **工具粒度适中** | 一个工具做一件事；太粗难复用，太细模型决策成本高 |
| **描述即接口** | docstring 写清"何时用、参数含义"，比换模型更影响效果 |
| **权限边界前置** | 提示词声明行为边界，工具后端强制校验身份、订单归属与写入权限 |
| **失败优雅降级** | 工具内部捕获异常，返回错误信息而非崩溃，让 Agent 自我纠正 |
| **有状态会话** | 每次调用显式传入消息；跨请求持久化另接检查点（第11章） |

需要跨请求持久状态、中断与恢复时，继续学习第11章；这里每次调用都显式传入消息。

---

## 小结

本章通过多功能客服 Agent，综合运用了 LangChain 的核心能力。系统的核心套路是：**把业务拆成独立工具函数，再让 LLM Agent 按用户意图自主选工具响应**——这种"工具驱动"架构在实际项目中极其常见，也是 LangChain Agent 最典型的使用方式。

| 技能 | 要点 |
|------|------|
| 工具定义 | `@tool` 装饰器，docstring 即描述 |
| Agent | `create_agent` + 宿主权限检查 |
| 会话历史 | 显式消息状态 |

---

*下一章：[第11章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md)*
