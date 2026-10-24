# 第10章 LangChain 深入实战

**阅读层次**：工程进阶；需要集成抽象时选用。

已有助手可以完成任务，但模型消息、工具定义和错误处理可能散落在业务代码中。本章用 LangChain 研究这些接口怎样复用，同时保留原来的来源、权限、预算和验收契约。是否采用该库，由维护成本与验证结果决定。

## 本章迭代

- **上一版的不足**：最小实现中的模型与工具适配重复，接入变化容易影响业务逻辑。
- **本章增加**：把消息、工具与调用结果放到明确的接口中，并观察链、Agent 与追踪如何承担部分集成工作。
- **怎样验证**：先运行固定模型输出的客服示例，再由读者把原任务与失败用例迁移到候选接口，核对调用、权限和独立验收；不能把两个不同 fixture 的成功当成同题比较。
- **下一步**：需要分支、暂停或恢复时读[第11章 LangGraph](../chapter_langgraph/README.md)；集成已满足需求时可转到[第18章 评估](../chapter_20_evaluation/README.md)。

## 本章结构

| 小节 | 本节关注的问题 |
|---|---|
| [10.1 LangChain 架构全景](./01_langchain_overview.md) | 模型、工具、消息与包边界 |
| [10.2 Chain：构建处理管道](./02_chains.md) | 显式输入输出与管道组合 |
| [10.3 使用 LangChain 构建 Agent](./03_langchain_agents.md) | 工具注册、结果与错误边界 |
| [10.4 LCEL：LangChain 表达式语言](./04_lcel.md) | 组合表达与调试记录 |
| [10.5 实战：多功能客服 Agent](./05_practice_customer_service.md) | 客服任务中的工具调用与权限 |
| [10.6 LangSmith 集成与可观测性](./06_langsmith_integration.md) | 记录调用轨迹与评估输入 |
| [10.7 LangChain 生态 2026](./07_langchain_ecosystem_2026.md) | 版本依赖与迁移记录 |
| [10.8 LangChain 生产化模式](./08_production_patterns.md) | 流式、并发、重试与资源约束 |

## 阅读路径

工具和证据契约可回看[第3章 工具调用](../chapter_tools/README.md)与[第6章 RAG](../chapter_rag/README.md)。环境准备见[附录F](../chapter_setup/README.md)。优先阅读 10.1、10.3 和 10.5，再按需要选用管道、观测与运行模式。

## 验证与材料边界

配套入口是 `examples/chapter12/customer_service.py`，目录保留历史编号。默认注入固定模型结果，实际执行框架工具循环；真实模型模式需要单独配置并记录结果。客服 fixture 与第11章的修改审查 fixture 不是同一任务，也不构成框架效果排名。

LangChain 与 LangGraph 可以组合，也可以分别使用。采用其中一个不会免除应用对来源、权限、失败状态和预算的责任。具体接口应与使用的依赖版本一起记录，避免混用不同版本的示例。

---

*下一节：[10.1 LangChain 架构全景](./01_langchain_overview.md)*
