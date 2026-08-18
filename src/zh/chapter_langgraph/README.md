# 第13章 LangGraph：构建有状态的 Agent

> 🕸️ *"LangGraph 用图结构取代线性链，让 Agent 拥有真正的状态管理和复杂工作流能力。"*

---

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解为什么复杂 Agent 需要图结构来管理状态
- ✅ 掌握 LangGraph 的核心概念：节点、边、状态
- ✅ 实现条件路由、循环控制和 Human-in-the-Loop 人机协作
- ✅ 从零构建一个工作流自动化 Agent

## 📚 本章结构

LangGraph 是 LangChain 团队推出的下一代 Agent 框架。相比于线性的 Chain，LangGraph 通过有向图（Directed Graph）构建 Agent 工作流，原生支持循环、条件分支和持久状态——这些正是构建复杂 Agent 系统所必需的能力。

| 小节 | 内容 | 难度 |
|------|------|------|
| 13.1 为什么需要图结构？ | 线性链的局限性 | ⭐⭐ |
| 13.2 核心概念：节点、边、状态 | LangGraph 基础 | ⭐⭐ |
| 13.3 第一个 Graph Agent | 动手实践 | ⭐⭐⭐ |
| 13.4 条件路由与循环控制 | 高级工作流 | ⭐⭐⭐ |
| 13.5 Human-in-the-Loop | 人机协作 | ⭐⭐⭐ |
| 13.6 实战：工作流自动化 Agent | 综合应用 | ⭐⭐⭐⭐ |

## 🔗 学习路径

> **前置知识**：[第12章 LangChain 深入实战](../chapter_langchain/README.md)
>
> **后续推荐**：
> - 👉 [第18章 多 Agent 协作](../chapter_18_multi_agent/README.md) — 用 LangGraph 构建多 Agent 系统
> - 👉 [第23章 项目实战：AI 编程助手](../chapter_23_coding_agent/README.md) — 在实战项目中综合运用 LangGraph

---

*下一节：[13.1 为什么需要图结构？](./01_why_graph.md)*
