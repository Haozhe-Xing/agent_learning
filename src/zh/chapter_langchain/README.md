# 第12章 LangChain 深度实践

> 🔗 *"LangChain 是当前最流行的 Agent 开发框架，本章深入讲解其核心架构和实战技巧。"*

---


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解 LangChain 的整体架构和核心组件
- ✅ 掌握 Chains 和 LCEL 表达式语言的使用
- ✅ 用 LangChain 构建一个完整的对话式 Agent
- ✅ 实现多功能客服 Agent 的完整项目

## ⏱️ 预计学习时间

约 **120-150 分钟**（含实战）

## 💡 前置知识

- 完成第4章（工具调用）、第5章（记忆）、第7章（RAG）
- pip install langchain langchain-openai

---

## 🔗 学习路径

> **后续推荐**：
> - 👉 [第13章 LangGraph](../chapter_langgraph/README.md)

---

## 本章概览

LangChain 通过提供标准化的抽象层，大大简化了 LLM 应用开发。从简单的 Prompt 链到复杂的 Agent 系统，LangChain 都提供了优雅的实现方式。本章从架构到实战，带你全面掌握 LangChain 的核心特性。

## 本章目标

学完本章，你将能够：

- ✅ 理解 LangChain 的分层架构和核心组件
- ✅ 使用 Chain 构建灵活的处理管道
- ✅ 掌握 LCEL 表达式语言的声明式写法
- ✅ 用 LangChain 构建具有工具调用能力的 Agent
- ✅ 完成一个多功能客服 Agent 实战项目

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 11.1 LangChain 架构全景 | 核心概念和组件关系 | ⭐⭐ |
| 11.2 Chain：构建处理管道 | LCEL 链式调用 | ⭐⭐⭐ |
| 11.3 使用 LangChain 构建 Agent | Tools、AgentExecutor | ⭐⭐⭐ |
| 11.4 LCEL：表达式语言 | 声明式管道构建 | ⭐⭐⭐ |
| 11.5 实战：多功能客服 Agent | 完整系统 | ⭐⭐⭐⭐ |

## ⏱️ 预计学习时间

约 **120-150 分钟**（含实战练习）

## 💡 前置知识

- 已完成第 4-7 章的核心能力学习（工具调用、记忆、规划、RAG）
- 已安装 LangChain（参考第 2 章）
- 了解 Python 的装饰器和异步编程基础

## 🔗 学习路径

> **前置知识**：[第4-7章 核心能力篇](../chapter_tools/README.md)
>
> **后续推荐**：
> - 👉 [第13章 LangGraph](../chapter_langgraph/README.md) — 从线性 Chain 升级到图结构
> - 👉 [第13章 其他框架](../chapter_frameworks/README.md) — 对比不同框架的设计取舍

---

*下一节：[11.1 LangChain 架构全景](./01_langchain_overview.md)*
