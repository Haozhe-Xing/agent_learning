# 第4章 工具调用（Tool Use / Function Calling）

> 🔧 *"Agent 的核心能力在于能够'动手'——调用工具与外部世界交互。"*

---


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解为什么 Agent 必须具备工具调用能力
- ✅ 掌握 OpenAI Function Calling / Anthropic Tool Use 的完整机制
- ✅ 能够设计并实现自定义工具，编写有效的工具描述
- ✅ 实战构建一个集成搜索引擎和计算器的 Agent
- ✅ 了解工具学习领域的前沿研究进展

## ⏱️ 预计学习时间

约 **90-120 分钟**（含实战）

## 💡 前置知识

- 完成第3章（LLM 基础）
- 熟悉 Python 函数和字典操作
- 了解 JSON 格式

---

## 🔗 学习路径

> **后续推荐**：
> - 👉 [第5章 记忆系统](../chapter_memory/README.md)
> - 👉 [第12章 LangChain](../chapter_langchain/README.md)

---

## 本章概览

工具调用（Tool Use / Function Calling）是 Agent 区别于普通聊天机器人的关键能力。通过工具，Agent 可以搜索网络、执行代码、操作数据库、调用外部 API……几乎可以做任何事情。本章深入探讨 Function Calling 机制，教你设计和实现高质量的 Agent 工具。

## 本章目标

学完本章，你将能够：

- ✅ 理解 Function Calling 的完整机制
- ✅ 设计和实现自定义工具
- ✅ 掌握工具描述的编写技巧
- ✅ 完成搜索引擎 + 计算器 Agent 实战

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 4.1 为什么 Agent 需要工具？ | 工具的价值与分类 | ⭐⭐ |
| 4.2 Function Calling 机制详解 | 完整的调用流程 | ⭐⭐⭐ |
| 4.3 自定义工具的设计与实现 | 工具开发最佳实践 | ⭐⭐⭐ |
| 4.4 工具描述的编写技巧 | 让 LLM 正确选择工具 | ⭐⭐ |
| 4.5 实战：搜索引擎 + 计算器 Agent | 完整项目实现 | ⭐⭐⭐⭐ |

## ⏱️ 预计学习时间

约 **90-120 分钟**（含实战练习）

## 💡 前置知识

- 已完成第 2-3 章的环境搭建和 LLM 基础知识
- 了解 OpenAI API 的基本调用方式
- 熟悉 Python 函数定义和 JSON 数据格式

## 🔗 学习路径

> **前置知识**：[第2章 环境搭建](../chapter_setup/README.md)、[第3章 LLM 基础](../chapter_llm/README.md)
>
> **后续推荐**：
> - 👉 [第5章 记忆系统](../chapter_memory/README.md) — 让 Agent 拥有"回忆"
> - 👉 [第9章 技能系统](../chapter_skill/README.md) — 从单个工具升级为完整技能
> - 👉 [第15章 通信协议](../chapter_protocol/README.md) — MCP 等标准化工具协议

---

*下一节：[4.1 为什么 Agent 需要工具？](./01_why_tools.md)*
