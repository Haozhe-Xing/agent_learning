# 第2章 大语言模型基础

> 工欲善其事，必先利其器。在开始构建 Agent 之前，我们需要深入理解它的"大脑"——大语言模型（LLM）。

---

## 本章概览

本章从直觉层面解释大语言模型的工作原理，然后系统讲解如何通过 Prompt Engineering 与模型高效对话，介绍常见的提示策略，并手把手带你完成第一次 API 调用。最后，我们深入探讨 Token、Temperature 等关键参数，帮助你真正"驾驭"语言模型。

## 本章目标

学完本章，你将能够：

- ✅ 用直觉理解 LLM 的工作机制（不需要数学推导）
- ✅ 掌握 Prompt Engineering 的核心原则和技巧
- ✅ 灵活运用 Zero-shot、Few-shot、CoT 等提示策略
- ✅ 熟练调用 OpenAI API 及常见开源模型接口
- ✅ 理解 Token、Temperature 等参数对输出的影响

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 2.1 LLM 是如何工作的？ | 直觉理解 Transformer、预训练与涌现能力 | ⭐⭐ |
| 2.2 Prompt Engineering | 系统消息、角色扮演、结构化输出 | ⭐⭐ |
| 2.3 提示策略 | Zero-shot、Few-shot、CoT、ToT | ⭐⭐⭐ |
| 2.4 模型 API 调用入门 | OpenAI SDK、开源模型、流式调用 | ⭐⭐ |
| 2.5 Token 与模型参数 | Token 计数、Temperature、Top-p 等 | ⭐⭐ |

## 核心概念速览

![LLM 核心概念速览](../svg/chapter_llm_readme_llm_tree.svg)

## 为什么 Agent 开发者需要理解 LLM？

很多 Agent 框架（LangChain、LangGraph 等）将模型调用封装得很好，初学者可以快速上手。但当你的 Agent 出现以下问题时，理解 LLM 底层机制就至关重要：

- 输出不稳定，同样的问题得到不同答案
- 模型"幻觉"——信心十足地给出错误答案
- Token 超限，长对话被截断
- 成本过高，需要优化 Prompt 减少消耗

理解 LLM 就像理解发动机原理——即使你不造发动机，懂原理也能让你成为更好的驾驶员。

---

*下一节：[2.1 LLM 是如何工作的？（直觉理解）](./01_how_llm_works.md)*
