# 第4章 记忆系统（Memory）

> 🧩 *"没有记忆的 Agent，每次对话都是从零开始。记忆系统让 Agent 能够'记住'过去，提供真正个性化的体验。"*

---


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解 Agent 为什么需要记忆，以及不同记忆类型的用途
- ✅ 实现对话历史管理的多种策略（固定窗口、摘要压缩、向量检索）
- ✅ 使用 Chroma / FAISS 等向量数据库构建长期记忆
- ✅ 掌握工作记忆（草稿本模式）的设计与实现
- ✅ 构建一个具备持久记忆能力的个人助手 Agent

## ⏱️ 预计学习时间

约 **90-120 分钟**（含实战）

## 💡 前置知识

- 已完成第3章（工具调用）
- 了解向量相似度搜索的基本概念
- Python 面向对象编程基础

---

## 🔗 学习路径

> **前置知识**：[第3章 工具调用（Tool Use / Function Calling）](../chapter_tools/README.md)
>
> **后续推荐**：
> - 👉 [第5章 规划与推理（Planning & Reasoning）](../chapter_planning/README.md) — 赋予 Agent "思考力"
> - 👉 [第6章 检索增强生成（RAG）](../chapter_rag/README.md) — 用检索增强 Agent 的知识库

---

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 4.1 为什么 Agent 需要记忆？ | 记忆的价值与挑战 | ⭐⭐ |
| 4.2 短期记忆：对话历史管理 | 滑动窗口、摘要压缩 | ⭐⭐ |
| 4.3 长期记忆：向量数据库 | ChromaDB、相似度检索 | ⭐⭐⭐ |
| 4.4 工作记忆：Scratchpad 模式 | 推理过程记录 | ⭐⭐⭐ |
| 4.5 实战：带记忆的个人助理 | 完整系统实现 | ⭐⭐⭐⭐ |

## 🚀 扩展项目

| 项目 | 简介 | Stars |
|------|------|-------|
| [supermemory](https://github.com/supermemoryai/supermemory) | AI 时代的记忆与上下文引擎。支持自动事实提取、用户画像构建、遗忘曲线式记忆衰减、混合搜索（RAG + Memory）。在 LongMemEval、LoCoMo、ConvoMem 三大基准测试中均排名第一。提供 API、MCP 服务及 LangChain/LangGraph 集成。 | 17.5k+ |

---

*下一节：[4.1 为什么 Agent 需要记忆？](./01_why_memory.md)*
