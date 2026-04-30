# 第6章 检索增强生成（RAG）

> 📚 *"RAG 是解决 LLM 知识局限性的最实用方案——让 Agent 能够'查阅'外部知识库，给出有依据的回答。"*

---


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解 RAG 的核心原理和工作流程
- ✅ 掌握文档加载、文本分割和向量嵌入的全流程
- ✅ 能够使用 FAISS / Chroma 等工具构建向量检索系统
- ✅ 实现多种检索策略：密集检索、稀疏检索、混合检索与重排序
- ✅ 构建一个完整的智能文档问答 Agent
- ✅ 掌握 GraphRAG / LightRAG 知识图谱增强检索
- ✅ 用 LangGraph 实现生产级 Agentic RAG 系统

## ⏱️ 预计学习时间

约 **150-180 分钟**（含实战，含 6.7 进阶内容）

## 💡 前置知识

- 完成第4章（记忆系统）
- 了解向量和余弦相似度的基本概念
- `pip install langchain faiss-cpu sentence-transformers graphrag lightrag-hku langgraph`

---

## 🔗 学习路径

> **后续推荐**：
> - 👉 [第7章 上下文工程](../chapter_context_engineering/README.md)
> - 👉 [第12章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md) — 深入 Agentic RAG 所用的状态机框架

---

## 本章概览

RAG（Retrieval-Augmented Generation，检索增强生成）是当前最重要的 AI 应用技术之一。LLM 的知识有截止日期，而且无法访问你的私有数据。RAG 通过"先检索、再生成"的方式，让 Agent 能够基于最新的、特定领域的知识来回答问题。本章从原理到实战，全面讲解如何构建 RAG 系统。

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 6.1 RAG 的概念与工作原理 | 为什么需要 RAG？如何工作？ | ⭐⭐ |
| 6.2 文档加载与文本分割 | 处理各种格式的文档 | ⭐⭐ |
| 6.3 向量嵌入与向量数据库 | 语义存储与检索 | ⭐⭐⭐ |
| 6.4 检索策略与重排序 | 提升检索精准度 | ⭐⭐⭐ |
| 6.5 实战：智能文档问答 Agent | 完整系统实现 | ⭐⭐⭐⭐ |
| 6.6 论文解读：RAG 前沿进展 | Self-RAG / CRAG / GraphRAG / Agentic RAG | ⭐⭐⭐ |
| **6.7 进阶：GraphRAG 与 Agentic RAG 工程实战** | **知识图谱检索 + LangGraph 编排** | ⭐⭐⭐⭐⭐ |

## 🔗 学习路径

> **前置知识**：[第4章 记忆系统（Memory）](../chapter_memory/README.md)（尤其是向量数据库部分）
>
> **后续推荐**：
> - 👉 [第7章 上下文工程](../chapter_context_engineering/README.md) — 系统化管理 RAG 检索到的上下文信息

---

*下一节：[6.1 RAG 的概念与工作原理](./01_rag_concepts.md)*
