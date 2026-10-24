# Chapter 4: Memory Systems

> 🧩 *"An Agent without memory starts from scratch every conversation. Memory systems allow Agents to 'remember' the past and provide truly personalized experiences."*

---

## Chapter Overview

Memory systems are the key that distinguishes "ordinary chatbots" from "true personal assistants." This chapter introduces three types of memory: short-term memory (conversation history), long-term memory (vector databases), and working memory (Scratchpad), and builds a personal assistant Agent with memory in a hands-on project.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Understand the role and applicable scenarios of three memory types
- ✅ Implement conversation history management and window trimming
- ✅ Use vector databases to build long-term memory
- ✅ Build a complete personal assistant Agent with memory

## Chapter Structure

| Section | Content | Difficulty |
|---------|---------|-----------|
| 5.1 Why Do Agents Need Memory? | The value and challenges of memory | ⭐⭐ |
| 5.2 Short-Term Memory: Conversation History Management | Sliding window, summary compression | ⭐⭐ |
| 5.3 Long-Term Memory: Vector Databases | ChromaDB, similarity search | ⭐⭐⭐ |
| 5.4 Working Memory: Scratchpad Pattern | Recording the reasoning process | ⭐⭐⭐ |
| 5.5 Hands-on: Personal Assistant with Memory | Complete system implementation | ⭐⭐⭐⭐ |

## ⏱️ Estimated Study Time

Approximately **90–120 minutes** (including hands-on exercises)

## 💡 Prerequisites

- Completed Chapter 4 (Tool Calling)
- Familiar with Python lists and dictionary operations
- Intuitive understanding of "vectors" (no linear algebra background required)

## 🔗 Learning Path

> **Prerequisites**: [Chapter 3: Tool Calling](../chapter_tools/README.md)
>
> **Recommended Next Steps**:
> - 👉 [Chapter 5: Planning and Reasoning](../chapter_planning/README.md) — Give your Agent "thinking power"
> - 👉 [Chapter 6: RAG](../chapter_rag/README.md) — Enhance the Agent's knowledge base with retrieval

## 🚀 Extended Projects

| Project | Description | Stars |
|---------|-------------|-------|
| [supermemory](https://github.com/supermemoryai/supermemory) | A memory and context engine for the AI era. Supports automatic fact extraction, user profile building, forgetting-curve-style memory decay, and hybrid search (RAG + Memory). Ranked #1 on three major benchmarks: LongMemEval, LoCoMo, and ConvoMem. Provides API, MCP service, and LangChain/LangGraph integration. | 17.5k+ |

---

*Next section: [4.1 Why Do Agents Need Memory?](./01_why_memory.md)*
