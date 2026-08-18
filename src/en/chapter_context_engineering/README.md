# Chapter 7: Context Engineering

> 📖 *"Prompt engineering teaches you how to talk to an LLM; context engineering teaches you how to help an LLM see the whole world."*

---

## Chapter Overview

If Prompt Engineering is about "how to ask a good question," then **Context Engineering** is about "how to build a high-quality information environment for an LLM."

As Agents take on increasingly complex tasks — from simple Q&A to long-horizon tasks spanning hundreds of interactions — simply optimizing prompts is far from enough. You need to **systematically manage** all the information the LLM can see during each inference: conversation history, tool returns, retrieved documents, task state... The total amount of this information can easily exceed 100K tokens, while the context window is finite.

This is one of the **most overlooked yet most impactful** topics in Agent development. Anthropic CEO Dario Amodei has explicitly stated: "I prefer to call it context engineering, not just prompt engineering" [1]. Mastering context engineering is the key leap from "being able to write prompts" to "being able to build production-grade Agents."

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Understand the essential difference between prompt engineering and context engineering, and establish a **systematic context design mindset**
- ✅ Identify and diagnose **context corruption**, and master methods for handling the Lost-in-the-Middle effect
- ✅ Flexibly apply the three major long-horizon strategies: **compression, structured notes, and sub-agent architecture**
- ✅ Implement a complete **GSSC context-building pipeline** and apply it to your own Agent projects

## Chapter Structure

| Section | Content | Key Takeaways | Difficulty |
|---------|---------|--------------|-----------|
| 8.1 From Prompt Engineering to Context Engineering | Define context engineering, compare the two | Six-source information model + three design principles | ⭐⭐ |
| 8.2 Context Window Management and Attention Budget | Context corruption, attention distribution, management techniques | Attention budget allocation + three management techniques | ⭐⭐⭐ |
| 8.3 Context Strategies for Long-Horizon Tasks | Compression, structured notes, sub-agent architecture | Principles and combined use of three strategies | ⭐⭐⭐ |
| 8.4 Practice: Building a Context Manager | Complete GSSC pipeline implementation | Reusable context management infrastructure | ⭐⭐⭐⭐ |

## ⏱️ Estimated Study Time

Approximately **90–120 minutes** (including hands-on exercises)

## Why Must Agent Developers Master Context Engineering?

A typical scenario: your Agent performs perfectly in round 1, but after round 20 it starts "forgetting things," "repeating itself," and "going off-topic." The LLM hasn't gotten dumber — the **context space has been filled with low-quality information**.

Context engineering is the systematic methodology for solving these problems. It upgrades you from "writing prompts by trial and error" to "engineering information flow":

- **Information collection**: aggregate candidate information from multiple sources (conversation, tools, RAG, task state)
- **Intelligent filtering**: select the optimal subset within the token budget, by priority and relevance
- **Dynamic compression**: summarize verbose content to free up space for new information
- **Optimal layout**: use attention distribution patterns to place key information where the LLM "pays most attention"

## 💡 Prerequisites

- Completed Chapter 3 (LLM Fundamentals), understanding how LLMs work
- Completed Chapters 4–7 (tool calling, memory systems, planning and reasoning, RAG), understanding the core information sources of Agents
- Understanding of the basic concepts of tokens and context windows
- Familiarity with Python dataclasses and basic data structures

## 🔗 Learning Path

> **Prerequisites**: [Chapter 2: LLM Fundamentals](../chapter_llm/README.md), [Chapters 4–7: Core Capabilities](../chapter_tools/README.md)
>
> **Recommended Next Steps**:
> - 👉 [Chapter 12: LangChain](../chapter_langchain/README.md) — Implement context management strategies with a framework
> - 👉 [Chapter 18: Evaluation and Optimization](../chapter_20_evaluation/README.md) — Evaluate the effectiveness of context strategies

---

## References

[1] AMODEI D. Context engineering vs prompt engineering[EB/OL]. 2025.

---

*Next: [7.1 From Prompt Engineering to Context Engineering](./01_context_vs_prompt.md)*
