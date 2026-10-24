# Chapter 2: LLM Fundamentals

**Reading level**: Core; sections 2.6–2.8 are optional.

## This chapter's iteration

- **Previous limitation**: Chapter 1 defines a task, but there is no reproducible record of how the model receives evidence, responds, or fails. A fluent numeric answer may still be wrong.
- **This chapter adds**: Turn the task card into messages containing the evidence, requested unit, and output requirements. Keep the model name, parameters, raw response, and error status as a baseline for the tool-assisted version.
- **How to verify**: Reader exercise: save two calls for the same task, checking non-empty output, format, and arithmetic separately; add a missing-evidence case. Deliver the call records and manual verdicts. Two calls do not establish overall model quality. Live calls require credentials; deterministic offline outputs do not replace this experiment.
- **Next step**: Move arithmetic into tools in [chapter 3](../chapter_tools/README.md). Revisit model architecture in 2.7 and training data in 2.8 with [chapter 24](../chapter_agentic_rl/README.md), after building an application; neither is a prerequisite for the first tool exercise.

> A craftsman must first sharpen his tools. Before we start building Agents, we need to deeply understand their "brain" — the Large Language Model (LLM).

---

## Chapter Overview

This chapter explains how large language models work at an intuitive level, then covers Prompt Engineering, prompting strategies, and the first API call. Tokens and model parameters complete the application baseline. Foundation model architecture and training data are optional extensions; automatic prompt optimization is developed in chapter 25 after applications.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Intuitively understand how LLMs work (no math required)
- ✅ Master the core principles and techniques of Prompt Engineering
- ✅ Flexibly apply prompting strategies like Zero-shot, Few-shot, and CoT
- ✅ Proficiently call the OpenAI API and common open-source model interfaces
- ✅ Understand how parameters like Token and Temperature affect output
- ✅ Understand the architectural components of mainstream models (MHA/GQA/MLA, RoPE, SwiGLU, MoE) and 2026 breakthroughs (Hybrid Attention, Attention Residuals, MuonClip, Engram Memory)
- ✅ Master the latest advances in foundation models and model selection strategies for Agent development
- ✅ Understand the core principles of SFT and RL training data preparation: data volume selection, quality evaluation, and reward function design

## Chapter Structure

| Section | Content | Difficulty |
|---------|---------|-----------|
| 2.1 How Does an LLM Work? | Intuitive understanding of Transformers, pre-training, and emergent abilities | ⭐⭐ |
| 2.2 Prompt Engineering | System messages, role-playing, structured output | ⭐⭐ |
| 2.3 Prompting Strategies | Zero-shot, Few-shot, CoT, ToT | ⭐⭐⭐ |
| 2.4 Model API Basics | OpenAI SDK, open-source models, streaming | ⭐⭐ |
| 2.5 Tokens & Model Parameters | Token counting, Temperature, Top-p, etc. | ⭐⭐ |
| 2.6 Foundation Model Landscape | Industry landscape, model ecosystem (Kimi K2/K2.5, DeepSeek V4, Qwen3.5), Agent selection guide | ⭐⭐⭐ |
| 2.7 Foundation Model Architecture | MHA→GQA→MLA, RoPE, SwiGLU, MoE, and 2026 breakthroughs: Hybrid Attention, Attention Residuals, MuonClip, Engram Memory | ⭐⭐⭐⭐ |
| 2.8 SFT & RL Training Data Preparation | Data volume selection, quality evaluation, SFT data creation, RL reward function design, difficulty calibration and curriculum learning | ⭐⭐⭐ |

## Core Concepts at a Glance

![LLM Core Concepts Overview](../svg/chapter_llm_readme_llm_tree.svg)

## Why Do Agent Developers Need to Understand LLMs?

Many Agent frameworks (LangChain, LangGraph, etc.) wrap model calls very cleanly, allowing beginners to get started quickly. But when your Agent encounters the following issues, understanding the underlying LLM mechanisms becomes critical:

- Unstable output — the same question gets different answers
- Model "hallucination" — confidently giving wrong answers
- Token limit exceeded — long conversations get truncated
- High costs — needing to optimize Prompts to reduce consumption

Understanding LLMs is like understanding how an engine works — even if you don't build engines, knowing the principles makes you a better driver.

## 🔗 Learning Path

> **Prerequisites**: [Chapter 1: What is an Agent?](../chapter_intro/README.md), [Appendix F: Development Environment Setup](../chapter_setup/README.md)
>
> **Recommended next steps**:
> - 👉 [Chapter 3: Tool Calling](../chapter_tools/README.md) — The core capability of Agents
> - 👉 [Chapter 7: Context Engineering](../chapter_context_engineering/README.md) — Upgrade from Prompt Engineering to systematic context management

---

*Next section: [2.1 How Does an LLM Work? (Intuitive Understanding)](./01_how_llm_works.md)*
