# Chapter 10: Agentic-RL: Agent Reinforcement Learning Training

> 📖 *"If Prompt Engineering is writing an 'instruction manual' for an Agent, then Agentic-RL is letting the Agent figure out the optimal solution through repeated practice."*

## Chapter Overview

In previous chapters, we built Agents using **prompts + tool calling** — all of the Agent's capabilities came from the base model's pre-training knowledge plus carefully designed prompts. This approach is simple and flexible, but has a fundamental bottleneck:

> **The Agent's capability ceiling = the base model's general capability ceiling.**

**Agentic-RL (Agentic Reinforcement Learning)** provides an alternative path: **training models through reinforcement learning to autonomously learn optimal strategies for completing Agent tasks**. Works such as DeepSeek-R1 [1] and DeepSWE [2] have demonstrated that RL-trained models can develop reasoning strategies that never appeared in the training data, significantly surpassing pure prompt approaches in reasoning and tool use capabilities.

## What You'll Learn

| Section | Content | Key Takeaway |
|---------|---------|-------------|
| 11.1 | What Is Agentic-RL | Understand the essential difference between Agentic-RL and traditional post-training; master MDP framework modeling |
| 11.2 | SFT + LoRA Basic Training | Master the formal principles of supervised fine-tuning and LoRA parameter-efficient training |
| 11.3 | PPO: Proximal Policy Optimization | Starting from policy gradients, systematically understand importance sampling, advantage functions, GAE, and the Clip mechanism |
| 11.4 | DPO: Direct Preference Optimization | Master the complete mathematical derivation from RLHF to DPO; understand the implicit reward concept |
| 11.5 | GRPO/GSPO + Reward Function Design | Understand the principle of intra-group comparison replacing the Critic; master GSPO's sequence-level optimization improvements; multi-dimensional reward function design and reward hacking defense |
| 11.6 | Practice: Complete Training Pipeline | Complete a full Agentic-RL training from data preparation to model deployment based on GSM8K |
| 11.7 | Latest Research Progress (2025–2026) | Survey frontier work including DeepSeek-R1, DAPO, VAPO, SAR; stay current with the field |

## Prerequisites

- Understanding of basic LLM working principles (Chapter 3)
- Familiarity with Python and PyTorch basics
- Basic concepts in machine learning / deep learning

## 🔗 Learning Path

> **Prerequisites**: [Chapter 2: LLM Fundamentals](../chapter_llm/README.md)
> Recommended but not required: [Chapter 5: Planning and Reasoning](../chapter_planning/README.md), [Appendix E: KL Divergence Explained](../appendix/kl_divergence.md)
>
> **Recommended Next**:
> - 👉 [Chapter 12: LangChain](../chapter_langchain/README.md) — quickly practice with your trained model using a framework
> - 👉 [Chapter 18: Evaluation and Optimization](../chapter_evaluation/README.md) — evaluate Agent performance after RL training

---

## References

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
