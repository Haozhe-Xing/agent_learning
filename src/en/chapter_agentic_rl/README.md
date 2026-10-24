# Chapter 10 Agentic-RL: Agent Reinforcement Learning Training

> 📖 *"If Prompt Engineering is writing a 'user manual' for an Agent, then Agentic-RL is letting the Agent figure out the optimal approach through repeated practice."*

## 🎓 Learning Objectives

After completing this chapter, you will be able to:

- ✅ Understand the concept and technical pathways of Agentic-RL
- ✅ Master the implementation of SFT + LoRA supervised fine-tuning
- ✅ Deeply understand the principles of PPO, DPO, and GRPO reinforcement learning algorithms
- ✅ Learn about DP / TP / PP / SP / ZeRO distributed training methods
- ✅ Complete a full SFT + GRPO training pipeline
- ✅ Understand intra-group sampling, relative advantage, and KL constraints in GRPO/RLVR through minimal runnable demos
- ✅ Master data construction methods tailored for Agent fine-tuning (tool call trajectories, Gorilla/ToolBench datasets)
- ✅ Track the latest research developments in the GRPO family, Agent RL, process rewards, and training efficiency optimization from 2025–2026

---

## 🔗 Learning Path

> **Prerequisites**: [Chapter 2: Large Language Model Foundations](../chapter_llm/README.md)
> Recommended but not required: [Chapter 5: Planning & Reasoning](../chapter_planning/README.md), [Appendix E: KL Divergence Explained](../appendix/kl_divergence.md)
>
> **Recommended Next Steps**:
> - 👉 [Chapter 12: LangChain In-Depth Practice](../chapter_langchain/README.md) — Quickly put your trained model into practice with frameworks
> - 👉 [Chapter 18: Agent Evaluation and Optimization](../chapter_20_evaluation/README.md) — Evaluate the effectiveness of RL-trained Agents

---

## Chapter Overview

In previous chapters, we built Agents using **prompts + tool calling** — all of the Agent's capabilities came from the base model's pre-training knowledge combined with carefully designed prompts. This approach is simple and flexible, but it has a fundamental bottleneck:

> **The upper bound of an Agent's capability = the upper bound of the base model's general capability.**

**Agentic-RL (Agentic Reinforcement Learning)** provides an alternative path: **through reinforcement learning training, the model autonomously learns the optimal policy for completing Agent tasks.** Works such as DeepSeek-R1 [1] and DeepSWE [2] have demonstrated that RL-trained models can exhibit reasoning strategies that never appeared in the training data, significantly outperforming pure prompt-based approaches in reasoning and tool-use capabilities.

## 📚 Chapter Structure

| Section | Content | Key Takeaways |
|---------|---------|--------------|
| 10.1 | What is Agentic-RL | Understand the essential differences between Agentic-RL and traditional post-training, master the MDP framework modeling approach |
| 10.2 | SFT + LoRA Basic Training | Master the formal principles of supervised fine-tuning and LoRA parameter-efficient training methods |
| 10.2b | Distributed Training Fundamentals | DP / TP / PP / SP / ZeRO distributed training methods |
| 10.3 | PPO: Proximal Policy Optimization | Starting from policy gradients, systematically understand importance sampling, advantage functions, GAE, and the Clip mechanism |
| 10.4 | DPO: Direct Preference Optimization | Master the complete mathematical derivation from RLHF to DPO, understand the idea of implicit rewards |
| 10.5 | GRPO/GSPO + Reward Function Design | Understand the principle of intra-group comparison replacing the Critic, along with multi-dimensional reward function design and reward hacking defense |
| 10.6 | Hands-On: Complete Training Pipeline | First run through the core mechanisms with a minimal GRPO/RLVR demo, then complete the full training from data preparation to model deployment based on GSM8K |
| 10.7 | Latest Research Developments (2025–2026) | Survey cutting-edge works including DeepSeek-R1, DAPO, VAPO, SAR, T-STAR, UCPO, DelTA, SRPO, GROW, and grasp the technical landscape of the GRPO family and Agent RL |
| 10.8 | Fine-Tuning Tailored for Agents | Tool call data construction, three major open-source datasets (Gorilla/ToolBench/AgentInstruct), Agent SFT-specific configurations |

> 📌 **Extended Reading**: The original 10.9 Agentic Data Flywheel, original 10.10/10.11 Self-Evolution Agent, and self-evolution frontier content have been independently expanded into [Chapter 11: Self-Evolution and Continual Learning](../chapter_self_evolving/README.md). It is recommended to continue reading after completing this chapter.

---

## References

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
