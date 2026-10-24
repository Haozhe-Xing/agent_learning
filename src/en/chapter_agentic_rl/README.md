# Chapter 24 Agentic-RL: Agent Reinforcement Learning Training

**Reading level**: Research extension, after a capstone and independent evaluation; not an application prerequisite.

## This chapter's iteration

- **Previous limitation**: An application, execution boundaries, and regression set exist, yet evaluation reveals recurring tool-selection or argument-generation errors. Rule out evidence, interface, and control-flow defects before deciding to train the model.
- **This chapter adds**: Build SFT examples or verifiable rewards from project traces, separating training, tuning, and final testing. Deliver model/data versions, training configuration, and controlled comparisons rather than treating higher training reward as task improvement.
- **How to verify**: Reader experiment: select the toy GRPO or larger training design in section 24.6; compare task success, invalid tool arguments, and call cost on a fixed independent set, including unsolvable tasks and tool faults. The text supplies teaching code, not a complete companion training project. Existing application tests cover neither GPU training nor training gains; retain actual experiment records before reporting an improvement.
- **Next step**: Study [chapter 25](../chapter_self_evolving/README.md) for prompt or skill improvement without changing weights. Model changes return through [chapter 18's evaluation](../chapter_20_evaluation/README.md) and [chapter 19's safety checks](../chapter_21_security/README.md). If you have not built an application, start with the [capstones](../part6.md).

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

> **Practical prerequisites**: Complete [a capstone](../part6.md) and establish an independent baseline with [chapter 18](../chapter_20_evaluation/README.md).
> Theory: [chapter 2](../chapter_llm/README.md); revisit [chapter 5](../chapter_planning/README.md) and [appendix E on KL divergence](../appendix/kl_divergence.md) as needed.
>
> **Recommended Next Steps**:
> - 👉 [Chapter 25: Self-Evolution and Continual Learning](../chapter_self_evolving/README.md) — compare prompt, skill, and model improvement
> - 👉 [Chapter 18: Agent Evaluation and Optimization](../chapter_20_evaluation/README.md) — compare before and after training with the same tasks and budgets

---

## Chapter Overview

The earlier applications combine an existing model with tools, evidence, and execution controls. Consider changing model parameters when independent evaluation still exposes recurring behavioral limitations after improving context, interfaces, and workflow. Training is not a required stage for every Agent project.

**Agentic-RL (Agentic Reinforcement Learning)** explores improving task policies through interaction traces and rewards. DeepSeek-R1 [1] and DeepSWE [2] are research cases discussed here. Transfer to your application requires controlled tests on tasks excluded from training and tuning; training reward or one benchmark cannot establish it.

## 📚 Chapter Structure

| Section | Content | Key Takeaways |
|---------|---------|--------------|
| 24.1 | What is Agentic-RL | Understand the essential differences between Agentic-RL and traditional post-training, master the MDP framework modeling approach |
| 24.2 | SFT + LoRA Basic Training | Master the formal principles of supervised fine-tuning and LoRA parameter-efficient training methods |
| 24.2b | Distributed Training Fundamentals | DP / TP / PP / SP / ZeRO distributed training methods |
| 24.3 | PPO: Proximal Policy Optimization | Starting from policy gradients, systematically understand importance sampling, advantage functions, GAE, and the Clip mechanism |
| 24.4 | DPO: Direct Preference Optimization | Master the complete mathematical derivation from RLHF to DPO, understand the idea of implicit rewards |
| 24.5 | GRPO/GSPO + Reward Function Design | Understand the principle of intra-group comparison replacing the Critic, along with multi-dimensional reward function design and reward hacking defense |
| 24.6 | Hands-On: Complete Training Pipeline | First run through the core mechanisms with a minimal GRPO/RLVR demo, then complete the full training from data preparation to model deployment based on GSM8K |
| 24.7 | Latest Research Developments (2025–2026) | Survey cutting-edge works including DeepSeek-R1, DAPO, VAPO, SAR, T-STAR, UCPO, DelTA, SRPO, GROW, and grasp the technical landscape of the GRPO family and Agent RL |
| 24.8 | Fine-Tuning Tailored for Agents | Tool call data construction, three major open-source datasets (Gorilla/ToolBench/AgentInstruct), Agent SFT-specific configurations |

> 📌 **Extended reading**: [Chapter 25](../chapter_self_evolving/README.md) covers prompt optimization, skill improvement, and data flywheels. It can follow model training or provide an alternative route; training weights is not required first.

---

## References

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
