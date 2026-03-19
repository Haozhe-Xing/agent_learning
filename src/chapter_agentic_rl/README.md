# 第18章 Agentic-RL：智能体强化学习训练

> 📖 *"如果说 Prompt Engineering 是给 Agent 写'使用说明书'，那么 Agentic-RL 就是让 Agent 通过反复实践，自己悟出最优解法。"*

## 章节概述

在前面的章节中，我们一直以**提示词 + 工具调用**的方式构建 Agent——Agent 的所有能力来自基座模型的预训练知识加上精心设计的 prompt。这种方式简单灵活，但存在一个根本性瓶颈：

> **Agent 的能力上界 = 基座模型的通用能力上界。**

**Agentic-RL（Agentic Reinforcement Learning）** 提供了另一条路径：**通过强化学习训练，让模型自主习得完成 Agent 任务的最优策略**。DeepSeek-R1 [1] 和 DeepSWE [2] 等工作已经证明，经过 RL 训练的模型可以涌现出训练数据中从未出现过的推理策略，在推理和工具使用能力上显著超越纯 prompt 方式。

## 你将学到

| 节 | 内容 | 核心收获 |
|----|------|---------|
| 18.1 | 什么是 Agentic-RL | 理解 Agentic-RL 与传统后训练的本质区别，掌握 MDP 框架建模方法 |
| 18.2 | SFT + LoRA 基础训练 | 掌握监督微调的形式化原理与 LoRA 参数高效训练方法 |
| 18.3 | PPO：近端策略优化 | 从策略梯度出发，系统理解重要性采样、优势函数、GAE 和 Clip 机制 |
| 18.4 | DPO：直接偏好优化 | 掌握从 RLHF 到 DPO 的完整数学推导，理解隐式奖励的思想 |
| 18.5 | GRPO + 奖励函数设计 | 理解组内比较替代 Critic 的原理，掌握多维度奖励函数设计与奖励黑客防御 |
| 18.6 | 实战：完整训练 Pipeline | 基于 GSM8K 完成从数据准备到模型部署的完整 Agentic-RL 训练 |

## 前置知识

- 理解 LLM 的基本工作原理（第2章）
- 了解 Python 和 PyTorch 基础
- 对机器学习/深度学习有基本概念

---

## 参考文献

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
