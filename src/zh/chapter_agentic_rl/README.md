# 第10章 Agentic-RL：智能体强化学习训练

> 📖 *"如果说 Prompt Engineering 是给 Agent 写'使用说明书'，那么 Agentic-RL 就是让 Agent 通过反复实践，自己悟出最优解法。"*


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解 Agentic-RL 的概念和技术路线
- ✅ 掌握 SFT + LoRA 监督微调的实现方法
- ✅ 深入理解 PPO、DPO、GRPO 三大强化学习算法的原理
- ✅ 了解 DP / TP / PP / SP / ZeRO 分布式训练方法
- ✅ 完成一个完整的 SFT + GRPO 训练 Pipeline
- ✅ 掌握专为 Agent 微调的数据构建方法（工具调用轨迹、Gorilla/ToolBench 数据集）
- ✅ 理解 Agentic 数据飞轮：如何让 Agent 用自身运行数据持续自我进化

## ⏱️ 预计学习时间

约 **180-240 分钟**（含实战，建议分两天完成）

## 💡 前置知识

- 完成第2章（LLM 基础）和第9章（技能系统）
- 了解神经网络基础和反向传播
- 具备 PyTorch 基础操作经验
- GPU 环境（推荐）或 Google Colab

---

## 🔗 学习路径

> **前置知识**：[第2章 大语言模型基础](../chapter_llm/README.md)
> 推荐但非必须：[第5章 规划与推理（Planning & Reasoning）](../chapter_planning/README.md)、[附录 E：KL 散度详解](../appendix/kl_divergence.md)
>
> **后续推荐**：
> - 👉 [第11章 LangChain 深入实战](../chapter_langchain/README.md) — 用框架快速实践你训练出的模型
> - 👉 [第17章 Agent 的评估与优化](../chapter_evaluation/README.md) — 评估 RL 训练后的 Agent 效果

---

## 章节概述

在前面的章节中，我们一直以**提示词 + 工具调用**的方式构建 Agent——Agent 的所有能力来自基座模型的预训练知识加上精心设计的 prompt。这种方式简单灵活，但存在一个根本性瓶颈：

> **Agent 的能力上界 = 基座模型的通用能力上界。**

**Agentic-RL（Agentic Reinforcement Learning）** 提供了另一条路径：**通过强化学习训练，让模型自主习得完成 Agent 任务的最优策略**。DeepSeek-R1 [1] 和 DeepSWE [2] 等工作已经证明，经过 RL 训练的模型可以涌现出训练数据中从未出现过的推理策略，在推理和工具使用能力上显著超越纯 prompt 方式。

## 本章结构

| 节 | 内容 | 核心收获 |
|----|------|---------|
| 10.1 | 什么是 Agentic-RL | 理解 Agentic-RL 与传统后训练的本质区别，掌握 MDP 框架建模方法 |
| 10.2 | SFT + LoRA 基础训练 | 掌握监督微调的形式化原理与 LoRA 参数高效训练方法 |
| 10.2b | 分布式训练基础 | DP / TP / PP / SP / ZeRO 分布式训练方法 |
| 10.3 | PPO：近端策略优化 | 从策略梯度出发，系统理解重要性采样、优势函数、GAE 和 Clip 机制 |
| 10.4 | DPO：直接偏好优化 | 掌握从 RLHF 到 DPO 的完整数学推导，理解隐式奖励的思想 |
| 10.5 | GRPO/GSPO + 奖励函数设计 | 理解组内比较替代 Critic 的原理，以及多维度奖励函数设计与奖励黑客防御 |
| 10.6 | 实战：完整训练 Pipeline | 基于 GSM8K 完成从数据准备到模型部署的完整 Agentic-RL 训练 |
| 10.7 | 最新研究进展（2025—2026） | 纵览 DeepSeek-R1、DAPO、VAPO、SAR 等前沿工作，掌握领域最新动态 |
| 10.8 | 专为 Agent 的微调 | 工具调用数据构建、三大开源数据集（Gorilla/ToolBench/AgentInstruct）、Agent SFT 专项配置 |
| 10.9 | Agentic 数据飞轮 | 轨迹收集→质量过滤→奖励标注→训练迭代的完整闭环系统设计 |

---

## 参考文献

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
