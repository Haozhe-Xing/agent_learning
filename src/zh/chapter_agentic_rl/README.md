# 第24章 Agentic-RL：智能体强化学习训练

**阅读层次**：研究选读；在综合项目与独立评估之后学习，不是应用开发前置。

## 本章迭代

- **上一版的不足**：已有应用、执行边界和回归集，但评估仍显示稳定的工具选择或参数生成错误。先排除资料、接口和控制流缺陷，才能判断是否值得用训练改模型。
- **本章增加**：以已有项目轨迹构建 SFT 数据或可验证奖励，分开训练、调参与最终测试；形成模型版本、数据版本、训练配置和对照结果，而不是凭训练奖励上涨宣布 Agent 变好。
- **怎样验证**：读者实验：在第24.6节的玩具 GRPO 与完整训练方案中选择合适规模，比较固定独立任务上的成功率、非法工具参数率与调用成本，并测试不可解题和工具故障。正文含教学代码，仓库未提供完整训练配套工程，既有应用测试不覆盖 GPU 训练或训练收益；须保存实际运行记录后才能报告改善。
- **下一步**：需要改提示或技能而不改模型权重时，可直接学习[第25章](../chapter_self_evolving/README.md)；模型变更必须重新经过[第18章评估](../chapter_20_evaluation/README.md)和[第19章安全检查](../chapter_21_security/README.md)。若尚未做过应用，先回到[综合项目篇](../part6.md)。

> 📖 *"如果说 Prompt Engineering 是给 Agent 写'使用说明书'，那么 Agentic-RL 就是让 Agent 通过反复实践，自己悟出最优解法。"*

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解 Agentic-RL 的概念和技术路线
- ✅ 掌握 SFT + LoRA 监督微调的实现方法
- ✅ 深入理解 PPO、DPO、GRPO 三大强化学习算法的原理
- ✅ 了解 DP / TP / PP / SP / ZeRO 分布式训练方法
- ✅ 完成一个完整的 SFT + GRPO 训练 Pipeline
- ✅ 通过最小可运行 demo 理解 GRPO/RLVR 的组内采样、相对优势和 KL 约束
- ✅ 掌握专为 Agent 微调的数据构建方法（工具调用轨迹、Gorilla/ToolBench 数据集）
- ✅ 跟踪 2025—2026 年 GRPO 家族、Agent RL、过程奖励与训练效率优化的最新研究脉络

---

## 🔗 学习路径

> **实践前置**：至少完成[一个综合项目](../part6.md)，并建立[第18章](../chapter_20_evaluation/README.md)的独立评估基线。
> 理论基础：[第2章](../chapter_llm/README.md)；按需回查[第5章](../chapter_planning/README.md)与[附录 E：KL 散度](../appendix/kl_divergence.md)。
>
> **后续推荐**：
> - 👉 [第25章 自我进化与持续学习](../chapter_self_evolving/README.md) — 比较提示、技能和模型改进的边界
> - 👉 [第18章 Agent 的评估与优化](../chapter_20_evaluation/README.md) — 用相同任务和预算比较训练前后效果

---

## 本章概览

前面的应用以已有模型配合工具、资料和执行控制完成任务。若改进上下文、工具接口与工作流后，独立评估仍反复暴露模型行为上的瓶颈，可以考虑调整模型参数；训练不是每个 Agent 项目都需要的阶段。

**Agentic-RL（Agentic Reinforcement Learning）** 探索通过交互轨迹和奖励改进任务策略。DeepSeek-R1 [1] 与 DeepSWE [2] 是本章讨论的研究案例。能否迁移到自己的应用，需要用未参与训练和调参的任务进行对照验证，不能从训练奖励或单一基准直接推断。

## 📚 本章结构

| 节 | 内容 | 核心收获 |
|----|------|---------|
| 24.1 | 什么是 Agentic-RL | 理解 Agentic-RL 与传统后训练的本质区别，掌握 MDP 框架建模方法 |
| 24.2 | SFT + LoRA 基础训练 | 掌握监督微调的形式化原理与 LoRA 参数高效训练方法 |
| 24.2b | 分布式训练基础 | DP / TP / PP / SP / ZeRO 分布式训练方法 |
| 24.3 | PPO：近端策略优化 | 从策略梯度出发，系统理解重要性采样、优势函数、GAE 和 Clip 机制 |
| 24.4 | DPO：直接偏好优化 | 掌握从 RLHF 到 DPO 的完整数学推导，理解隐式奖励的思想 |
| 24.5 | GRPO/GSPO + 奖励函数设计 | 理解组内比较替代 Critic 的原理，以及多维度奖励函数设计与奖励黑客防御 |
| 24.6 | 实战：完整训练 Pipeline | 先用最小 GRPO/RLVR demo 跑通核心机制，再基于 GSM8K 完成从数据准备到模型部署的完整训练 |
| 24.7 | 最新研究进展（2025—2026） | 纵览 DeepSeek-R1、DAPO、VAPO、SAR、T-STAR、UCPO、DelTA、SRPO、GROW 等前沿工作，掌握 GRPO 家族与 Agent RL 的技术脉络 |
| 24.8 | 专为 Agent 的微调 | 工具调用数据构建、三大开源数据集（Gorilla/ToolBench/AgentInstruct）、Agent SFT 专项配置 |

> 📌 **延伸阅读**：[第25章](../chapter_self_evolving/README.md)讨论提示优化、技能改进和数据飞轮。它可以作为模型训练的替代改进路线，也可以接续本章，不要求先完成权重训练。

---

## 参考文献

[1] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[2] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.
