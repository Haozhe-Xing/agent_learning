# 智能体发展史：从符号主义到大模型驱动

> 📖 *"不了解历史，就无法真正理解当下。Agent 的每一次跃迁，都站在前人的肩膀之上。"*

## 为什么要了解发展史？

"Agent"这个概念并非 LLM 时代的发明。从 1950 年代 AI 学科诞生之日起，**如何构建能够自主行动的智能系统**就是核心研究命题。了解这段历史，能让我们：

1. 理解当前 Agent 架构设计背后的**学术渊源**
2. 避免"重新发明轮子"——很多经典方法仍在发挥作用
3. 预判未来 Agent 技术的**演进方向**

![智能体发展史时间线](../svg/chapter_intro_06_history_timeline.svg)

## 第一阶段：符号主义智能体（1950s—1980s）

### 图灵的预言

1950 年，Alan Turing 发表了划时代的论文《Computing Machinery and Intelligence》[1]，提出了"机器能思考吗？"这个根本性问题，并设计了著名的"图灵测试"。这篇论文为整个 AI 领域奠定了哲学基础。

### 早期专家系统

符号主义（Symbolism）认为智能的本质是**符号操作**：通过逻辑规则推理来模拟人类思维。这一时期的代表性系统包括：

- **SHRDLU（1970）**[2]：MIT 的 Terry Winograd 开发的自然语言理解系统，能在一个由积木组成的虚拟世界中理解并执行指令（如"把红色方块放在蓝色方块上面"）。它是最早的"能理解语言并执行动作"的系统——某种意义上，这就是最原始的 Agent。

- **MYCIN（1976）**[3]：斯坦福大学开发的医学诊断专家系统，使用约 600 条规则来诊断细菌感染并推荐抗生素。在临床测试中，MYCIN 的诊断准确率达到 69%，超过了当时大多数非专科医生。

- **STRIPS（1971）**[4]：SRI International 开发的自动化规划系统，为机器人 Shakey 提供行动规划能力。STRIPS 提出的"前置条件-动作-效果"规划范式，至今仍是 AI 规划领域的基础框架。

```
# STRIPS 规划表示示例（伪代码）
# 这种"条件-动作-效果"的思路在今天的 Agent 工具调用中依然可见

Action: move_block(block, from, to)
  Preconditions: 
    - on(block, from)        # 方块在源位置上
    - clear(block)           # 方块顶部没有其他方块
    - clear(to)              # 目标位置是空的
  Effects:
    - on(block, to)          # 方块移到目标位置
    - clear(from)            # 源位置变空
    - NOT on(block, from)    # 方块不再在源位置
```

### 符号主义的局限

符号主义 Agent 在**封闭领域**内表现出色，但面临根本性瓶颈：

| 问题 | 说明 |
|------|------|
| 知识获取瓶颈 | 规则需要人工手动编写，数量呈指数增长 |
| 脆弱性 | 遇到规则未覆盖的情况就完全失效 |
| 常识缺失 | 无法处理"不言自明"的常识知识 |
| 可扩展性差 | 规则库越大，规则之间的冲突越难管理 |

> 💡 **与当代 Agent 的联系**：符号主义的"规则+推理"思路并未消亡。在今天的 LLM Agent 中，**系统提示词（System Prompt）** 其实就是一种软性"规则"，而**工具的参数约束**则是硬性规则。区别在于 LLM 用统计学习取代了手工编码的逻辑推理。

## 第二阶段：心智社会与分布式智能（1980s—1990s）

### 明斯基的"心智社会"

1986 年，MIT 人工智能实验室的创始人之一 Marvin Minsky 出版了《心智社会（The Society of Mind）》[5]。他提出了一个革命性的观点：

> **智能不是单一能力的体现，而是大量"不那么聪明"的小 Agent（Minsky 称之为 "agency"）协作的结果。**

这个理论的核心思想是：

- 每个小 Agent 只负责一件简单的事（如"识别颜色"、"计算距离"）
- 复杂的智能行为由这些小 Agent 的层级协作**涌现**出来
- 不同的小 Agent 之间会竞争和合作

```python
# 心智社会的思想在今天的多 Agent 系统中得到了完美体现
# 以下是一个简化的概念演示

class SimpleAgent:
    """一个只擅长单一任务的简单 Agent"""
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    def can_handle(self, task: str) -> bool:
        return self.specialty in task.lower()
    
    def handle(self, task: str) -> str:
        return f"[{self.name}] 正在处理与 {self.specialty} 相关的任务..."

class SocietyOfMind:
    """心智社会：多个简单 Agent 的协作"""
    def __init__(self):
        self.agents = [
            SimpleAgent("搜索者", "搜索"),
            SimpleAgent("分析师", "分析"),
            SimpleAgent("写手", "撰写"),
            SimpleAgent("审核员", "检查"),
        ]
    
    def solve(self, task: str) -> list[str]:
        """将任务分发给能处理的 Agent"""
        results = []
        for agent in self.agents:
            if agent.can_handle(task):
                results.append(agent.handle(task))
        return results

# 当你在 CrewAI 或 AutoGen 中定义多个角色时，
# 本质上就是在实践明斯基的"心智社会"理论
```

### BDI 架构

1990 年代，Rao 和 Georgeff 提出了 **BDI（Belief-Desire-Intention）架构** [6]，成为理性 Agent 的标准理论框架：

- **Belief（信念）**：Agent 对世界的认知（"我认为现在的交通状况很拥堵"）
- **Desire（愿望）**：Agent 想要达成的目标（"我想在 30 分钟内到达公司"）
- **Intention（意图）**：Agent 决定采取的行动计划（"我选择坐地铁而不是开车"）

```python
from dataclasses import dataclass, field

@dataclass
class BDIAgent:
    """BDI 架构的 Agent（概念演示）"""
    
    # Belief: Agent 对世界的认知
    beliefs: dict = field(default_factory=lambda: {
        "traffic": "congested",    # 交通拥堵
        "weather": "rainy",        # 下雨
        "time": "08:30",           # 当前时间
        "has_umbrella": True,      # 有伞
    })
    
    # Desire: Agent 想要达成的目标
    desires: list = field(default_factory=lambda: [
        "到达公司",
        "不要迟到",
        "不要被淋湿",
    ])
    
    # Intention: Agent 选择的行动计划
    intentions: list = field(default_factory=list)
    
    def deliberate(self):
        """基于信念和愿望，形成意图（决策过程）"""
        if self.beliefs["traffic"] == "congested":
            self.intentions.append("选择地铁出行")
        if self.beliefs["weather"] == "rainy" and self.beliefs["has_umbrella"]:
            self.intentions.append("带上雨伞")
        return self.intentions

# BDI 的思想在今天的 Agent 中体现为：
# - Belief → Agent 的上下文/记忆（工具返回值、对话历史）
# - Desire → 用户的任务目标（System Prompt 中的任务描述）
# - Intention → Agent 的规划结果（ReAct 中的 Thought 部分）
```

> 💡 **BDI 与 ReAct 的联系**：如果你对比 BDI 架构和 ReAct 框架 [7]，会发现惊人的相似性——ReAct 中的 Thought 对应 BDI 的 Belief + 推理过程，Action 对应 Intention 的执行，Observation 对应 Belief 的更新。ReAct 本质上是用 LLM 实现了 BDI 架构中的"审慎推理"过程。

## 第三阶段：联结主义与深度学习（1990s—2020s）

### 从统计学习到神经网络

1990 年代后期，随着计算能力的提升和数据量的增长，**联结主义（Connectionism）** 逐渐占据主流。其核心思想是：智能可以通过大量简单计算单元（神经元）的**连接和学习**来涌现。

关键里程碑：

- **1997 年 Deep Blue**：IBM 的国际象棋程序击败世界冠军卡斯帕罗夫，但本质仍是搜索+启发式算法
- **2012 年 AlexNet** [8]：深度卷积神经网络在 ImageNet 竞赛中取得突破性成绩，开启深度学习革命
- **2016 年 AlphaGo** [9]：DeepMind 的围棋程序击败李世石，将深度强化学习推向大众视野。AlphaGo 可以被视为一个复杂的"游戏 Agent"——它能感知棋盘状态、推理最优走法、并执行落子动作

### 强化学习 Agent

深度强化学习（Deep RL）为 Agent 领域带来了一套系统的数学框架。Agent 被建模为在环境中采取行动以最大化累计奖励的实体 [10]：

```python
# 强化学习中的 Agent-环境交互循环
# 这个循环在 LLM Agent 中依然是核心模式

class RLAgentLoop:
    """
    强化学习的 Agent 循环：
    State → Action → Reward → New State → ...
    
    对比 LLM Agent 循环：
    Observation → Thought → Tool Call → Result → ...
    """
    
    def __init__(self, environment, policy):
        self.env = environment    # 环境
        self.policy = policy      # 策略（RL 中是神经网络，LLM Agent 中是 LLM）
    
    def run_episode(self, max_steps: int = 100):
        state = self.env.reset()               # 初始状态
        total_reward = 0
        
        for step in range(max_steps):
            action = self.policy(state)        # 根据策略选择动作
            next_state, reward, done = self.env.step(action)  # 执行并观察
            total_reward += reward
            
            if done:
                break
            state = next_state
        
        return total_reward
```

> 💡 **RL 到 LLM Agent 的传承**：强化学习的 Agent 循环（State→Action→Reward→NewState）直接映射到今天 LLM Agent 的工作循环（Observation→Thought→Action→Result）。唯一的区别是：RL Agent 的策略由数值化的神经网络驱动，而 LLM Agent 的策略由自然语言推理驱动。

### Attention Is All You Need

2017 年，Google 的研究团队发表了具有里程碑意义的 Transformer 论文 [11]，提出了完全基于注意力机制的序列到序列模型。这篇论文直接催生了：

- **GPT 系列**（OpenAI）：生成式预训练 + 指令微调
- **BERT**（Google）：双向编码器，在理解任务上取得突破
- **T5、PaLM、LLaMA** 等后续模型

Transformer 架构使得模型规模可以高效地扩展到数千亿参数，为 LLM 时代的到来奠定了技术基础。

## 第四阶段：LLM 驱动的智能体（2023—至今）

### 从语言模型到通用智能体

2022-2023 年，以 ChatGPT/GPT-4 为代表的大语言模型证明了一个关键假设：**足够大的语言模型能够涌现出推理、规划、工具使用等高层认知能力** [12]。这使得"Agent"的实现方式发生了根本性转变：

| 维度 | 传统 Agent | LLM 驱动的 Agent |
|------|-----------|-----------------|
| 决策引擎 | 规则/搜索/RL 策略网络 | 大语言模型 |
| 知识来源 | 手动编码的知识库 | 预训练中习得的世界知识 |
| 交互方式 | 结构化输入/输出 | 自然语言 |
| 泛化能力 | 仅限训练域 | 跨领域泛化 |
| 开发成本 | 需要大量领域工程 | Prompt + Tool 即可构建 |

### 标志性里程碑

**2023 年：概念验证阶段**

- **ReAct** [7]：将推理和行动统一到一个 LLM 交互循环中，成为 Agent 的基础范式
- **AutoGPT**（2023.3）：第一个引起全球关注的自主 Agent 项目，证明 LLM 可以自主规划和执行复杂任务
- **Generative Agents** [13]（2023.4）：Stanford 的"AI 小镇"实验，25 个 Agent 在虚拟小镇中自主生活、社交和记忆，展示了 Agent 的社会行为涌现
- **Voyager** [14]（2023.5）：NVIDIA 的 Minecraft Agent，能自主探索、学习技能并编写代码，展示了终身学习能力

**2024 年：工程化阶段**

- **Devin**（2024.3）：Cognition AI 推出的首个"AI 软件工程师"，在 SWE-bench 上取得突破
- **SWE-Agent** [15]（2024.6）：Princeton 的开源代码 Agent，系统性地设计了 Agent-Computer Interface (ACI)
- **MCP 协议**（2024.11）：Anthropic 发布 Model Context Protocol，标准化工具集成
- **A2A 协议**（2024.4）：Google 发布 Agent-to-Agent Protocol，标准化 Agent 间通信

**2025 年：规模化应用阶段**

- **Claude Code / Codex CLI**：Agent 进入开发者日常工作流
- **OpenAI Agents SDK**：官方 Agent 开发框架
- **DeepSWE**：纯 RL 训练的开源代码 Agent，在 SWE-bench Verified 上达到 59% SOTA
- **Anthropic** 发布 *Building Effective Agents* 指南 [16]，强调"简单组合优于复杂框架"
- **OpenAI** 发布 *A Practical Guide to Building Agents* [17]，提供完整的工程化最佳实践

### 当代 Agent 的三大范式

经过几年的快速发展，LLM 驱动的 Agent 形成了三大主要范式：

![当代 Agent 三大范式](../svg/chapter_intro_06_three_paradigms.svg)

## 发展脉络总结

整个智能体发展史可以用一条清晰的主线串联：

> **从"写规则"到"学规则"，再到"用语言替代规则"——Agent 的能力来源在不断抽象化和通用化。**

| 时期 | 范式 | Agent 能力来源 | 代表 |
|------|------|--------------|------|
| 1950s-1980s | 符号主义 | 人工编写的逻辑规则 | MYCIN, SHRDLU, STRIPS |
| 1980s-1990s | 分布式智能 | 多 Agent 协作涌现 | BDI 架构, 心智社会 |
| 1990s-2020s | 联结主义 | 数据驱动的统计学习 | AlphaGo, DQN |
| 2023-至今 | LLM 驱动 | 预训练知识 + 自然语言推理 | GPT-4 Agent, Claude Agent |

## 本节小结

- Agent 的概念贯穿了 AI 70 余年的发展史，远比 LLM 时代更早
- 符号主义奠定了"规则+推理"的基础，BDI 架构定义了理性 Agent 的理论框架
- 深度学习和强化学习提供了数据驱动的学习能力
- LLM 的出现实现了"质变"：Agent 首次获得了**跨领域的通用推理和规划能力**
- 理解历史有助于我们更好地设计和构建当代 Agent 系统

## 🤔 思考练习

1. MYCIN 的 600 条规则和今天 Agent 的 System Prompt，本质区别在哪里？
2. 明斯基的"心智社会"理论如何启发了今天的多 Agent 框架设计？
3. 从 STRIPS 到 ReAct，Agent 的"规划"能力经历了怎样的演变？
4. 你认为下一个阶段的 Agent 会是什么样的？

---

## 参考文献

[1] TURING A M. Computing machinery and intelligence[J]. Mind, 1950, 59(236): 433-460.

[2] WINOGRAD T. Understanding Natural Language[M]. New York: Academic Press, 1972.

[3] SHORTLIFFE E H, BUCHANAN B G. A model of inexact reasoning in medicine[J]. Mathematical Biosciences, 1975, 23(3-4): 351-379.

[4] FIKES R E, NILSSON N J. STRIPS: A new approach to the application of theorem proving to problem solving[J]. Artificial Intelligence, 1971, 2(3-4): 189-208.

[5] MINSKY M. The Society of Mind[M]. New York: Simon & Schuster, 1986.

[6] RAO A S, GEORGEFF M P. BDI agents: From theory to practice[C]//Proceedings of the First International Conference on Multi-Agent Systems (ICMAS). 1995: 312-319.

[7] YAO S, ZHAO J, YU D, et al. ReAct: Synergizing reasoning and acting in language models[C]//ICLR. 2023.

[8] KRIZHEVSKY A, SUTSKEVER I, HINTON G E. ImageNet classification with deep convolutional neural networks[C]//NeurIPS. 2012: 1097-1105.

[9] SILVER D, HUANG A, MADDISON C J, et al. Mastering the game of Go with deep neural networks and tree search[J]. Nature, 2016, 529(7587): 484-489.

[10] SUTTON R S, BARTO A G. Reinforcement Learning: An Introduction[M]. 2nd ed. Cambridge: MIT Press, 2018.

[11] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//NeurIPS. 2017: 5998-6008.

[12] WEI J, TAY Y, BOMMASANI R, et al. Emergent abilities of large language models[J]. Transactions on Machine Learning Research, 2022.

[13] PARK J S, O'BRIEN J C, CAI C J, et al. Generative agents: Interactive simulacra of human behavior[C]//UIST. 2023.

[14] WANG G, XIE Y, JIANG Y, et al. Voyager: An open-ended embodied agent with large language models[R]. arXiv preprint arXiv:2305.16291, 2023.

[15] YANG J, JIMENEZ C E, WETTIG A, et al. SWE-agent: Agent-computer interfaces enable automated software engineering[R]. arXiv preprint arXiv:2405.15793, 2024.

[16] ANTHROPIC. Building effective agents[EB/OL]. 2024. https://www.anthropic.com/engineering/building-effective-agents.

[17] OPENAI. A practical guide to building agents[R]. 2025. https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf.

---

## 📚 第一章总结

恭喜你完成了第一章的学习！🎉 让我们回顾一下学到的核心概念：

![第一章核心知识图谱](../svg/chapter_intro_05_knowledge_map.svg)

在下一章中，我们将深入了解 Agent 的"大脑"——大语言模型（LLM），理解它是如何工作的，以及如何有效地与它沟通。

*让我们继续前进！ 🚀*
