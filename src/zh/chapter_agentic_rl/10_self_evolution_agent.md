# 10.10 Self-Evolution Agent：从会执行到会自我改进

> 🧬 *“真正长期有价值的 Agent，不只是完成当前任务，而是能从每次任务中提取经验，让下一次表现更好。”*

第 10.9 节介绍了 **Agentic 数据飞轮**：通过收集轨迹、过滤标注、训练迭代，让模型本身不断增强。本节进一步讨论一个更贴近 Agent 系统工程的概念：**Self-Evolution Agent（自我进化智能体）**。

Self-Evolution Agent 不一定每次都更新模型权重。很多时候，它通过更新记忆、提示词、工具策略、技能库、评估规则和任务流程来完成“自我进化”。因此，它是连接 **记忆系统、反思机制、技能系统、评估体系和 Agentic-RL** 的综合形态。

---

## 什么是 Self-Evolution Agent？

**Self-Evolution Agent** 是一种能够基于自身运行经验持续改进的 Agent 系统。它会把每次任务中的成功经验、失败原因、用户反馈和环境变化沉淀为可复用资产，并在后续任务中自动应用。

可以用一句话概括：

> **普通 Agent 解决任务；Self-Evolution Agent 解决任务，并学习如何更好地解决下一类任务。**

| 维度 | 普通 Agent | Self-Evolution Agent |
|------|------------|----------------------|
| **目标** | 完成当前请求 | 完成请求，并提取可复用经验 |
| **反馈使用** | 只在当前对话中修正 | 写入长期记忆、技能库或训练数据 |
| **能力变化** | 会话结束后基本不变 | 随运行次数积累而改进 |
| **改进对象** | Prompt 或单次推理 | 记忆、工具选择、流程、技能、模型策略 |
| **评估方式** | 看最终答案是否正确 | 同时评估过程、成本、稳定性和可迁移性 |

Self-Evolution Agent 的核心不是“让模型自己随意修改自己”，而是建立一个受控闭环：

```text
执行任务 → 记录轨迹 → 评估表现 → 归因失败/成功 → 生成改进 → 验证改进 → 安全部署
```

只有经过评估和验证的改进，才会进入长期系统。

---

## 自我进化的四个层级

Self-Evolution 可以发生在不同层级。越往下，收益越大，但风险和成本也越高。

| 层级 | 进化对象 | 典型方式 | 成本 | 风险 |
|------|----------|----------|------|------|
| **L1 记忆进化** | 长期记忆、偏好、经验教训 | 把成功策略和失败教训写入记忆 | 低 | 记忆污染 |
| **L2 Prompt 进化** | System Prompt、任务模板、工具说明 | 自动生成更好的指令和约束 | 低-中 | 过拟合少数案例 |
| **L3 Skill 进化** | 可复用技能、脚本、工作流 | 把高频任务封装成 Skill | 中 | 技能错误被复用 |
| **L4 Model 进化** | SFT / DPO / RL 训练数据 | 通过数据飞轮更新模型权重 | 高 | 灾难性遗忘、安全回退困难 |

生产系统通常从 L1 和 L2 开始：先让 Agent 学会“记住教训”和“改写流程”，等轨迹数据足够稳定后，再进入 L3 技能沉淀和 L4 模型训练。

---

## 代表性前沿工作：Self-Evolution 的论文脉络

Self-Evolution Agent 不是单一论文提出的固定架构，而是由多条研究线汇合而成：**反思学习、输出自修正、工具辅助批评、技能库终身学习、自动设计 Agent 系统、自我修改代码库**。理解这些工作，才能看清“自我进化”到底可以发生在哪一层。

### Reflexion：把奖励信号转化为语言记忆

- **论文链接**：[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- **当时的核心贡献**：在 2023 年，这篇论文提出了“语言强化学习”范式：不通过更新模型权重，而是把环境反馈、奖励信号或测试结果转化为自然语言反思，写入记忆缓冲区，指导 Agent 在后续尝试中改进行为。

**Reflexion: Language Agents with Verbal Reinforcement Learning** 是 Self-Evolution Agent 的基础论文之一。它提出的问题是：如果不更新模型权重，Agent 能否仍然从失败中学习？

Reflexion 的答案是：可以。它把任务反馈、环境奖励或单元测试结果转化为一段自然语言反思，并写入记忆缓冲区。下一次尝试同类任务时，Agent 会把这些反思读回上下文，从而避免重复犯错。

它的典型结构包括三部分：

- **Actor**：执行任务，生成动作或答案。
- **Evaluator**：根据环境反馈、测试结果或规则判断任务表现。
- **Self-Reflection**：把失败原因总结成自然语言经验，供下一轮使用。

这篇工作的关键意义在于提出了“语言强化学习”视角：**学习不一定发生在模型参数里，也可以发生在可检索、可编辑、可遗忘的语言记忆里**。它对应前面四层中的 L1 记忆进化。

### Self-Refine：用自反馈迭代改进单次输出

- **论文链接**：[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- **当时的核心贡献**：在 2023 年，这篇论文系统化证明了 LLM 可以在没有额外训练数据、监督信号或强化学习的情况下，通过“生成—自评—改写”的迭代循环提升输出质量。

**Self-Refine: Iterative Refinement with Self-Feedback** 关注的是另一类自我改进：模型生成初稿后，能否自己提出反馈，再根据反馈改写输出？

它的流程非常简单：

```text
Generate 初稿 → Feedback 自评 → Refine 改写 → 再次 Feedback → ... → 停止
```

Self-Refine 的贡献不在于长期记忆，而在于证明了很多任务可以通过“生成—反馈—改写”的循环获得更好结果，且不需要额外训练数据或强化学习。它适合写作、代码优化、格式约束生成等任务。

对 Self-Evolution Agent 来说，Self-Refine 对应 L2 Prompt / 流程进化：系统可以把一次成功的自修正过程沉淀为模板，例如“先生成候选方案，再从准确性、完整性、约束满足三个维度批评，最后重写”。

### CRITIC：让外部工具参与批评与修正

- **论文链接**：[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)
- **当时的核心贡献**：在 2023—2024 年，这篇工作把自我修正从“模型主观自评”推进到“工具交互式批评”，让模型调用搜索、计算器、代码解释器等外部工具验证声明，再根据工具结果修正答案。

**CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing** 进一步指出：只靠模型自评很容易自信地重复错误。更可靠的做法是让模型调用外部工具来验证自己的输出，例如搜索引擎、代码解释器、计算器或数据库。

CRITIC 的典型流程是：

```text
Generate → Verify with Tools → Critique → Correct
```

它和 Self-Refine 的区别在于，批评不再只来自模型的“主观判断”，而是来自外部工具的客观反馈。对自我进化系统来说，这一点非常重要：**没有外部验证的自我改进很容易变成自我强化幻觉**。

因此，生产级 Self-Evolution Agent 在写入长期记忆、更新 Prompt 或沉淀 Skill 前，都应该经过工具验证、测试集验证或人类审核。

### Voyager：自动课程 + 技能库 + 环境反馈的终身学习 Agent

- **论文链接**：[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- **当时的核心贡献**：在 2023 年，这篇工作首次展示了由 LLM 驱动的开放式具身终身学习 Agent：它能在 Minecraft 中自动生成课程、持续探索环境，并把成功行为沉淀为可检索、可复用的代码技能库。

**Voyager: An Open-Ended Embodied Agent with Large Language Models** 是 Self-Evolution Agent 走向 L3 Skill 进化的代表工作。它在 Minecraft 这样的开放世界中，让 Agent 持续探索环境、发现新目标、学习新技能，而不需要人类逐步指定任务。

Voyager 的三个核心组件非常经典：

1. **Automatic Curriculum**：自动生成适合当前能力水平的新任务，让 Agent 持续探索。
2. **Skill Library**：把成功行为保存为可执行代码技能，后续遇到相似任务时检索复用。
3. **Iterative Prompting Mechanism**：结合环境反馈、执行错误和自我验证，不断修正行为。

Voyager 的意义在于，它把“经验”从自然语言记忆推进到“可执行技能”。这比单纯写反思更强，因为技能可以直接调用、组合和迁移。它对应 L3 Skill 进化，也是很多现代 Agent 框架中“工具化技能库”的思想来源。

### ADAS：自动搜索和设计 Agent 系统本身

- **论文链接**：[Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)
- **当时的核心贡献**：在 2024—2025 年，这篇论文把 Agent 架构设计本身定义为可搜索、可优化的问题，提出由元智能体自动探索 Prompt、模块组合、控制流和多 Agent 结构，而不是完全依赖人工设计 ReAct、Reflection 等固定范式。

**Automated Design of Agentic Systems (ADAS)** 把自我进化推进到更高层：不仅让 Agent 学会任务，还让系统自动搜索“什么样的 Agent 架构更有效”。

ADAS 把 Agent 系统设计看成一个搜索问题：

- **搜索空间**：可组合的 Prompt、模块、控制流、多 Agent 结构、工具调用策略。
- **搜索算法**：如何提出新架构、变异旧架构、筛选候选系统。
- **评估函数**：用任务集、成本、准确率或鲁棒性衡量候选系统。

这类工作的重要性在于：过去 Agent 系统主要靠研究者手工设计，例如 ReAct、Plan-and-Execute、Reflection、多 Agent Debate；ADAS 则尝试让元智能体自动生成和评估新的 Agent 设计。

从 Self-Evolution 的角度看，ADAS 对应 L2 到 L3 之间的系统级进化：更新的不只是单条记忆或单个技能，而是 Agent 的整体工作流。

### SICA：能编辑自身代码库的自我改进编码 Agent

- **论文链接**：[A Self-Improving Coding Agent](https://arxiv.org/abs/2504.15228)
- **当时的核心贡献**：在 2025 年，这篇工作进一步让编码 Agent 直接作用于自己的代码库，通过评估当前版本、提出改动、修改自身实现、重新跑基准测试的闭环，探索代码级自我改进的可行性与安全边界。

2025 年的 **A Self-Improving Coding Agent (SICA)** 进一步触及一个更激进的问题：编码 Agent 能否直接修改自己的代码库，让自己在后续任务中更快、更便宜或更强？

与 ADAS 中“元智能体优化目标智能体”的设定不同，SICA 尝试消除元智能体与目标智能体的边界：Agent 观察自己的运行瓶颈，提出代码级改动，修改自身系统，然后用基准测试验证改动是否有效。

这代表了 Self-Evolution Agent 的高风险形态。它的潜力很大，因为系统可以直接优化实现；但风险也很高，因为错误改动可能破坏安全边界、工具协议或任务稳定性。因此，任何“自改代码”的系统都必须具备：

- 沙箱执行环境；
- 回归测试集；
- 版本控制和回滚机制；
- 权限边界；
- 人类审批或灰度发布流程。

### 论文脉络小结

| 代表工作 | 自我进化层级 | 核心问题 | 核心机制 | 工程启发 |
|----------|--------------|----------|----------|----------|
| **Reflexion** | L1 记忆进化 | 不改权重能否从失败中学习 | 反馈 → 语言反思 → 记忆缓冲区 | 把失败归因写成可检索经验 |
| **Self-Refine** | L2 流程进化 | 单次输出能否自我改写变好 | Generate → Feedback → Refine | 把自评改写沉淀为任务模板 |
| **CRITIC** | L2/L3 验证进化 | 自我批评如何避免幻觉 | 工具验证 → 批评 → 修正 | 重要改进必须经过外部验证 |
| **Voyager** | L3 技能进化 | Agent 能否在开放环境中终身学习 | 自动课程 + 可执行技能库 + 环境反馈 | 把成功轨迹沉淀为可调用 Skill |
| **ADAS** | 系统级进化 | 能否自动设计更好的 Agent 架构 | 搜索空间 + 搜索算法 + 评估函数 | 让工作流、模块组合和控制流参与进化 |
| **SICA** | 代码级进化 | Agent 能否修改自身代码库 | 自诊断 + 代码修改 + 基准验证 | 自改系统必须有沙箱、测试和回滚 |

这些论文共同说明：Self-Evolution 不是一句“让 Agent 自我改进”的口号，而是一组逐层增强的机制。从 Reflexion 的语言记忆，到 Voyager 的技能库，再到 ADAS/SICA 的系统级和代码级改进，每一层都需要更严格的评估和安全边界。

---

## Self-Evolution Agent 的系统架构

一个可控的自我进化系统通常包含六个模块：

1. **执行器（Executor）**：完成用户任务，调用工具、检索资料、生成结果。
2. **轨迹记录器（Trajectory Logger）**：保存输入、计划、工具调用、观察结果、最终输出和成本。
3. **评估器（Evaluator）**：判断任务是否成功、过程是否可靠、是否存在安全问题。
4. **归因器（Critic / Diagnoser）**：分析成功或失败原因，定位可改进点。
5. **进化器（Evolution Engine）**：把改进点转化为记忆、Prompt patch、Skill 或训练样本。
6. **验证器（Validator）**：在回归测试和沙箱中验证改进是否真的有效。

```text
┌──────────────┐
│ 用户任务      │
└──────┬───────┘
       ↓
┌──────────────┐      ┌──────────────┐
│ Executor     │─────→│ Trajectory   │
│ 执行任务      │      │ Logger       │
└──────┬───────┘      └──────┬───────┘
       ↓                     ↓
┌──────────────┐      ┌──────────────┐
│ 用户结果      │      │ Evaluator    │
└──────────────┘      └──────┬───────┘
                              ↓
                       ┌──────────────┐
                       │ Diagnoser    │
                       └──────┬───────┘
                              ↓
                       ┌──────────────┐
                       │ Evolution    │
                       │ Engine       │
                       └──────┬───────┘
                              ↓
                       ┌──────────────┐
                       │ Validator    │
                       └──────┬───────┘
                              ↓
                    记忆 / Prompt / Skill / 训练数据
```

关键原则是：**执行路径和进化路径要分离**。用户请求应该被稳定处理，进化逻辑最好异步运行，避免每次对话都被“自我分析”拖慢。

---

## 自我进化循环：从一次失败中学习

假设一个代码 Agent 在修改项目时犯了错误：没有先读取最新文件内容，就直接基于过期上下文做替换，导致补丁失败。

Self-Evolution Agent 不应该只返回“替换失败”，而应该提取一条可复用经验：

```json
{
  "event": "patch_failed",
  "failure_reason": "used_stale_context_before_search_replace",
  "lesson": "在执行精确替换前，必须先读取目标文件的最新内容，并复制真实上下文作为 old_string。",
  "trigger": "replace_in_file 或 search-and-replace 编辑",
  "future_rule": "如果需要精确替换，先读取文件；不要使用摘要中的旧内容作为替换依据。",
  "confidence": 0.92
}
```

下一次遇到类似任务时，Agent 就可以自动应用这条规则，而不是重复犯错。

这就是自我进化的最小闭环：

1. **检测失败**：工具失败、测试失败、用户纠正、评估低分。
2. **归因失败**：不是简单记录“失败了”，而是找出可操作原因。
3. **抽象经验**：把具体错误转化为未来可复用规则。
4. **验证经验**：确认这条规则不会伤害其他任务。
5. **应用经验**：在相似场景中自动触发。

---

## 实现骨架：一个轻量 Self-Evolution Agent

下面是一个简化版本，展示自我进化系统的关键数据结构和控制流。

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import json


@dataclass
class AgentEvent:
    """一次 Agent 运行事件"""
    task: str
    plan: list[str]
    actions: list[dict]
    final_answer: str
    success: bool
    feedback: str | None = None
    cost_tokens: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvolutionPatch:
    """一次候选进化补丁"""
    patch_type: Literal["memory", "prompt", "skill", "training_sample"]
    content: str
    trigger: str
    expected_benefit: str
    risk: str
    confidence: float


class SelfEvolutionAgent:
    """轻量级自我进化 Agent"""

    def __init__(self, base_agent, evaluator, memory_store):
        self.base_agent = base_agent
        self.evaluator = evaluator
        self.memory_store = memory_store
        self.pending_patches: list[EvolutionPatch] = []

    def run(self, task: str) -> str:
        """执行用户任务，并异步产生进化候选"""
        event = self._execute(task)

        # 用户结果优先返回；进化逻辑可以放到后台队列
        patches = self._reflect_and_propose(event)
        verified = self._validate_patches(patches)
        self._apply_patches(verified)

        return event.final_answer

    def _execute(self, task: str) -> AgentEvent:
        """执行任务并记录轨迹"""
        result = self.base_agent.run(task)
        score = self.evaluator.evaluate(task, result)

        return AgentEvent(
            task=task,
            plan=result.get("plan", []),
            actions=result.get("actions", []),
            final_answer=result.get("answer", ""),
            success=score["success"],
            feedback=score.get("feedback"),
            cost_tokens=result.get("cost_tokens", 0),
        )

    def _reflect_and_propose(self, event: AgentEvent) -> list[EvolutionPatch]:
        """基于成功/失败轨迹生成改进建议"""
        patches = []

        if not event.success:
            patches.append(EvolutionPatch(
                patch_type="memory",
                content=f"任务失败经验：当遇到类似任务 `{event.task}` 时，先检查失败原因：{event.feedback}",
                trigger=self._infer_trigger(event),
                expected_benefit="减少相同错误重复发生",
                risk="可能把一次偶然失败泛化为通用规则",
                confidence=0.75,
            ))

        if event.success and len(event.actions) >= 3:
            patches.append(EvolutionPatch(
                patch_type="skill",
                content=self._summarize_successful_workflow(event),
                trigger=self._infer_trigger(event),
                expected_benefit="把多步成功流程沉淀为可复用技能",
                risk="流程可能只适用于当前环境",
                confidence=0.68,
            ))

        return patches

    def _validate_patches(self, patches: list[EvolutionPatch]) -> list[EvolutionPatch]:
        """验证候选补丁，过滤高风险或低置信度改进"""
        verified = []
        for patch in patches:
            if patch.confidence < 0.7:
                continue
            if "绕过权限" in patch.content or "忽略安全" in patch.content:
                continue
            verified.append(patch)
        return verified

    def _apply_patches(self, patches: list[EvolutionPatch]):
        """应用通过验证的改进"""
        for patch in patches:
            if patch.patch_type == "memory":
                self.memory_store.save({
                    "trigger": patch.trigger,
                    "content": patch.content,
                    "confidence": patch.confidence,
                    "created_at": datetime.now().isoformat(),
                })
            else:
                self.pending_patches.append(patch)

    def _infer_trigger(self, event: AgentEvent) -> str:
        """从任务和动作中推断未来触发条件"""
        if any(action.get("tool") == "search" for action in event.actions):
            return "需要检索或调研资料的任务"
        if any(action.get("tool") == "code_edit" for action in event.actions):
            return "需要修改代码的任务"
        return "相似任务"

    def _summarize_successful_workflow(self, event: AgentEvent) -> str:
        """把成功轨迹总结为技能草案"""
        steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(event.plan))
        return f"成功工作流：\n{steps}\n适用任务：{event.task}"
```

这个示例故意保持简单，但已经体现了 Self-Evolution Agent 的核心思想：**不是每次都训练模型，而是先把经验转化为低成本、可验证、可回滚的系统资产**。

---

## 如何判断一次“进化”是否值得保留？

自我进化系统最危险的地方在于：它可能把错误经验固化下来。因此，每个进化补丁都应该经过评估。

| 检查项 | 问题 | 不通过时的处理 |
|--------|------|----------------|
| **可复现性** | 这个问题是否多次出现？ | 只作为临时记忆，不写入长期规则 |
| **泛化性** | 经验是否适用于一类任务，而非单个案例？ | 降低触发范围 |
| **安全性** | 是否鼓励绕过权限、隐藏错误或忽略用户意图？ | 直接拒绝 |
| **收益** | 是否显著提升成功率、速度或质量？ | 不部署，只保留观察 |
| **回归风险** | 是否会让其他任务变差？ | 进入 A/B 测试或人工审核 |
| **可回滚性** | 出问题时能否撤销？ | 不允许自动上线 |

一个实用规则是：

> **记忆可以自动写入，Prompt 和 Skill 要半自动审核，模型权重更新必须离线评估后再部署。**

---

## Self-Evolution 与 Agentic 数据飞轮的关系

Self-Evolution Agent 和第 10.9 节的数据飞轮不是两个独立概念，而是同一个闭环在不同层级的表现。

| 视角 | Self-Evolution Agent | Agentic 数据飞轮 |
|------|----------------------|------------------|
| **关注点** | 系统行为如何自我改进 | 模型能力如何通过数据训练增强 |
| **更新对象** | 记忆、Prompt、Skill、流程、评估规则 | 训练数据、奖励模型、策略模型 |
| **迭代速度** | 快，可以按天甚至按任务更新 | 慢，通常按周或月训练发布 |
| **风险控制** | 规则校验、沙箱、回滚 | 离线评估、基准测试、灰度发布 |
| **最佳用途** | 快速吸收经验、修复流程问题 | 提升模型底层能力和泛化能力 |

在成熟团队里，两者通常是串联的：

```text
Self-Evolution Agent
  ↓ 产生高质量经验、失败归因、技能草案
Agentic 数据飞轮
  ↓ 过滤、标注、训练、评估
更强的 Agent 模型
  ↓ 部署回执行系统
产生更高质量轨迹
```

---

## 风险与边界

Self-Evolution Agent 听起来很诱人，但必须避免“失控自改”。生产环境要坚持以下边界：

1. **不能自动放宽安全策略**：任何降低权限、审计、沙箱和隐私保护的改动都必须人工审批。
2. **不能把单次用户偏好当作全局规则**：用户个性化偏好应写入用户级记忆，而不是系统级策略。
3. **不能只学习成功样本**：失败样本同样重要，否则 Agent 会学到脆弱的捷径。
4. **不能没有回归测试**：Prompt、Skill 和模型更新都可能造成隐性退化。
5. **不能让进化逻辑影响当前任务稳定性**：进化应异步执行，用户任务优先。

---

## 实战落地路线

从零构建 Self-Evolution Agent，可以按以下路线推进：

```text
第 1 阶段：记录轨迹
  - 保存任务、计划、工具调用、结果、用户反馈

第 2 阶段：自动评估
  - 建立成功率、成本、工具错误率、用户满意度指标

第 3 阶段：失败归因
  - 把失败分为工具错误、规划错误、信息不足、权限不足、格式错误等类型

第 4 阶段：记忆进化
  - 将高置信度经验写入长期记忆，并按触发条件检索

第 5 阶段：技能进化
  - 将高频成功流程封装成 Skill，并经过回归测试

第 6 阶段：数据飞轮
  - 把高质量轨迹和失败对比样本送入 SFT / DPO / RL 训练
```

---

## 参考文献

1. Shinn et al. [**Reflexion: Language Agents with Verbal Reinforcement Learning**](https://arxiv.org/abs/2303.11366). NeurIPS 2023.
2. Madaan et al. [**Self-Refine: Iterative Refinement with Self-Feedback**](https://arxiv.org/abs/2303.17651). NeurIPS 2023.
3. Gou et al. [**CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**](https://arxiv.org/abs/2305.11738). ICLR 2024.
4. Wang et al. [**Voyager: An Open-Ended Embodied Agent with Large Language Models**](https://arxiv.org/abs/2305.16291). 2023.
5. Hu et al. [**Automated Design of Agentic Systems**](https://arxiv.org/abs/2408.08435). ICLR 2025.
6. Robeyns et al. [**A Self-Improving Coding Agent**](https://arxiv.org/abs/2504.15228). 2025.

## 小结

Self-Evolution Agent 的本质是：**让 Agent 从自身运行轨迹中提取经验，并把经验转化为可复用、可验证、可回滚的能力资产**。

关键要点：

- **自我进化不等于自动改模型权重**：记忆、Prompt、Skill 和流程优化往往更低成本、更安全。
- **进化必须闭环**：执行、记录、评估、归因、改进、验证缺一不可。
- **失败样本非常重要**：它们能暴露边界、触发规则修正，并生成偏好学习数据。
- **风险控制优先**：任何自我修改都要有权限边界、回归测试和回滚机制。
- **与数据飞轮互补**：Self-Evolution 负责快速系统级改进，数据飞轮负责长期模型级增强。

当 Agent 能够稳定完成任务、记住教训、沉淀技能，并把轨迹反哺训练系统时，它就不再只是一个“会调用工具的聊天机器人”，而开始具备持续成长的能力。

---

*下一章：[第11章 LangChain 深入实战](../chapter_langchain/README.md)*
