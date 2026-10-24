# 5.5 实战：自动化研究助手 Agent

综合本章所学的规划、推理和反思能力，构建一个能够自主进行研究的 Agent。

> **设计说明**：本项目采用"Plan-then-Execute"的多阶段 Pipeline 架构，而非纯 ReAct 循环。这是因为在研究任务中，各阶段（规划→搜索→分析→质量检查）有明确的先后顺序，Pipeline 模式更易于控制流程和调试。在 Pipeline 内部，每个阶段仍然运用了 ReAct 思想——Agent 根据当前阶段的输出"思考"下一步行动，并在质量检查阶段进行"反思"。这体现了第 5.1 节讨论的"将合适的推理框架应用到合适的场景"原则。

> **前沿定位**：本节的研究助手是 **Deep Research Agent** 的入门形态。真正的 Deep Research Agent 不只是"搜索几次然后总结"，而是能围绕开放问题持续提出子问题、跨来源验证证据、管理引用、识别矛盾，并在多轮研究中逐步收敛结论。

## 什么是 Auto Research？

**Auto Research（自动化研究）** 指的是让 Agent 像初级研究员一样，围绕一个开放主题自动完成"提出问题 → 制定计划 → 搜索资料 → 阅读与摘录 → 交叉验证 → 综合成报告 → 自我审查"的完整研究流程。

它和普通搜索最大的区别是：普通搜索关注"找到一个答案"，Auto Research 关注"形成一个可信结论"。因此，Auto Research 不是把搜索引擎结果做摘要，而是把研究过程拆成可追踪、可验证、可迭代的工作流。

| 能力 | 普通搜索 | Auto Research |
|------|----------|---------------|
| **输入** | 一个明确问题 | 一个开放主题或模糊目标 |
| **过程** | 搜索 → 摘要 | 规划 → 多轮检索 → 阅读 → 验证 → 综合 → 审查 |
| **证据** | 常依赖前几条结果 | 维护来源、时间、可信度和引用链 |
| **质量控制** | 用户自行判断 | Agent 主动检查遗漏、矛盾和反方观点 |
| **输出** | 简短答案 | 结构化报告、结论、不确定性与后续问题 |

一个成熟的 Auto Research Agent 通常包含五个核心模块：

1. **研究规划器**：把主题拆解成研究问题、搜索词和报告大纲。
2. **资料收集器**：调用搜索、网页阅读、论文检索、数据库查询等工具。
3. **证据管理器**：把资料转成"证据卡片"，记录来源、摘要、可信度和关联结论。
4. **综合写作者**：按大纲组织材料，生成结构化分析，而不是简单拼接摘要。
5. **质量审查器**：检查覆盖度、引用支撑、矛盾信息、时效性和潜在偏见。

## 代表性前沿工作：Auto Research 的研究脉络

Auto Research 不是凭空出现的产品概念，而是从 **长答案问答、浏览器辅助问答、检索增强写作、Web Agent、自动科学发现** 等方向逐步演化出来的。下面几篇工作分别代表了这一演化链条中的关键能力。

### WebGPT：把"浏览网页"变成可学习的问答行为

- **论文链接**：[WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)
- **当时的核心贡献**：在 2021 年，这篇论文把"搜索网页—阅读网页—引用证据—生成长答案"变成了一个可以通过模仿学习和人类反馈优化的任务，为后来的浏览器 Agent、Deep Research Agent 和带引用问答系统奠定了早期范式。

OpenAI 的 **WebGPT: Browser-assisted question-answering with human feedback** 是早期非常重要的工作。它关注的问题是：当模型面对开放域长答案问题时，能否像人一样使用浏览器查资料、摘录证据，并生成带引用的答案？

WebGPT 的核心不是简单接入搜索 API，而是把浏览过程建模成一系列可学习动作：搜索、打开网页、滚动、引用片段、组合答案。研究者再用人类偏好反馈训练模型，让模型学会"哪些浏览轨迹和答案更可信"。这对 Auto Research 的启发是：**研究能力不仅来自最终生成，还来自可监督、可评估的资料搜集过程**。

从工程角度看，WebGPT 对应本节的三个模块：

- **资料收集器**：搜索和浏览不是一次性调用，而是可追踪的行动序列。
- **证据管理器**：答案中的关键 claim 应能回到具体网页片段。
- **质量审查器**：人类偏好或自动评估可以用来训练"更可信的研究轨迹"。

### STORM：用多视角提问生成高质量长文大纲

- **论文链接**：[Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
- **当时的核心贡献**：在 2024 年，这篇工作把自动研究写作的重点从"直接生成正文"前移到"预写阶段"，提出通过检索和多视角提问来合成主题大纲，让 LLM 更系统地覆盖开放主题的关键维度。

Stanford OVAL 的 **STORM: Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking** 进一步把问题推进到"从零写一篇类 Wikipedia 长文"。它解决的不是单个问答，而是长篇知识整理中的"预写阶段"：写之前如何决定应该覆盖哪些角度、提出哪些问题、检索哪些材料。

STORM 的关键机制是 **多视角提问（multi-perspective question asking）**。系统会模拟不同背景的提问者，从多个角度追问同一主题，再基于检索结果合成文章大纲。这样做的意义在于：开放主题最难的不是生成文字，而是知道"还缺哪些维度"。

这对 Auto Research 的启发非常直接：

- 不要只让 Agent 生成一个搜索词，而要让它生成一组互补视角。
- 报告大纲应该来自"问题空间探索"，而不是模型凭直觉列目录。
- 质量评估要检查覆盖度：是否遗漏历史背景、核心机制、争议观点、应用案例和局限性。

### MindSearch：用 WebPlanner + WebSearcher 构造深度搜索图

- **论文链接**：[MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183)
- **当时的核心贡献**：在 2024 年，这篇工作将复杂 Web 搜索显式建模为"规划器构建问题图 + 搜索器逐点检索"的多 Agent 过程，使搜索从一次性查询升级为可扩展、可回溯的深度信息探索。

**MindSearch: Mimicking Human Minds Elicits Deep AI Searcher** 把深度搜索建模为多智能体协作框架。它的典型架构包含 `WebPlanner` 和 `WebSearcher`：前者负责把复杂问题拆成一张动态扩展的子问题图，后者负责针对每个子问题执行搜索和阅读。

MindSearch 的关键价值在于，它不把研究计划看成一次性列表，而看成会随着搜索结果不断扩展的图结构。每当发现新实体、新关系或新缺口，规划器都可以继续添加节点。这使它更接近人类研究过程：先有粗略框架，再在阅读中不断发现新的分支。

对本节实现来说，MindSearch 提醒我们：

- `research_questions` 不应只是静态数组，可以升级为"研究问题图"。
- 每次搜索结果都应反馈给规划器，决定是否新增子问题。
- 对复杂主题，停止条件不应只是搜索次数，而应是关键节点是否被充分覆盖。

### WebSailor：面向高不确定性 Web 任务的后训练范式

- **论文链接**：[WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592)
- **当时的核心贡献**：在 2025 年，这篇工作面向 BrowseComp 这类高不确定性、多跳、强干扰 Web 任务，提出用复杂任务生成、推理轨迹重构和后训练方法，让开源 Web Agent 学会系统性降低不确定性。

2025 年的 **WebSailor: Navigating Super-human Reasoning for Web Agent** 代表了更前沿的方向：不是只靠 Prompt 让模型会搜索，而是通过专门的数据构造和后训练，让 Web Agent 学会处理高度不确定、路径不明确、需要多跳推理的复杂检索任务。

它关注的任务类似 BrowseComp：答案往往隐藏在多个网页、多个实体和间接线索之间。普通搜索模型容易停在表面结果，而强 Web Agent 需要系统性降低不确定性：先定位线索，再排除干扰，再跨页面验证，最后给出答案。

WebSailor 对 Auto Research 的启发是：

- 高难研究任务需要"主动消除不确定性"，而不是被动摘要搜索结果。
- 训练数据不能只包含答案，还要包含有效的搜索轨迹和推理轨迹。
- 复杂 Web Agent 的能力来自"检索 + 推理 + 验证"的联合训练，而不是单独增强某一个模块。

### The AI Scientist：从研究报告走向自动科学发现

- **论文链接**：[The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)
- **当时的核心贡献**：在 2024 年，这篇工作把研究 Agent 从"写综述/报告"推进到"提出研究想法、写代码实验、分析结果、撰写论文、模拟评审"的端到端自动科学发现流程，展示了开放式科研自动化的雏形。

Sakana AI 等提出的 **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** 把 Auto Research 推向更激进的方向：不只是写研究综述，而是尝试自动提出研究想法、检索相关工作、设计实验、运行代码、分析结果、撰写论文，甚至进行自动评审。

这类系统还远不能替代真正的科学家，但它揭示了 Auto Research 的上限形态：研究 Agent 不再只是"资料整理器"，而可能成为"假设生成器 + 实验执行器 + 论文写作者 + 审稿人"的组合系统。

对工程实践的启发是：如果研究对象涉及可实验验证的问题，就不能只停留在网页检索，还需要加入：

- **实验计划生成**：把研究问题转化为可执行实验。
- **代码执行环境**：运行实验并记录结果。
- **结果解释器**：区分真实发现、偶然波动和实验错误。
- **自动评审器**：从新颖性、有效性、可复现性角度审查结论。

### 论文脉络小结

上面的工作更像是 Auto Research 的“能力地基”：WebGPT 解决浏览与引用，STORM 解决多视角大纲，MindSearch 解决动态搜索图，WebSailor 解决高不确定性 Web 推理，The AI Scientist 解决从研究报告到实验闭环的延伸。真正进入 2025 年之后，前沿焦点进一步从“能不能做研究”转向三个更硬的问题：**能否产品化地完成长程研究任务、能否被高难基准稳定评测、能否用开源模型和开源框架复现闭源 Deep Research 能力**。

| 代表工作 | 研究问题 | 核心机制 | 对 Auto Research 的启发 |
|----------|----------|----------|--------------------------|
| **WebGPT** | 如何让模型使用浏览器回答开放域长问题 | 浏览动作建模 + 引用证据 + 人类反馈 | 研究轨迹要可监督、可引用、可评估 |
| **STORM** | 如何从零生成结构化长篇知识文章 | 多视角提问 + 检索 + 大纲合成 | 先探索问题空间，再写报告 |
| **MindSearch** | 如何像人一样逐步展开复杂搜索 | `WebPlanner` + `WebSearcher` + 子问题图 | 研究计划应动态扩展，而不是静态列表 |
| **WebSailor** | 如何处理高不确定性、多跳 Web 推理任务 | 复杂任务生成 + 轨迹重构 + 后训练 | 训练 Agent 主动降低不确定性 |
| **The AI Scientist** | 能否自动完成开放式科学发现流程 | 想法生成 + 文献检索 + 实验 + 写作 + 评审 | Auto Research 可进一步扩展为自动实验系统 |

### 2025–2026 的前沿进展：从研究原型到 Deep Research 竞赛

如果只介绍上面的论文，确实会显得“不够前沿”。因为 2025 年之后，Auto Research 已经不再只是论文里的研究原型，而是变成了闭源产品、开源框架、专用模型和评测基准共同竞争的方向。

#### OpenAI Deep Research：把长程研究流程产品化

- **官方链接**：[Introducing deep research](https://openai.com/index/introducing-deep-research/)
- **核心贡献**：2025 年，OpenAI 将 Deep Research 作为面向真实用户的 Agent 能力推出，把“研究计划 → 多轮浏览 → 做笔记 → 证据整合 → 生成带引用报告”包装成可交互的产品流程。

它的重要性不在于提出了某个单点算法，而在于把 Auto Research 从“模型能不能联网搜索”推进到“能否在几十分钟内完成一个人类研究员需要数小时的信息综合任务”。这意味着 Deep Research 的评价标准不再只是回答准确率，还包括报告结构、证据覆盖、引用质量、任务坚持度和不确定性表达。

#### BrowseComp：用“难找但易验证”的问题评测浏览型 Agent

- **项目链接**：[OpenAI simple-evals / BrowseComp](https://github.com/openai/simple-evals)
- **核心贡献**：2025 年，OpenAI 发布 BrowseComp，用 1266 个高难网页检索问题评估 Agent 在真实 Web 上寻找隐藏信息的能力，问题通常需要跨多个网页、多跳线索和较长浏览轨迹才能回答。

BrowseComp 的价值在于，它把 Web Agent 的评测从“是否能回答简单事实”升级为“是否能在开放互联网中定位难以直接搜索到的信息”。这对 Auto Research 很关键：一个真正的研究 Agent 不能只会读搜索结果首页，而要会在噪声中追踪线索、排除干扰、验证唯一答案。

#### BrowseComp-ZH：中文互联网让 Deep Research 更难

- **论文链接**：[BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese](https://arxiv.org/abs/2504.19314)
- **核心贡献**：2025 年，BrowseComp-ZH 将高难浏览评测扩展到中文互联网，强调中文语境下的信息碎片化、平台分散、表达省略、搜索入口差异和多跳线索问题。

这项工作提醒我们：Deep Research 不是把英文 Web Agent 平移到中文就可以解决。中文互联网有大量信息分布在百科、新闻、论坛、政务网站、社交平台和视频平台中，关键词常常不直接命中答案。因此，中文 Auto Research Agent 需要更强的查询改写、实体消歧、跨平台检索和来源可信度判断能力。

#### Open Deep Research：开源框架开始复现 Deep Research 工作流

- **项目链接**：[LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- **核心贡献**：LangChain 的 Open Deep Research 把 Deep Research 的工程模式开源化，用 `LangGraph` 等组件实现研究计划、搜索、内容读取、报告生成和人工反馈调整，让开发者可以复现和定制研究 Agent 流程。

它的意义在于推动 Deep Research 从闭源能力走向可组合工程框架。对开发者来说，重点不只是“调用哪个模型”，而是如何设计状态机、如何管理中间笔记、如何控制并发搜索、如何把用户反馈反馈到报告计划中。

#### Tongyi DeepResearch：开源专用 Deep Research 模型与系统

- **项目链接**：[Alibaba-NLP DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- **模型链接**：[Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- **核心贡献**：2025 年，通义 DeepResearch 将 Deep Research 从“通用模型 + Agent 框架”推进到“面向长周期深度信息搜索任务训练的专用开源模型与系统”，并围绕数据生成、持续训练、强化学习、推理和评估形成完整工程栈。

这代表了一个重要趋势：未来的 Deep Research 可能不会完全依赖通用对话模型加 Prompt，而会出现专门为搜索轨迹、证据整合、长程推理和报告生成优化的 Agent 模型。

#### DeepResearch Bench / BrowseComp-Plus：评测从单题答案走向端到端报告质量

- **项目示例**：[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)
- **核心贡献**：2026 年前后，评测开始从“给出一个可验证答案”扩展到“评估完整研究报告”。新的评测更关注全面性、洞察力、指令遵循、可读性、引用质量和端到端研究流程表现。

这说明 Auto Research 的前沿评测正在分化成两类：一类像 BrowseComp，考察 Agent 是否能找到隐藏事实；另一类像 DeepResearch Bench，考察 Agent 是否能产出高质量研究报告。前者更像“搜索推理能力测试”，后者更像“研究员工作质量测试”。

### 前沿脉络小结

| 阶段 | 代表进展 | 关注问题 | 前沿意义 |
|------|----------|----------|----------|
| **早期浏览问答** | **WebGPT** | 模型如何浏览网页并引用来源 | 让搜索轨迹和证据引用变成可学习对象 |
| **检索增强写作** | **STORM** | 如何为开放主题生成高覆盖大纲 | 把研究写作前移到问题空间探索 |
| **深度搜索图** | **MindSearch** | 如何动态拆解和扩展复杂搜索 | 让研究计划从静态列表变成问题图 |
| **高不确定性 Web 推理** | **WebSailor** | 如何训练 Agent 解决极难多跳检索 | 后训练开始面向搜索轨迹和不确定性消除 |
| **产品化 Deep Research** | **OpenAI Deep Research** | 如何让用户真正完成长程研究任务 | Deep Research 成为通用知识工作入口 |
| **高难浏览评测** | **BrowseComp / BrowseComp-ZH** | 如何评测真实 Web 上的深度检索能力 | 逼迫 Agent 处理多跳、噪声和跨语言 Web 环境 |
| **开源复现与专用模型** | **Open Deep Research / Tongyi DeepResearch** | 如何复现闭源 Deep Research 能力 | 从 Prompt 工程走向框架化、模型化和训练化 |
| **端到端研究评测** | **DeepResearch Bench / BrowseComp-Plus** | 如何评估完整报告质量 | 评测从答案准确率走向研究质量 |

因此，一个更前沿的 Auto Research Agent 不应只停留在“搜索 + 总结”。它需要综合这些进展：像 WebGPT 一样保留证据，像 STORM 一样多视角提问，像 MindSearch 一样维护问题图，像 WebSailor 一样主动降低不确定性，像 OpenAI Deep Research 一样产品化长程流程，像 BrowseComp/BrowseComp-ZH 一样接受高难检索评测，并进一步吸收 Open Deep Research 和 Tongyi DeepResearch 的工程化、开源化和训练化经验。

## Auto Research 的典型工作流

```text
用户主题
  ↓
研究问题拆解
  ↓
生成搜索计划与资料来源列表
  ↓
多轮搜索与阅读
  ↓
提取证据卡片
  ↓
交叉验证与冲突检测
  ↓
生成报告初稿
  ↓
自我审查：是否遗漏关键维度？引用是否支持结论？
  ↓
补充检索或输出最终报告
```

在工程上，最重要的是让 Agent 始终知道自己"为什么要搜这一次"。一次好的搜索不应该只是关键词命中，而应该对应一个明确的研究缺口：

- **概念缺口**：还没有定义清楚核心概念。
- **事实缺口**：缺少关键数据、时间线或案例。
- **证据缺口**：已有结论没有可靠来源支持。
- **反例缺口**：只看到了支持观点，没有看到反方观点。
- **时效缺口**：资料可能过旧，需要最新信息验证。

## Auto Research 的工程挑战

Auto Research 看起来像"多调用几次搜索工具"，但真正难点在于长程任务控制。

| 挑战 | 常见问题 | 工程解法 |
|------|----------|----------|
| **搜索漂移** | 搜着搜着偏离原主题 | 每轮搜索都绑定研究问题和预期证据 |
| **证据污染** | 摘录了低质量、重复或过期资料 | 为来源打可信度、时效性和独立性标签 |
| **引用幻觉** | 报告里的引用并不支持结论 | 生成前做 claim-to-source 检查 |
| **过早总结** | 信息不足就输出结论 | 设置覆盖度阈值和强制反向搜索 |
| **上下文膨胀** | 搜索结果太多导致窗口爆炸 | 使用证据卡片、分层摘要和检索式记忆 |
| **无限研究** | Agent 不断搜索，迟迟不输出 | 设置预算：最大轮数、最大来源数、最大成本 |

> **实践建议**：Auto Research 的目标不是"穷尽所有资料"，而是在给定时间和成本预算内，最大化结论的可信度、覆盖度和可追溯性。

## 从 Search Agent 到 Deep Research Agent

传统搜索 Agent 的目标是"找到答案"；Deep Research Agent 的目标是"形成可信结论"。二者的差异不在于是否联网，而在于是否具备**长程研究流程**。

| 能力维度 | 搜索 Agent | Deep Research Agent |
|---------|------------|---------------------|
| **任务目标** | 回答一个具体问题 | 研究一个开放主题并形成报告 |
| **规划方式** | 一次性生成搜索词 | 动态拆解研究问题，持续补充子问题 |
| **信息处理** | 摘要前几条结果 | 多来源交叉验证、去重、冲突检测 |
| **上下文管理** | 保存搜索结果 | 管理研究笔记、证据卡片、引用链 |
| **质量控制** | 简单检查完整性 | 检查覆盖度、可信度、时效性、反方观点 |
| **输出形式** | 简短回答 | 带引用、结构化论证和不确定性说明的报告 |

可以把 Deep Research Agent 理解为由多个子能力组成的研究流水线：

![Deep Research Agent 研究流水线](../svg/chapter_planning_05_research_pipeline.svg)

本节代码为了教学简洁，只实现其中的核心骨架：规划、搜索、综合、质量检查。你可以在此基础上逐步扩展成完整的 Deep Research Agent。

## 研究助手功能设计

![研究助手 Agent 功能设计](../svg/chapter_planning_05_research_arch.svg)

## 完整实现

```python
import json
import datetime
from openai import OpenAI
import requests

client = OpenAI()

class ResearchAssistant:
    """自动化研究助手"""
    
    def __init__(self):
        self.research_notes = []
        self.sources = []
    
    def _search(self, query: str) -> str:
        """搜索工具（使用 DuckDuckGo）"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1}
            response = requests.get(url, params=params, timeout=8)
            data = response.json()
            
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
                if data.get("AbstractURL"):
                    self.sources.append(data["AbstractURL"])
            
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"][:300])
            
            return "\n".join(results) if results else "未找到相关结果"
        except Exception as e:
            return f"搜索失败：{e}"
    
    def _take_notes(self, content: str, source: str = ""):
        """记录研究笔记"""
        self.research_notes.append({
            "content": content,
            "source": source,
            "time": datetime.datetime.now().isoformat()
        })
    
    def research(self, topic: str, depth: str = "standard") -> str:
        """
        执行研究
        
        Args:
            topic: 研究主题
            depth: "quick"=快速概览, "standard"=标准研究, "deep"=深度研究
        """
        
        depth_config = {
            "quick": {"max_searches": 2, "sections": 3},
            "standard": {"max_searches": 4, "sections": 5},
            "deep": {"max_searches": 8, "sections": 7}
        }
        config = depth_config.get(depth, depth_config["standard"])
        
        print(f"\n🔬 开始研究：{topic}")
        print(f"研究深度：{depth}\n")
        
        # ===== 阶段1：规划研究 =====
        print("📋 阶段1：制定研究计划...")
        plan_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""你是一位研究分析师。为以下主题制定研究计划：

主题：{topic}
研究目标：全面理解该主题，生成{config['sections']}个核心章节的报告

请生成JSON格式的研究计划：
{{
  "research_questions": ["核心问题1", "核心问题2", ...],
  "search_queries": ["搜索词1", "搜索词2", ...（最多{config['max_searches']}个）],
  "report_outline": ["章节1标题", "章节2标题", ...]
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(plan_response.choices[0].message.content)
        search_queries = plan.get("search_queries", [topic])[:config["max_searches"]]
        report_outline = plan.get("report_outline", [f"{topic}概述"])
        
        print(f"  搜索计划：{len(search_queries)} 个查询")
        print(f"  报告结构：{len(report_outline)} 个章节")
        
        # ===== 阶段2：搜索信息 =====
        print("\n🔍 阶段2：搜索信息...")
        all_findings = []
        
        for i, query in enumerate(search_queries, 1):
            print(f"  搜索 [{i}/{len(search_queries)}]：{query}")
            result = self._search(query)
            
            self._take_notes(result, source=f"搜索：{query}")
            all_findings.append(f"【查询：{query}】\n{result}")
        
        findings_text = "\n\n".join(all_findings)
        
        # ===== 阶段3：分析和综合 =====
        print("\n🧠 阶段3：分析综合...")
        
        analysis_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""基于以下研究资料，对主题"{topic}"进行深度分析。

研究资料：
{findings_text[:4000]}

报告大纲：{report_outline}

请按大纲生成完整的研究报告，要求：
1. 每个章节有实质性内容（200-400字）
2. 包含具体的数据、案例或观点
3. 在报告末尾给出结论和建议
4. 使用Markdown格式"""
                }
            ]
        )
        
        report = analysis_response.choices[0].message.content
        
        # ===== 阶段4：质量检查 =====
        print("\n✅ 阶段4：质量检查...")
        
        review_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""简要评估以下研究报告的质量（JSON格式）：

主题：{topic}
报告（前1000字）：{report[:1000]}

评估：
{{
  "completeness_score": 1-10,
  "accuracy_indicators": "高/中/低",
  "missing_aspects": ["遗漏点1"],
  "overall_quality": "优秀/良好/一般"
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        review = json.loads(review_response.choices[0].message.content)
        
        # 生成最终报告
        final_report = f"""# 研究报告：{topic}

> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
> 研究深度：{depth}
> 质量评分：{review.get('completeness_score', 'N/A')}/10
> 信息来源：{len(self.research_notes)} 条

---

{report}

---

## 研究说明

- 本报告基于 {len(search_queries)} 次网络搜索
- 信息截止日期：{datetime.datetime.now().strftime('%Y-%m-%d')}
- 建议结合最新资料进行验证
"""
        
        print(f"\n📄 报告生成完成！")
        print(f"质量：{review.get('overall_quality', 'N/A')} | "
              f"完整性：{review.get('completeness_score', 'N/A')}/10")
        
        return final_report


# 使用示例
assistant = ResearchAssistant()

report = assistant.research(
    topic="大语言模型在软件开发中的应用",
    depth="standard"
)

# 保存报告
filename = f"research_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n📁 报告已保存到：{filename}")
```

## 关键代码解读

这个研究助手虽然代码不长，但已经具备 Deep Research Agent 的雏形：

- **规划阶段**：先生成 `research_questions` 和 `search_queries`，避免边搜边迷路。
- **信息收集阶段**：每次搜索都写入 `research_notes`，形成可追溯的研究轨迹。
- **综合阶段**：不是简单拼接搜索结果，而是按 `report_outline` 重组信息。
- **质量检查阶段**：引入第二次模型调用评估覆盖度，模拟研究员的自审流程。

如果要把它升级为生产级 Deep Research Agent，建议补上四个模块：

| 模块 | 作用 | 实现要点 |
|------|------|----------|
| **证据卡片** | 保存事实、来源 URL、发布时间、可信度 | 每条结论都能追溯到来源 |
| **反向搜索** | 主动寻找反例和不同观点 | 避免只采纳支持性证据 |
| **引用检查** | 验证报告中的引用是否真实支持结论 | 防止"引用幻觉" |
| **研究状态机** | 控制研究阶段切换 | 防止无限搜索或过早总结 |

Deep Research Agent 的关键不是"多搜"，而是**让每一次搜索都服务于一个明确的研究缺口**。这也是长程规划能力在真实 Agent 应用中的典型落点。

## 运行研究助手

```bash
pip install openai python-dotenv requests rich
python research_agent.py
```

示例输出：
```markdown
# 研究报告：大语言模型在软件开发中的应用

> 生成时间：2024-03-15 14:30
> 研究深度：standard
> 质量评分：8/10

## 1. 概述
...

## 2. 代码生成与补全
...

## 3. 代码审查与 Bug 检测
...
```

## 参考文献

1. Nakano et al. [**WebGPT: Browser-assisted question-answering with human feedback**](https://arxiv.org/abs/2112.09332). OpenAI, 2021.
2. Shao et al. [**Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models**](https://arxiv.org/abs/2402.14207). Stanford OVAL, 2024.
3. Chen et al. [**MindSearch: Mimicking Human Minds Elicits Deep AI Searcher**](https://arxiv.org/abs/2407.20183). 2024.
4. Li et al. [**WebSailor: Navigating Super-human Reasoning for Web Agent**](https://arxiv.org/abs/2507.02592). Alibaba Tongyi Lab, 2025.
5. Lu et al. [**The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**](https://arxiv.org/abs/2408.06292). Sakana AI, 2024.
6. OpenAI. [**Introducing Deep Research**](https://openai.com/index/introducing-deep-research/). 2025.
7. OpenAI. [**BrowseComp in simple-evals**](https://github.com/openai/simple-evals). 2025.
8. Li et al. [**BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese**](https://arxiv.org/abs/2504.19314). 2025.
9. LangChain AI. [**Open Deep Research**](https://github.com/langchain-ai/open_deep_research). 2025.
10. Alibaba-NLP. [**Tongyi DeepResearch**](https://github.com/Alibaba-NLP/DeepResearch). 2025.
11. Alibaba-NLP. [**Tongyi-DeepResearch-30B-A3B**](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B). 2025.
12. Texttron. [**BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent**](https://github.com/texttron/BrowseComp-Plus). 2026.

## 小结

本节完成了一个自动化研究助手，并展示了 Deep Research Agent 的基础架构：

- ✅ 研究计划生成：把开放主题拆成问题和搜索词
- ✅ 多轮信息收集：记录搜索结果和来源
- ✅ 分析综合：按大纲生成结构化报告
- ✅ 质量检查：对完整性和遗漏点进行自审
- ✅ 可扩展方向：证据卡片、反向搜索、引用检查、研究状态机

真正的 Deep Research Agent 是"长程规划 + Web/文档工具 + 证据治理 + 质量评估"的组合体。它会在后续的 Web Agent、上下文工程、评估与安全章节中继续展开。

---

*下一节：[5.6 Plan-and-Execute 与 Test-time Compute Scaling](./07_plan_and_execute.md)*