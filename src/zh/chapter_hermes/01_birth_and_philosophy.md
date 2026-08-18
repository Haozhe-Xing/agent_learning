# 15.1 Hermes Agent 的诞生与"成长型 Agent"理念

> ☤ *"The agent that grows with you."*——Hermes Agent 自描述

---

## 一、从 OpenClaw 的同源兄弟说起

上一章我们看到了 OpenClaw——它把 Agent 推到了每个聊天 App 里，但**所有能力都需要人来维护**。Skill 文件要人写、人发版、人升级。

Hermes Agent 走了另一条路：**让 Agent 自己写 Skill**。它由 Nous Research 在 2025 年中后期发布，作者把它定位为"OpenClaw 的迭代继承者 + 自主进化分支"——直到今天，Hermes 都还为那些从 OpenClaw 迁过来的用户提供一键迁移命令。

但 Hermes 与 OpenClaw **不只是改名关系**——它们的**核心差异化在一个字：学**。

| 项目 | 核心策略 | 技能来自 |
|------|---------|---------|
| OpenClaw | "用户/社区手写技能" | **人** |
| Hermes | "Agent 自创技能 + 自迭代" | **人 + Agent** |
| OpenCode/Codex | "模型能力 + 内置工具" | **厂商** |

这一节我们首先把 Hermes 的**诞生背景、设计哲学、与 OpenClaw 的边界**讲清楚。后 6 节再逐层深入。

---

## 二、Nous Research 与"Hermes"系列

Hermes Agent 出自 **Nous Research**——一个以"开源模型 + 开源 Agent"为核心的非营利研究组织，旗下模型包括 Hermes 2、3、4 系列（多个尺寸、从 7B 到 405B）。

**Hermes** 这个名字有两层含义：

1. **神话学**：Hermes（赫尔墨斯）——希腊神话里的使者之神，连接神界与人界、把消息送到各处——这与 Agent 的角色高度匹配；
2. **品牌延续**：Nous 的开源模型系列本身就叫 Hermes（Hermes 3 LLM 等），所以开源自进化 Agent 也沿用了这个名字。

**Hermes Agent**（本书语境特指）发布于 2025 年 7 月，本节整理的事实以官方仓库 `NousResearch/hermes-agent` `main` 分支为准，时间锚 2026-08。

---

## 三、"成长型 Agent"的设计哲学

Hermes 的产品定位里反复出现一句话：**"The agent that grows with you"**——一个会和你一起成长的 Agent。它的设计哲学可以拆成 4 个层次：

### 3.1 自我学习（Self-Learning）

Agent 在完成任务后，**自主**把执行轨迹提炼为一个 Skill（`SKILL.md`），下回遇到类似任务时直接调用。

> 这和 LangChain / AutoGPT 那种"每次从头规划"模式截然相反——一旦某个流程被固化，它就成为 Agent 的永久能力。

### 3.2 持续自我迭代（Continuous Self-Improvement）

不只是"创建 Skill"，还要"**更新** Skill"。Hermes 在每次使用 Skill 时记录使用结果，基于 DSPy/GEPA 的思路**离线**重新调优 Skill——成功的步骤强化，失败的步骤弱化或替换。

### 3.3 主动反思（Active Reflection）

Skill 自动生成是被动的。Hermes 还内置一个 **Nudge Engine**——定期主动问自己：

> "我刚解决了一个问题。有什么经验值得沉淀为 Skill 吗？"

> "用户偏好我怎样总结？有没有反复出现的偏好可以记到 USER.md 里？"

### 3.4 跨平台持续人格

和 OpenClaw 一样，Hermes 也支持多渠道——同一真人在 WhatsApp / Telegram / Discord / Slack / Signal 上保持同一人格（通过手机号 / OAuth 关联）。

---

## 四、为什么"自进化"这件事重要

在解释工程细节之前，先把"为什么"讲清楚：

| 角度 | 没有自进化的 Agent | 有自进化的 Agent |
|------|------------------|-----------------|
| **首次使用** | 需要用户配置 / 装一堆 Skill | 默认能力 + 自创 Skill |
| **30 天后** | 和 30 天前几乎一样（除了记忆数据） | 累积了十几个针对该用户的 Skill |
| **90 天后** | 用户仍然要"教"它新事 | 已经能主动预测用户偏好 |
| **1 年后** | 与新装一个 Agent 没差别 | 每个用户都养出了"自己的版本" |
| **迁移成本** | 配置可以迁移，但能力不行 | 能力也迁移（Skill 文件可移植） |

> 📌 **核心观察**：自进化让"长期使用"出现复利效应——这是普通 Agent 不会有的属性。15.4 我们会看到这件事的工程实现。

---

## 五、Hermes 与 OpenClaw 的精确边界

很多读者会把"OpenClaw"与"Hermes Agent"混为一谈。下面给一张"易混点澄清表"：

| 维度 | OpenClaw（前一章） | Hermes Agent（本章） |
|------|---------------------|---------------------|
| **许可** | MIT | MIT |
| **作者** | Peter Steinberger 社区 | Nous Research |
| **主线焦点** | "多渠道 + Skill 生态" | "自进化技能 + 跨会话记忆" |
| **Skill 来源** | 仅人工 | **人工 + 自动生成 + 自动迭代** |
| **记忆** | SQLite 长期记忆 | **三层记忆**（长期语义 / 工作记忆 / 情景日志） |
| **用户建模** | 无显式机制 | **Honcho dialectic 用户建模** |
| **迁移工具** | — | **`hermes claw migrate` 一键平迁 OpenClaw** |
| **Node.js 后端** | TypeScript | TypeScript |
| **主模型** | 任意 | 任意（推荐 Nous Portal） |
| **门槛** | 极低（npx） | 极低（curl \| bash） |

两者不是替代关系——OpenClaw 是"现在能跑"，Hermes 是"还能自我进化"。

---

## 六、Hermes 在 Harness 谱系中的位置

![行动型 Agent Harness 谱系](../svg/chapter_openclaw_01_harness_spectrum.svg)

Hermes 在"自进化"路线上是**当前开源生态做得最深的一个**——其他项目（OpenHuman 等）的"技能自创"想法都还在实验阶段，Hermes 已经在 v1.x 上跑了近一年。

---

## 七、官方事实清单（以仓库 `main` 分支为准）

下面给一张"可核验事实"清单——任何会变化的数字（如 Star 数、PR 数）我都标了**不引用**，只列可独立核验的事实：

| 项目 | 事实 |
|------|------|
| 仓库 | `github.com/NousResearch/hermes-agent` |
| 许可证 | MIT（以仓库 `LICENSE` 文件为准） |
| 主语言 | Python（核心 agents 子系统） + TS（gateway） |
| 安装 | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` |
| 文档站 | `hermes-agent.nousresearch.com/docs/` |
| 技能市场协议 | 兼容 `agentskills.io` 开放标准 |
| 消息平台 | 15+（Telegram / Discord / Slack / WhatsApp / Signal / CLI / TUI / …） |
| 模型后端 | 多个主流后端（Nous Portal / OpenRouter / OpenAI / Anthropic / 本地 Ollama / 自定义，数量以仓库为准） |
| 执行后端 | 6 种（local / Docker / SSH / Singularity / Modal / Daytona） |
| OpenClaw 迁移 | `hermes claw migrate` 一键（skills / memories / API keys / persona） |
| 自进化机制 | DSPy 风格 + GEPA 风格的离线迭代 |
| 三层记忆 | `~/.hermes/MEMORY.md`（长期）/ `USER.md`（用户建模）/ Session 日志（情景） |
| 全文检索 | SQLite FTS5 + LLM 摘要 |

> 所有"会随时间变化"的具体数字（Star 数 / PR 数 / 安装数 / 用户数）本书**不引用**——这是 working memory 中明确约定的"内容可信度约定"。

---

## 八、本节小结

| 主题 | 关键要点 |
|------|---------|
| 项目定位 | 自进化个人 Agent，技能自动生成并自动迭代 |
| 作者 | Nous Research（Hugo Botev 等） |
| 设计哲学 | 自学习 / 持续自迭代 / 主动反思 / 跨平台人格 |
| 与 OpenClaw 关系 | 同源继承 + 自进化分支；`hermes claw migrate` 一键迁移 |
| 谱系坐标 | 多渠道 · 自进化 · 开源；与 Claude Code / OpenClaw / DeepSeek Harness 并列 |
| 核心差异化 | "Skill 自动生成 + 自动迭代"——这件事的工程实现见 15.4 |

---

*下一节：[15.2 安装与快速上手（含 OpenClaw 迁移）](./02_install_and_migration.md)*
