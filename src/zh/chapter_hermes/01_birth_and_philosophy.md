# 13.1 Hermes Agent 的诞生与"成长型 Agent"理念

> ☤ *"The agent that grows with you."*——Hermes Agent 自描述

---

## 一、从迁移入口看两个独立项目

[第12章](../chapter_openclaw/README.md)从聊天入口分析 OpenClaw。本章换一个观察重点：经验怎样形成记忆或技能候选，以及候选怎样经过检验。

[Hermes Agent 官方仓库](https://github.com/NousResearch/hermes-agent)归属 Nous Research。固定版本的 README 记录了技能创建与更新目标，也提供 OpenClaw 迁移入口；迁移功能不证明两个项目同源或存在继承关系。

以下比较是本章的阅读重点，不是其他项目缺少相同能力的排他性结论。

| 项目 | 核心策略 | 技能来自 |
|------|---------|---------|
| OpenClaw | 本章对照聊天入口与扩展边界 | 不在此断言技能仅由人工编写 |
| Hermes | 本章关注技能候选与记忆 | 区分内置、导入与自动生成 |

这一节我们首先把 Hermes 的**诞生背景、设计哲学、与 OpenClaw 的边界**讲清楚。后 6 节再逐层深入。

---

## 二、Nous Research 与"Hermes"系列

Hermes Agent 出自 **Nous Research**——开发开放模型与 Agent 的研究团队，旗下模型包括 Hermes 2、3、4 系列（多个尺寸、从 7B 到 405B）。

**Hermes** 这个名字有两层含义：

1. **神话学**：Hermes（赫尔墨斯）——希腊神话里的使者之神，连接神界与人界、把消息送到各处——这与 Agent 的角色高度匹配；
2. **品牌延续**：Nous 的开源模型系列本身就叫 Hermes（Hermes 3 LLM 等），所以开源自进化 Agent 也沿用了这个名字。

本次材料核验锚定[发行版 v0.22.4](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.4)及[提交 4d04303](https://github.com/NousResearch/hermes-agent/commit/4d04303818b61bf3aa09278af48356315515965b)，核验日为2026-10-22。这里不保留缺少一手依据的首次发布日期。

---

## 三、"成长型 Agent"的设计哲学

Hermes 的产品定位里反复出现一句话：**"The agent that grows with you"**——一个会和你一起成长的 Agent。它的设计哲学可以拆成 4 个层次：

### 3.1 自我学习（Self-Learning）

Agent 在完成任务后，**自主**把执行轨迹提炼为一个 Skill（`SKILL.md`），下回遇到类似任务时直接调用。

> 技能文件只是候选产物；是否改善未见任务、是否仍有正确权限，必须单独评估。不能据此断言其他框架每次都从头规划或候选必然成为可靠能力。

### 3.2 持续自我迭代（Continuous Self-Improvement）

不只是"创建 Skill"，还要"**更新** Skill"。官方 README 描述技能会在使用中改进，但本次没有核实 DSPy/GEPA、离线优化进程或每次调用后的自动更新算法。这些不能写成已证实的实现；本书将候选评估作为单独的实验设计。

### 3.3 主动反思（Active Reflection）

Skill 自动生成是被动的。Hermes 还内置一个 **Nudge Engine**——定期主动问自己：

> "我刚解决了一个问题。有什么经验值得沉淀为 Skill 吗？"

> "用户偏好我怎样总结？有没有反复出现的偏好可以记到 USER.md 里？"

### 3.4 跨平台持续人格

和 OpenClaw 一样，Hermes 也支持多渠道——同一真人在 WhatsApp / Telegram / Discord / Slack / Signal 上保持同一人格（通过手机号 / OAuth 关联）。

---

## 四、为什么"自进化"这件事重要

在解释工程细节之前，先把"为什么"讲清楚：

| 角度 | 基线记录 | 启用候选后的检查 |
|------|------------------|-----------------|
| **首次使用** | 记录配置和固定任务结果 | 将相同记录作为启用候选技能前的基线 |
| **长期使用** | 保留人工维护记录 | 用未参与生成的任务比较改进、退化与成本 |
| **迁移** | 核对配置与资料的导入范围 | 额外核对技能依赖、权限及失败后的恢复 |

> 长期使用是否带来收益，需要结果证明；不能从运行天数预测能力增长。技能候选与验收设计见[13.4节](./04_self_evolving_skills.md)。



---

## 五、Hermes 与 OpenClaw 的精确边界

很多读者会把"OpenClaw"与"Hermes Agent"混为一谈。下面给一张"易混点澄清表"：

| 维度 | 应保留的比较证据 |
|---|---|
| **许可** | 两者的固定版本均需查对应 LICENSE，不能用许可推断安全性 |
| **Skill 来源** | 分别记录内置、人工导入和自动候选，不能预设一方“仅人工” |
| **记忆与用户建模** | 按版本核对存储、隔离、同意和删除，而不是从产品定位推断有无 |
| **迁移工具** | Hermes 的固定 README 记录迁移入口；导入范围与冲突处理需单独验证 |
| **运行入口** | 根据当前任务选择渠道、CLI 或其他界面；不以安装口号判断使用成本 |



两者是否适合替换当前实现，应依任务、权限与实际结果判断，不能用产品定位直接作结论。

---

## 六、Hermes 在 Harness 谱系中的位置

![行动型 Agent Harness 谱系](../svg/chapter_openclaw_01_harness_spectrum.svg)

本章选 Hermes 研究记忆与技能候选，不作“最深”排名；产品自述也不能证明在未见任务上的改进。

---

## 七、已核验的来源与范围

| 项目 | 本次核验结果 |
|---|---|
| 仓库与材料 | [固定提交 README](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/README.md)，不是可变 `main` 的替代描述 |
| 许可证 | 固定提交的 [MIT LICENSE](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/LICENSE) |
| 文档 | [官方文档入口](https://hermes-agent.nousresearch.com/docs/)，网站内容可能与固定提交不同 |
| Gateway 来源 | 固定提交包含 Python [gateway/run.py](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/gateway/run.py)；不能将其写成 TypeScript Gateway |
| 迁移入口 | 固定 README 记录 `hermes claw migrate` 和 `--dry-run`；本书未实际执行迁移或验证资料完整性 |
| 学习与记忆 | 技能创建、记忆和会话检索为项目自述；具体触发、数据路径与效果需要逐项核验 |

不以 Star、安装量或运行天数作为能力依据；固定版本仅便于复查，不代表本章所有命令与教学代码已得到验证。



---

## 八、本节小结

| 主题 | 关键要点 |
|------|---------|
| 项目定位 | 自进化个人 Agent，技能自动生成并自动迭代 |
| 作者 | Nous Research（Hugo Botev 等） |
| 设计哲学 | 自学习 / 持续自迭代 / 主动反思 / 跨平台人格 |
| 与 OpenClaw 关系 | 两个独立项目；迁移入口不等于共享谱系或完全兼容 |
| 谱系坐标 | 多渠道 · 自进化 · 开源；与 Claude Code / OpenClaw / DeepSeek Harness 并列 |
| 核心差异化 | 技能候选、验证与回退的设计见[13.4节](./04_self_evolving_skills.md) |

---

*下一节：[13.2 安装与快速上手（含 OpenClaw 迁移）](./02_install_and_migration.md)*
