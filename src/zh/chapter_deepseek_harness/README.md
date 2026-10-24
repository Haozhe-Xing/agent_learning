# 第15章 DeepSeek Harness：一切皆插件的开源底座

**阅读层次**：架构案例选读；需要研究组件替换或插件管理时阅读。

本章把 DeepSeek Harness 作为组件边界与插件生命周期的架构案例，围绕模型适配、工具、状态和扩展提出可验证问题。目标是判断哪些接口值得借鉴，以及替换一个组件需要保留哪些契约。

## 本章迭代

- **上一版的不足**：已有能力可以运行，但模型、工具或状态实现发生变化时，替换范围和清理责任尚未明确。
- **本章增加**：分析插件接口、服务与事件边界，并把加载、调用、失败和卸载纳入生命周期。
- **怎样验证**：本书没有统一 DeepSeek Harness 适配器。读者应固定源码或发行版本、配置与一个替换目标，检查相同输入输出、错误状态、权限和清理结果；未经运行的材料保留为架构分析。
- **下一步**：用[第15.7节 选择记录](./07_summary_and_decision_matrix.md)整理证据；只有存在可独立分工的任务时，再读[第16章 多 Agent 协作](../chapter_18_multi_agent/README.md)。

## 本章结构

| 小节 | 本节关注的问题 |
|---|---|
| [15.1 DeepSeek Harness 是什么：Cordis 内核与"一切皆插件"](./01_what_is_dsh.md) | 把插件化目标转化为组件契约 |
| [15.2 安装与四种运行模式](./02_install_and_modes.md) | 核验选定版本的入口与运行模式 |
| [15.3 架构：Cordis 微内核与插件拓扑](./03_cordis_architecture.md) | 梳理服务、事件与组件依赖 |
| [15.4 插件开发：tool / llm / skill / subagent 的插件接口](./04_plugin_development.md) | 为一种扩展定义输入输出和失败清理 |
| [15.5 横向对比：DeepSeek Harness vs Claude Code / OpenClaw / Hermes](./05_comparison.md) | 按任务与权限比较候选，记录未知项 |
| [15.6 借鉴：可换内核 / 模型无关的工程哲学](./06_lessons_philosophy.md) | 区分可迁移设计与产品专有实现 |
| [15.7 总结：按任务与验证证据选择实现](./07_summary_and_decision_matrix.md) | 用固定任务、失败用例与停止条件作选择 |

## 阅读路径

建议先理解[第8章 Harness](../chapter_harness/README.md)与[第9章 Skill](../chapter_skill/README.md)。[第14章 Claude Code](../chapter_claude_code/README.md)可作为代码操作案例对照，不是必修前置。本章也不要求先学习强化学习或其他产品的安装流程。

## 验证与材料边界

本章概览不承诺某个发行版具备全部接口或运行模式，也不依据发布口号推荐生产平台。读者实验需要记录确切版本和证据；章节中的安装命令、组件实现与功能描述都应在该版本中核验。

插件接口相似不证明产品可以互换内核或直接作为彼此的界面。与 OpenClaw、Hermes 或 Claude Code 的比较应保持任务和权限一致，无法运行或保持同等条件的项目分别标记“未运行”或“不可直接比较”。

---

*下一节：[15.1 DeepSeek Harness 是什么：Cordis 内核与"一切皆插件"](./01_what_is_dsh.md)*
