# 第13章 Hermes Agent：自我进化的 Agent

**阅读层次**：产品与架构案例选读；需要跨会话记忆或技能积累时阅读。

本章用 Hermes Agent 讨论经验怎样形成记忆或技能候选，以及候选怎样经过验证、审查和回退。写出一份技能文件只说明产物形成；它是否改善未见任务，必须由独立结果判断。

## 本章迭代

- **上一版的不足**：已有记忆和 Skill 接口能保存内容，但还没有定义经验何时值得保留、如何验证改动及如何撤回。
- **本章增加**：分析跨会话记忆、技能积累与调度的边界，把生成候选、评估和启用分开。
- **怎样验证**：本书没有统一 Hermes 适配器。读者实验应固定版本、任务与权限，在未用于编写技能的任务上比较启用前后，并验证用户隔离、删除、停用和回退；未运行时只提交分析记录。
- **下一步**：有代码操作需求时选读[第14章 Claude Code](../chapter_claude_code/README.md)；要独立衡量记忆或技能改动时进入[第18章 评估](../chapter_20_evaluation/README.md)。

## 本章结构

| 小节 | 本节关注的问题 |
|---|---|
| [13.1 Hermes Agent 的诞生与"成长型 Agent"理念](./01_birth_and_philosophy.md) | 把成长目标转化为可检验问题 |
| [13.2 安装与快速上手（含 OpenClaw 迁移）](./02_install_and_migration.md) | 记录环境与迁移范围，保留原数据 |
| [13.3 三层架构：Gateway / Engine / Plugin 子系统](./03_three_layer_architecture.md) | 区分入口、执行与扩展边界 |
| [13.4 核心：Self-Evolving Skills 闭环](./04_self_evolving_skills.md) | 技能候选、验证、启用与回退 |
| [13.5 三层记忆系统](./05_memory.md) | 隔离、召回、删除与跨会话证据 |
| [13.6 Nudge Engine 与跨会话学习](./06_nudge_engine.md) | 主动任务的触发、授权与停止 |
| [13.7 借鉴哲学：从 Hermes 学到的"自进化"工程原则](./07_lessons_philosophy.md) | 提炼可迁移设计并标明验证边界 |

## 阅读路径

相关基础是[第4章 记忆](../chapter_memory/README.md)、[第8章 Harness](../chapter_harness/README.md)和[第9章 Skill](../chapter_skill/README.md)。[第12章 OpenClaw](../chapter_openclaw/README.md)可用于对照入口设计，不是必修前置，也不表示两者存在直接迁移或组件兼容关系。

## 验证与材料边界

本章的“自我进化”是一组待检验的设计问题，不保证每次任务都会产生有效技能，也不把产品自述当作质量证据。具体后端、迁移命令和存储实现需要在明确版本中核验，不能仅据本章概览断言可用。

本书的记忆和技能离线示例验证自己的模块契约，不能代替 Hermes 产品实验。应用开发无需先学习研究篇；只有需要研究学习算法时，再按目标选择[第七部分](../part7.md)。

---

*下一节：[13.1 Hermes Agent 的诞生与"成长型 Agent"理念](./01_birth_and_philosophy.md)*
