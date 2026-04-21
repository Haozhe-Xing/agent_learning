# 第10章 Agent 技能系统（Skill System）

> 🎯 *"工具让 Agent 能做一件事，技能让 Agent 能做好一类事。技能是工具、提示、流程和经验的有机组合。"*

---


## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解技能系统在 Agent 架构中的地位和作用
- ✅ 掌握技能的定义、封装和生命周期管理
- ✅ 了解技能学习的主要路径（蒸馏、RL、人工设计）
- ✅ 实现一个完整的可复用技能系统
- ✅ 掌握 Tool、Skill、Sub-Agent 三层能力抽象的设计原则

## ⏱️ 预计学习时间

约 **120-150 分钟**

## 💡 前置知识

- 完成第4章（工具调用）和第9章（Harness 工程）
- 了解 YAML / TOML 配置文件格式

---

## 🔗 学习路径

> **后续推荐**：
> - 👉 [第11章 Agentic-RL](../chapter_agentic_rl/README.md)
> - 👉 [第16章 多 Agent 协作](../chapter_multi_agent/README.md)

---

## 本章概览

在前面的章节中，我们学会了如何让 Agent 调用工具。但"会用锤子"和"会做木工"是两回事——**工具是单个动作，技能是解决一类问题的完整能力**。本章将介绍 Agent 技能系统的完整体系：什么是技能、如何定义和封装技能、Agent 如何自主学习新技能、以及如何在多 Agent 系统中发现和共享技能。

## 为什么需要单独一章？

你可能会问：工具调用不就够了吗？

![Tool vs Skill 概念对比](../svg/chapter_skill_readme_tool_vs_skill.svg)

一个技能通常包含：多个工具的组合使用、专业领域的提示知识、特定的处理流程、以及从经验中积累的最佳实践。

## 本章目标

学完本章，你将能够：

- ✅ 理解 Skill 与 Tool 的核心区别和层次关系
- ✅ 掌握三种技能封装方式：Prompt-based、Code-based、Workflow-based
- ✅ 了解 Agent 如何通过经验自主学习新技能（Voyager 范式）
- ✅ 掌握技能发现与注册机制（A2A Agent Card、MCP 等）
- ✅ 实战构建一个可复用的技能系统

## 本章结构

| 小节 | 内容 | 难度 |
|------|------|------|
| 10.1 技能系统概述 | Skill vs Tool、技能的三层架构 | ⭐⭐ |
| 10.2 技能的定义与封装 | 三种封装方式及实战 | ⭐⭐⭐ |
| 10.3 技能学习与获取 | Voyager、CRAFT、自主技能进化 | ⭐⭐⭐ |
| 10.4 技能发现与注册 | A2A Skill 声明、动态发现 | ⭐⭐⭐ |
| 10.5 实战：构建可复用的技能系统 | 完整项目实现 | ⭐⭐⭐⭐ |
| 10.6 论文解读：技能系统前沿研究 | Voyager、CRAFT 等论文解读 | ⭐⭐⭐ |
| 10.7 Tool、Skill 与 Sub Agent：三层能力抽象 | 三层能力模型与协作模式 | ⭐⭐⭐ |
| 10.8 Skills 圣经：Superpowers 工程实践指南 | obra/superpowers 完整工作流 | ⭐⭐⭐⭐ |

## ⏱️ 预计学习时间

约 **90-120 分钟**（含实战练习）

## 💡 前置知识

- 已完成第 4 章（工具调用 / Function Calling）
- 了解 JSON Schema 和 Python 装饰器
- 对 Agent 的基本工作流程有初步认识

## 🔗 学习路径

> **前置知识**：[第4章 工具调用](../chapter_tools/README.md)
>
> **后续推荐**：
> - 👉 [第14章 多 Agent 协作](../chapter_multi_agent/README.md) — 在多 Agent 系统中共享和发现技能
> - 👉 [第15章 通信协议](../chapter_protocol/README.md) — MCP/A2A 中的技能声明机制

---

*下一节：[10.1 技能系统概述](./01_skill_overview.md)*
