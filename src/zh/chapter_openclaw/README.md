# 第14章 OpenClaw：跨平台个人 AI 助理

> 🦞 *"你的龙虾管家，已经住在你每一个聊天 App 里。"*
> ——改编自 OpenClaw 社区的一句戏言

---

## 本章导读

你可能已经在用 Claude Code 或 Hermes 这种"行动型 Agent"了——但它们的体验大多停留在 **CLI / TUI**。如果你想让 Agent 直接住在你**每天都在用的聊天工具**里（WhatsApp、Telegram、Discord、Slack、Signal），那么 **OpenClaw** 就是当下生态最完整的一种实现。

OpenClaw 是 Peter Steinberger 在 2025 年 12 月以一个周末项目的形式开始的**开源个人 AI 助理**，最初叫 `Clawdbot`、后改名 `Moltbot`、再于 2026 年 1 月定型为 `OpenClaw`（避开商标 + 强调开源）。它的发展速度可以用"现象级"形容：一个周末项目在短短几个月内就成为 GitHub 上增长最快的开源项目之一（具体规模以仓库主页为准，动态数字不做引用）。有意思的是，正是 OpenClaw 这种开源 Harness 的快速普及，让"个人 Agent"从一个极客玩具，变成了任何人都可以在自己机器上启用的日常工具——后续我们还会看到 Hermes Agent（第 15 章）甚至专门提供了 `hermes claw migrate` 命令来平迁 OpenClaw 的数据。

这一章我们要做的事情有三件：

1. **拆解 OpenClaw 的工程骨架**——弄清楚"一个聊天 App 里的 Agent"到底由哪些模块组成、消息流转如何被网关和 Agent 循环消费；
2. **演示 4 种部署方式**——`npx` 一键、安装脚本、源码构建、Docker 沙箱，让你不管什么场景都能跑起来；
3. **与前几章的范式对齐**——把 OpenClaw 的 `Skills` 系统、`Toolbox` 系统、`Memory` 系统映射回本书第 8 章 Harness 的"六大工程支柱"，让你看到真实生产项目如何落地第 8 章讲过的抽象。

通过本章你将获得：**实战层面**——你能在 30 分钟内让 OpenClaw 在你的 WhatsApp / Telegram 上线，对它说话、让它操作你的文件系统；**架构层面**——你将理解"消息中枢 + Agent Loop + 技能市场"这种范式，与 Claude Code（第 16 章）的 CLI 路线、DeepSeek Harness（第 17 章）的"内核 + 插件"路线的根本差异，从而真正掌握"行动型 Agent"的多样生态。

---

## 本章内容概览

| 小节 | 内容 | 你将学到 |
|------|------|---------|
| 14.1 OpenClaw 全景：从 Clawdbot 到 OpenClaw 的演变 | 项目起源、命名史、核心定位、与 Claude Code / Hermes / DeepSeek Harness 的边界 | 建立 OpenClaw 在 Harness 谱系中的坐标 |
| 14.2 安装与四种部署方式 | `npx @openclaw/cli` 一键、官方安装脚本、git 源码、Docker 沙箱化运行 | 在 macOS / Linux / Windows 上线 |
| 14.3 架构深度解析：Gateway / Agent Loop / Skills | 四层架构、消息流转、插件系统、状态管理 | 读懂 OpenClaw 的工程骨架 |
| 14.4 多渠道路由：WhatsApp / Telegram / Discord / Slack / Signal | 渠道适配器、消息解析、权限分级、跨渠道人格一致 | 让 Agent 接管所有聊天入口 |
| 14.5 Skills 与插件生态：ClawHub 与社区贡献 | Skill 文件结构、ClawHub 技能市场、社区改写分支（Rust/Go/Zig 版） | 找到、定制、发布 OpenClaw 技能 |
| 14.6 实战：基于 OpenClaw 打造个人助理 | 完整搭建一个"能读邮件、能整理飞书文档、能跑命令"的本地 Agent | 端到端走通 OpenClaw 的工程实践 |
| 14.7 借鉴点：从消费品 Agent 学到的工程经验 | "用户键就在聊天框里"这件事如何重塑了 Agent 设计取舍 | 把 OpenClaw 的设计哲学搬回自己的系统 |

---

## 阅读建议

本章适合以下读者：

- ✅ **想真正"用上" Agent 的人**：希望 Agent 不只是 CLI，而是在日常聊天软件里就能召唤——从 14.1 顺序读起
- ✅ **跨平台应用开发者**：想研究"消息中枢"如何把多渠道统一成一个 Agent——重点阅读 14.3、14.4
- ✅ **想了解开源 Harness 的工程师**：对比 Claude Code / Hermes / DeepSeek Harness 三种路线——重点阅读 14.7

**前置知识建议**：建议先阅读第 8 章 Harness Engineering（理解六大工程支柱）、第 9 章 Skill System（理解技能系统如何工作）。这两章的抽象会在 OpenClaw 的每个子系统里一一落地。

> 📌 **章节关系**：OpenClaw 是"行动型 Agent 谱系"的**消费产品路线**代表；Claude Code（第 16 章）是**工业 IDE 路线**代表；DeepSeek Harness（第 17 章）是**开发者工坊路线**代表。读完本章后，建议继续读第 15 章 Hermes Agent——它在"多渠道"路线上和 OpenClaw 同源，但在"自我进化"上更激进。

---

## 引用约定

本章所有可核验事实（提交数、版本号、安装命令、协议、关键文件路径）以 **OpenClaw `main` 分支** 与官方 `openclaw.ai` 网站为准；任何随时间变化的数据（如 Star 数、扩展仓库数）以"截至 2026 年 8 月"为时间锚，并在文中明确标注"以仓库 `main` 分支为准"。

> 💡 **核心洞察**：OpenClaw 真正改变的不是"Agent 能力"，而是"Agent 触达用户的方式"。它证明了一件事——**当用户键变成聊天框里的 `@bot`**，Agent 才真正进入了日常生活。后面 14.7 节我们会展开讲这件事。

---

*下一节：[14.1 OpenClaw 全景：从 Clawdbot 到 OpenClaw 的演变](./01_history_and_positioning.md)*
