# 17.1 DeepSeek Harness 是什么：Cordis 内核与"一切皆插件"

> 🐋 *"Model + Harness = Agent；模型负责想，Harness 负责做。"*

---

## 一、一句话定义

**DeepSeek Harness（dsh / DSH）** 是 DeepSeek AI 于 2026 年 8 月 13 日开源的 Agent 运行框架，采用 **MIT 协议**，定位是"让大模型真正具备执行工作流的能力"——读写文件、执行命令、调用工具、派出子任务，并在授权范围内把长任务持续做完。

它不是模型，也不是 API 客户端——它是 **AI 的操作系统外壳**。

---

## 二、为什么 DeepSeek 要做"开源 Harness"

### 2.1 行业背景

到 2025 年底，"行动型 Agent"赛道已经出现两个清晰的对立：

| 阵营 | 代表 | 模型支持 | 开源 |
|------|------|---------|------|
| **商业闭源 Harness** | Claude Code、OpenAI Codex、Cursor | 主要绑自家模型 | ❌ |
| **开源消费级 Harness** | OpenClaw（第 14 章）、Hermes（第 15 章） | 模型无关 | ✅ |
| **开源工坊型 Harness** | **DeepSeek Harness（本章）** | 模型无关 | ✅ |

DeepSeek 的策略是"工坊型"——把"能力"和"模型"完全解耦，让开源社区用同一套 Harness 跑任何模型，同时在 v0.1 阶段就内置一套完整插件（插件数量以仓库为准）和 4 种运行模式。

### 2.2 团队自述的口号

发布时，DeepSeek 团队说了两句话：

- **"一切皆插件"（Everything is a Plugin）** —— 没有特权组件，所有能力都可换可加可减；
- **"闭源为用户提供现在，开源让用户自己掌握未来"** —— 公开承诺长期开源路线。

这句话被国内外的开发者大量引用，因为它**承认了商业 Harness 的局限**（闭源、版本不可控、模型绑定）——同时给出了一个可被审计、可被 fork、可被改写的替代。

---

## 三、"一切皆插件"是什么

DeepSeek Harness 的底层使用的是 **Cordis**（Koishi Team 开发的插件元框架）。Cordis 提供：

- 微内核，只负责插件的加载 / 卸载 / 依赖管理；
- 插件之间靠 **服务（service）** 和 **事件（event）** 协作；
- 每个插件挂到稳定的 **上下文键（context key）**（如 `ctx.llm`、`ctx.tools`、`ctx.agentLoop`）。

在 Cordis 之上，DeepSeek Harness 把**所有 Agent 能力**做成插件：

| Agent 能力 | Cordis 插件 | 默认实现 |
|-----------|------------|---------|
| **LLM 调用** | `llm-*` 插件 | anthropic / openai / openai-compatible / ollama |
| **工具执行** | `tool-*` 插件 | `core.tool.shell` / `core.tool.fs` / `core.tool.edit` |
| **Skill 系统** | `skill-*` 插件 | `core.skill.loader`（Markdown） |
| **上下文** | `context-*` 插件 | `context.summary` / `context.window` |
| **子 Agent** | `subagent-*` 插件 | `subagent.spawn` |
| **会话状态** | `session-*` 插件 | `session.sqlite` |
| **持久化** | `storage-*` 插件 | `storage.local` / `storage.redis` |
| **沙箱** | `sandbox-*` 插件 | `sandbox.local` / `sandbox.docker` |
| **调度** | `scheduler-*` 插件 | `scheduler.cron` |
| **UI** | `tui-*` / `web-*` 插件 | `tui.ink`（终端）/`web.react`（浏览器） |

合计上百个插件（数量以仓库为准）——这就是"一切皆插件"的真实含义。

### 3.1 它给开发者的 3 个直接好处

1. **换模型不动 Harness**：从 Anthropic 换到 OpenAI 只需要改一行配置；
2. **加新能力不动核心**：写一个插件，启用，重启即生效；
3. **降级或拆功能很容易**：CLI 不想要 Web UI？禁掉 `web-*` 插件；不需要 shell？禁掉 `core.tool.shell`。

> 📌 **和前两章的对比**：OpenClaw 把能力做成 Skill / Plugin 但**保留特权核心**；Hermes 把能力做成 Plugin 但**有内置运行循环**。DeepSeek Harness 走得更彻底——**连核心循环都是插件**。

---

## 四、核心特性清单（基于仓库 `main` 分支）

| 特性 | 实现要点 |
|------|---------|
| **MIT 开源** | 完整源码可商用 / 改 / 分发 |
| **一切皆插件** | Cordis 微内核 + 100+ 默认插件 |
| **模型无关** | 任意 OpenAI 兼容端点 + 自定义适配器 |
| **四种模式** | standard / minimal / PTC / create（详见 17.2） |
| **事件流+轨迹回放** | Trajectory（轨迹视图）支持 git 风格回放 / 分叉 |
| **MCP 兼容** | 内置 MCP server，让外部工具按标准接入 |
| **SKILL.md 兼容** | 兼容 Anthropic 风格 SKILL.md，与 Claude Code / OpenClaw / Hermes 互通 |
| **本地 + 沙箱** | 默认本地、可切 docker / kubernetes 沙箱 |
| **CLI + Web** | `dsh` 命令行 + 浏览器 127.0.0.1:3080 Web UI |
| **Python SDK** | `import deepseek_harness` 直接驱动 Agent |

> 与 Claude Code / OpenClaw / Hermes 同样的能力，但**路线不同**——这是后面 17.5 我们会展开的核心。

---

## 五、官方事实清单（可独立核验）

下面所有事实以 `deepseek-ai/deepseek-harness` `main` 分支 + 官方文档站为准：

| 项目 | 事实 |
|------|------|
| 仓库 | `github.com/deepseek-ai/deepseek-harness` |
| 许可证 | MIT（以仓库 `LICENSE` 为准） |
| 主语言 | TypeScript（核心 + CLI/Web）+ Python（SDK） |
| Cordis 上游 | `github.com/koishijs/koishi` 同名子项目 |
| 默认安装 | `npx @deepseek-ai/dsh web`（开发预览版） |
| 文档站 | `deepseek-harness.github.io/deepseek-harness/` |
| 协议兼容 | MCP（Model Context Protocol）、Anthropic `SKILL.md` |
| 模型后端 | 可换多种主流模型（数量以仓库为准） |
| 默认后端 | DeepSeek V4-Pro / V4-Flash（同时支持 OpenAI / Anthropic / 自定义） |
| 关联论文 | "A Programming Paradigm for Spatiotemporal Composability" |
| 发布时间 | 2026-08-13（v0.1 开发者预览） |

> **会随时间变化的数字**（Star / 安装 / 插件市场条目数等）本书**不引用**——这是 working memory 明确约定的"内容可信度约定"。

---

## 六、谱系坐标

![行动型 Agent Harness 谱系](../svg/chapter_openclaw_01_harness_spectrum.svg)

DeepSeek Harness 的**真正差异化**：

| 维度 | OpenClaw | Hermes | Claude Code | DeepSeek Harness |
|------|----------|--------|-------------|------------------|
| **能力归属** | 用户/社区写 Skill | Agent + 用户 | 厂商封闭 | **社区插件** |
| **核心可换** | ❌（核心写死） | 部分 | ❌ | ✅（连循环都是插件） |
| **可换内核** | ❌ | ❌ | ❌ | ✅ |
| **模型绑定** | 无 | 无 | 强 | **弱** |

一句话：**OpenClaw 给你房间的家具，DeepSeek 给你房间蓝图**。

---

## 七、本节小结

| 主题 | 关键要点 |
|------|---------|
| 项目定位 | 开源 Agent 运行框架，模型无关、插件化 |
| 团队 | DeepSeek AI（负责人崔添翼 / Tianyi Cui） |
| 上游 | Cordis（Koishi Team）作为插件微内核 |
| 核心差异化 | "一切皆插件"——连 Agent 循环都是可换插件 |
| 谱系坐标 | 开发者工坊路线，与 OpenClaw/Hermes/Claude Code 并列 |
| 协议兼容 | MCP + Anthropic SKILL.md |

---

*下一节：[17.2 安装与四种运行模式](./02_install_and_modes.md)*
