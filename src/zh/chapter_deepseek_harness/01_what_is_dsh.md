# 15.1 DeepSeek Harness 是什么：Cordis 内核与"一切皆插件"

> 🐋 *"Model + Harness = Agent；模型负责想，Harness 负责做。"*

---

## 一、一句话定义

**DeepSeek Harness（dsh / DSH）** 是 DeepSeek AI 的开源 Agent Harness。本次核验使用[预发行版 dsh-v0.2.1-rc.2](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.2.1-rc.2)及[固定提交 687f282](https://github.com/deepseek-ai/deepseek-harness/commit/687f282e8625c6b8d11ff58211a90524be2e4317)，该提交采用 [MIT 许可证](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/LICENSE)。首次发布日期尚未由本次一手材料核验，不将仓库创建时间等同于产品发布。

它不是模型，也不是 API 客户端——它是 **AI 的操作系统外壳**。

---

## 二、为什么 DeepSeek 要做"开源 Harness"

### 2.1 行业背景

下面按本书的阅读问题区分案例用途；这不是截至某一日期的行业全景或互斥产品分类：

| 阵营 | 代表 | 模型支持 | 开源 |
|------|------|---------|------|
| 代码操作案例 | [第14章 Claude Code](../chapter_claude_code/README.md) | 按版本核验 | 另查适用条款 |
| 个人助理案例 | [第12章 OpenClaw](../chapter_openclaw/README.md)、[第13章 Hermes](../chapter_hermes/README.md) | 按版本核验 | 固定源码为 MIT |
| 插件化案例 | **DeepSeek Harness（本章）** | 适配器与工具协议分别核验 | 固定源码为 MIT |

“工坊型”是本书的教学视角，指组件和扩展的研究问题，不是官方发行类别，也不保证任意模型或所有模式已经兼容。

### 2.2 区分项目自述与验证结论

固定提交的 [README](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md)将项目描述为基于 Cordis 的插件化 Harness，同时标明 developer preview、接口可能发生破坏性变化。它支持我们研究组件边界，不足以证明任意能力均可替换或某类实现必然优于其他产品。

---

## 三、"一切皆插件"是什么

DeepSeek Harness 的底层使用的是 [Cordis（cordiverse/cordis）](https://github.com/cordiverse/cordis)插件元框架。Cordis 提供：

- 微内核，只负责插件的加载 / 卸载 / 依赖管理；
- 插件之间靠 **服务（service）** 和 **事件（event）** 协作；
- 每个插件挂到稳定的 **上下文键（context key）**（如 `ctx.llm`、`ctx.tools`、`ctx.agentLoop`）。

下表是本书用于分析插件边界的能力分类；其中组件名与默认实现尚未逐项在固定提交中核实，不能作为官方 API 清单：

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

插件数量不作为设计结论。应先确认具体服务、依赖和卸载行为，再讨论“一切皆插件”的实际边界。

### 3.1 它给开发者的 3 个直接好处

1. **换模型不动 Harness**：从 Anthropic 换到 OpenAI 只需要改一行配置；
2. **加新能力不动核心**：写一个插件，启用，重启即生效；
3. **降级或拆功能很容易**：CLI 不想要 Web UI？禁掉 `web-*` 插件；不需要 shell？禁掉 `core.tool.shell`。

> **比较循环边界**：在选定版本中查明循环入口、状态读写、工具授权与终止清理分别由哪一层承担，再用固定任务验证替换。不能仅凭插件名称判断循环可换，也不能据此排除其他实现。

---

## 四、后续需要逐项核验的能力

| 书稿涉及的能力 | 需要核验的接口或行为 |
|------|---------|
| **MIT 开源** | 完整源码可商用 / 改 / 分发 |
| **插件化** | Cordis 服务、依赖和卸载边界；不引用未核验数量 |
| **模型无关** | 任意 OpenAI 兼容端点 + 自定义适配器 |
| **四种模式** | standard / minimal / PTC / create 的版本对应关系，见[15.2节](./02_install_and_modes.md) |
| **事件流+轨迹回放** | Trajectory（轨迹视图）支持 git 风格回放 / 分叉 |
| **MCP 兼容** | 内置 MCP server，让外部工具按标准接入 |
| **SKILL.md 迁移检查** | 核对元数据、依赖工具、状态、权限与固定任务结果；同名文件不证明跨产品兼容 |
| **本地 + 沙箱** | 默认本地、可切 docker / kubernetes 沙箱 |
| **CLI + Web** | `dsh` 命令行 + 浏览器 127.0.0.1:3080 Web UI |
| **Python SDK** | `import deepseek_harness` 直接驱动 Agent |

> 与其他案例比较时，先固定任务与权限，再核对实际接口和结果。[第15.5节](./05_comparison.md)不把功能名称相似当作能力等价。

---

## 五、已核验的来源与范围

| 项目 | 一手材料与核验范围 |
|---|---|
| 仓库 | [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)，本次只对照上述固定提交 |
| 文档 | [官方开发者文档](https://deepseek-harness.github.io/deepseek-harness/guide/quickstart)，网站未固定到发行版 |
| Cordis 上游 | README 指向 [cordiverse/cordis](https://github.com/cordiverse/cordis)，不是 `koishijs/koishi` 的同名子目录 |
| 运行入口 | 固定 README 记录 `npx @deepseek-ai/dsh web` 与默认本地地址 `127.0.0.1:3080`；本次未实际安装运行 |
| 发行状态 | 固定 README 标明开发者预览；所选 release 标记为 Pre-release |
| 默认模型、Python SDK及组件清单 | 本次未逐项验证；不从官网截图、名称或论文引用推断实现 |

数字与宣传口号不代替行为证据。具体接口、模式和生产适用性，需要在相应版本上验证；未验证部分保留为待核验项。

---

## 六、谱系坐标

![行动型 Agent Harness 谱系](../svg/chapter_openclaw_01_harness_spectrum.svg)

对第12–15章的案例，分别记录以下问题；结论限于实际运行的版本与配置：

| 检查层次 | 要核对的问题 |
|---|---|
| 循环入口 | 能否替换入口，原有路由和停止条件由谁负责？ |
| 状态 | 替换前后的消息、任务状态与恢复数据是否保持契约？ |
| 工具权限 | 授权和确认是否仍在执行前由宿主检查？ |
| 失败清理 | 出错或取消后，进程、连接与临时产物如何清理？ |
| 固定任务 | 成功、拒绝和预算耗尽用例是否保持，何时需要回退？ |

不需要替换循环时可以保留原实现；尚未验证的候选记录为“未知”，不作排他性推荐。

---

## 七、本节小结

| 主题 | 关键要点 |
|------|---------|
| 项目定位 | 开源 Agent 运行框架，模型无关、插件化 |
| 团队 | DeepSeek AI；个人职责与发布史未在本次核验 |
| 上游 | 官方 README 指向 Cordis；接口与生命周期需按固定版本核对 |
| 循环替换检查 | 在指定版本验证入口、状态、权限与停止清理，记录失败与回退 |
| 谱系坐标 | 开发者工坊路线，与 OpenClaw/Hermes/Claude Code 并列 |
| 协议与格式边界 | 按版本核对 MCP 与 SKILL.md 的接口、权限及任务结果 |

---

*下一节：[15.2 安装与四种运行模式](./02_install_and_modes.md)*
