# 16A.3 自定义 Agents、Skills 与 MCP

这三个接口解决不同的问题。Agent 定义角色、运行方式和权限。Skill 提供能按需加载的操作步骤。MCP 给 Agent 接入由另一个进程或服务提供的工具。

## 把审阅规则做成只读 Agent

项目可在 `.opencode/agents/` 下放 Markdown 文件。frontmatter 描述它的模式和权限，正文是给模型的系统指令。文件名成为 Agent ID。配套练习把上游的 `task-reviewer` 改为检查排名规则，保留「找证据、输出判定」的做法。它拒绝修改文件和执行 shell。[官方 Agent 配置](https://opencode.ai/docs/agents/) · [上游审阅员示例](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/examples/task-reviewer.md)

| `mode` | 用法 |
|---|---|
| `primary` | 在会话中直接选择 |
| `subagent` | 由父 Agent 在子会话里调用 |
| `all` | 两种用法都支持。练习采用这个模式，方便直接用 CLI 调用 |

父 Agent 可通过 `subagent` 权限限定它允许调用哪些子 Agent。被调用的 Agent 用自己的权限执行。练习中可先在 `/agents` 选择 `task-reviewer`，再附上 `ranking.py` 和验收脚本，观察它给出的证据和结论。[Agent 模式和委托权限](https://opencode.ai/docs/agents/)

## 把固定的讲解步骤做成 Skill

`.opencode/skills/explain-ranking/SKILL.md` 定义了一个讲解过程。名称和描述帮助模型决定何时加载，正文给出需要执行的步骤。

```markdown
{{#include ../../../examples/opencode-workshop/.opencode/skills/explain-ranking/SKILL.md}}
```

可问「用 explain-ranking Skill 解释这次并列处理的改动」。检查回答是否给了样例、代码位置和验收结果。在第 9 章 Skill System 学到的按需加载，在终端里可以直接观察。[OpenCode Skills 官方文档](https://opencode.ai/docs/skills/) · [第 9 章](../chapter_skill/README.md)

## 给需要的项目接一个 MCP

在临时项目中试用一个文档检索服务时，可用官方示例的 Context7 地址：

```bash
opencode mcp add context7 --url https://mcp.context7.com/mcp
opencode mcp list
```

添加命令会更新项目配置。用 `/mcps` 查看状态和所需认证，只给有需要的项目和 Agent 开放工具。V2 手写配置时用 `mcp.servers`，本地服务通过 stdio 启动，远程服务通过 Streamable HTTP 连接。排名练习只使用本地文件，可以独立完成。[MCP 设置与管理](https://opencode.ai/docs/mcp-servers/) · [第 19 章 MCP 原理](../chapter_19_protocol/01_mcp_protocol.md)
