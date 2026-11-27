# 16A.3 Custom Agents, Skills, and MCP

Agents define a role, invocation mode, and permissions. Skills supply procedures the agent can load when needed. MCP connects tools exposed by another process or service.

## Make a read-only reviewer

Put a Markdown definition under `.opencode/agents/`. The frontmatter sets mode and permissions, the body supplies the system instructions, and the filename becomes the agent ID. The workshop adapts the upstream `task-reviewer` to check the ranking rule while keeping the evidence-and-verdict exercise. The reviewer denies edits and shell calls. [Official agent configuration](https://opencode.ai/docs/agents/) · [Upstream reviewer example](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/examples/task-reviewer.md)

| `mode` | Where it runs |
|---|---|
| `primary` | Directly in the current session |
| `subagent` | In a child session started by a parent |
| `all` | In either role. The workshop uses this mode so the reviewer is directly selectable from the CLI |

A parent agent's `subagent` permissions determine which child agents it can start. Each child uses its own permissions. In the lab, select `task-reviewer` with `/agents`, attach `ranking.py` and the acceptance script, and inspect the evidence and verdict. [Agent modes and delegation rules](https://opencode.ai/docs/agents/)

## Package a procedure as a skill

`.opencode/skills/explain-ranking/SKILL.md` defines a procedure. Its name and description guide when the model loads it, and the body gives it steps to follow.

```markdown
{{#include ../../../examples/opencode-workshop/.opencode/skills/explain-ranking/SKILL.md}}
```

Ask the agent to use the explain-ranking skill on the latest tie-break change. Check for an example, a code location, and the validation result. You can observe the on-demand loading mechanism introduced in Chapter 9. [OpenCode skills guide](https://opencode.ai/docs/skills/) · [Chapter 9](../chapter_skill/README.md)

## Connect MCP in a project that needs it

To try a documentation search server in a temporary project, use the Context7 endpoint from the official guide:

```bash
opencode mcp add context7 --url https://mcp.context7.com/mcp
opencode mcp list
```

Adding a server updates project configuration. Use `/mcps` to inspect status and any authentication requirements, and make tools available only where useful. In handwritten V2 configuration, put connections in `mcp.servers`. Local servers use stdio and remote servers use Streamable HTTP. The ranking lab uses local files and runs independently of MCP integrations. [MCP setup and management](https://opencode.ai/docs/mcp-servers/) · [Chapter 19 on MCP](../chapter_19_protocol/01_mcp_protocol.md)
