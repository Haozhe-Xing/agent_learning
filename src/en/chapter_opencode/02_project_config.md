# 16A.2 Project Instructions, Permissions, and V1 Migration

## Put project conventions in the repository

Use `AGENTS.md` for the information a new contributor needs, including entry points, test commands, generated files, and acceptance checks. Use `opencode.json` or `opencode.jsonc` for models, tool permissions, and integrations. Version both with the project. Display and keyboard preferences belong in the global terminal client file, `cli.json`. [Project instructions](https://opencode.ai/docs/instructions/) · [Configuration precedence](https://opencode.ai/docs/config/) · [Terminal settings](https://opencode.ai/docs/cli/config/)

## Scope the workshop with permissions

The workshop starts in Plan, blocks access outside its workspace, and requires approval for editing `ranking.py` or running shell commands. Plan explicitly denies all edits and shell calls. Switch to Build for the approval steps. Use these rules from the workshop directory.

```json
{{#include ../../../examples/opencode-workshop/opencode.json}}
```

A V2 rule has three fields, `action`, `resource`, and `effect`. The **last matching rule** wins for ordinary permissions, so the file-specific exception follows the wildcard. When debugging, account for project, user, agent, and managed settings. The shell still runs as the host user, and permission matching depends on the commands and paths OpenCode can identify. Use a separate working directory for unfamiliar repositories. [Permission matching and directory rules](https://opencode.ai/docs/permissions/) · [Policies](https://opencode.ai/docs/policies/)

## Recognize V1 instructions in older tutorials

| V1 tutorial pattern | V2 equivalent |
|---|---|
| Install `opencode-ai` | Install `@opencode/cli` |
| Set `permission` for `bash` and `task` | Use `shell` and `subagent` in the `permissions` list |
| Put server names directly under `mcp` | Put them under `mcp.servers` and use `disabled` to disable a connection |
| Edit a project's `tui.json` | Use global `cli.json` |
| Reuse a V1 plugin or server SDK | Migrate each integration to the current API |

Use the [official V1 documentation](https://opencode.ai/v1/docs/) when maintaining a V1 workspace. Before upgrading a project, back up its configuration and important sessions and check the features you use against the [migration guide](https://opencode.ai/docs/migrate-v1/).
