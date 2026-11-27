# 16A.1 Install, Connect a Model, and Start a Session

## Check that you are using V2

These commands target OpenCode V2. The npm package is `@opencode/cli`:

```bash
npm install -g @opencode/cli
opencode --version
```

If you already have V1, use `command -v opencode` on macOS/Linux or `Get-Command opencode` in PowerShell to check which binary starts. Both versions use the same command name. See the [official installation page](https://opencode.ai/docs/) for alternative installers and Windows binaries, and the [migration guide](https://opencode.ai/docs/migrate-v1/) before moving existing sessions, plugins, or API clients.

## Open a project and pick a model

From the repository root, run `opencode` to open the terminal UI. Use `/connect` to select and authenticate a provider, then `/models` to choose a model for the session. Use an available `provider/model` value when adding a project default. Enter credentials through provider authentication or the environment so the repository configuration remains shareable. [Provider authentication](https://opencode.ai/docs/cli/providers/) · [Model configuration](https://opencode.ai/docs/models/)

## Try a small interaction

1. Select **Plan** with `/agents`. Ask for the project entry point, the test command, and two important files. Watch which files it reads.
2. Attach one file with `@` and ask a focused question. For example, ask where a function handles empty input.
3. If a change is needed, switch to **Build** with `/agents`. Specify the goal, the file it may edit, and the acceptance check. Inspect the resulting `git diff` and run the check yourself.

Plan and Build are separate built-in agents. Plan blocks edits to ordinary project files by default and has its own shell and plan-file rules. Build starts with broader tool permissions. [Built-in agents](https://opencode.ai/docs/agents/) · [Terminal interaction](https://opencode.ai/docs/cli/tui/)

| Action | V2 control |
|---|---|
| Change agent | `/agents` or the default `Shift+Tab` |
| Start or reopen a session | `/new` or `/sessions` |
| Undo the last turn | `/undo` stages the rollback. Submitting accepts it. `/redo` cancels it before then |
| Find available commands | Type `/` or open the command palette with `Ctrl+P` |

Undo follows OpenCode's snapshot scope. Use separate recovery mechanisms for commits, remote state, and other external effects. [Snapshots and rollback](https://opencode.ai/docs/snapshots/)
