# Chapter 16A: OpenCode V2 in Practice

> Commands and configuration target OpenCode V2, checked on 2026-11-27.

OpenCode is an open-source coding agent with terminal, desktop, and web clients. In V2, a server manages sessions, model connections, permissions, and tool execution. You can work interactively from the terminal or run a scoped task against the same project from the CLI. [Official introduction](https://opencode.ai/docs/) · [CLI overview](https://opencode.ai/docs/cli/) · [OpenCode source and license](https://github.com/anomalyco/opencode)

This chapter follows Damien Sileo's workshop sequence. Read and plan, make and verify a small change, then turn the review into a reusable agent. The exercise here uses only the Python standard library and has been adapted to OpenCode V2. The upstream tutorial is MIT-licensed. Its copyright and license accompany the example. [Upstream guided workshop](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/GUIDED.md) · [MIT license](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/LICENSE)

| Section | What you will practice |
|---|---|
| [16A.1 Install, connect a model, and start a session](./01_getting_started.md) | Set up V2 and try Plan and Build |
| [16A.2 Project instructions, permissions, and migration](./02_project_config.md) | Configure `AGENTS.md` and permissions and recognize V1 syntax |
| [16A.3 Custom agents, skills, and MCP](./03_extensions.md) | Scope a reviewer, package a procedure, and connect a useful external tool |
| [16A.4 Workshop: fix a ranking bug](./04_workshop.md) | Reproduce a failure, inspect the diff, and run a read-only review from the CLI |

You will need Git, Python 3.10 or later, and access to a model. Your provider determines account requirements, quotas, and data handling. [Model provider guide](https://opencode.ai/docs/cli/providers/)

**Further practice.** Follow the [full English workshop](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/README.md) for batch evaluation or the [OpenCode Chinese Community course](https://www.opencodechina.com/) for more practice scenarios. Check older setup instructions against the [official V1 migration guide](https://opencode.ai/docs/migrate-v1/).
