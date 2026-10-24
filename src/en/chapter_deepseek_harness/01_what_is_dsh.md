# 15.1 What Is DeepSeek Harness: Cordis Kernel and "Everything Is a Plugin"

> 🐋 *"Model + Harness = Agent; the model thinks, the harness acts."*

---

## One-Sentence Definition

**DeepSeek Harness (dsh / DSH)** is an open-source Agent harness from DeepSeek AI. This review anchors [dsh-v0.2.1-rc.2](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.2.1-rc.2), [commit 687f282](https://github.com/deepseek-ai/deepseek-harness/commit/687f282e8625c6b8d11ff58211a90524be2e4317), and its [MIT license](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/LICENSE). This review has not established the first-release date; repository creation is not a product-launch date.

## Why DeepSeek Built an Open Harness

The **workshop** perspective is this book's teaching lens: study component boundaries and extension lifecycles. It is not an official release category, an exhaustive industry classification, or evidence that every model or mode is supported.

The fixed [README](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md) describes plugin composition and marks the project as a developer preview with breaking changes expected. A project description does not establish universal replaceability or superiority.

## What "Everything Is a Plugin" Means

DeepSeek Harness runs on [Cordis (cordiverse/cordis)](https://github.com/cordiverse/cordis) — a microkernel that only loads/unloads plugins and manages their dependencies. The following is this book's analysis inventory. Component names and defaults have not been individually checked at the fixed commit, so this is not an official API reference:

| Agent capability | Cordis plugin | Default implementation |
|------------------|---------------|------------------------|
| LLM call | `llm-*` | anthropic / openai / openai-compatible / ollama |
| Tool execution | `tool-*` | shell / fs / edit |
| Skill system | `skill-*` | Markdown loader |
| Context | `context-*` | summary / window |
| Sub-agent | `subagent-*` | spawn |
| Session | `session-*` | sqlite |
| Storage | `storage-*` | local / redis |
| Sandbox | `sandbox-*` | local / docker |
| Scheduler | `scheduler-*` | cron |
| UI | `tui-*` / `web-*` | ink / react |

Plugin counts do not establish the architecture. Check services, dependencies, and unload behavior directly.

### Three direct benefits

1. **Swap models without touching the harness** — one config line;
2. **Add capabilities without touching the core** — write a plugin, enable, restart;
3. **Remove/disable features easily** — disable `web-*`, `tool.shell`, etc.

> **Compare loop boundaries**: In a selected version, identify loop entry, state access, tool authorization, and termination cleanup, then test replacement on fixed tasks. Plugin names alone establish neither replaceability nor the absence of alternatives.

## Capabilities Requiring Further Checks

| Capability in the manuscript | Interface or behavior to check |
|---------|------|
| MIT open source | full source usable commercially |
| Everything is a plugin | Cordis service, dependency, and unload boundaries |
| Model-agnostic | any OpenAI-compatible endpoint + custom adapters |
| Four modes | standard / minimal / PTC / create |
| Event stream + trajectory replay | git-style replay / fork |
| MCP compatible | built-in MCP server |
| SKILL.md migration checks | Verify metadata, tool dependencies, state, permissions, and fixed task results; a filename does not establish cross-product compatibility |
| Local + sandbox | default local, docker/kubernetes options |
| CLI + Web | `dsh` + browser at 127.0.0.1:3080 |
| Python SDK | `import deepseek_harness` |

## Checked Sources and Their Scope

| Item | Evidence and limit |
|---|---|
| Repository | [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), anchored to the commit above |
| Documentation | [Official quickstart](https://deepseek-harness.github.io/deepseek-harness/guide/quickstart); website content is not pinned to the release |
| Cordis upstream | The README links [cordiverse/cordis](https://github.com/cordiverse/cordis), not a subdirectory of `koishijs/koishi` |
| Entry point | The fixed README documents `npx @deepseek-ai/dsh web` and local address `127.0.0.1:3080`; no installation was executed here |
| Release status | Developer preview in the README; Pre-release on the selected release |
| Defaults, Python SDK, and component inventory | Not individually verified in this review |

Counts and slogans do not replace execution evidence. Interfaces, modes, and deployment suitability require version-specific checks.

## Spectrum Position

```
Action-oriented Harness spectrum (2026)
├── multi-channel · self-evolving (open)      → Hermes (Ch.13)
├── multi-channel · consumer · skills (open)  → OpenClaw (Ch.12)
├── terminal IDE · industrial · closed        → Claude Code (Ch.14)
└── terminal+web · workshop · all-plugins (open) → DeepSeek Harness (this chapter)
```

For each case in chapters 12–15, record these checks in an exact version and configuration:

| Layer | Question |
|---|---|
| Loop entry | Can the entry be replaced, and who owns routing and stop conditions? |
| State | Do messages, task state, and recovery data preserve their contracts? |
| Permissions | Does the host still authorize and confirm before tool execution? |
| Cleanup | How are processes, connections, and temporary artifacts handled on failure or cancellation? |
| Fixed task | Do success, refusal, and exhaustion cases remain valid, and when is rollback needed? |

Keep the original loop when replacement is unnecessary. Mark an unverified candidate unknown rather than excluding alternatives.

## Section Summary

| Topic | Key point |
|-------|-----------|
| Positioning | open Agent runtime, model-agnostic, pluginized |
| Team | DeepSeek AI; individual responsibilities and launch history are not checked here |
| Upstream | Cordis, linked by the official README |
| Loop replacement | Verify entry, state, permissions, stopping, cleanup, and rollback in a selected version |
| Protocol and format boundaries | Verify MCP and SKILL.md interfaces, permissions, and task results by version |

---

*Next section: [15.2 Installation & 4 Run Modes](./02_install_and_modes.md)*
