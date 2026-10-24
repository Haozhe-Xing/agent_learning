# 17.1 What Is DeepSeek Harness: Cordis Kernel and "Everything Is a Plugin"

> 🐋 *"Model + Harness = Agent; the model thinks, the harness acts."*

---

## One-Sentence Definition

**DeepSeek Harness (dsh / DSH)** is an Agent runtime framework open-sourced by DeepSeek AI on August 13, 2026 under the MIT license. It is not a model, not an API client — it is an **AI operating-system shell**.

## Why DeepSeek Built an Open Harness

By late 2025, the action-Agent race had two clear poles: commercial closed harnesses (Claude Code, OpenAI Codex, Cursor — mostly model-bound) and open consumer harnesses (OpenClaw, Hermes — model-agnostic). DeepSeek chose a third path — the **workshop route**: decouple "capability" from "model" completely, ship 100+ default plugins, and four run modes, all MIT.

The team's launch slogans: **"Everything is a plugin"** and **"Closed source gives users the present; open source lets users own the future."**

## What "Everything Is a Plugin" Means

DeepSeek Harness runs on **Cordis** (a plugin meta-framework by the Koishi team) — a microkernel that only loads/unloads plugins and manages their dependencies. On top of it, **every Agent capability is a plugin**:

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

100+ default plugins — that's what "everything is a plugin" really means.

### Three direct benefits

1. **Swap models without touching the harness** — one config line;
2. **Add capabilities without touching the core** — write a plugin, enable, restart;
3. **Remove/disable features easily** — disable `web-*`, `tool.shell`, etc.

> 📌 vs the previous two chapters: OpenClaw makes capabilities into Skills/plugins but keeps a privileged core; Hermes does the same but has a built-in loop; DeepSeek Harness goes further — **even the Agent loop itself is a plugin**.

## Core Feature List (per `main` branch)

| Feature | Note |
|---------|------|
| MIT open source | full source usable commercially |
| Everything is a plugin | Cordis microkernel + 100+ default plugins |
| Model-agnostic | any OpenAI-compatible endpoint + custom adapters |
| Four modes | standard / minimal / PTC / create |
| Event stream + trajectory replay | git-style replay / fork |
| MCP compatible | built-in MCP server |
| SKILL.md compatible | interoperable with Claude Code / OpenClaw / Hermes |
| Local + sandbox | default local, docker/kubernetes options |
| CLI + Web | `dsh` + browser at 127.0.0.1:3080 |
| Python SDK | `import deepseek_harness` |

## Verifiable Facts

Repository `deepseek-ai/deepseek-harness`; MIT; TypeScript core + Python SDK; Cordis upstream `koishijs/koishi`; default install `npx @deepseek-ai/dsh web`; docs `deepseek-harness.github.io/deepseek-harness/`; ~40 swappable models; MCP + Anthropic `SKILL.md` compatible; associated paper "A Programming Paradigm for Spatiotemporal Composability"; released 2026-08-13 (v0.1 preview). Time-varying numbers are **not cited**.

## Spectrum Position

```
Action-oriented Harness spectrum (2026)
├── multi-channel · self-evolving (open)      → Hermes (Ch.15)
├── multi-channel · consumer · skills (open)  → OpenClaw (Ch.14)
├── terminal IDE · industrial · closed        → Claude Code (Ch.16)
└── terminal+web · workshop · all-plugins (open) → DeepSeek Harness (this chapter)
```

The true differentiator: **only DeepSeek Harness makes the Agent loop itself swappable**.

## Section Summary

| Topic | Key point |
|-------|-----------|
| Positioning | open Agent runtime, model-agnostic, pluginized |
| Team | DeepSeek AI (lead Tianyi Cui) |
| Upstream | Cordis (Koishi team) |
| Core differentiator | "everything is a plugin" — even the loop |
| Protocol | MCP + Anthropic SKILL.md |

---

*Next section: [17.2 Installation & 4 Run Modes](./02_install_and_modes.md)*
