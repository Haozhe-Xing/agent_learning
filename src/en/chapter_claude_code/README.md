# Chapter 16: Deep Dive into Claude Code — How an Industrial Harness Works

> 🛠️ *"Tools are not just for using — they are for understanding. Truly mastering a tool means you can predict its limits, control its behavior, and even reshape its direction."*
> — Adapted from Richard Feynman

---

## Chapter Introduction

In earlier chapters we learned to **build** Agent systems with LangChain and LangGraph; in Chapters 14–15 we saw OpenClaw and Hermes push the Agent toward **chat apps** and **self-evolution** respectively. This chapter studies the **working principles** of an industrial-grade AI coding Agent: Anthropic's **Claude Code**.

Claude Code ships as an npm package under a **source-available** license (not an OSI open-source license), but its internals have been extensively analyzed by the community at the source level. This makes it one of the few industrial-grade Agents in this book that you can *read line by line* — we don't need to guess how it works; we can look at its source organization, System Prompt, permission logic, and context management directly.

This chapter is about **principles**, answering three questions:

1. **How does it run?** — the six-layer architecture, the QueryEngine main loop, the TAOR (Think-Act-Observe-Repeat) execution core;
2. **How is it constrained?** — the 4-module, ~915-line System Prompt, the 6-stage permission pipeline, the static/dynamic partitioning of the Prompt Cache;
3. **How is it extended?** — MCP, Hooks, Skills, and sub-agents.

Through this chapter you gain value on two levels:

- **Practical**: truly master Claude Code — installation, MCP integration, Hooks automation, fine-grained CLAUDE.md configuration;
- **Principle**: see how a production-grade AI coding Agent handles permissions, manages context, and orchestrates multi-Agent collaboration — all directly transferable to your own Agent systems.

---

## Chapter Content Overview

| Section | Content | What You'll Learn |
|---------|---------|-------------------|
| 16.1 The Prehistory of Industrial Harnesses | AutoGPT → BabyAGI → OpenHands → Claude Code | What Claude Code stands on |
| 16.2 Getting to Know Claude Code | Installation, interaction modes, commands, differences from Copilot/Cursor | Get started; understand Agent vs. code-completion |
| 16.3 Deep Dive into Core Architecture | Six-layer architecture, QueryEngine, TAOR core, 52 tools, React+Ink | How Claude Code works end to end |
| 16.4 System Prompt, Permission Engineering & Prompt Cache | 915-line System Prompt's 4 modules, static/dynamic zones, 6-stage permission pipeline | Industrial System Prompt design & permission modeling |
| 16.5 Advanced Usage: MCP, Hooks, and Skills | MCP, Hooks, Skills, sub-agents | Turn Claude Code into a team workflow engine |
| 16.6 Production Practice | CLAUDE.md best practices, team config, cost, security | Deploy stably, efficiently, safely |

---

## Reading Recommendations

- ✅ **Engineers looking to boost productivity** — read 16.2 in order
- ✅ **Agent system builders** — focus on 16.3 and 16.4
- ✅ **Team leads / architects** — focus on 16.5 and 16.6

**Prerequisites**: Chapters 7 (Context Engineering) and 8 (Harness Engineering) before the architectural sections. 16.2 and 16.5 are relatively independent.

> 💡 **Note**: Claude Code's internals evolve between versions. This chapter's analysis is based on **community source analysis** of the source-available code plus officially documented behavior — no undisclosed internal information. Verifiable claims (Prompt Cache savings, permission modes, CLAUDE.md injection) each come with a way you can reproduce them on your own machine. Security analysis is for learning only.

---

*Next section: [16.1 The Prehistory of Industrial Harnesses: From AutoGPT to Claude Code](./01_industry_history.md)*
