# Part III: Framework Practice

> 🛠️ *"Frameworks won't make your Agent smarter, but they let you ship the smart parts faster. The cost of picking the wrong framework isn't rewriting code — it's being forced into a mental model that doesn't fit your problem."*

---

## What We Actually Buy in a "Framework"

In Parts I–II you hand-built Agents with **Prompt + tool calling + context engineering**. You gained full control — and felt the weight of boilerplate: message loops, tool-schema parsing, error handling, retries, streaming.

**A framework's entire purpose is to converge those recurring "engineering cross-cutting concerns" into reusable abstractions.** But 2026's "frameworks" have split into two routes:

| Route | Representative | What it abstracts | For |
|-------|---------------|-------------------|-----|
| **Code framework** | LangChain / LangGraph | an Agent as a tunable library | writing code, shipping products |
| **Action harness** | OpenClaw / Hermes / Claude Code / DeepSeek Harness | an Agent as a whole product you *use* | wanting a capable assistant now |

This part focuses on the latter — **6 deep chapters**, one project each, from usage to architecture.

---

## The 6-Chapter / Harness Decision Matrix

| Ch | Project | Route | Type | License | One-liner |
|----|---------|-------|------|---------|-----------|
| 12 | **LangChain** | code framework | general orchestration | MIT | the widest Agent assembly library |
| 13 | **LangGraph** | code framework | stateful graph | MIT | complex flows: state graph + checkpoints + HITL |
| 14 | **OpenClaw** | action harness | consumer personal assistant | MIT | an Agent living in all your chat apps |
| 15 | **Hermes Agent** | action harness | self-evolving assistant | MIT | "the agent that grows with you" |
| 16 | **Claude Code** | action harness | industrial IDE | source-available | the benchmark harness: six layers |
| 17 | **DeepSeek Harness** | action harness | developer workshop | MIT | "everything is a plugin" foundation |

### Decision Tree

```
Q1: "Use" an Agent, or "build" one?
├─ use → action-harness subtree
│   ├─ chat apps? → OpenClaw (Ch.14)
│   ├─ grow to know me? → Hermes (Ch.15)
│   ├─ software-engineering IDE? → Claude Code (Ch.16)
│   └─ long-term controllable platform? → DeepSeek Harness (Ch.17)
└─ build → code-framework subtree
    ├─ complex flows + state? → LangGraph (Ch.13)
    ├─ fast multi-tool integration? → LangChain (Ch.12)
    └─ DIY? → read any "borrowing" section + Ch.8
```

### Selection Dimensions

| Dimension | LangChain | LangGraph | OpenClaw | Hermes | Claude Code | DeepSeek Harness |
|-----------|-----------|-----------|----------|--------|-------------|------------------|
| Core paradigm | Runnable chain | state graph | multi-channel | self-evolving | industrial IDE | all-plugins |
| Swappable kernel | — | — | ❌ | ❌ | ❌ | ✅ |
| Model-agnostic | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Customization | high (code) | high (code) | mid (skill) | mid (skill) | low (config) | very high (plugin) |
| Production-ready | high | high | high | high | high | preview (v0.1) |
| License | MIT | MIT | MIT | MIT | source-avail | MIT |

> ⚠️ Licenses per each repo's `LICENSE`. DeepSeek Harness is a **v0.1 developer preview** with expected breaking changes. Time-varying numbers (stars, installs) are **not cited**.

---

## What Good Open-Source Projects Look Like

Top open-source Agent projects converge on the same paradigms, regardless of surface:

| Paradigm | Meaning | In this part |
|----------|---------|--------------|
| Event stream | typed actions/observations on a central bus | Claude Code (QueryEngine) |
| State graph + checkpoint | graph + snapshots for pause/resume/time-travel | LangGraph |
| Code-as-action | LLM emits executable code to call tools | DeepSeek Harness (PTC) |
| Layered packages | protocol / orchestration / integration decoupled | LangChain, DSH |
| Model-agnostic | unified layer shields model differences | DSH / OpenClaw / Hermes |
| Sandboxing | tools run isolated; dangerous actions confirm | Claude Code / DSH / Hermes |
| Plugin meta-framework | all capabilities are plugins | DSH (Cordis), OpenClaw, Hermes |
| Self-evolving skills | Agent distills skills from its own trajectories | Hermes |

### Two Routes to "Extending" an Agent

- **Pluginization** (OpenClaw Skill / DSH plugin / Claude Code Skills+MCP): capabilities explicitly written by developers, loaded at runtime — simple, controllable, auditable.
- **Self-evolution** (Hermes): capabilities learned by the Agent from experience — powerful but harder to audit.

They're not mutually exclusive — Hermes' generated skills also enter a skill directory.

---

## Part III Chapter Map

| Chapter | Content | Key takeaway |
|---------|---------|-------------|
| 12 LangChain | architecture, Chain, Agent, LCEL, observability, production | master the most popular framework |
| 13 LangGraph | graph, state, routing, HITL | build complex stateful Agents |
| 14 OpenClaw | channels, Skills, consumer harness | put the Agent in chat apps |
| 15 Hermes | self-evolving skills, memory, Nudge | how a "growing Agent" is engineered |
| 16 Claude Code | six layers, System Prompt, permissions, cache | reverse-engineer industrial design |
| 17 DeepSeek Harness | Cordis kernel, 4 modes, plugins | open + model-agnostic + pluginized |

After these 6 chapters, you'll have the ability to read **any new harness** — judging which branch it belongs to, what it solves, and what it can't.

---

*Start: [Chapter 12: LangChain In-Depth](./chapter_langchain/README.md)*
