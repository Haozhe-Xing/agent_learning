# Chapter 15: Hermes Agent — The Self-Evolving Agent

> ☤ *"The longer you use it, the more it understands you — not marketing, a closed loop."*

---

## Chapter Introduction

The previous chapter showed OpenClaw — a "consumer-grade" personal assistant living in chat apps, where **all capabilities are maintained by humans**. Hermes Agent takes a different road: **the Agent writes its own skills**.

Hermes Agent (nicknamed "爱马仕" by the Chinese community) was released by Nous Research in mid-2025. It shares deep lineage with OpenClaw (Node + TypeScript, personal local execution, multi-platform), but its core differentiator is one word: **learn**.

- **OpenClaw** is "configure-and-use" — skills are pre-written by users/community;
- **Hermes** is "learn-in-use" — after each complex task, the Agent **itself** distills the execution trajectory into a Markdown skill file (`SKILL.md`) and iterates those skills offline (DSPy / GEPA-inspired).

This "autonomous skill creation + continuous self-iteration" is called **Self-Evolving Skills** — the sharpest feature distinguishing Hermes from every other Agent framework. Combined with a three-layer memory system and a proactive Nudge Engine, Hermes is "the agent that grows with you."

This chapter:

1. **Dissects the closed learning loop** — how "write skills automatically + improve skills automatically + reflect proactively" forms an engineered self-evolution loop;
2. **Demonstrates install / migration** — `hermes claw migrate` one-click migration from OpenClaw;
3. **Analyzes three-layer memory + FTS5 + LLM summarization** — how SQLite full-text search + LLM summarization resolve "limited context vs. long-term memory."

---

## Chapter Content Overview

| Section | Content | What You'll Learn |
|---------|---------|-------------------|
| 15.1 Birth & the "Growing Agent" Philosophy | Nous Research, the naming, the self-evolution design | Hermes' place in the new-gen Agent spectrum |
| 15.2 Installation & Migration (from OpenClaw) | one-liner, `hermes claw migrate` | Run it on macOS/Linux/VPS/Termux |
| 15.3 Three-Layer Architecture | Gateway / Engine / Plugin subsystems | Read Hermes' engineering skeleton |
| 15.4 Core: Self-Evolving Skills Loop | skill distillation, DSPy/GEPA offline iteration | How "Agent writes its own skills" works |
| 15.5 Three-Layer Memory | long-term / working / episodic, FTS5 + LLM summary | Cross-session memory design |
| 15.6 Nudge Engine & Cross-Session Learning | active reflection, Honcho user modeling, cron | Make the Agent proactively ask "what to remember" |
| 15.7 Borrowing the Philosophy | closed loop, model-agnostic, runs on $5 VPS | Bring Hermes' philosophy to your system |

---

## Reading Recommendations

- ✅ **Engineers wanting to understand Agent self-evolution** — focus on 15.4, 15.6
- ✅ **OpenClaw users wanting to upgrade** — focus on 15.2
- ✅ **Long-term memory / cross-session developers** — focus on 15.5

**Prerequisites**: Chapters 8 (Harness), 9 (Skill System), 14 (OpenClaw). This chapter and Ch.14 form the "action-Agent twin chapters" — one multi-channel, one self-evolving.

---

## Citation Convention

Verifiable facts are anchored to the Hermes Agent `main` branch and `hermes-agent.nousresearch.com/docs/`. Time-varying numbers are **not cited**.

> 💡 **Core insight**: Hermes proves an Agent can *distill experience into skills like a person* — provided it's designed as a closed system loop, not maintained by hand.

---

*Next section: [15.1 Birth & the "Growing Agent" Philosophy](./01_birth_and_philosophy.md)*
