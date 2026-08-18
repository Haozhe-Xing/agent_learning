# 15.1 Birth & the "Growing Agent" Philosophy

> ☤ *"The agent that grows with you."* — Hermes Agent's self-description

---

## From OpenClaw's Cousin

OpenClaw puts the Agent in chat apps, but every capability must be maintained by humans. Hermes took another road: **let the Agent write its own skills**. Hermes is published by Nous Research, positioned as "OpenClaw's successor + the autonomous-evolution branch" — to this day it offers `hermes claw migrate` for OpenClaw users.

| Project | Core strategy | Skills come from |
|---------|--------------|------------------|
| OpenClaw | "humans write skills" | people |
| Hermes | "Agent writes + iterates skills" | people + Agent |
| OpenCode / Codex | "model + built-in tools" | vendor |

## Nous Research & the "Hermes" Name

Hermes comes from Nous Research (a non-profit known for open models and open Agents). The name has two layers: **mythology** (Hermes the messenger god, connecting worlds) and **brand continuity** (their model series is also named Hermes).

## The 4-Layer Design Philosophy

1. **Self-Learning** — after a task, distill the trajectory into a Skill (`SKILL.md`);
2. **Continuous Self-Improvement** — not just create, but *update* skills offline (DSPy/GEPA);
3. **Active Reflection** — a Nudge Engine periodically asks "what did I learn worth persisting?";
4. **Cross-platform persona** — same human, one persona across channels.

## Why Self-Evolution Matters

| Horizon | Without self-evolution | With self-evolution |
|---------|------------------------|---------------------|
| Day 1 | user configures everything | default + self-created skills |
| Day 30 | same as day 1 (+memory data) | a dozen user-specific skills |
| Day 90 | still teaching it | predicts user preferences |
| Year 1 | like a fresh install | each user has "their own version" |

> 📌 Self-evolution produces **compounding returns** over long-term use — a property ordinary Agents don't have.

## Hermes vs OpenClaw (Precise Boundary)

| Dimension | OpenClaw | Hermes |
|-----------|----------|--------|
| License | MIT | MIT |
| Author | Peter Steinberger / community | Nous Research |
| Focus | multi-channel + skill ecosystem | self-evolving skills + memory |
| Skill source | human only | human + auto + auto-iterate |
| Memory | SQLite long-term | three-layer |
| User modeling | none | Honcho dialectic |
| Migration | — | `hermes claw migrate` |

## Verifiable Facts (per `main` branch)

Repository `NousResearch/hermes-agent`; MIT license; Python core + TS gateway; one-liner install; docs at `hermes-agent.nousresearch.com/docs/`; compatible with the `agentskills.io` open standard; 15+ messaging platforms; 200+ model backends; 6 execution backends (local/Docker/SSH/Singularity/Modal/Daytona); `hermes claw migrate` for OpenClaw.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| Positioning | self-evolving personal Agent; skills auto-created & auto-iterated |
| Author | Nous Research |
| Philosophy | self-learn / self-improve / active reflection / cross-platform persona |
| vs OpenClaw | same lineage + self-evolution branch; `hermes claw migrate` |
| Core differentiator | "Skill auto-creation + auto-iteration" (see 15.4) |

---

*Next section: [15.2 Installation & Migration (from OpenClaw)](./02_install_and_migration.md)*
