# 13.1 Birth & the "Growing Agent" Philosophy

> ☤ *"The agent that grows with you."* — Hermes Agent's self-description

---

## Two Projects and a Migration Entry Point

[Chapter 12](../chapter_openclaw/README.md) examines chat entry points. This chapter examines memory and skill candidates. Hermes Agent belongs to [Nous Research's official repository](https://github.com/NousResearch/hermes-agent). Its README describes skill creation and improvement and provides an OpenClaw migration entry point. Migration does not establish shared lineage or succession.

The source-check anchor is [v0.22.4](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.4), [commit 4d04303](https://github.com/NousResearch/hermes-agent/commit/4d04303818b61bf3aa09278af48356315515965b), checked on October 22, 2026. An unsupported first-release date is not retained here.



## Nous Research & the "Hermes" Name

Hermes comes from Nous Research, a team developing open models and Agents. The name has two layers: **mythology** (Hermes the messenger god, connecting worlds) and **brand continuity** (their model series is also named Hermes).

## The 4-Layer Design Philosophy

1. **Self-Learning** — after a task, distill the trajectory into a Skill (`SKILL.md`);
2. **Continuous Self-Improvement** — investigate skill updates; this review has not verified DSPy/GEPA or an automatic offline optimization process;
3. **Active Reflection** — a Nudge Engine periodically asks "what did I learn worth persisting?";
4. **Cross-platform persona** — same human, one persona across channels.

## Why Self-Evolution Matters

| Horizon | Baseline record | Check after enabling a candidate |
|---------|------------------------|---------------------|
| Initial use | Retain configuration and task results | Use the same records as a pre-skill baseline |
| Long-term use | Keep human-maintenance records | Compare gains, regressions, and cost on held-out tasks |
| Migration | Check imported configuration and data | Also check skill dependencies, permissions, and recovery |

Longer use does not establish improved capability. See [13.4](./04_self_evolving_skills.md) for candidate generation and acceptance design.



## Hermes vs OpenClaw (Precise Boundary)

| Dimension | Comparison evidence to retain |
|---|---|
| License | Check each selected version's LICENSE; licensing does not establish safety |
| Skill sources | Record bundled, imported, and generated candidates instead of assuming one product is human-only |
| Memory and user modeling | Verify storage, isolation, consent, and deletion by version |
| Migration | The fixed Hermes README documents an entry point; imported data and conflicts need separate tests |
| Entry points | Select an interface for the task and measure setup effort |



## Checked Sources and Their Scope

| Item | Evidence |
|---|---|
| Repository material | [README at the fixed commit](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/README.md) |
| License | Fixed-commit [MIT LICENSE](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/LICENSE) |
| Documentation | [Official documentation](https://hermes-agent.nousresearch.com/docs/); the website may differ from the fixed commit |
| Gateway | The fixed source includes Python [gateway/run.py](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/gateway/run.py), not a TypeScript Gateway |
| Migration | The README documents `hermes claw migrate` and `--dry-run`; migration and data integrity have not been executed here |
| Learning and memory | Skill creation, memory, and session search are project descriptions; triggers, storage, and effectiveness need separate checks |

Popularity and time in use are not capability evidence. The source anchor does not verify every command or teaching example in this chapter.



---

## Section Summary

| Topic | Key point |
|-------|-----------|
| Positioning | self-evolving personal Agent; skills auto-created & auto-iterated |
| Author | Nous Research |
| Philosophy | self-learn / self-improve / active reflection / cross-platform persona |
| vs OpenClaw | Separate projects; a migration entry does not establish lineage or complete compatibility |
| Core differentiator | Candidate generation and acceptance design in [13.4](./04_self_evolving_skills.md) |

---

*Next section: [13.2 Installation & Migration (from OpenClaw)](./02_install_and_migration.md)*
