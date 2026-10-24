# Chapter 13: Hermes Agent — The Self-Evolving Agent

**Reading level**: Optional product and architecture case for cross-session memory or skill accumulation.

Use Hermes Agent to study how experience becomes a memory or skill candidate, and how that candidate is validated, reviewed, and reverted. Creating a skill file establishes an artifact; improvement on unseen tasks requires independent evidence.

## This chapter's iteration

- **Previous limitation**: Memory and Skill interfaces can store content, but do not yet decide what to retain, how to validate a change, or how to withdraw it.
- **This chapter adds**: A case for cross-session memory, skill accumulation, and scheduling, separating candidate generation, evaluation, and activation.
- **How to verify**: No uniform Hermes adapter is supplied. Pin the version, tasks, and permissions; compare enabled and disabled behavior on held-out tasks and test user isolation, deletion, disabling, and rollback. Without execution, submit an analysis record only.
- **Next step**: For code operations, select [chapter 14, Claude Code](../chapter_claude_code/README.md). For independent memory or skill evaluation, continue to [chapter 18, Evaluation](../chapter_20_evaluation/README.md).

## Chapter structure

| Section | Focus |
|---|---|
| [13.1 Birth & the "Growing Agent" Philosophy](./01_birth_and_philosophy.md) | Turn growth goals into testable questions |
| [13.2 Installation & Migration (from OpenClaw)](./02_install_and_migration.md) | Record environment and migration scope; preserve original data |
| [13.3 Three-Layer Architecture: Gateway / Engine / Plugin](./03_three_layer_architecture.md) | Separate entry, execution, and extension boundaries |
| [13.4 Core: Self-Evolving Skills Loop](./04_self_evolving_skills.md) | Skill candidates, validation, activation, and rollback |
| [13.5 Three-Layer Memory](./05_memory.md) | Isolation, retrieval, deletion, and cross-session evidence |
| [13.6 Nudge Engine & Cross-Session Learning](./06_nudge_engine.md) | Triggers, authorization, and stopping for proactive tasks |
| [13.7 Borrowing the Philosophy](./07_lessons_philosophy.md) | Extract transferable designs and their verification limits |

## Reading path

Relevant foundations are [chapter 4, Memory](../chapter_memory/README.md), [chapter 8, Harness](../chapter_harness/README.md), and [chapter 9, Skills](../chapter_skill/README.md). [Chapter 12, OpenClaw](../chapter_openclaw/README.md) provides an optional entry-point comparison, not a prerequisite or evidence of direct migration and component compatibility.

## Verification and material boundaries

“Self-evolution” here identifies design questions to test. It does not guarantee an effective skill after each task or treat a product description as quality evidence. Specific backends, migration commands, and storage implementations require version-specific verification.

The book's offline memory and skill examples check their own contracts and do not replace Hermes experiments. Application development does not require research chapters first; select [Part VII](../part7.md) later when learning algorithms become a research goal.

---

*Next section: [13.1 Birth & the "Growing Agent" Philosophy](./01_birth_and_philosophy.md)*
