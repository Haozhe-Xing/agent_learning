# 13.7 Borrowing the Philosophy: Self-Evolution as Engineering Principle

> Treat long-running operation, retained experience, and candidate improvement as separate design questions. Select automatic learning according to the task.

---

## The One Thing to Take Away

After sections 13.4–13.6, ask separately whether the task needs cross-session state, whether experience is worth retaining, and whether evidence supports automatic skill changes. Reliable fixed workflows, human maintenance, and automatic candidate improvement can each be valid choices.

Before an automatic-improvement experiment, fix versions, tasks, interfaces, and permissions. Check held-out acceptance, refusal paths, and rollback. Long-running use alone establishes neither the need for learning nor candidate effectiveness. The following principles organize these checks:

## 1. Determine Long-Running Requirements from the Task

| Dimension | Short-term | Long-term (Hermes) |
|-----------|-----------|--------------------|
| State | in RAM | on disk + rollback |
| Crash recovery | ignore | resume |
| Dependencies | freeze | gray release |
| Security | local perms | credential rotation + audit |
| Backups | none | periodic + incremental |

> Most "dead-in-a-month" Agent projects failed one of these.

## 2. Make "Persistent Distillation" a Core Capability

Three distillation loops: dialogue → Skill, dialogue → USER.md, dialogue → MEMORY.md. Each product is **machine-readable** (LLM can use it) **and** **human-readable** (you can open/edit/merge it).

## 3. Close the Feedback Loop

Three feedback forms: explicit user feedback, task outcomes (success/failure), and behavioral patterns. Combine all three — one alone isn't enough.

## 4. Decide Your Cross-Channel Strategy

| Strategy | Trade-off |
|----------|-----------|
| Full isolation | independent persona per channel; poor UX |
| Full merge | consistent persona; security risk |
| Partial merge | preferences merged, work-context isolated |

Hermes defaults to full merge — pick what fits your scenario.

## 5. Swappable Kernel = Long-Term Control

Your production Agent **will** be swapped — model, storage, a tool. Make "swapping" cheap: stable interfaces, config overrides, graceful degradation, observability, documented plugin APIs.

---

## Squeezed Into One Sentence

> **Select state storage, feedback, and replacement mechanisms by task. Enable automatic improvement only when independent acceptance supports it.**

Identify state that must not be lost and credentials the task may use; test storage, isolation, recovery, and withdrawal. No single configuration guarantees reliable long-term operation.

---

## Chapter Summary

You should now be able to answer:

- ✅ What is a self-evolving Agent — distill-after-task + offline skill iteration + proactive reflection
- ✅ Three-layer memory — long-term / working / episodic
- ✅ What the Nudge Engine does
- ✅ Risks & boundaries — draft + audit + rollback + veto
- ✅ Transferable principles — long-running, distillation, feedback, cross-channel, swappable kernel

---

*Next chapter: [Chapter 14: Deep Dive into Claude Code](../chapter_claude_code/README.md)*
