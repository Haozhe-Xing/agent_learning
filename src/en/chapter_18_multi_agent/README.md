# Chapter 16: Multi-Agent Collaboration

**Reading level**: Engineering extension; verify single-agent limitations first.

## This chapter's iteration

- **Previous limitation**: A single Agent already has tools, explicit state and acceptance. Some research subtasks may be independent, while extra roles can merely add handoff and call costs.
- **This chapter adds**: Define handoff inputs, artifacts, permissions and a total budget; accept subtasks before aggregation. The developer-team example separates repeated failures, timeouts and delivery.
- **How to verify**: Supplied `examples/dev_team/tests/test_dev_team.py` checks actual test results, failed delivery, process cleanup and workspace isolation. As a reader experiment, compare one Agent and a team on identical tasks and total budgets, recording acceptance, handoff failures and calls. These offline cases do not establish a team advantage.
- **Next step**: Read [chapter 17](../chapter_19_protocol/README.md) for process boundaries, or proceed to [evaluation](../chapter_20_evaluation/README.md). Collaboration is optional.

> 👥 *"Collaboration is worth adding when its measured benefit exceeds handoff costs."*

---

## Chapter Overview

When a single Agent faces complex tasks requiring multiple areas of expertise, multiple specialized Agents need to collaborate with a division of labor. This chapter introduces the design patterns, communication mechanisms, and collaboration strategies of multi-Agent systems, and walks through building a multi-Agent software development team composed of an architect, developers, testers, and DevOps engineers.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Understand the limitations of single Agents and the advantages of multi-Agent systems
- ✅ Master communication patterns (message queue, shared state, direct call) and negotiation mechanisms (voting, debate, contract-net bidding)
- ✅ Handle concurrent-write conflicts in shared state (reducer / single-writer / optimistic concurrency)
- ✅ Design reasonable role division, task allocation, and load balancing strategies
- ✅ Implement system-level fault tolerance (timeout, bounded retry, circuit breaker, heartbeat, idempotency)
- ✅ Distinguish Supervisor / Handoff / decentralized modes and identify the six collaboration patterns' applicable scenarios
- ✅ Use LangGraph to build a multi-Agent software development team as a hands-on project

## Chapter Structure

| Section |
|---|
| [16.1 Limitations of Single Agents](./01_single_agent_limits.md) |
| [16.2 Multi-Agent Communication Patterns](./02_communication_patterns.md) |
| [16.3 Role Assignment and Task Allocation](./03_role_assignment.md) |
| [16.4 Supervisor Mode vs. Decentralized Mode](./04_supervisor_vs_decentralized.md) |
| [16.5 Practice: Multi-Agent Software Development Team](./05_practice_dev_team.md) |
| [16.6 Paper Reading: Frontiers in Multi-Agent Systems](./06_paper_readings.md) |
| [16.7 Framework Supplement: CrewAI (Role-Playing Multi-Agent Framework)](./07_crewai.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Read [chapter 17](../chapter_19_protocol/README.md) for process boundaries, or proceed to [evaluation](../chapter_20_evaluation/README.md). Collaboration is optional.

---

*Next section: [16.1 Limitations of Single Agents](./01_single_agent_limits.md)*
