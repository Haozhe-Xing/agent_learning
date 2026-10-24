# Chapter 11: LangGraph: Building Stateful Agents

**Reading level**: Engineering extension; select for explicit state, routing, or recovery.

Once tools can be called, the next questions are which step is running, why a retry occurs, when confirmation is needed, and how to stop. This chapter uses nodes, edges, and shared state to express those transitions while preserving task inputs, tool results, and acceptance criteria.

## This chapter's iteration

- **Previous limitation**: Implicit control flow makes branches, revision counts, and stop reasons difficult to inspect.
- **This chapter adds**: Explicit state, conditional edges, and bounded loops, with state requirements for confirmation and recovery.
- **How to verify**: Use the fixed-score review companion to check routing and counts. Port the same contract and failure cases when comparing an original task. Durable recovery, confirmation, and side-effect deduplication each need additional experiments; a running graph does not establish them.
- **Next step**: For independently assignable work, read [chapter 16, Multi-Agent Collaboration](../chapter_18_multi_agent/README.md). Otherwise turn state records into regression cases in [chapter 18, Evaluation](../chapter_20_evaluation/README.md).

## Chapter structure

| Section | Focus |
|---|---|
| [11.1 Why Graph Structures?](./01_why_graph.md) | Decide whether explicit state helps the task |
| [11.2 LangGraph Core Concepts: Nodes, Edges, and State](./02_core_concepts.md) | Node inputs, state updates, and edges |
| [11.3 Build Your First Graph Agent](./03_first_graph_agent.md) | Run and inspect a state graph |
| [11.4 Conditional Routing and Loop Control](./04_conditional_routing.md) | Failure branches, revision limits, and stopping |
| [11.5 Human-in-the-Loop: Human-AI Collaboration](./05_human_in_the_loop.md) | Confirmation points, resume state, and side effects |
| [11.6 Practice: Workflow Automation Agent](./06_practice_workflow_agent.md) | Accept state transitions in a review task |

## Reading path

Useful prerequisites are result contracts from [chapter 3, Tools](../chapter_tools/README.md), failures and budgets from [chapter 5, Planning](../chapter_planning/README.md), and Python functions and dictionaries. Consult [chapter 10, LangChain](../chapter_langchain/README.md) for integration needs; completing that chapter or adopting both libraries is not required.

## Verification and material boundaries

The companion is `examples/chapter13/content_workflow.py`; its directory retains a historical chapter number. Fixed outputs control review scores while the graph actually executes. This checks routing and revision counts, not model review quality, and does not provide universal acceptance coverage for recovery mechanisms.

Learning state graphs does not require multiple Agents. Role assignment, communication across services, and graph orchestration should each follow a separate task requirement.

---

*Next section: [11.1 Why Graph Structures?](./01_why_graph.md)*
