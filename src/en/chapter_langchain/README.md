# Chapter 10: LangChain In-Depth

**Reading level**: Engineering extension; select when integration abstractions are needed.

The existing assistant can complete its task, but model messages, tool definitions, and error handling may be scattered through business code. Study how LangChain can share these interfaces while preserving source, permission, budget, and acceptance contracts. Adoption depends on maintenance cost and verification results.

## This chapter's iteration

- **Previous limitation**: Repeated model and tool adapters make integration changes affect business logic.
- **This chapter adds**: Explicit interfaces for messages, tools, and call results, with chains, Agents, and traces managing selected integration concerns.
- **How to verify**: Run the customer-service companion with fixed model outputs, then port the original task and failures to the candidate interface. Check calls, permissions, and independent acceptance; successful runs of different fixtures are not a controlled comparison.
- **Next step**: For branching, pause, or recovery, read [chapter 11, LangGraph](../chapter_langgraph/README.md). If integration is sufficient, continue to [chapter 18, Evaluation](../chapter_20_evaluation/README.md).

## Chapter structure

| Section | Focus |
|---|---|
| [10.1 LangChain Architecture Overview](./01_langchain_overview.md) | Model, tool, message, and package boundaries |
| [10.2 Chains: Building Processing Pipelines](./02_chains.md) | Explicit inputs and outputs in composed pipelines |
| [10.3 Building Agents with LangChain](./03_langchain_agents.md) | Tool registration, results, and error boundaries |
| [10.4 LCEL: LangChain Expression Language](./04_lcel.md) | Composition and debugging records |
| [10.5 Practice: Multi-Function Customer Service Agent](./05_practice_customer_service.md) | Tool calls and permissions in a customer-service task |
| [10.6 LangSmith Integration and Observability](./06_langsmith_integration.md) | Call traces and evaluation inputs |
| [10.7 LangChain Ecosystem 2026](./07_langchain_ecosystem_2026.md) | Version dependencies and migration records |
| [10.8 LangChain Production Patterns](./08_production_patterns.md) | Streaming, concurrency, retries, and resource bounds |

## Reading path

Review contracts in [chapter 3, Tools](../chapter_tools/README.md) and [chapter 6, RAG](../chapter_rag/README.md) as needed. Environment setup is in [appendix F](../chapter_setup/README.md). Start with 10.1, 10.3, and 10.5, then select pipeline, observability, and operational material according to the task.

## Verification and material boundaries

The companion is `examples/chapter12/customer_service.py`; its directory retains a historical chapter number. It injects fixed model results by default while running the framework tool loop. Live-model operation needs separate configuration and evidence. Its customer-service fixture differs from chapter 11's review fixture and does not establish a framework ranking.

LangChain and LangGraph can be combined or used separately. Neither removes the application's responsibility for sources, permissions, failure states, and budgets. Record dependency versions with the interfaces used instead of mixing examples across versions.

---

*Next section: [10.1 LangChain Architecture Overview](./01_langchain_overview.md)*
