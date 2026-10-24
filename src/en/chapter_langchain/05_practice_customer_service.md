# 10.5 Practice: Multi-Function Customer Service Agent

## Iteration and Acceptance

Move orchestration into a framework while retaining acceptance and permission checks.

- **Success criteria**: Use fixed orders, FAQs, and complaints to check tool selection and replies; use test backends and distinguish reads from writes.
- **Failure cases**: Try nonexistent orders, another user’s order, and repeated submissions; failures must not be described as completed actions.
- **Safety boundary**: Enforce identity, ownership, and write permissions in the backend; prompt instructions do not replace authorization.

Record inputs, expected results, and actual results before extending the system. See [Chapter 18](../chapter_20_evaluation/README.md) for evaluation and [Chapter 19](../chapter_21_security/README.md) for trust and permissions.

Combining all LangChain features, we build a complete multi-function customer service Agent system.

![Multi-Function Customer Service Agent System Architecture](../svg/chapter_langchain_05_customer_service.svg)

This exercise uses a complete LangChain v1 Agent with fixture orders, FAQs, and idempotent complaints. The host binds the authenticated owner and the write permission; the model cannot change them.

## Run the Companion

```python
from examples.chapter12.customer_service import build_agent

# Fixture identity is set by the host. Real services must authenticate it.
agent = build_agent(owner="alice", allow_writes=False)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "check ORD-1"}]},
    config={"recursion_limit": 8},
)
print(result["messages"][-1].content)  # shipped
```

Run from the repository root after installing `examples/requirements.txt`:

```bash
python examples/chapter12/customer_service.py --query "check ORD-1"
python examples/chapter12/customer_service.py --query "check ORD-2"
python examples/chapter12/customer_service.py --query "complaint ORD-1"
python examples/chapter12/customer_service.py --query "complaint ORD-1" --allow-writes
```

Expected results are `shipped`, access denied, write denied, and `TICKET-1`. Repeating the same complaint on the same Agent returns the same ticket. The default model is deterministic, but LangChain executes the actual tool loop. Add `--real --model MODEL` and configure credentials for model-backed operation.

## What to Inspect

Check tool messages as well as the final answer. Ownership and write checks run in the backend. The recursion limit bounds framework execution, and a denied tool result must remain a denial in the response. Fixture identity selection demonstrates host binding; production systems need authentication.

This replaces the old `AgentExecutor` example. The interface follows the [official LangChain quickstart](https://docs.langchain.com/oss/python/langchain/quickstart). Continue to [Chapter 11](../chapter_langgraph/README.md) for state and recovery.
