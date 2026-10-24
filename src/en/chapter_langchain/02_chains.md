# 10.2 Chains: Building Processing Pipelines

The previous section exercised one model-message interface. This section adds normalization, one classification, routing, and result gathering. The task is an **offline customer-service intake exercise**: English return/refund substrings are supported; other requests ask for clarification. The fixed provider supplies neither real intent recognition nor retrieval. Chinese questions, negation, and ambiguous keywords are outside this rule's quality claims.

Use the [10.1 environment](./01_langchain_overview.md). Complete code is in `reference_code/langchain_patterns/workflows.py`.

## Start with Data Flow

![LCEL sequence, preserved input, parallel, and conditional patterns](../svg/chapter_langchain_02_chain_patterns.svg)

| Composition | Input → output | Exercise responsibility |
|---|---|---|
| `A | B` | A's value becomes B's input | Normalize before classification |
| `RunnablePassthrough.assign` | Preserve dictionary fields and add others | Classify once and retain `intent` |
| `RunnableBranch` | First matching branch or default | Distinguish supported work from clarification |
| `RunnableParallel` | Same input → dictionary of branch outputs | Retain question, intent, and answer |

These are real control-flow operators. A Chain is not limited to a linear sequence. Graph adoption depends on state lifetimes, recovery, and human decisions, rather than the presence of an `if`. The [official Runnable entry](https://reference.langchain.com/python/langchain-core/runnables) lists these primitives.

## A Complete Runnable Intake Flow

`normalize_request` accepts a nonempty `question` and rejects extra fields. `classify_fixture` is an explicit English substring rule. Both are short ordinary Python functions in the companion file; read them before replacing classification.

```python
from langchain_core.runnables import (
    RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch,
)
from langsmith import tracing_context
from reference_code.langchain_patterns import FIXTURE_ANSWER, build_answer_chain
from reference_code.langchain_patterns.workflows import normalize_request, classify_fixture

policy = RunnableLambda(lambda x: {
    "fact": FIXTURE_ANSWER, "question": x["question"],
}) | build_answer_chain()
route = RunnableBranch(
    (lambda x: x["intent"] == "policy", policy),
    RunnableLambda(lambda _: "Unsupported fixture question; clarify the task."),
)
flow = (
    RunnableLambda(normalize_request)
    | RunnablePassthrough.assign(intent=RunnableLambda(classify_fixture))
    | RunnableParallel(
        question=RunnableLambda(lambda x: x["question"]),
        intent=RunnableLambda(lambda x: x["intent"]), answer=route,
    )
)
with tracing_context(enabled=False):
    result = flow.invoke({"question": "  Refund policy?  "})
assert result == {"question": "Refund policy?", "intent": "policy", "answer": FIXTURE_ANSWER}
print(result)
```

The result retains the normalized question. Conditions do not call the model again; the default branch states the fixture's limitation. A future model classifier would be a budgeted, fallible call requiring independent misclassification cases. Retaining `_intent` does not create a cross-request cache.

`RunnableParallel` can run independent branches concurrently; it must not run a report before the statistics that report needs. No latency gain or real concurrent model requests were measured here. If one parallel branch raises, default `invoke` does not return a complete success dictionary. External side effects that already happened are not automatically rolled back.

## Keep Per-Input Batch Failures

```python
from langsmith import tracing_context
from reference_code.langchain_patterns import build_intake_flow

with tracing_context(enabled=False):
    results = build_intake_flow().batch(
        [{"question": "refund policy?"}, {"question": ""}],
        config={"max_concurrency": 2}, return_exceptions=True,
    )
assert results[0]["intent"] == "policy"
assert isinstance(results[1], ValueError)
print([type(item).__name__ for item in results])
```

The first result is a dictionary and the second is a `ValueError`. `return_exceptions=True` lets the caller handle each input; it does not turn errors into accepted answers. `max_concurrency` limits this batch, not a service-wide quota or requests per second.

A script can use the asynchronous interface as follows:

```python
import asyncio
from langsmith import tracing_context
from reference_code.langchain_patterns import build_intake_flow

async def main():
    with tracing_context(enabled=False):
        return await build_intake_flow().ainvoke({"question": "refund"})

assert asyncio.run(main())["intent"] == "policy"
```

Inside an existing notebook/service event loop, `await main()` instead of nesting `asyncio.run`. An `ainvoke` name does not establish native asynchronous I/O; a default Runnable implementation may run synchronous work in a thread.

## When Does the Workflow End?

A scripted response is an execution result. Business acceptance still requires the question scope, factual evidence, and required points. The `unsupported` route requests clarification; it is distinct from generation failure and an answered question. See [10.5](./05_practice_customer_service.md) for actual fixture order facts and write permissions.

Exercise: submit a blank question, an unsupported question, and an extra `permissions` field. Record where each ends. Then make a parallel branch raise `PermissionError` and confirm that remaining fields are not packaged as complete success. Do not remove validation to make a case pass.

*Previous: [10.1 Architecture Overview](./01_langchain_overview.md)*

*Next: [10.3 Building Agents with LangChain](./03_langchain_agents.md)*
