# 10.8 LangChain Production Patterns

Move from “it can be invoked” to “its failure still reports the actual state.” This section supplies runnable local boundary exercises, not a deployed production service. Real HTTP connections, cross-process state, Redis, deployment services, and online models were not accepted here. See [10.1](./01_langchain_overview.md) for versions and the entry point.

## Keep Three Results Separate

| Result | Record |
|---|---|
| Execution | Actual attempts, errors, stop reasons, and cleanup |
| Delivery | What was emitted, receiver confirmation, and retransmission needs |
| Task acceptance | Whether frozen requirements were met and what remains missing |

A finished stream, fallback response, or cache hit cannot replace task acceptance. Exercises use real Runnables in `reference_code/langchain_patterns/`; tests live in `examples/integration_tests/test_langchain_chapter.py`.

## Failure Before and After the First Chunk Differ

`with_fallbacks` can try alternatives for selected exceptions, but it cannot retract emitted content. This actual local counterexample emits `partial`, then raises `TimeoutError`; the fallback is not called:

```python
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from reference_code.langchain_patterns import stream_events

fallback_calls = []
def interrupted(_):
    yield "partial"
    raise TimeoutError("fixed failure after first output")

backup = RunnableLambda(lambda _: fallback_calls.append(True) or "backup")
chain = RunnableLambda(interrupted).with_fallbacks(
    [backup], exceptions_to_handle=(TimeoutError,),
)
with tracing_context(enabled=False):
    events = list(stream_events(chain, "fixture"))
assert [item["event"] for item in events] == ["delta", "error"]
assert fallback_calls == []
print(events)
```

The result contains `delta` and `error`, without `done`. The companion `stream_events` is an in-process adapter. Errors include only a type rather than potentially sensitive exception messages. `done` means normal iterator exhaustion, not accepted answer quality.

If failure occurs before the first chunk, the same fallback configuration can try the alternative; that alternative may also fail. Retrying after emitted output requires an explicit attempt boundary, replacement/retransmission protocol, and idempotency policy. Silently concatenating different answers is not recovery, and a successful fragment cannot erase the failed turn.

No SSE server is supplied here. HTTP integration must encode separate frames and handle disconnects, backpressure, and cancellation. A JSON event list is not a tested network stream. Services may schedule synchronous generators in worker threads, so `stream()` does not universally prevent every other user from being served. Avoid direct blocking work inside async routes and verify that underlying operations cooperate.

## Count Retries, Fallbacks, and the Whole Task

Section 10.4 retries only `TimeoutError`, at most three attempts. Negative cases also establish that `ValueError` gets one attempt and exhausted transient failures remain exceptions. Policies depend on error meaning, not a blanket `Exception` recovery rule.

| Failure | Possible handling | What must not happen |
|---|---|---|
| Identified transient read/transport failure | Budgeted backoff or fallback with attempts retained | Unlimited retries or hidden consumption |
| Argument/schema error | Locate the problem; bounded correction if justified | Unchanged retry as if it were rate limiting |
| Permission denial | Stop the unauthorized action and report it | Rephrase or switch tools to bypass permission |
| Timed-out side-effecting request | Query its outcome or use an idempotency contract | Blindly repeat a write |
| Failed task acceptance | Retain missing requirements and revise under the task contract | Declare success because another text response arrived |

SDK retries, Runnable attempts, and Agent loops can multiply work. Account for them in one task budget; backoff also consumes deadline time. The model/tool limits in 10.3 and graph-step guard are distinct measurements, not one universal `max_iterations` parameter.

## Async, Concurrency, and Timeouts Bound Different Things

Async I/O can free execution while waiting; it does not automatically parallelize CPU work or forcibly cancel synchronous functions. This example interrupts a cooperative coroutine:

```python
import asyncio
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

async def slow(value):
    await asyncio.sleep(1)
    return value

async def main():
    try:
        with tracing_context(enabled=False):
            await asyncio.wait_for(RunnableLambda(slow).ainvoke("fixture"), timeout=0.01)
    except TimeoutError:
        return "deadline_exceeded"
    raise AssertionError("expected the fixture to time out")

assert asyncio.run(main()) == "deadline_exceeded"
```

Another test starts a controlled synchronous worker. Canceling the `ainvoke` wait leaves that worker running until the test explicitly releases it. This proves no subprocess termination. Real process tools need their own timeout, termination, and cleanup; a returned `wait_for` timeout does not establish that side effects stopped.

`max_concurrency` in `batch`/`abatch` bounds activity in that batch. Tests observe at most two cooperative tasks at once. It is not requests per second, a token allowance, or a multi-process quota. Provider rates, tenant quotas, queues, and service capacity need separate controls; no fixed throughput is claimed for an undeployed limiter.

## Verify Exact Caching Before Adding More Backends

This uses an instance-specific `InMemoryCache`, leaving global cache settings alone. The fixed provider has two different scripted responses: the second identical request reuses the first result, while a source-revision change consumes the next:

```python
from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langsmith import tracing_context
from reference_code.langchain_patterns import build_scoped_chain

model = FakeListChatModel(responses=["first", "second"], cache=InMemoryCache())
chain = build_scoped_chain(model)
request = {"scope": "team-a", "revision": "v1", "question": "fixture"}
with tracing_context(enabled=False):
    values = [chain.invoke(request), chain.invoke(request),
              chain.invoke({**request, "revision": "v2"})]
assert values == ["first", "first", "second"]
print(values)
```

The host chooses `scope` and `revision`; arbitrary client declarations are not authorization. They enter the model input to demonstrate key separation. Callback metadata alone need not affect a cache key. Never put credentials into a prompt merely to distinguish permissions. A production host cache can use nonsensitive permission/data digests with actual access checks.

This cache is neither persistent nor shared across processes. TTL, invalidation queues, and capacity policy are not implemented here. Cached results must still satisfy current task and access requirements. Even `temperature=0` establishes neither permanent determinism nor fresh data; tasks requiring diverse responses may not want reuse.

Redis caches and backend-specific semantic caches are separate integrations with package-specific constructors. Similar questions can require different answers. Pin the concrete package, backend, thresholds, and false-hit cases before adopting one. This section verifies only the in-process exact cache; readers must test other backends separately.

## Observation Cannot Manufacture Success

The local recorder in [10.6](./06_langsmith_integration.md) deliberately omits prompt/output text and preserves typed failures. Its spans cannot establish actual request counts, tokens, or cost. Real services must reconcile provider usage, caching, retries, and billing; absent fields remain unknown. Trace inspection and business acceptance are separate.

Before delivery, add task-required controls rather than checking framework names:

- Frozen cases/expectations locate normal, unanswerable, denied, and budget-failure outcomes.
- Side effects, cancellation, and cleanup agree; missing mandatory artifacts prevent completion.
- Cache keys, source/permission versions, expiry, and revocation agree.
- Logging/tracing data scope is explicit; input cleaning alone does not establish prompt-injection protection.
- A service separately needs authentication, disconnects, health checks, graceful shutdown, and actual recovery tests; local tests here cover none of those.

## Run and Inspect Counterexamples

```bash
python -m reference_code.langchain_patterns
python -m unittest examples.integration_tests.test_langchain_chapter -v
```

Integration tests use real Runnables and fixed providers, intercept network entry points, and require zero requests. They cover structured rejection, branch failure, selected retries, interrupted streams, cache separation, local records, and coroutine/thread cancellation differences. They establish neither online model quality, service availability, nor performance gains.

Exercise: fail the primary before and after its first chunk. Record fallback execution and the deliverable state. Change the source revision and inspect caching. Deliver event sequences, actual attempt counts, and independent acceptance results. Without persistent-state needs, stop at this verified boundary; read Chapter 11 when recovery points are required.

*Previous: [10.7 LangChain Ecosystem 2026](./07_langchain_ecosystem_2026.md)*

*Next: [Chapter 11 LangGraph](../chapter_langgraph/README.md)*
