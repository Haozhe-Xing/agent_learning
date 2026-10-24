# 3.1 Why Do Agents Need Tools?

A model can write “looked up” or “calculated” without any operation occurring. Tools first provide inspectable actions and observations: retrieved sources, actual arguments, function results, and identifiable failures. Task needs determine whether a tool is useful; availability alone does not justify calling it.

## Choose Capabilities from the Missing Requirement

| Missing capability | Possible tool | Remaining checks |
|---|---|---|
| Required material is absent from current input | Local retrieval or authorized data API | Source, version, visibility, empty results |
| Numbers need defined operations | Bounded arithmetic or domain calculator | Units, precision, input range, independent expectation |
| An external action is required | Authorized write interface | Resource scope, confirmation policy, idempotency, receipt |
| Rewriting supplied text only | Possibly no tool | Output requirements |

Model knowledge can be stale or lack private task material. Tools introduce observations, but retrieved information can also be wrong. A proposed call supplies neither connectivity, credentials, nor database privileges; the host accesses resources.

## An Actual Function Observation

```python
import json
from reference_code.tool_calling import ToolHost

host = ToolHost(allowed_tools=("calculate",))
observation = host.execute("calculate", json.dumps({"expression": "12345 * 67890"}))
assert observation == {"ok": True, "value": {"value": 838102050}}
print(observation)
```

The shared bounded interpreter computes 838102050. This establishes the result for that expression in this implementation, not correct business inputs or currency units. Tool selection, argument provenance, execution, and task acceptance remain separate.

## Three Records: Proposal, Execution, Acceptance

1. **Proposal**: The model requests `calculate` with JSON text; neither name nor arguments is trusted yet.
2. **Execution**: The host checks registration, current permission, argument types/values, and remaining budgets before invoking a function.
3. **Acceptance**: Independent requirements check final values, units, sources, and omissions; a normal tool return does not complete the task.

Calling a tool “read-only” does not enforce read-only access. Files, SQL, and messages need resource authorization and operation-specific execution limits. This chapter provides no arbitrary-path, shell, or sending tool.

## Failures Are Part of the Tool Contract

Bad arguments, permission denial, empty retrieval, timeouts, and failed acceptance mean different things. Empty text or an answer-shaped error string can obscure those distinctions. The companion uses `ok/code/value` objects and preserves an incomplete task state.

Record diagnostics by purpose. Call IDs, tool names, statuses, and necessary provenance aid inspection. Full arguments, pages, customer records, or exception messages may be sensitive and should not all be retained indefinitely. A visible fixture trace is not a production logging policy.

Exercise: define boundaries for “continue yesterday's release checks.” Identify required reads, calculations, and writes; list principal, maximum input, failure states, and completion evidence. Keep the minimum read-only path before adding the next section's protocol.

*Next: [3.2 Function Calling](./02_function_calling.md)*
