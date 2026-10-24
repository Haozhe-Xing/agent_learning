# 15.5 Comparison: DeepSeek Harness vs Claude Code / OpenClaw / Hermes

Start a comparison by identifying the layer to replace and the task it must satisfy. Chat entry points, coding products, cross-session memory, and component extension address different questions. A feature list does not put them in a meaningful ranking.

This section compares the teaching focus of four cases. Verify concrete capabilities in a specified version and configuration. Keep unexecuted items unknown instead of treating an overview as product validation.

## Separate the Layers and Experiment Subjects

| Case | Observation layer in this book | Check in a fixed task | Skip when |
|---|---|---|---|
| [12, OpenClaw](../chapter_openclaw/README.md) | Chat entry points and sessions in a personal-assistant product | Choose one channel; inspect message identity, session scope, allowed tools, and failures | Chat access is unnecessary or access conditions cannot be met |
| [13, Hermes](../chapter_hermes/README.md) | Product/architecture case for memory, skill candidates, and scheduling | Compare enabled and disabled behavior on held-out tasks; verify deletion, disabling, and rollback | Long-term accumulation is unnecessary or improvement cannot be judged independently |
| [14, Claude Code](../chapter_claude_code/README.md) | Code operations, context, and authorization in a coding product | Fix one defect; inspect the patch, real tests, edit scope, and failure records | The task is unrelated to coding or execution cannot be reproduced |
| [15, DeepSeek Harness](./README.md) | Architecture case for replaceable components and plugin lifecycle | Specify one replacement and inspect inputs, outputs, state, permissions, and failure cleanup | Replacement is unnecessary or maintenance costs exceed benefits |

A product is the subject used in an experiment. An architecture case is a way of reading and borrowing a design. One subject can serve both purposes, but understanding a diagram does not establish acceptance on the reader's task.

If the problem is repeated model and tool adapters, consider [chapter 10, LangChain](../chapter_langchain/README.md). For explicit state, branching, and recovery, consider [chapter 11, LangGraph](../chapter_langgraph/README.md). Libraries can work with the reader's host implementation; selecting a case does not require installing four products.

## Keep the Task and Conditions Fixed

For the existing evidence-and-calculation task, retain the [continuing project's](../learning_project.md) inputs, expectations, sources, permissions, budgets, and independent acceptance. Change only the layer under study. For example, a chat-entry comparison preserves the backend calculation contract; a component comparison preserves inputs, outputs, and failure criteria.

| Condition | Fix before comparison | Retain after execution |
|---|---|---|
| Task | One input and expected result, including missing evidence, conflicts, and unauthorized actions | Artifacts, execution status, and independent acceptance verdict |
| Permissions | The same identity, data access, allowed tools, and confirmation rules | Allowed and denied actions, including where checks occur |
| Model mode | Fixed outputs or a live model; record the model and parameters for live operation | Separate model and tool call counts, duration, and errors |
| Lifecycle | The same budget, stop rules, and specified fault points | Retry, cancellation, recovery, or cleanup records; identify uncovered behavior |
| Version | Exact product or source version, configuration, and environment | Reproduction steps, dependency differences, and unmatched conditions |

For coding tasks, fix the defective repository and unchanged acceptance tests. For memory or skill experiments, hold out tasks from candidate generation. Unrelated demonstrations cannot be assembled into a controlled performance comparison.

The book supplies no uniform offline adapter for these four products. The evidence-and-calculation fixture checks its own contracts and orchestration without calling them. The two existing library companions also use different tasks. Port the same contract and actually run each candidate experiment. Mark an unexecuted candidate “not run,” and mismatched conditions “not directly comparable.”

## Similar Extension Files Do Not Establish Interchangeability

A shared filename such as `SKILL.md` does not establish compatibility. A skill migration needs checks for:

- Content and metadata parsing, including ignored and rejected fields.
- Tool names, arguments, results, and errors.
- Execution environment, credential scope, and paths.
- The layer enforcing authorization, confirmation, timeouts, and failure handling.
- The original independent acceptance cases in the target environment.

Likewise, configuring a model does not establish its tool protocol, streaming errors, or recovery behavior. Loading a plugin does not establish that any kernel can be replaced. Limit compatibility conclusions to the versions, interfaces, and cases actually verified.

Using several products separately on one machine establishes separate use. Sharing memory, tools, or skills still requires data ownership, synchronization interfaces, conflict handling, and permissions. Coexistence is not evidence of integration.

## Record Adoption or a Decision to Skip

| Decision field | Record |
|---|---|
| Problem | A reproducible failure or maintenance difficulty in the existing implementation |
| Candidate and scope | Entry point, state mechanism, component replacement, or a product completing an independent task |
| Evidence | Successes and failures on the same task, with version, configuration, calls, and permissions |
| Decision | Adopt one verified change, retain the baseline, or investigate one explicit unknown |
| Stop and rollback | Stop migration and preserve existing artifacts when correctness, permissions, data, or maintenance requirements fail |

Skip migration when it addresses no current maintenance problem. Retain the minimal implementation when a candidate fails acceptance. One successful demonstration does not establish suitability for other tasks or future versions; changed versions and configurations need relevant acceptance checks again.

Carry this record to [section 15.7](./07_summary_and_decision_matrix.md) and use the same order as the [Part III choice guide](../part3.md): problem, layer, candidate, fixed task, and stop conditions.

---

*Next section: [15.6 Borrowing the Philosophy](./06_lessons_philosophy.md)*
