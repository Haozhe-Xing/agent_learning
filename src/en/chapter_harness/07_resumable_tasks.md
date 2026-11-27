# 8.7 Practice: Resume a Long-Running Task Across Processes

The Performance Factory, described in November 2026, stores the stage, findings, and next steps of a performance regression investigation in a durable task record. The agent proposes a fix, engineers decide what ships, and the agent checks recovery after deployment. Another run can pick up the work when a process or session ends. See the [original case study](https://openai.com/index/how-openai-performance-factory-keeps-chatgpt-fast/).

This offline exercise simulates publishing a report. SQLite records the job state, a command stands in for approval, and a separate query verifies the artifact. Each command starts a new process.

```bash
python examples/chapter08/resumable_task.py start --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db
python examples/chapter08/resumable_task.py approve --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db --stop-after-apply
python examples/chapter08/resumable_task.py run --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db
```

The expected states are `proposed → proposed → approved → applied → verified → verified`. The second command checks that execution waits for approval. The fourth simulates a process stopping after it commits the artifact and before verification. The final commands resume verification and retry the task. Only one artifact is stored.

```python
{{#include ../../../examples/chapter08/resumable_task.py}}
```

The artifact and its checkpoint share a transaction in this example. For effects in external systems, persist the operation identifier, supply an idempotency key, and reconcile the external result before retrying. Bind approval to the artifact revision and enforce it on an authenticated server. See [7.3, context strategies for long tasks](../chapter_context_engineering/03_long_horizon.md), and [13.5, human-in-the-loop](../chapter_langgraph/05_human_in_the_loop.md).
