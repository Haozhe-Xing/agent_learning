# Part II: Core Capabilities

Bring the task card and model baseline from chapters 1–2: read a distance source and return cited light-travel time in the requested unit, stopping when evidence is insufficient. Chapters 3–8 progressively add execution, memory, planning, evidence, context and host controls. Chapter 9 optionally packages a verified workflow as a skill.

The six calculation stages are foundational checks. For the moderately sized application, run `python -m reference_code.evidence_assistant` and observe local project documents becoming a cited report through retrieval, planning, and context. The tasks share mechanisms but have separate evidence and acceptance criteria. See the [reference project](./learning_project.md) for entry points and reading order.

| Chapter | Where the previous version fails | Artifact added | Keep this failure case |
|---|---|---|---|
| [3 Tools](./chapter_tools/README.md) | A model's numeric answer cannot be replayed | Tool inputs, results and call trace | Invalid expressions do not execute |
| [4 Memory](./chapter_memory/README.md) | A new session forgets the output unit | Consented, isolated and deletable preference | Another user cannot read it |
| [5 Planning](./chapter_planning/README.md) | Remembering milliseconds does not prevent a missing extraction or conversion step | Dependency order, execution state and stopping rules | Extraction alone is not task completion |
| [6 RAG](./chapter_rag/README.md) | A completed plan may still use an unsupported quantity | Retrieved text, source IDs and answer checks | Missing evidence, unknown citations and conflicts cannot pass |
| [7 Context](./chapter_context_engineering/README.md) | Growing history pushes out required evidence | Constructed messages, required items and measured budget | Exhaustion must be explicit, not a silent deletion |
| [8 Harness](./chapter_harness/README.md) | Good input organization cannot prevent unauthorized actions or self-reported success | Host permissions, call budget and independent acceptance | Denied actions never execute; wrong artifacts never pass |
| [9 Skills](./chapter_skill/README.md) | The same verified workflow is copied repeatedly | A skill contract for input, permissions, artifact and failure | Reuse must not expand caller permissions |

Memory retains background; planning controls task progress. They solve different problems. Context controls what the model sees; a Harness controls what may execute, when execution stops and who accepts the result.

Before continuing, run the [shared project](./learning_project.md) and preserve success, missing-evidence, wrong-unit, denied-action and exhausted-budget records. Each chapter distinguishes supplied runnable checks from exercises you implement yourself.

Choose [Part III](./part3.md) when integration code or state recovery needs justify it. Otherwise keep the small implementation and proceed to [evaluation](./chapter_20_evaluation/README.md). Training and self-improvement are in [Part VII](./part7.md); neither is a prerequisite for a first application.
