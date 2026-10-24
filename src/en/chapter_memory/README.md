# Chapter 4: Memory Systems

**Reading level**: Core; papers and layered memory can wait.

## This chapter's iteration

- **Previous limitation**: The calculation works, but a new session may forget “use milliseconds from now on.” Persisting the entire history instead risks unauthorized writes and mixing users' data.
- **This chapter adds**: Store a consented, user-scoped `output_unit` preference and consume the retrieved value during conversion. Separate lasting preferences from the current task's working state.
- **How to verify**: `python -m reference_code.evidence_assistant.demo --stage memory --output-unit milliseconds` demonstrates memory-driven conversion. Run `python -m unittest reference_code.evidence_assistant.tests.test_learning_path.MemoryTests` for reopen persistence, isolation, deletion, and consent. Deliver storage records and a unit-bearing result. The shared stage uses a temporary SQLite file (removed when the stage ends); separate tests establish persistence. Semantic retrieval quality requires a real task set.
- **Next step**: Remembering “milliseconds” still cannot prevent skipped extraction or repeated calculation. In [chapter 5](../chapter_planning/README.md), track steps, dependencies, and completion criteria for the same task. Memory retains information; planning selects the next action and determines when to stop.

Memory consists of inspectable storage, access, and lifecycle mechanisms. Conversation history, current task state, and information authorized for future sessions have different roles. Persistent memory can be a structured database; it is not synonymous with a vector store.

## Chapter Goals

After the core exercises, you should be able to:

- Distinguish conversation history, working task state, and persistent memory by purpose and lifecycle.
- Preserve fixed rules, the current question, and complete turns within a byte budget, explaining how it differs from a model token budget.
- Exercise consent, owner isolation, reopening, and deletion with SQLite or explicitly supplied Chroma vectors.
- Retain provenance and independent verification status for intermediate artifacts, separating proposals from facts.
- Run the declared personal-assistant paths and identify the separate requirements for models, storage, and host authentication.

## Chapter Structure

| Section | Content | Difficulty |
|---|---|---|
| 4.1 Why Do Agents Need Memory? | What to retain and what not to write | ⭐⭐ |
| 4.2 Short-Term Memory | Complete turns, input byte budgets, and summary loss | ⭐⭐ |
| 4.3 Long-Term Memory | SQLite/Chroma mechanisms, explicit consent, isolation, deletion | ⭐⭐⭐ |
| 4.4 Working Memory: Scratchpad | Observable task state, provenance, independent checks | ⭐⭐⭐ |
| 4.5 Personal Assistant Practice | Memory contracts and companion entry points | ⭐⭐⭐ |
| 4.6 Paper Readings: Eight Design Views on Agent Memory | Same eight fixed texts, design tradeoffs, evaluation exercise; optional | ⭐⭐⭐ |
| 4.7 MemGPT/Letta Memory Practice | The book's layered baseline and lifecycle; optional second pass | ⭐⭐⭐⭐ |

## Reference Code and Scope

`reference_code/evidence_assistant/` contains the continuing project and SQLite memory. `reference_code/memory_rag/` supplies history, task-state, and vector exercises for 4.1–4.4. Start with `python -m reference_code.memory_rag history`, then choose `working`, `chunks`, or `vectors`. Vector mode requires pinned Chroma dependencies; fixed vectors establish database mechanics only.

Explain sources, permissions, budgets, and failures alongside results. Successful storage does not establish correct recall, and logical deletion does not establish backup erasure. Automatic summarization/preference extraction, real embedding quality, remote stores, and product SDKs require separate validation. No external memory product is a required dependency or selected using an unreplicated current ranking.

## Learning Path

Prerequisites: [Chapter 3: Tool Calling](../chapter_tools/README.md), Python lists/dictionaries, and basic file operations. Continue to [Chapter 5](../chapter_planning/README.md) for steps and stopping conditions, then [Chapter 6: RAG](../chapter_rag/README.md) for evidence recall, answer support, and coverage.

*Next: [4.1 Why Do Agents Need Memory?](01_why_memory.md)*
