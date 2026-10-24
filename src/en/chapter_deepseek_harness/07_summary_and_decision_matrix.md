# 15.7 Summary: Choose an Implementation with Task Evidence

Close Part III by returning to the existing assistant, identifying one concrete problem, and comparing candidates with fixed inputs. Keeping the minimal implementation, adopting a development library, and borrowing one mechanism from a case study can each be a complete conclusion.

## Six Chapters, Different Questions

| Problem | Layer | Candidate chapter | Fixed-task check | Skip or stop when |
|---|---|---|---|---|
| Repeated adapters are hard to maintain | Library | [10, LangChain](../chapter_langchain/README.md) | With identical inputs and permissions, are results, errors, and call traces consistent? | Maintenance does not improve or earlier acceptance cases break |
| Pause, recovery, confirmation, and branching are hard to track | Library | [11, LangGraph](../chapter_langgraph/README.md) | After the same failure point, does execution resume correctly without duplicate side effects? | A simple implementation already meets these requirements |
| An assistant needs a chat entry point | Product case | [12, OpenClaw](../chapter_openclaw/README.md) | Do identity, session, and tool boundaries hold in one specified channel? | Access conditions cannot be met or only a backend is needed |
| Cross-session memory and skill accumulation need study | Product/architecture case | [13, Hermes](../chapter_hermes/README.md) | Compare held-out tasks and verify review, disabling, and rollback | There is no long-term use case or independent improvement criterion |
| A coding assistant or code-operation case is needed | Product/architecture case | [14, Claude Code](../chapter_claude_code/README.md) | Inspect a fixed coding task's patch, real tests, authorization, and failures | The task is unrelated to coding or execution cannot be reproduced |
| Component replacement and plugin management need study | Architecture case | [15, DeepSeek Harness](./README.md) | Replace one component in a pinned version; inspect interfaces, state, and failure cleanup | Replacement is unnecessary or validation cost outweighs the benefit |

This is a reading guide organized by problem. LangChain and LangGraph can be combined, but are not a required consecutive stack. Select chapters 12–15 by scenario instead of installing every product. Product cases supply observable examples; architecture cases supply design questions. Neither establishes deployment suitability on its own.

## Run One Reproducible Comparison

Keep an offline baseline from the [continuing project](../learning_project.md):

```bash
python -m reference_code.evidence_assistant
```

The main application answers several questions from fictional project documents and records retrieved evidence, plan state, context budgets, and a cited report. It extracts original text without a live model or online search. Preserve the documents, questions, and acceptance requirements when comparing a candidate implementation.

Reuse inputs, expectations, tool permissions, and budgets in the candidate; change only the layer under study. Retain at least these comparisons:

| Fixed input or fault | Expected observation |
|---|---|
| Default project documents and three questions | Each finding has original evidence and sources; plan status and acceptance remain separate |
| A document's rollback requirement changes | The answer follows the new text instead of retaining old evidence |
| A question has no matching evidence | The finding retains a failure such as `no_evidence`; the whole task is incomplete and cannot pass |
| Required evidence exceeds the context byte budget | Explicit failure instead of silently omitting evidence and reporting success |
| Unauthorized tool or exhausted budget | A recorded stop reason without the prohibited action |
| Recovery or human confirmation is required | Interrupt at a specified point and inspect resumption, confirmation, and duplicate actions |

Include the final row only when relevant. The research application supplies neither durable recovery nor human confirmation; add candidate-specific experiments. For unit handling and numeric contracts, use the separate foundation exercise, `python -m reference_code.evidence_assistant.demo --all`. Success on these two different tasks is not a controlled comparison.

Existing library companions are `examples/chapter12/customer_service.py` and `examples/chapter13/content_workflow.py`; directory names retain historical chapter numbers. They use different tasks and fixed model outputs by default to demonstrate tool orchestration and state graphs. Success in each is not a controlled comparison. Chapters 12–15 also do not supply four uniform offline product adapters. Mark candidates that were not executed as “not run,” and experiments with unmatched conditions as “not directly comparable.”

## Write Down the Decision

| Field | Record |
|---|---|
| Problem | A reproducible maintenance or execution problem in the existing assistant |
| Baseline and candidate | Original implementation and the interface, state mechanism, or product under consideration; exact version and configuration |
| Fixed conditions | Inputs, expected results, model mode, tool permissions, budgets, and failure cases |
| Evidence | Test outcomes, call traces, sources, recovery records, duration, and call counts; identify anything not executed |
| Adopted change | The replaced layer and how business logic and acceptance criteria remain consistent |
| Decision and stop conditions | Adopt, retain the baseline, or investigate further; specific rollback triggers |

Model quality requires a separate representative task set. One successful demonstration, product description, or plugin interface cannot establish overall correctness or security boundaries. Keep publicly observable behavior, version-specific implementation details, and architectural inferences separate in the record.

## Continue from the Next Problem

When one Agent meets the requirements, proceed to regression gates, permissions, and deployment. Only when tasks can be assigned independently should [chapter 16, Multi-Agent Collaboration](../chapter_18_multi_agent/README.md) test collaboration against a single Agent. Read [chapter 17, Communication Protocols](../chapter_19_protocol/README.md) when interoperability crosses process or service boundaries. Judge protocol needs separately from the number of roles.

Use [18, Evaluation](../chapter_20_evaluation/README.md), [19, Security](../chapter_21_security/README.md), and [20, Deployment](../chapter_22_deployment/README.md) to develop the existing evidence and controls. Then choose a capstone in chapters 21–23: [coding](../chapter_23_coding_agent/README.md), [data analysis](../chapter_24_data_agent/README.md), or [multimodal work](../chapter_25_multimodal/README.md). Research extensions in chapters 24–25 follow the capstones and can be selected later according to research goals.

Completion evidence for this part is a reviewable decision record: the problem, chosen implementation, passing and failing cases, and conditions for stopping or reverting.

---

*Back to [chapter 15, DeepSeek Harness](./README.md), or continue as needed to [chapter 16, Multi-Agent Collaboration](../chapter_18_multi_agent/README.md).*
