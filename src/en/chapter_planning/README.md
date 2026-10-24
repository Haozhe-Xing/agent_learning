# Chapter 5: Planning and Reasoning

**Reading level**: Core; papers and search extensions are optional. Plans, reflections and public explanations are inspectable data, not private reasoning logs or independent factual evidence.

## This chapter's iteration

- **Previous limitation**: Chapter 4 retains preferences and task state without enforcing step dependencies, goal coverage or invalidation after failure. Text generation or queue completion can still be mistaken for delivery.
- **This chapter adds**: A release-brief task exercises structured actions, DAG validation, goal mapping, feedback revision and total budgets. Reuse the existing queue while independently checking approval records, budget and rollback rather than trusting self-scores.
- **How to verify**: Run `python -m reference_code.planning_patterns revise`, `python -m reference_code.planning_patterns stubborn` and `python -m unittest examples.integration_tests.test_planning_chapter -v`. Retain plans, public actions/observations, missing points, budgets and the verdict. Fixed adapters neither understand arbitrary language nor publish anything.
- **Next step**: A valid plan can still use incorrect evidence. In [Chapter 6](../chapter_rag/README.md), add loading, retrieval and context, then check source support and coverage separately. The [shared project](../learning_project.md) already combines multiple documents/questions.

## Complete One Bounded Task First

The brief needs attributed approval, a 30-unit budget and restoration of the release-1.0 snapshot. `reference_code/planning_patterns/` uses the standard library without models/network. A fixed writer can omit points or revise after feedback. An approval record is data, not permission to publish. No arbitrary code, shell or file-writing tool is present.

```bash
python -m reference_code.planning_patterns plan
python -m reference_code.planning_patterns react
python -m reference_code.planning_patterns revise
python -m reference_code.planning_patterns stubborn
```

The first three defaults pass this fixed contract. Persistent omission returns `budget_exhausted/accepted=false` and exits 1. `trace` preserves actual public actions, observations and errors. `verification` separates missing points, unsupported claims and unmet frozen requirements. Execution ending is not business completion.

The existing `examples/chapter05/plan_and_execute.py` and `reference_code.evidence_assistant.tests.test_learning_path.PlanningTests` remain for queue-replacement practice. Their `execution.completed` is not artifact acceptance. The new wrapper adds dependency/goal contracts without modifying the shared core.

## Chapter Structure and Deliverables

| Section | Exercise | Deliverable |
|---|---|---|
| [5.1 How Do Agents “Think”?](./01_how_agents_think.md) | Public plans, observations, assumptions and evidence | Task contract and stop reasons |
| [5.2 ReAct: Reasoning + Acting](./02_react_framework.md) | Structured actions, host observations and finish proposals | Normal/invalid-action/premature-finish traces |
| [5.3 Task Decomposition](./03_task_decomposition.md) | Bounded dependencies and declared coverage | DAG, goal map and rejected plans |
| [5.4 Reflection and Self-Correction](./04_reflection.md) | Feedback guides revision while acceptance stays independent | Missing-point repair versus false self-approval |
| [5.5 Research Assistant Practice](./05_practice_research_agent.md) | A separate smaller QA interface exercise | Configured verifier and citation boundaries |
| [5.6 Plan-and-Execute and Test-time Compute Scaling](./07_plan_and_execute.md) | Queue replacement, total budgets and compute tradeoffs | Failure states, attempts and comparison records |
| [5.7 Paper Readings](./06_paper_readings.md) | Optional research with original settings | Differences from this project, not reproduced scores |

The smaller 5.5 API uses a different fixture. Its default citation-label checks do not establish full source support/coverage. Run `python -m reference_code.evidence_assistant` for the complete local-document application, whose default tasks have separate frozen specifications. Neither fixture's passes establish model quality.

## Prerequisites and Stopping Conditions

Prerequisites: [Chapter 3 tools](../chapter_tools/README.md), [Chapter 4 memory](../chapter_memory/README.md), Python functions and dictionaries. Defaults need no SDK account/framework. Close dependencies, sources and independent checks first, then use [Chapter 10 components](../chapter_langchain/README.md) or [Chapter 11 state/recovery](../chapter_langgraph/README.md) when needed. Neither is required here.

Only passing the fixed task's independent requirements establishes completion. Steps, decisions, replans and bytes have different units; none automatically cancels blocking callbacks or reverses effects. Durable recovery, real models, parallel scheduling, HTN/MCTS and production publishing are outside this small companion's implemented scope.

*Next: [5.1 How Do Agents “Think”?](./01_how_agents_think.md)*
