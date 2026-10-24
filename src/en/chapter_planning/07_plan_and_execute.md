# 5.6 Plan-and-Execute and Test-time Compute Scaling

Combine the existing bounded queue with 5.3's dependencies and 5.4's limited revisions. Record execution separately from acceptance. Extra computation is useful only if it improves outcomes under the same contract and total budget; pattern names do not establish a ranking.

## Queue Completion Is Not Accepted Delivery

In `reference_code/evidence_assistant/planning.py`, `completed` means the queue ended without a tool exception, not that its content passed. The original `examples/chapter05/plan_and_execute.py` entry and queue tests remain. This runnable example distinguishes the outcomes:

```python
from reference_code.evidence_assistant.planning import PlanAndExecuteAgent, Step
from reference_code.planning_patterns import FACTS, check_brief

def planner(task):
    return [Step("proposal")]

def proposal():
    return {"budget": {"value": "300 units", "source": "budget-v1#release"}}

execution = PlanAndExecuteAgent(planner, {"proposal": proposal}, max_steps=1).run("brief")
assert execution["status"] == "completed"  # the tool returned, not task acceptance
candidate = execution["trace"][0]["result"]
assert check_brief(candidate, FACTS)["status"] == "failed"
```

`planning_patterns.run_plan` reuses that queue instead of copying another same-named Agent. Its wrapper validates plans and observations and performs independent checks before reporting `status/accepted`. From the repository root:

```bash
python -m reference_code.planning_patterns plan
python -m reference_code.planning_patterns revise
python -m reference_code.planning_patterns stubborn
python -m unittest examples.integration_tests.test_planning_chapter -v
```

These demonstrate direct acceptance, acceptance after revision and persistent missing requirements. The last case exits 1. Reports retain candidates, sources, missing/unsupported points, public actions and actual budgets.

## Bounded Queues and Replacement Contracts

| Budget | Meaning here | Not a guarantee of |
|---|---|---|
| `max_steps` | Whole-task tool attempts, including failures and replacement work; integer 1..30 | Tokens or elapsed seconds |
| `max_replans` | Replacement opportunities after requirement failure; integer 0..3 | Safe reuse of unchecked old artifacts |
| Plan size | At most 12 nodes per version, validated DAG/goal mapping | HTN planning or parallel execution |
| Observation/candidate/feedback size | 8192/8192/4096 serialized bytes | Complete provider request-token accounting |

Replace the pending queue rather than rebinding a list within `for enumerate(plan)`. This exercise invalidates artifacts and reruns dependencies conservatively, charging failed actions to the total. Empty replacement plans, invalid feedback, read errors and denied access cannot become success; alternative tools never bypass permission denial.

```python
from reference_code.planning_patterns import OmitRollbackOnce, run_plan

result = run_plan(writer=OmitRollbackOnce(), max_steps=3, max_replans=1)
assert result["status"] == "budget_exhausted"
assert result["accepted"] is False
assert result["budgets"]["steps_used"] == 3
assert result["verification"]["status"] == "not_started"

try:
    run_plan(max_replans=True)
except ValueError:
    print("a Boolean is not a replanning budget")
else:
    raise AssertionError("invalid budget was accepted")
```

Three attempts cover the first read, failed composition and reread. No budget remains to form a new candidate, so old verification cannot establish completion. Budgets are validated before execution; Booleans, floats and negative values are not valid counts.

## Choose by Lifecycle, Not Pattern Labels

Known dependencies favor validating a complete small plan. New observations that determine the next action can use 5.2's structured reactive decisions. Both require state, permission, budgets and independent acceptance. ReAct can maintain plans, and plan-based systems can adapt to observations; they are not mutually exclusive product stacks.

Here reactive revision reuses loaded observations while queue replacement deliberately reloads them, exposing a cost tradeoff. This is not a controlled model quality or latency experiment. Evaluate state graphs when recovery/collaboration requires them, as in [Chapter 11](../chapter_langgraph/README.md). Compiling a graph supplies neither disk persistence nor exactly-once effects automatically.

## Test-Time Compute: Compare Within a Budget

[Snell et al., v1](https://arxiv.org/abs/2408.03314v1) study inference-compute allocation under specified models, difficulty and compute-matched conditions. This does not establish that more reasoning always beats a larger model. Generation, search, checking and retries all cost resources; incorrect feedback can degrade results.

| Strategy | Potential contribution | Separate measurements |
|---|---|---|
| Finite candidates and independent checks | More opportunities to find an acceptable artifact | Total generation/scoring calls, correlated errors, rejection rates |
| Feedback revision | Another proposal after an explicit gap | Fixed gaps, new errors and total budget |
| Tree search/MCTS | Search over defined states/actions/rewards | Branch/depth/node/time limits, biased rewards and final acceptance |

This section implements no MCTS, model difficulty classifier or reasoning-token API. An unfinished skeleton is not a complete searcher, and the most visited path is not necessarily optimal. Provider reasoning settings vary by model/SDK; there is no universal “16000+ tokens” instruction or benefit from indiscriminately adding all history.

Exercise: hold sources and requirements fixed while comparing direct writing, one feedback revision and stubborn failure. Record attempts, gaps, input scale and stop reasons. Real-model comparisons need independent questions, provider usage and timing; fixed fixture call counts establish neither price nor performance rankings.

*Next: [5.7 Paper Readings: Frontiers in Planning and Reasoning](./06_paper_readings.md)*
