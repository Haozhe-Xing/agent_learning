# Bounded planning: a local release brief

Run from the repository root with Python 3.11+. These exercises use only the
standard library and the existing `evidence_assistant.planning` queue. No model,
network, arbitrary code execution, or publication/write tool is involved.

```bash
python -m reference_code.planning_patterns plan
python -m reference_code.planning_patterns react
python -m reference_code.planning_patterns revise
python -m reference_code.planning_patterns stubborn
python -m unittest examples.integration_tests.test_planning_chapter -v
```

The task gathers three fixture observations (approval record, budget, rollback)
and prepares an attributed brief. Its frozen requirements are independent of
the writer. An approval *record in a source* is data, not authority to publish.
The `stubborn` case exits 1; the other three default cases pass the small fixed
contract and exit 0. This is not a quality benchmark or a paper reproduction.

`workflow.py` contains a bounded DAG validator, two registered operations,
independent source/goal checks, and two execution styles. `run_plan` reuses the
shared bounded executor. The exercise retains one latest source snapshot;
composition rejects it unless the producing load is a declared dependency
ancestor. Multiple loads are not automatically merged. A replacement plan
discards its old pending queue and
invalidates all prior artifacts; failed attempts remain charged to the total
step limit. Reading sources again also invalidates the prior brief and its
acceptance. `run_react` accepts one structured action or an explicit finish
signal, with separate decision and tool-call limits. Finish still requires
independent checks. A reflection is only feedback for another proposal.

Plans contain at most 12 nodes; step/decision/call budgets are integers up to
30 and replanning is limited to 0..3. Source observations and candidate briefs
are bounded at 8192 serialized bytes each, feedback at 4096 bytes. These limits
are not model-token accounting or wall-clock cancellation. Callbacks are trusted
local functions; the exercise does not isolate hostile code or interrupt a
blocking callback. New external adapters need authorization, deadlines,
cancellation and side-effect idempotency of their own.

The report preserves `status`, `accepted`, candidate, verification details,
actual public actions/observations, and budget usage/limits. It never requests
or stores a model's private chain of thought. General language planning, HTN,
MCTS, durable recovery, and automatic source discovery are not implemented.
