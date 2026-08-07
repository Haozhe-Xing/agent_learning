# 11.3 Agentic Data Flywheel: Letting Agents Improve Themselves

> 🔄 *"The best training data is not manually labeled in isolation; it is the trajectory left behind when Agents succeed, fail, recover, and complete tasks in real environments."*

---

## Why the Data Flywheel Matters

Agentic RL needs feedback from real tasks. But high-quality Agent data is expensive because it includes multi-step reasoning, tool calls, observations, failures, and final outcomes.

A data flywheel turns production usage into training improvement:

```text
Deploy Agent
    ↓
Collect trajectories
    ↓
Evaluate outcomes
    ↓
Filter and label data
    ↓
Train / fine-tune / optimize
    ↓
Deploy improved Agent
    ↓
Collect better trajectories
```

The more the Agent is used, the more opportunities you have to improve it—if the data pipeline is designed correctly.

---

## What Data Should Be Collected?

A useful Agent trajectory includes more than the final answer:

```python
from dataclasses import dataclass, field


@dataclass
class AgentTrajectory:
    task_id: str
    user_query: str
    steps: list[dict] = field(default_factory=list)
    final_answer: str = ""
    success: bool | None = None
    user_feedback: int | None = None
    cost: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
```

Each step should record:

- model input and output;
- tool name and arguments;
- tool observation;
- errors and retries;
- intermediate decisions;
- timestamps and token usage.

Without trajectories, you cannot train better Agent behavior. You can only train better final text.

---

## Feedback Sources

| Feedback Source | Signal | Strength | Weakness |
|----------------|--------|----------|----------|
| User rating | Satisfaction | Direct product signal | Sparse and noisy |
| Task outcome | Success / failure | Strong for objective tasks | Hard to define for open-ended tasks |
| Tool result | API success, test pass, validation result | Deterministic | May not reflect user value |
| Human review | Expert judgment | High quality | Expensive |
| LLM/Agent judge | Scalable quality scoring | Cheap and broad | Needs calibration |
| Runtime metrics | Cost, latency, retries | Always available | Not a direct quality signal |

A mature system combines multiple signals instead of relying on a single score.

---

## Data Filtering

Not every production trace should become training data. Filter aggressively:

```python
def is_training_candidate(trace: AgentTrajectory) -> bool:
    if trace.success is False and trace.user_feedback is None:
        return False
    if trace.cost > 5.0:
        return False
    if trace.metadata.get("contains_sensitive_data"):
        return False
    if len(trace.steps) == 0:
        return False
    return True
```

Recommended filters:

- remove sensitive or private data;
- remove corrupted traces;
- deduplicate near-identical tasks;
- keep both successes and informative failures;
- balance categories and difficulty levels;
- separate training, validation, and regression sets.

---

## Turning Trajectories into Training Data

Different training methods need different data transformations:

| Target Method | Data Format |
|--------------|-------------|
| SFT | Expert or corrected trajectories |
| DPO / preference tuning | Chosen vs rejected responses or trajectories |
| PPO / GRPO | Environment interaction with reward signals |
| Reward model | Trajectory + human or judge score |
| Evaluation set | Fixed tasks with expected outcomes |

Example preference pair:

```json
{
  "prompt": "Analyze this CSV and find anomalies.",
  "chosen": "Uses Python tool, validates columns, reports anomalies with evidence.",
  "rejected": "Guesses anomalies without reading the file.",
  "reason": "Chosen trajectory grounds the answer in tool output."
}
```

---

## Reward Design

Agent rewards should combine outcome quality with process quality.

```python
def compute_agent_reward(trace: AgentTrajectory) -> float:
    reward = 0.0
    if trace.success:
        reward += 1.0
    if trace.user_feedback is not None:
        reward += 0.2 * trace.user_feedback

    # Penalize inefficient or unstable behavior.
    reward -= 0.01 * len(trace.steps)
    reward -= 0.1 * count_tool_errors(trace)
    reward -= 0.001 * trace.cost

    return max(min(reward, 2.0), -1.0)
```

Reward design should be monitored carefully. A poorly designed reward can teach the Agent to optimize metrics while hurting real user value.

---

## Human-in-the-Loop Data Improvement

The most valuable traces are often failure cases. A reviewer can convert them into training data by:

1. identifying the first wrong decision;
2. correcting the tool call or reasoning step;
3. adding the corrected trajectory to SFT data;
4. adding the original failure as a rejected preference example;
5. adding the task to the regression suite.

```text
Failed trajectory
      ↓
Human correction
      ↓
SFT example + preference pair + regression test
```

This turns incidents into long-term capability improvements.

---

## Flywheel Governance

A data flywheel can be dangerous without governance:

- it may store private user data;
- it may amplify biased or low-quality behavior;
- it may train on prompt-injection traces;
- it may overfit to frequent but low-value tasks;
- it may silently degrade rare but important workflows.

Governance checklist:

- user consent and data retention policy;
- PII detection and redaction;
- safety filtering;
- dataset versioning;
- human review for high-risk domains;
- separate evaluation sets that are never trained on.

---

## Production Data Flywheel Architecture

```text
Agent Runtime
   ↓ emits traces
Trace Store
   ↓
Privacy Filter / Redaction
   ↓
Outcome Evaluator
   ↓
Data Selection
   ├── SFT Dataset
   ├── Preference Dataset
   ├── Reward Dataset
   └── Regression Dataset
   ↓
Training / Fine-Tuning
   ↓
Offline Evaluation
   ↓
A/B Test
   ↓
Production Rollout
```

The key principle is separation: raw traces, filtered data, training sets, and evaluation sets should be versioned independently.

---

## Best Practices

- Collect full trajectories, not only final answers.
- Treat production traces as raw material, not automatically trusted training data.
- Keep informative failures and human corrections.
- Use privacy filtering before any training pipeline.
- Build separate datasets for SFT, preferences, rewards, and regression.
- Version datasets, prompts, tools, models, and evaluation results together.
- Close the loop with A/B testing before production rollout.

---

## Chapter Takeaways

- An Agentic data flywheel converts real Agent usage into better training data.
- The most valuable data includes trajectories, tool calls, observations, failures, and outcomes.
- Feedback can come from users, tools, humans, judges, and runtime metrics.
- Data filtering and privacy governance are mandatory.
- A mature flywheel turns failures into SFT examples, preference pairs, reward data, and regression tests.

---

*Previous section: [11.2 Self-Evolution Agent](./02_self_evolution_agent.md)*

*Next chapter: [Chapter 12: LangChain In-Depth](../chapter_langchain/README.md)*
