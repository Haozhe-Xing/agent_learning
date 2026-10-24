# 5.6 Plan-and-Execute and Test-time Compute Scaling

> **Section Objectives**: Master the architecture and implementation of the Plan-and-Execute pattern, and understand how Test-time Compute Scaling transforms the reasoning paradigm.

---

## From ReAct to Plan-and-Execute: The Evolution of Reasoning Paradigms

The ReAct pattern introduced earlier (Section 5.2) lets the Agent "think while doing" — reasoning and acting simultaneously at each step. However, for complex tasks, this "one step at a time" strategy leads to two problems:

1. **Short-sightedness Trap**: The Agent only focuses on the next action, lacking a global perspective, and easily walks into dead ends
2. **Context Bloat**: Every step includes thought + action + observation, and the context window is quickly consumed over long chains

The **Plan-and-Execute** pattern decouples "planning" from "execution":

> **ReAct Pattern**: Think1 → Act1 → Observe1 → Think2 → Act2 → Observe2 → ... → Answer (decisions made at each step, easy to go astray)
>
> **Plan-Execute**: Planner generates a complete plan → Executor executes step by step → Replan when deviations occur (see the whole picture first, then execute details)

> 📄 **Background**: The Plan-and-Execute pattern was first proposed by LangGraph officially in 2024 as a recommended pattern. It combines the early HuggingGPT idea of "LLM as a task planner" with LangGraph's state graph architecture. By 2025–2026, this pattern had become the mainstream architectural choice for production-grade Agents.

---

## Plan-and-Execute Architecture

### Core Components

Plan-and-Execute consists of two independent Agents:

- **Planner**: Uses a large model to generate a complete step-by-step plan
- **Executor**: Called individually for each step, may decide that replanning is needed
- **Replan Trigger**: Determines whether the current step has severely deviated from expectations

```python
class PlanAndExecuteAgent:
    """Decoupled planning and execution: plan → execute → replan → synthesize."""

    def __init__(self, model: str = "gpt-4.1", max_replans: int = 3):
        self.model = model
        self.max_replans = max_replans

    def run(self, task: str) -> str:
        plan = self._plan(task)
        executed, replan_count = [], 0

        for i, step in enumerate(plan):
            result = self._execute_step(step, executed)
            executed.append({"step": step, "result": result})

            # Key: detect deviations during execution and trigger replanning
            if self._should_replan(task, executed, plan[i+1:]) \
               and replan_count < self.max_replans:
                replan_count += 1
                plan = plan[:i+1] + self._replan(task, executed, plan[i+1:])

        return self._synthesize(task, executed)

    def _plan(self, task: str) -> list[dict]:
        """Let the LLM break the task into a structured plan of 3–8 steps."""
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"""You are a task planning expert.
Task: {task}
Break the task into 3–8 executable steps. Each step must include description/tool/expected_output.
Return as a JSON array. Only generate the plan, do not execute."""}],
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("steps", result.get("plan", []))

    def _execute_step(self, step: dict, history: list[dict]) -> str:
        """Execute a single step: send the current step + historical results to the LLM."""
        history_text = "\n".join(
            f"- {h['step']['description']}: {h['result'][:200]}" for h in history
        )
        prompt = f"""Please execute the following step.
Step: {step['description']}
Tool: {step.get('tool', 'General Reasoning')}
Expected Output: {step.get('expected_output', '')}
Completed:
{history_text or '(This is the first step)'}
Please execute directly and return the result."""
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

    def _should_replan(self, task, executed, remaining) -> bool:
        """Use a small model to determine whether the most recent step has severely deviated from expectations."""
        if not remaining:
            return False
        last = executed[-1]
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",   # Use a small model for judgment to save costs
            messages=[{"role": "user", "content": f"""Determine whether the execution result has deviated from expectations.
Step Objective: {last['step']['description']}
Expected Output: {last['step'].get('expected_output', '')}
Actual Result: {last['result'][:500]}
If it has severely deviated (e.g. execution failed, retrieved incorrect data), answer YES, otherwise NO."""}],
            max_tokens=10
        )
        return "YES" in resp.choices[0].message.content.upper()

    def _replan(self, task, executed, old_remaining) -> list[dict]:
        """Replan the remaining part based on previously executed steps."""
        # Actual: send history + original remaining steps together to the LLM to generate a new plan
        # See the repository for the full implementation
        ...

    def _synthesize(self, task, executed) -> str:
        """Synthesize all step results to produce the final answer."""
        history_text = "\n".join(
            f"Step {i+1} - {h['step']['description']}:\n{h['result'][:300]}"
            for i, h in enumerate(executed)
        )
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"""Answer based on the following step results:
Task: {task}
Results:
{history_text}"""}]
        ).choices[0].message.content
```

> 📦 **See the repository for the complete code** `examples/chapter05/plan_and_execute.py`, which includes the full `_replan` implementation, debug logging, and error recovery.

### Three Key Design Decisions in Plan-and-Execute

| Decision | Choice | Rationale |
|---|---|---|
| **Decouple planning and execution** | Required | Global planning avoids short-sightedness; stepwise execution keeps details controllable |
| **Use a small model for the replan trigger** | `gpt-4.1-mini` | Determining "whether it deviated" is a simple classification task that doesn't need a large model |
| **`max_replans` hard limit** | 3 times | Prevents the task from running out of control; exceeding 3 means the problem itself needs re-decomposition |

---

## Implementing Plan-and-Execute with LangGraph

In production environments, it is recommended to use LangGraph (Chapter 13) to implement Plan-and-Execute — it natively supports state graphs, conditional routing, and loops:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class PlanExecuteState(TypedDict):
    task: str
    plan: list[dict]
    current_step: int
    executed: Annotated[list, operator.add]   # Append mode
    replan_count: int

def should_replan_or_continue(state) -> str:
    """Conditional routing: continue / replan / synthesize."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    last = state["executed"][-1] if state["executed"] else {}
    if state.get("replan_count", 0) < 3 \
       and ("failed" in last.get("result", "") or "error" in last.get("result", "")):
        return "replan"
    return "execute"

# Build the graph with nodes and edges
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", plan_step)
graph.add_node("executor", execute_step)
graph.add_node("replanner", replan_step)
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges(
    "executor", should_replan_or_continue,
    {"execute": "executor", "replan": "replanner", "synthesize": END}
)
graph.add_edge("replanner", "executor")
app = graph.compile()
```

**Advantages of LangGraph**: Serializable state (supports checkpointing and resumption from breakpoints); graph structure natively supports complex branching; visual debugging.

---

## Test-time Compute Scaling: Dynamically Scaling Computation at Inference Time

### Core Idea

In 2024–2025, reasoning models (o1/o3/DeepSeek-R1) revealed a profound discovery: **investing more computation at inference time is more effective than training larger models**.

```
Training-time Scaling: larger models + more data = stronger capabilities (cost grows exponentially)
Inference-time Scaling: same model + more reasoning steps = stronger results (invest on demand)
```

### Three Test-time Compute Strategies

| Strategy | Principle | Representative Implementations | Suitable For |
|---|---|---|---|
| **Search-based Reasoning** | Generate multiple reasoning paths, search for the optimal solution | Tree of Thoughts, MCTS, LATS | Decision problems with clear evaluation criteria |
| **Self-correction** | Generate draft → self-critique → revise → repeat | Self-Refine, CRITIC | Tasks with objective verification (code, math) |
| **Extended Chain-of-Thought** | Let the model generate longer reasoning chains | o1/o3, R1, Claude Extended Thinking | Complex reasoning tasks |

### Hands-on: Adaptive Reasoning Depth

Not all problems require deep reasoning. A good Agent should **automatically adjust based on problem difficulty**:

```python
class AdaptiveReasoningAgent:
    """Classify by difficulty: simple → direct answer, medium → CoT, hard → multi-path search."""

    THRESHOLDS = {
        "simple": {"max_tokens": 500, "strategy": "direct"},
        "medium": {"max_tokens": 2000, "strategy": "cot"},
        "hard":   {"max_tokens": 8000, "strategy": "search"},
    }

    def run(self, question: str) -> dict:
        # 1. Quickly assess difficulty
        difficulty = self._assess_difficulty(question)
        cfg = self.THRESHOLDS[difficulty]
        # 2. Choose strategy based on difficulty
        if cfg["strategy"] == "direct":
            answer = self._direct(question)
        elif cfg["strategy"] == "cot":
            answer = self._cot(question, cfg["max_tokens"])
        else:    # search
            answer = self._multi_path_search(question, cfg["max_tokens"])
        return {"question": question, "difficulty": difficulty, "answer": answer}

    def _assess_difficulty(self, question: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": f"""Assess question difficulty.
Question: {question}
Criteria: simple = factual Q&A / simple calculation; medium = multi-step reasoning / comprehensive analysis; hard = creative thinking / complex proof
Answer with simple/medium/hard only."""}],
            max_tokens=10
        )
        result = resp.choices[0].message.content.strip().lower()
        return result if result in self.THRESHOLDS else "medium"

    def _direct(self, q: str) -> str:
        return client.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": q}], max_tokens=500
        ).choices[0].message.content

    def _cot(self, q: str, max_tokens: int) -> str:
        return client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": f"{q}\nPlease think step by step."}],
            max_tokens=max_tokens
        ).choices[0].message.content

    def _multi_path_search(self, q: str, max_tokens: int) -> str:
        # Generate 3 solutions → LLM selects the best
        paths = [
            client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": f"{q}\nSolve using method {i+1}"}],
                max_tokens=max_tokens // 3
            ).choices[0].message.content
            for i in range(3)
        ]
        # Comprehensive evaluation (see the repository for full implementation)
        ...
```

> 📦 **See the repository for the complete code** `examples/chapter05/adaptive_reasoning.py`, which includes the "compare + select" logic for multi-path search.

---

## MCTS in Agent Reasoning

**Monte Carlo Tree Search (MCTS)** is the core algorithm of AlphaGo, introduced to LLM reasoning by the LATS paper:

> 📄 **Paper Source**: *"Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models"* (Zhou et al., 2024, arXiv:2310.04406). LATS models the LLM's reasoning process as a search tree — each node is a "thought state" and each edge is a "reasoning step." MCTS searches this tree to find the optimal path from the initial state to the goal state.

```python
class MCTSNode:
    """Search tree node with UCB1 score."""
    def __init__(self, state: str, parent=None):
        self.state = state; self.parent = parent
        self.children = []; self.visits = 0; self.value = 0.0
        self.action = ""

    @property
    def ucb1(self) -> float:
        """UCB1 balances exploration and exploitation."""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSReasoningAgent:
    """Standard MCTS: Selection → Expansion → Simulation → Backprop."""

    def search(self, problem: str) -> str:
        root = MCTSNode(state=problem)
        for _ in range(self.max_iterations):
            node = self._select(root)         # Selection
            if node.visits > 0 and not self._is_terminal(node):
                self._expand(node, problem)   # Expansion
                if node.children:
                    node = random.choice(node.children)
            reward = self._simulate(node, problem)   # Simulation
            self._backpropagate(node, reward)  # Backpropagation
        return self._extract_best_path(root)  # Most-visited path = optimal solution

    # _select/_expand/_simulate/_backpropagate — four ~15-line methods following the standard MCTS flow
    # See the repository for the complete code
```

> 📦 **See the repository for the complete code** `examples/chapter05/mcts_agent.py`.

---

## The Era of Reasoning Models: How o1/o3 Transforms Agent Development

The emergence of reasoning models (o1/o3/DeepSeek-R1/Claude Extended Thinking) fundamentally changes how Agents are developed:

| Traditional Agent | Reasoning Model Agent |
|---|---|
| Requires carefully designed CoT prompts | Concise instructions work better |
| Thinking tokens are limited | Let the model decide its own reasoning depth |
| Context utilization limited by the model | The more context, the better |
| Tool calls planned by the LLM itself | Reasoning model handles planning, small model handles execution |

**Best Practices**:
1. **Simplify the System Prompt**: Reasoning models do not need detailed CoT instructions
2. **Let the model decide its own reasoning depth**: Set `max_tokens` to 16000+, only limit in time-sensitive scenarios
3. **Provide rich context**: Put relevant documents, history, and constraints all into the prompt
4. **Use as the Agent's "brain"**: Reasoning model for planning, small models for tool calls and simple tasks

```text
Architecture: Reasoning Model (Plan) → Small Model (Execute) → Reasoning Model (Verify)
```

---

## Comparison of Three Reasoning Patterns

| Dimension | ReAct | Plan-and-Execute | Test-time Compute |
|---|---|---|---|
| Core Idea | Think while doing | Plan first, then execute | Think more for hard problems, answer quickly for easy ones |
| Planning Depth | Single step | Global | Adaptive |
| Context Consumption | High (every step is verbose) | Medium (planning and execution are separated) | Controllable (adjusted by difficulty) |
| Suitable Scenarios | Simple interactive tasks | Multi-step complex tasks | Mixed tasks with widely varying difficulty |
| Error Recovery | Difficult (prone to loops) | Good (can replan) | Good (multi-path search) |
| Representative Implementations | LangChain Agent | LangGraph PlanExecute | o1/o3, MCTS, LATS |

> 💡 **Selection Guide**:
> - **Rapid Prototyping**: ReAct — simple to implement, complete code already in Section 3.2
> - **Production**: Plan-and-Execute — global perspective + flexible replanning
> - **High-difficulty Reasoning**: Test-time Compute + MCTS — multi-path search for optimal solutions
> - **Mixed Scenarios**: Adaptive Reasoning — automatically switch strategies based on difficulty

---

## Summary

| Concept | Description |
|---|---|
| Plan-and-Execute | Generate a complete plan first, then execute step by step, replan when deviations occur |
| Test-time Compute Scaling | Investing more computation at inference time is more efficient than training larger models |
| Adaptive Reasoning | Automatically adjust reasoning depth and strategy based on problem difficulty |
| MCTS Reasoning | Model reasoning as a search tree, find the optimal path via Monte Carlo search |
| Reasoning Models | o1/o3/R1 internalize CoT into the model, simplifying Agent prompt design |

> 📖 **Further Reading**:
> - LangGraph. "Plan-and-Execute Agent." LangGraph Documentation, 2024.
> - Zhou et al. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models." arXiv:2310.04406, 2024.
> - Snell et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters." ICLR, 2025.
> - OpenAI. "Learning to Reason with LLMs." OpenAI Blog, 2024.

---

*Next Section: [5.7 Paper Readings: Frontiers in Planning and Reasoning Research](./06_paper_readings.md)*
