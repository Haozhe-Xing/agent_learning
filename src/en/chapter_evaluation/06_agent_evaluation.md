# 18.6 Agent-Specific Evaluation Frameworks

> **Goal**: Master frontier methods for agent evaluation, including the Agent-as-Judge paradigm and benchmarks such as τ-bench / OSWorld / SWE-bench, and be able to implement a complete Agent-as-Judge evaluator.

---

## From LLM-as-Judge to Agent-as-Judge

In Chapter 18.1 we introduced LLM-as-Judge — using one LLM to judge the output quality of another LLM. But an Agent is different from an ordinary chat model: an Agent calls tools, executes multi-step operations, and interacts with an environment. Evaluating only the final output is not enough — we need to evaluate the Agent's **entire behavior trajectory**.

That is the core idea of **Agent-as-Judge**: use an Agent (not merely an LLM) to evaluate another Agent's complete execution process [1].

### LLM-as-Judge vs Agent-as-Judge

| Dimension | LLM-as-Judge | Agent-as-Judge |
|------|---------------|----------------|
| Evaluation target | Single-turn text output | Full execution trajectory (multi-step, multi-tool) |
| Evaluation method | One-shot scoring | Step-by-step review + interactive verification |
| Context understanding | Sees only the input and the output | Understands tool calls, intermediate state, error recovery |
| Evaluation depth | Semantic quality | Decision quality + execution efficiency + error handling |
| Cost | Lower | Higher (requires multi-turn reasoning) |
| Consistency | Higher | Medium (the evaluation process is more complex) |

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class TrajectoryAspect(Enum):
    """Aspects for evaluating an Agent's behavior trajectory"""
    GOAL_ACHIEVEMENT = "goal_achievement"    # did it accomplish the user's goal
    TOOL_SELECTION = "tool_selection"        # was the chosen tool reasonable
    TOOL_USAGE = "tool_usage"                # were the tool arguments correct
    ERROR_RECOVERY = "error_recovery"        # can it self-correct after an error
    EFFICIENCY = "efficiency"                # did it take unnecessary detours
    REASONING_QUALITY = "reasoning_quality"  # was the thinking process sound

@dataclass
class AgentTrace:
    """An Agent execution trajectory"""
    task_id: str
    user_query: str
    steps: list[dict] = field(default_factory=list)  # detailed record of every step
    final_output: str = ""
    success: bool = False
    total_tokens: int = 0
    total_time: float = 0.0

@dataclass
class TraceEvaluation:
    """Trajectory evaluation result"""
    task_id: str
    aspect: TrajectoryAspect
    score: float            # 0.0 - 1.0
    reasoning: str
    evidence: list[str] = field(default_factory=list)  # evidence extracted from the trajectory
```

---

## Agent-as-Judge Methodology

### Core Workflow

The Agent-as-Judge workflow has three stages:

1. **Trajectory collection**: record the complete execution process of the evaluated Agent.
2. **Step-by-step review**: the evaluating Agent inspects every single step.
3. **Holistic judgment**: aggregate the per-step evaluations into an overall verdict.

```python
import json
from langchain_openai import ChatOpenAI

class AgentAsJudge:
    """Use an Agent to evaluate another Agent's full execution trajectory"""

    def __init__(self, model: str = "gpt-4.1"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_trace(self, trace: AgentTrace) -> dict:
        """Evaluate a complete Agent execution trajectory"""

        # Stage 1: format the trajectory
        trajectory_text = self._format_trajectory(trace)

        # Stage 2: step-by-step review
        step_evaluations = self._review_steps(trajectory_text, trace.user_query)

        # Stage 3: holistic judgment
        overall_evaluation = self._synthesize_evaluation(
            trace, step_evaluations
        )

        return {
            "task_id": trace.task_id,
            "step_evaluations": step_evaluations,
            "overall": overall_evaluation
        }

    def _format_trajectory(self, trace: AgentTrace) -> str:
        """Format the execution trajectory into readable text"""
        lines = [f"User request: {trace.user_query}\n"]

        for i, step in enumerate(trace.steps, 1):
            lines.append(f"--- Step {i} ---")
            if "thought" in step:
                lines.append(f"Thought: {step['thought']}")
            if "action" in step:
                lines.append(f"Action: {step['action']}")
            if "tool" in step:
                lines.append(f"Tool: {step['tool']}")
            if "tool_input" in step:
                lines.append(f"Tool input: {json.dumps(step['tool_input'], ensure_ascii=False)}")
            if "observation" in step:
                lines.append(f"Observation: {step['observation']}")
            lines.append("")

        lines.append(f"Final output: {trace.final_output}")
        lines.append(f"Execution succeeded: {'Yes' if trace.success else 'No'}")
        return "\n".join(lines)

    def _review_steps(self, trajectory: str, query: str) -> list[dict]:
        """Review the Agent's behavior step by step"""
        prompt = f"""You are a professional reviewer of Agent behavior. Review the following Agent execution trajectory step by step.

{trajectory}

Review every step and analyze:
1. Is the reasoning in this step sound?
2. Is the selected tool/action appropriate?
3. Are the tool arguments correct?
4. Is the interpretation of the observation accurate?

Reply in JSON format:
{{
    "steps": [
        {{
            "step_number": 1,
            "thought_quality": "<good/fair/poor>",
            "action_appropriateness": "<good/fair/poor>",
            "parameter_correctness": "<good/fair/poor>",
            "observation_understanding": "<good/fair/poor>",
            "issues": ["issue 1", "issue 2"],
            "improvement": "improvement suggestion"
        }}
    ]
}}"""

        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return result.get("steps", [])
        except json.JSONDecodeError:
            return []

    def _synthesize_evaluation(self, trace: AgentTrace, step_evals: list) -> dict:
        """Synthesize the reviews into an overall verdict"""
        prompt = f"""Based on the information below, give a holistic judgment of the Agent's overall performance.

Task: {trace.user_query}
Number of steps: {len(trace.steps)}
Succeeded: {'Yes' if trace.success else 'No'}
Total time: {trace.total_time:.1f}s
Total tokens: {trace.total_tokens}

Step-by-step review results:
{json.dumps(step_evals, ensure_ascii=False, indent=2)}

Score the following dimensions (0-10) and give an overall assessment:
1. Goal achievement: did it accomplish the user's goal
2. Decision quality: was every step a reasonable decision
3. Execution efficiency: were there unnecessary steps
4. Error handling: how well did it cope with failures
5. Output quality: how good is the final answer

Reply in JSON format:
{{
    "goal_achievement": <0-10>,
    "decision_quality": <0-10>,
    "execution_efficiency": <0-10>,
    "error_handling": <0-10>,
    "output_quality": <0-10>,
    "overall_score": <0-10>,
    "summary": "overall assessment (2-3 sentences)",
    "key_strengths": ["strength 1", "strength 2"],
    "key_weaknesses": ["weakness 1", "weakness 2"],
    "recommendations": ["recommendation 1", "recommendation 2"]
}}"""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"overall_score": 0, "summary": "Failed to parse the evaluation"}
```

### Worked Example: Evaluating a Search Agent

```python
# Build a trajectory to be evaluated
sample_trace = AgentTrace(
    task_id="search_eval_001",
    user_query="Compare the pros and cons of Python and Rust for web backend development",
    steps=[
        {
            "thought": "The user wants a comparison of two languages, so I need to search for the pros and cons of each",
            "action": "call the search tool",
            "tool": "web_search",
            "tool_input": {"query": "Python web backend development pros and cons 2025"},
            "observation": "Found 5 results: 1. Python strengths: rich ecosystem... 2. Django/Flask..."
        },
        {
            "thought": "Now search for information about Rust",
            "action": "call the search tool",
            "tool": "web_search",
            "tool_input": {"query": "Rust web backend development pros and cons 2025"},
            "observation": "Found 5 results: 1. Rust strengths: high performance... 2. Actix/Axum..."
        },
        {
            "thought": "I have enough information; I can now write the comparison",
            "action": "generate the final answer",
            "tool": None,
            "tool_input": None,
            "observation": None
        }
    ],
    final_output="Python and Rust each have their strengths for web backend development...\nPython: rich ecosystem, fast development...\nRust: high performance, memory safety...",
    success=True,
    total_tokens=3200,
    total_time=8.5
)

# Run the Agent-as-Judge evaluation
judge = AgentAsJudge(model="gpt-4.1")
result = judge.evaluate_trace(sample_trace)
print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
```

### Caveats

| Issue | Description | Mitigation |
|------|------|----------|
| Evaluation bias | The judge Agent may favor a particular style | Use several judge Agents and let them vote |
| Evaluation cost | Every evaluation needs multiple LLM calls | Use a rules + LLM hybrid for simple tasks |
| Consistency | Evaluating the same trajectory twice may give different results | temperature=0 plus averaging over several runs |
| Ceiling on judging ability | The judge Agent is limited by its own model capability | Use a stronger model for the judge than for the evaluated Agent |
| Trajectory formatting | Very long trajectories may exceed the context window | Summarize the trajectory or evaluate it in segments |

> 💡 **Best practice**: the judge model in Agent-as-Judge should be stronger than the Agent being evaluated. For example, use gpt-4.1 to evaluate an Agent driven by gpt-4.1-mini, so you avoid the "student grading their own paper" problem.

---

## τ-bench: A Benchmark for Tool Use

### What Is τ-bench?

τ-bench (tau-bench) is a benchmark proposed in 2024 specifically for evaluating the tool-use ability of LLM Agents [2]. Unlike traditional benchmarks, τ-bench focuses on an Agent's ability to use tools in a **realistic environment**, not merely on picking the right tool.

### Core Design of τ-bench

| Feature | Description |
|------|------|
| Evaluation dimensions | Tool selection, argument filling, multi-step reasoning, error handling |
| Environment | Simulated real API environments (flight search, hotel booking, etc.) |
| Difficulty levels | Simple single-tool → complex multi-tool collaboration |
| Evaluation method | End-to-end result matching + trajectory review |
| Key innovation | Introduces a "user simulator" that mimics real multi-turn user behavior |

### τ-bench Evaluation Dimensions in Detail

```python
@dataclass
class TauBenchResult:
    """τ-bench evaluation result"""
    task_id: str
    # Core metrics
    tool_selection_accuracy: float   # tool selection accuracy
    param_fill_accuracy: float       # argument filling accuracy
    multi_step_success_rate: float   # multi-step task success rate
    error_recovery_rate: float       # error recovery rate
    # Auxiliary metrics
    avg_steps_per_task: float        # average number of steps
    avg_redundant_steps: float       # average number of redundant steps
    total_token_usage: int           # total token usage

class TauBenchEvaluator:
    """A τ-bench-style evaluator"""

    def __init__(self, agent_func, user_simulator, env):
        self.agent_func = agent_func       # the Agent under evaluation
        self.user_simulator = user_simulator  # user simulator
        self.env = env                       # simulated environment

    def evaluate_task(self, task: dict) -> TauBenchResult:
        """Evaluate a single task"""
        steps = []
        tool_calls_correct = 0
        tool_calls_total = 0
        params_correct = 0
        params_total = 0
        errors_encountered = 0
        errors_recovered = 0

        # Simulate a multi-turn conversation
        conversation = [{"role": "user", "content": task["initial_query"]}]

        for step_idx in range(20):  # at most 20 steps
            agent_response = self.agent_func(conversation)

            # Extract tool calls
            if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
                for tc in agent_response.tool_calls:
                    tool_calls_total += 1
                    params_total += len(tc["args"])

                    # Check whether the tool selection is correct
                    expected_tools = task.get("expected_tool_sequence", [])
                    if step_idx < len(expected_tools):
                        if tc["name"] == expected_tools[step_idx]:
                            tool_calls_correct += 1

                        # Check the arguments
                        expected_args = task.get("expected_args", {}).get(step_idx, {})
                        for key, expected_val in expected_args.items():
                            if key in tc["args"] and tc["args"][key] == expected_val:
                                params_correct += 1

                    # Execute the tool and collect the result
                    try:
                        observation = self.env.execute(tc["name"], tc["args"])
                    except Exception as e:
                        errors_encountered += 1
                        observation = f"Error: {str(e)}"

                    steps.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                        "observation": observation,
                        "is_error": "Error" in str(observation)
                    })

                    conversation.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc]
                    })
                    conversation.append({
                        "role": "tool",
                        "content": str(observation)
                    })
            else:
                # The Agent produced its final answer
                break

        # Check whether the final result is correct
        final_success = self._check_final_result(task, steps)

        return TauBenchResult(
            task_id=task["id"],
            tool_selection_accuracy=(
                tool_calls_correct / tool_calls_total
                if tool_calls_total > 0 else 0.0
            ),
            param_fill_accuracy=(
                params_correct / params_total
                if params_total > 0 else 0.0
            ),
            multi_step_success_rate=1.0 if final_success else 0.0,
            error_recovery_rate=(
                errors_recovered / errors_encountered
                if errors_encountered > 0 else 1.0
            ),
            avg_steps_per_task=len(steps),
            avg_redundant_steps=self._count_redundant_steps(steps),
            total_token_usage=sum(
                len(str(m["content"]).split()) for m in conversation
            )
        )

    def _check_final_result(self, task: dict, steps: list) -> bool:
        """Check whether the final result matches expectations"""
        expected_results = task.get("expected_results", {})
        if not expected_results:
            return len(steps) > 0

        # Simplified: check that the key tools were called successfully
        for required_tool in expected_results.get("required_tools", []):
            found = any(s["tool"] == required_tool and not s["is_error"] for s in steps)
            if not found:
                return False
        return True

    def _count_redundant_steps(self, steps: list) -> int:
        """Count redundant steps (the same tool called again with the same arguments)"""
        redundant = 0
        seen = set()
        for step in steps:
            key = (step["tool"], json.dumps(step["args"], sort_keys=True))
            if key in seen:
                redundant += 1
            seen.add(key)
        return redundant
```

---

## OSWorld and VisualWebArena: Multimodal Agent Benchmarks

### OSWorld: Agent Evaluation in Real Desktop Environments

OSWorld [3] is the first benchmark (proposed in 2024) that evaluates multimodal Agents in a **real operating system environment**. Unlike earlier benchmarks built on simulated environments, OSWorld makes the Agent complete tasks on an actual Ubuntu / Windows / macOS desktop.

| Feature | Description |
|------|------|
| Environment | Real OS (Ubuntu 22.04, Windows 11, macOS) |
| Task types | File operations, application use, web browsing, cross-application workflows |
| Interaction | Screenshots + accessibility tree |
| Number of tasks | 369 real tasks |
| Evaluation method | Function-based verification of the execution result (not string matching) |

### VisualWebArena: A Multimodal Agent Benchmark for the Web

VisualWebArena [4] focuses on multimodal Agent evaluation in **web environments**, requiring the Agent to understand and operate web pages visually:

| Feature | Description |
|------|------|
| Environment | Self-hosted web applications (e-commerce, forums, CMS) |
| Task types | Information retrieval, content management, data manipulation |
| Interaction | Page screenshots + DOM operations |
| Core challenge | Understanding visual layout, filling forms, navigating across pages |

### Comparison of Multimodal Agent Benchmarks

| Benchmark | Environment type | Interaction | Number of tasks | Best success rate |
|------|----------|----------|----------|------------|
| OSWorld | Real desktop OS | Screenshots + keyboard/mouse | 369 | ~12.5% (2024) |
| VisualWebArena | Web applications | Screenshots + DOM operations | 910 | ~14.6% (2024) |
| WebArena | Web applications | HTML + DOM | 812 | ~35.9% (2024) |
| τ-bench | Simulated APIs | Text + tool calls | 200+ | ~68% (2024) |

> ⚠️ **Note**: the best success rates on OSWorld and VisualWebArena are far below those of text-only benchmarks, which shows that multimodal Agents still have enormous room for improvement.

### Key Metrics for Evaluating Multimodal Agents

```python
@dataclass
class MultimodalEvalMetrics:
    """Multimodal Agent evaluation metrics"""
    # Basic metrics
    task_success_rate: float          # task completion rate
    partial_success_rate: float       # partial completion rate

    # Visual understanding metrics
    screenshot_understanding_acc: float  # screenshot understanding accuracy
    element_localization_acc: float      # element localization accuracy
    ocr_accuracy: float                  # OCR accuracy

    # Operation metrics
    action_accuracy: float            # action selection accuracy
    coordinate_accuracy: float        # coordinate accuracy (for click tasks)
    typing_accuracy: float            # typing accuracy

    # Efficiency metrics
    avg_steps: int                    # average number of steps
    avg_time_per_task: float          # average time per task
    unnecessary_actions_rate: float   # ratio of unnecessary actions


class OSWorldStyleEvaluator:
    """An OSWorld-style multimodal Agent evaluator"""

    def __init__(self, agent_func, environment):
        self.agent_func = agent_func
        self.env = environment

    def evaluate(self, task: dict) -> MultimodalEvalMetrics:
        """Evaluate a single multimodal task"""
        steps_data = []
        action_correct = 0
        action_total = 0
        coord_errors = []
        typing_errors = []

        # Reset the environment
        self.env.reset(task["initial_state"])

        for step_idx in range(task.get("max_steps", 15)):
            # Get the current screenshot and accessibility information
            screenshot = self.env.get_screenshot()
            accessibility_tree = self.env.get_accessibility_tree()

            # The Agent decides what to do
            agent_action = self.agent_func(
                task["instruction"],
                screenshot,
                accessibility_tree,
                steps_data  # previous history
            )

            # Record the step
            step_info = {
                "step": step_idx,
                "action_type": agent_action.get("type"),
                "action_params": agent_action.get("params", {}),
            }

            # Evaluate action accuracy
            if step_idx < len(task.get("expected_actions", [])):
                expected = task["expected_actions"][step_idx]
                action_total += 1

                if agent_action["type"] == expected["type"]:
                    action_correct += 1

                    # Evaluate coordinate / typing accuracy
                    if expected["type"] == "click":
                        expected_coord = expected.get("coordinates", (0, 0))
                        actual_coord = agent_action["params"].get(
                            "coordinates", (0, 0)
                        )
                        error = (
                            (expected_coord[0] - actual_coord[0]) ** 2
                            + (expected_coord[1] - actual_coord[1]) ** 2
                        ) ** 0.5
                        coord_errors.append(error)

                    elif expected["type"] == "type":
                        expected_text = expected.get("text", "")
                        actual_text = agent_action["params"].get("text", "")
                        typing_errors.append(
                            self._edit_distance(expected_text, actual_text)
                        )

            # Execute the action
            self.env.execute_action(agent_action)
            steps_data.append(step_info)

            # Check whether the task is done
            if self.env.is_task_completed():
                break

        # Compute the final result
        success = self.env.verify_final_state(task["expected_state"])

        return MultimodalEvalMetrics(
            task_success_rate=1.0 if success else 0.0,
            partial_success_rate=self._partial_score(task, steps_data),
            screenshot_understanding_acc=0.0,  # needs a separate evaluation
            element_localization_acc=0.0,       # needs a separate evaluation
            ocr_accuracy=0.0,                    # needs a separate evaluation
            action_accuracy=(
                action_correct / action_total
                if action_total > 0 else 0.0
            ),
            coordinate_accuracy=(
                1.0 - min(1.0, sum(coord_errors) / len(coord_errors) / 100)
                if coord_errors else 1.0
            ),
            typing_accuracy=(
                1.0 - min(1.0, sum(typing_errors) / len(typing_errors) / 10)
                if typing_errors else 1.0
            ),
            avg_steps=len(steps_data),
            avg_time_per_task=0.0,  # needs real timing
            unnecessary_actions_rate=0.0  # needs human annotation
        )

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute the edit distance"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def _partial_score(self, task: dict, steps: list) -> float:
        """Compute the partial completion score"""
        expected = task.get("expected_actions", [])
        if not expected:
            return 0.0
        completed = min(len(steps), len(expected))
        correct = sum(
            1 for i in range(completed)
            if steps[i].get("action_type") == expected[i].get("type")
        )
        return correct / len(expected)
```

---

## SWE-bench Verified: The Gold Standard for Coding Agents

### SWE-bench Overview

SWE-bench [5] is a benchmark that measures how well coding Agents resolve real GitHub issues. The **SWE-bench Verified** version released in 2024 was human-reviewed to filter out problematic test cases, making the results far more reliable.

| Version | Number of issues | Description |
|------|-----------|------|
| SWE-bench Full | 2294 | Full dataset; some issue descriptions are unclear |
| SWE-bench Lite | 300 | Curated subset, but still has quality problems |
| SWE-bench Verified | 500 | Human-verified; every issue is confirmed solvable |

### How SWE-bench Verified Evaluates

```python
@dataclass
class SWEBenchResult:
    """SWE-bench evaluation result"""
    instance_id: str
    repo: str
    resolved: bool          # was the issue resolved
    patch_applied: bool     # could the patch be applied
    tests_passed: bool      # did the tests pass
    fail_to_pass: list[str]   # tests that went from failing to passing
    pass_to_pass: list[str]   # tests that passed all along
    fail_to_fail: list[str]   # tests that failed all along

class SWEBenchEvaluator:
    """A SWE-bench-style evaluator"""

    def __init__(self, agent_func, docker_env=None):
        self.agent_func = agent_func
        self.docker_env = docker_env

    def evaluate_instance(self, instance: dict) -> SWEBenchResult:
        """Evaluate a single SWE-bench instance"""
        # 1. Prepare the environment
        repo_path = self._setup_repo(instance)

        # 2. Let the Agent analyze the problem and produce a patch
        agent_patch = self.agent_func(
            problem_statement=instance["problem_statement"],
            repo_path=repo_path,
            hints_text=instance.get("hints_text", "")
        )

        # 3. Apply the patch
        patch_applied = self._apply_patch(repo_path, agent_patch)

        if not patch_applied:
            return SWEBenchResult(
                instance_id=instance["instance_id"],
                repo=instance["repo"],
                resolved=False,
                patch_applied=False,
                tests_passed=False,
                fail_to_pass=[],
                pass_to_pass=[],
                fail_to_fail=[]
            )

        # 4. Run the tests
        test_results = self._run_tests(
            repo_path,
            instance.get("test_patch", ""),
            instance.get("fail_to_pass", []),
            instance.get("pass_to_pass", [])
        )

        # 5. Decide whether the issue is resolved
        resolved = (
            len(test_results["fail_to_pass_resolved"])
            == len(instance.get("fail_to_pass", []))
            and len(test_results["pass_to_pass_failed"]) == 0
        )

        return SWEBenchResult(
            instance_id=instance["instance_id"],
            repo=instance["repo"],
            resolved=resolved,
            patch_applied=True,
            tests_passed=resolved,
            fail_to_pass=test_results["fail_to_pass_resolved"],
            pass_to_pass=test_results["pass_to_pass_passed"],
            fail_to_fail=test_results.get("fail_to_fail", [])
        )

    def _setup_repo(self, instance: dict) -> str:
        """Set up the Git repository at the specified revision"""
        import subprocess
        repo_dir = f"/tmp/swebench_{instance['instance_id']}"
        # Clone and check out the base commit
        subprocess.run(
            ["git", "clone", instance["repo"], repo_dir],
            capture_output=True
        )
        subprocess.run(
            ["git", "checkout", instance["base_commit"]],
            cwd=repo_dir, capture_output=True
        )
        return repo_dir

    def _apply_patch(self, repo_path: str, patch: str) -> bool:
        """Try to apply the patch"""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "apply"],
                input=patch.encode(),
                cwd=repo_path,
                capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_tests(self, repo_path, test_patch, fail_to_pass, pass_to_pass):
        """Run the tests and collect the results"""
        import subprocess
        # Apply the test patch
        subprocess.run(
            ["git", "apply"],
            input=test_patch.encode(),
            cwd=repo_path,
            capture_output=True
        )
        # Run the tests
        result = subprocess.run(
            ["python", "-m", "pytest", "-x", "--tb=short"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300
        )
        # Parse the test results (simplified)
        output = result.stdout + result.stderr
        return {
            "fail_to_pass_resolved": [],   # needs to parse output
            "pass_to_pass_passed": [],
            "pass_to_pass_failed": [],
            "fail_to_fail": []
        }
```

### Latest Progress on SWE-bench Verified (2025—2026)

| Rank | Method | Resolution rate | Description |
|------|------|--------|------|
| OpenHands + CodeAct | ~53% | Early 2025 | Best open source |
| Devin | ~50% | Early 2025 | Commercial product |
| SWE-Agent + GPT-4.1 | ~48% | 2025 | Agent framework |
| AutoCodeRover | ~45% | 2024 | Spectrum analysis + LLM |
| Amazon Q Developer | ~42% | 2024 | Built by Amazon |

> 💡 **Trend watch**: the resolution rate on SWE-bench Verified passed 50% in 2025, yet nearly half of all issues still cannot be resolved automatically. The core bottlenecks are long-context understanding, multi-file edits, and complex debugging reasoning.

---

## Full Walkthrough: Building a Runnable Evaluation Pipeline

> ⚠️ **An honest note**: the old version of this section used two classes, `TraceCollector` and `AgentAsJudgeEvaluator`, to "demonstrate" a complete evaluation system. But their `evaluate` methods depended on a real LLM API, and the accompanying "worked example" used a hand-written **mock trajectory** — in other words, it had never actually been run against a fixed dataset. It looked complete but was impossible to reproduce.

> The real difficulty of an evaluation system is not writing a `class Evaluator`. It is having **a fixed, reproducible task set and a runnable harness that returns a pass/fail verdict for every task**. Instead of reinventing the wheel, this section reuses the **already tested** evaluation module from `reference-agent/`, this book's shared foundation.

### A Real, Runnable Eval Harness

`reference-agent/src/reference_agent/evaluation/harness.py` provides three functions:

- `load_samples(path)` — load the fixed task set from a JSONL file;
- `run_case(agent, sample)` — run one task and decide `passed` (based on `expect_contains` keyword hits);
- `evaluate(agent, samples)` — run the whole batch and return a reproducible report with `total / passed / pass_rate / results`.

```python
# Real code from reference-agent (shipped with this book, reproducible via pytest)
from reference_agent.evaluation.harness import evaluate

report = evaluate()  # uses FakeProvider by default, runs fully offline
print(report["pass_rate"])  # e.g. 1.0
```

The task set `reference-agent/data/eval_samples.jsonl` has one JSON object per line with a fixed schema:

```json
{"id": "calc_001", "input": "what is 3 plus 5", "expect_contains": ["8"]}
```

### How to Reproduce It Locally

```bash
cd reference-agent
pip install -e .
pytest                      # includes the evaluation tests; passes offline
```

`FakeProvider` returns deterministic answers offline by default, so `pass_rate` is reproducible. To use a real model, set `AGENT_REAL=1` and configure an API key — the harness then switches to `OpenAIProvider` automatically.

### Design Principles (More Important Than "Writing a Class")

1. **A fixed task set beats ad-hoc trajectories**: evaluation must be based on a task set that does not drift over time, otherwise regression comparisons are impossible. After every prompt or model change, run the same `eval_samples.jsonl` and check whether `pass_rate` dropped.
2. **Verdicts must be executable**: `expect_contains` is a weak verdict (keyword hits). In production, replace it with something stronger — structured output validation, unit tests, or human annotation.
3. **Reports must be reproducible**: `pass_rate` plus a per-case `passed` flag can serve directly as a CI quality gate (see 17.7).

> Want to see how "multi-dimensional Agent-as-Judge scoring" translates into real code? The `reviewer` node in `examples/dev_team/` from Chapter 16 uses structured output to make quality judgments, following the same principles: executable verdicts and reproducible reports.


---

## A Panorama of Agent-Specific Evaluation Benchmarks

No single benchmark covers everything an Agent can do. Production teams usually combine several benchmarks by capability domain: τ-bench for tool calling, WebArena/VisualWebArena for web operation, SWE-bench for code changes, GAIA/HLE for research tasks, AgentDojo/InjecAgent/ASB for security, and LoCoMo/LongMemEval for memory systems.

### Choosing Benchmarks by Capability Domain

| Capability domain | Representative benchmarks | What it mainly evaluates | Suitable Agent types |
|--------|----------|--------------|-------------------|
| **Tool calling** | τ-bench, ToolBench, API-Bank | Tool selection, argument filling, multi-turn tool use | Tool Agents, customer-service Agents |
| **Web operation** | WebArena, VisualWebArena, Mind2Web | Web navigation, DOM/visual understanding, form handling | Browser Use / Web Agents |
| **Desktop operation** | OSWorld, AndroidWorld | GUI operation, cross-application workflows, environment state verification | Computer Use Agents |
| **Coding tasks** | SWE-bench Verified, HumanEval, RepoBench | Issue fixing, code generation, multi-file understanding | Coding Agents |
| **Deep research** | GAIA, HLE, BrowseComp, FRAMES | Multi-step retrieval, evidence synthesis, complex Q&A | Deep Research Agents |
| **Security robustness** | AgentDojo, InjecAgent, ASB, PromptBench | Indirect prompt injection, tool abuse, privilege escalation | Web/RAG/Tool Agents |
| **Long-term memory** | LoCoMo, LongMemEval, ConvoMem | Long-conversation memory, temporal reasoning, user-profile consistency | Memory Agents, personal assistants |

### Side-by-Side Comparison of Mainstream Benchmarks

| Benchmark | Domain | Core capability | Best performance | Limitations |
|------|------|----------|----------|--------|
| τ-bench | Tool use | Tool selection and argument filling | ~68% | Simulated environment, not real |
| OSWorld | Desktop operation | Cross-application workflows | ~12.5% | Low success rate, high cost |
| VisualWebArena | Web operation | Visual understanding + DOM operation | ~14.6% | Web environments only |
| SWE-bench Verified | Code fixing | Fault localization + patch generation | ~53% | Python projects only |
| WebArena | Web navigation | Information retrieval + operation | ~35.9% | Version without visual input |
| GAIA | General reasoning | Multi-step reasoning + tool calling | ~45% | Limited number of tasks |
| BrowseComp | Browsing research | Web retrieval + complex Q&A | Changes fast | Depends on the live web |
| AgentDojo | Agent security | Tool injection attack/defense | Task dependent | Security-specific |
| InjecAgent | Indirect injection | Injection attacks on tool-integrated Agents | Task dependent | Attack-focused evaluation |
| LoCoMo | Long-term memory | Long conversations, multi-hop memory reasoning | Varies widely by design | Mainly evaluates the memory layer |
| LongMemEval | Long-term memory | Long-context memory retrieval and Q&A | Varies widely by design | Tightly coupled to the memory architecture |
| AgentBench | Multi-task | A range of Agent capabilities | ~35% | Evaluation dimensions are not fine-grained enough |

### How Do You Combine Them Into a Production Evaluation Suite?

If you are evaluating a real Agent, do not report a single benchmark score. Build a "capability matrix" instead:

```text
Basic capabilities: tool-call accuracy, format validity rate, task completion rate
  +
Scenario capabilities: domain benchmarks for Web / Code / Research / Memory
  +
Security capabilities: indirect prompt injection, privilege-escalating tool calls, data leakage tests
  +
Production metrics: latency, cost, human-takeover rate, regression stability
```

For example, the evaluation suite for a Deep Research Agent could be:

| Evaluation layer | Metrics |
|--------|------|
| **Task results** | GAIA / BrowseComp accuracy |
| **Process quality** | Number of search rounds, source diversity, citation validity rate, conflict-handling rate |
| **Security** | Refusal rate for malicious web instructions, approval rate for external-link access |
| **Production** | Average latency, token cost, failure-retry rate, human-takeover rate |

This avoids a common misconception: **a high benchmark score does not mean production readiness**. Before an Agent goes live, it must satisfy all four dimensions — capability, security, cost, and stability.

---

## Summary

| Concept | Description |
|------|------|
| Agent-as-Judge | Uses an Agent to evaluate another Agent's full execution trajectory, going beyond the single-turn verdict of LLM-as-Judge |
| τ-bench | A benchmark dedicated to tool-use ability, focused on tool selection and argument filling |
| WebArena / VisualWebArena | The core benchmarks for browser and web-operation Agents |
| OSWorld | Multimodal Agent evaluation in real desktop environments; success rates are still very low |
| SWE-bench Verified | The gold standard for coding Agents; human verification makes the results reliable |
| GAIA / BrowseComp | Important references for Deep Research Agents and multi-step retrieval reasoning |
| AgentDojo / InjecAgent / ASB | Benchmarks for Agent security and indirect prompt injection |
| LoCoMo / LongMemEval | Benchmarks for long-term memory and memory governance |
| Evaluation system | Trajectory collection → step-by-step review → holistic judgment → report generation |

> **Coming next**: we will study A/B testing and regression test automation, and see how to continuously safeguard Agent quality inside a CI/CD pipeline.

---

## References

[1] ZHUGE M, WANG H, LIU J, et al. Agent-as-Judge: Evaluate Agents with Agents for Long Tasks[J]. arXiv preprint arXiv:2410.10934, 2024.

[2] SIYAN Z, YU G, JIAYI P, et al. τ-bench: A Benchmark for Tool-Using LLMs[J]. arXiv preprint arXiv:2406.12045, 2024.

[3] XUE Y, WU D, ZHENG Z, et al. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments[C]//NeurIPS. 2024.

[4] KOH J, LO R, JANG J, et al. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks[J]. arXiv preprint arXiv:2401.13649, 2024.

[5] JIMENEZ C E, YANG J, WETZIG A, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?[C]//ICLR. 2024.

---

[18.7 A/B Testing and Regression Test Automation](./07_ab_testing.md)
