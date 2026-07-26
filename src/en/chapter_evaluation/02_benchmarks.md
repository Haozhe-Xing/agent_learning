# Benchmarks and Evaluation Metrics

> **Section Goal**: Gain a deep understanding of mainstream Agent benchmarks and their evaluation principles; master the underlying algorithms of BFCL, GAIA, AgentBench, WebArena, SWE-bench, and other benchmarks; and learn how to design your own evaluation system.

---

## Why Do We Need Benchmarks?

Imagine you're interviewing two candidates — if you don't give them the same questions, how can you compare who is better? Benchmarks are the "standardized exam questions" for Agents — they use unified tasks, data, and scoring criteria to measure the performance of different Agents.

But different benchmarks evaluate completely different capability dimensions. The "evaluation panorama" below helps you quickly orient yourself:

![Agent Evaluation Benchmark Panorama](../svg/chapter_evaluation_02_benchmarks_map.svg)

---

## Agent Evaluation Benchmark Categories

Agent evaluation benchmarks can be categorized by capability dimension as follows [1]:

| Category | Representative Benchmarks | Capabilities Evaluated |
|----------|--------------------------|----------------------|
| **Tool calling** | BFCL, ToolBench, API-Bank | Correctly calling APIs, handling parameters, combining tools |
| **General reasoning** | GAIA, MMLU, GSM8K | Multi-step reasoning, knowledge breadth, math ability |
| **Comprehensive Agent** | AgentBench | End-to-end task completion across 8 domains |
| **Web operations** | WebArena, Mind2Web, ClawBench | Completing tasks in controlled and live Web environments |
| **Software engineering** | SWE-bench, HumanEval | Code generation, bug fixing, project-level changes |
| **Multimodal** | VisualWebArena, OSWorld | Visual understanding + action execution |

---

## 1. BFCL — Tool Calling Evaluation Benchmark

### Overview

**BFCL (Berkeley Function Calling Leaderboard)** [2] is a tool calling evaluation benchmark released by UC Berkeley that systematically evaluates LLMs' ability to call functions/APIs. It is one of the most authoritative tool calling benchmarks available.

### Four Test Scenarios

BFCL divides tool calling into four progressively difficult levels:

| Type | Description | Example |
|------|-------------|---------|
| **Simple** | Single function, complete parameters | `get_weather(city="Beijing")` |
| **Multiple** | Select the correct function from multiple candidates | Given 10 functions, select and correctly call 1 |
| **Parallel** | Call multiple functions at once | Query weather and book a flight simultaneously |
| **Irrelevance** | Identify irrelevant requests and refuse to call | Should not call any tool when the user says "Hello" |

### AST Matching Algorithm: Why String Matching Isn't Enough

BFCL's core innovation is using **AST (Abstract Syntax Tree) matching** rather than simple string matching to evaluate the correctness of tool calls.

**The problem with string matching**:

```python
# These two calls are semantically identical, but string matching would judge them as "unequal"
ground_truth = 'get_weather(city="Beijing", unit="celsius")'
prediction   = 'get_weather(unit="celsius", city="Beijing")'
# String match: 'city="Beijing", unit="celsius"' ≠ 'unit="celsius", city="Beijing"'
# Result: ❌ Incorrectly judged as wrong!
```

**How AST matching works**:

```python
import ast
from typing import Any

def ast_match(prediction: str, ground_truth: str) -> bool:
    """
    BFCL's AST matching algorithm (simplified)
    
    Core idea: Parse function calls into AST nodes,
    compare function names and argument sets (ignoring argument order)
    """
    try:
        # Parse into AST
        pred_ast = ast.parse(prediction, mode='eval').body
        true_ast = ast.parse(ground_truth, mode='eval').body
        
        # Check that both are function calls
        if not (isinstance(pred_ast, ast.Call) and isinstance(true_ast, ast.Call)):
            return False
        
        # Compare function names
        pred_func = ast.dump(pred_ast.func)
        true_func = ast.dump(true_ast.func)
        if pred_func != true_func:
            return False
        
        # Compare keyword arguments (ignoring order)
        pred_kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in pred_ast.keywords
        }
        true_kwargs = {
            kw.arg: ast.literal_eval(kw.value) 
            for kw in true_ast.keywords
        }
        
        if pred_kwargs != true_kwargs:
            return False
        
        # Compare positional arguments
        pred_args = [ast.literal_eval(a) for a in pred_ast.args]
        true_args = [ast.literal_eval(a) for a in true_ast.args]
        
        return pred_args == true_args
        
    except (SyntaxError, ValueError):
        return False


# Test
print(ast_match(
    'get_weather(city="Beijing", unit="celsius")',
    'get_weather(unit="celsius", city="Beijing")'
))
# ✅ True — AST matching correctly identifies argument order independence

print(ast_match(
    'get_weather(city="Beijing")',
    'get_weather(city="Shanghai")'
))
# ❌ False — argument values differ
```

### Type-Aware Matching

BFCL also handles **type equivalence**:

```python
def type_aware_match(pred_value: Any, true_value: Any) -> bool:
    """
    Type-aware argument value matching
    
    Handles common type equivalence scenarios:
    - Integer 1 and float 1.0
    - String "true" and boolean True
    - Single-element list ["a"] and string "a"
    """
    # Direct equality
    if pred_value == true_value:
        return True
    
    # Numeric type equivalence: 1 == 1.0
    if isinstance(pred_value, (int, float)) and isinstance(true_value, (int, float)):
        return abs(float(pred_value) - float(true_value)) < 1e-6
    
    # String vs boolean: "true" == True
    if isinstance(pred_value, str) and isinstance(true_value, bool):
        return pred_value.lower() == str(true_value).lower()
    
    # List vs set: ignore order
    if isinstance(pred_value, list) and isinstance(true_value, list):
        return sorted(str(x) for x in pred_value) == sorted(str(x) for x in true_value)
    
    return False
```

---

## 2. GAIA — General AI Assistant Evaluation

### Overview

**GAIA (General AI Assistant Benchmark)** [3] was released by Meta to evaluate AI assistants' ability to handle real-world tasks. Its unique characteristics are:

- **Easy for humans, hard for AI**: Questions are designed so humans can easily answer them (through search and reasoning), but AI needs to combine multiple capabilities to complete them
- **Short, definitive answers**: Each question has a short standard answer (usually a word or number), avoiding the subjectivity of open-ended evaluation
- **Three-level difficulty system**: From simple to complex, comprehensively evaluating different capability levels

### Three Difficulty Levels

| Level | Required Capabilities | Example |
|-------|----------------------|---------|
| **Level 1** | 1–2 steps, basic reasoning | "What is the capital of France?" |
| **Level 2** | 3–5 steps, requires tools | "What is the population of the birthplace of the 2024 Nobel Physics Prize winner?" |
| **Level 3** | 5+ steps, multiple tools + reasoning | "Find the data in row 2 of the table on page 3 of a PDF, and calculate its ratio to CPI" |

### Quasi-Exact Match Algorithm

GAIA's evaluation uses a **quasi-exact match** algorithm — more lenient than strict string matching, but maintaining objectivity:

```python
import re
import unicodedata

def quasi_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    GAIA's quasi-exact match algorithm
    
    Core idea: Normalize both the prediction and the ground truth before
    exact comparison, tolerating meaningless differences like case,
    punctuation, and whitespace
    """
    
    def normalize(text: str) -> str:
        """Normalization"""
        # Lowercase
        text = text.lower().strip()
        
        # Unicode normalization (handles full-width/half-width, accents, etc.)
        text = unicodedata.normalize("NFKD", text)
        
        # Remove punctuation (but keep decimal points and minus signs)
        text = re.sub(r'[^\w\s\.\-]', '', text)
        
        # Collapse extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove articles (English)
        text = re.sub(r'\b(a|an|the)\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_number(text: str) -> str | None:
        """Try to parse text into a standard numeric format"""
        try:
            # Remove thousands separators
            cleaned = text.replace(",", "").replace(" ", "")
            num = float(cleaned)
            # If integer, return integer format
            if num == int(num):
                return str(int(num))
            return f"{num:.6f}".rstrip('0').rstrip('.')
        except ValueError:
            return None
    
    # Normalized comparison
    norm_pred = normalize(prediction)
    norm_truth = normalize(ground_truth)
    
    if norm_pred == norm_truth:
        return True
    
    # Numeric comparison
    num_pred = normalize_number(prediction)
    num_truth = normalize_number(ground_truth)
    
    if num_pred is not None and num_truth is not None:
        return num_pred == num_truth
    
    # Containment (answer may be within a longer response)
    if norm_truth in norm_pred:
        # Ensure it's a complete match, not a substring
        pattern = r'\b' + re.escape(norm_truth) + r'\b'
        if re.search(pattern, norm_pred):
            return True
    
    return False


# Tests
print(quasi_exact_match("Paris", "paris"))                    # True: case
print(quasi_exact_match("42,000", "42000"))                   # True: thousands separator
print(quasi_exact_match("The answer is Paris.", "Paris"))     # True: containment
print(quasi_exact_match("3.14", "3.14000"))                   # True: decimal precision
print(quasi_exact_match("Beijing", "Shanghai"))               # False: different answers
```

### GAIA's Multimodal Data Handling

A unique challenge in GAIA is that **tasks may include attachments** — PDF files, Excel spreadsheets, images, etc. Agents need to:
1. Identify the attachment type
2. Use the appropriate tool to read the content
3. Extract key information from the content
4. Combine the information to complete the reasoning

---

## 3. AgentBench — Comprehensive Agent Evaluation

### Overview

**AgentBench** [4] is a comprehensive Agent benchmark released by Tsinghua University, covering tasks across **8 different domains** to comprehensively evaluate Agent performance in various environments:

| Domain | Task Type | Evaluation Focus |
|--------|----------|-----------------|
| **OS** | Operating system command execution | File operations, process management |
| **DB** | Database queries | SQL generation, data analysis |
| **KG** | Knowledge graph reasoning | Graph traversal, relational reasoning |
| **DCG** | Digital card game | Strategic decision-making, state tracking |
| **LTP** | Lateral thinking puzzles | Creative reasoning |
| **HouseHold** | Home environment operations | Spatial reasoning, object interaction |
| **WebShop** | Online shopping | Search, filtering, decision-making |
| **WebBrowse** | Web browsing | Information extraction, navigation |

### AgentBench Evaluation Framework

```python
class AgentBenchEvaluator:
    """
    AgentBench evaluation architecture (conceptual implementation)
    
    Key features:
    1. Each domain has its own environment and evaluator
    2. Agents interact with environments through a unified text interface
    3. The evaluation metric is task Success Rate
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.environments = {
            "os": OSEnvironment(),
            "db": DatabaseEnvironment(),
            "web_shop": WebShopEnvironment(),
            # ... other environments
        }
    
    def evaluate_task(self, env_name: str, task: dict) -> dict:
        """
        Evaluate a single task
        
        Process:
        1. Initialize the environment and provide the task description
        2. Agent interacts with the environment (up to N steps)
        3. Check whether the final state satisfies the success conditions
        """
        env = self.environments[env_name]
        observation = env.reset(task)
        
        for step in range(task.get("max_steps", 20)):
            # Agent generates an action based on the observation
            action = self.agent.act(observation)
            
            # Environment executes the action
            observation, reward, done, info = env.step(action)
            
            if done:
                break
        
        return {
            "success": info.get("success", False),
            "steps": step + 1,
            "reward": reward,
        }
    
    def evaluate_all(self, benchmark_data: dict) -> dict:
        """Evaluate all domains"""
        results = {}
        for env_name, tasks in benchmark_data.items():
            env_results = [
                self.evaluate_task(env_name, task) 
                for task in tasks
            ]
            results[env_name] = {
                "success_rate": sum(r["success"] for r in env_results) / len(env_results),
                "avg_steps": sum(r["steps"] for r in env_results) / len(env_results),
            }
        return results
```

### Top Model Performance by Domain (as of 2025)

| Environment | GPT-4o | Claude-3.5 | Open-source SOTA |
|-------------|--------|------------|-----------------|
| OS | ~45% | ~42% | ~30% (CodeLlama) |
| DB | ~52% | ~48% | ~35% |
| WebShop | ~60% | ~55% | ~40% |
| **Overall** | **~42%** | **~38%** | **~28%** |

> **Key insight**: Even the strongest closed-source models achieve only about 40% overall on AgentBench. This shows there is still enormous room for improvement in current LLMs' Agent capabilities.

---

## 4. WebArena — Web Operation Evaluation

### Overview

**WebArena** [5] is a Web Agent evaluation benchmark released by CMU that tests Agents' ability to complete tasks in **real Web application environments**. It deploys 4 complete Web applications:

- **Reddit** (forum)
- **GitLab** (code hosting)
- **Shopping** (e-commerce website)
- **CMS** (content management system)

### Task Example

```
Task: "On GitLab, create a new repository named 'ml-pipeline',
      add a .gitignore file using the Python template,
      then create an Issue named 'setup-ci'."

Evaluation criteria:
1. Does the repository 'ml-pipeline' exist? ✅/❌
2. Does the .gitignore contain Python template content? ✅/❌
3. Has the Issue 'setup-ci' been created? ✅/❌

Final score: Success only if all conditions are met
```

### Evaluation Method

WebArena uses **state-based evaluation** — checking whether the state of the Web application after the operation matches expectations:

```python
class WebArenaEvaluator:
    """WebArena evaluation logic (conceptual implementation)"""
    
    def evaluate(self, task: dict, final_state: dict) -> bool:
        """
        Check whether the final state of the Web application meets the task requirements
        
        Evaluation types:
        1. URL match: Is the final page correct?
        2. Element existence: Does the page contain specific elements?
        3. Database state: Is the backend data correct?
        """
        
        for condition in task["success_conditions"]:
            if condition["type"] == "url_match":
                if not self._check_url(final_state["url"], condition["pattern"]):
                    return False
                    
            elif condition["type"] == "element_exists":
                if not self._find_element(
                    final_state["page_html"], 
                    condition["selector"]
                ):
                    return False
                    
            elif condition["type"] == "db_check":
                if not self._query_db(
                    condition["query"], 
                    condition["expected"]
                ):
                    return False
        
        return True  # All conditions met
```

### Complementary Live-Web Evaluation: ClawBench

[**ClawBench**](https://claw-bench.com/) [8] ([repository](https://github.com/TIGER-AI-Lab/ClawBench)) evaluates browser Agents on everyday tasks across live third-party websites. V1 contains 153 tasks across 144 sites, and V2 adds 130 tasks. Every run takes place in an isolated container and records five synchronized evidence layers: session video, action screenshots, HTTP traffic, browser actions, and Agent messages. A request interceptor blocks matched irreversible final requests, allowing the run to be inspected and judged without completing the real-world side effect.

> **WebArena or ClawBench?** WebArena's self-hosted applications make reset and state-based evaluation reproducible. ClawBench exposes Agents to live-site drift and network or UI variation, trading some reproducibility for deployment realism and broader site coverage. Using both helps separate controlled-environment capability from live-site robustness.

---

## 5. SWE-bench — Software Engineering Evaluation

### Overview

**SWE-bench** [6] is a software engineering evaluation benchmark released by Princeton that tests Agents' ability to resolve **real GitHub Issues**. Each test case comes from a real open-source project (such as Django, Flask, scikit-learn) and includes an Issue description and corresponding test cases.

### Task Flow

```
1. Receive Issue description:
   "Django QuerySet.union() returns duplicate results when using values()"

2. Understand the codebase:
   Analyze Django's QuerySet implementation (thousands of files)

3. Locate the problem:
   Find the union logic in django/db/models/sql/query.py

4. Generate a fix:
   Generate a git patch file

5. Validate:
   Run the project's unit tests and check if they pass
```

### SWE-bench Variants

| Variant | Test Count | Description |
|---------|-----------|-------------|
| **SWE-bench Full** | 2,294 Issues | Complete collection |
| **SWE-bench Lite** | 300 Issues | Curated subset, more reproducible |
| **SWE-bench Verified** | 500 Issues | High-quality subset verified by humans |

### Evaluation Metric

SWE-bench's core metric is the **Resolved Rate**:

```python
def swe_bench_evaluate(
    patch: str,          # git patch generated by the Agent
    test_suite: str,     # original test cases
    repo_path: str,      # code repository path
) -> dict:
    """
    SWE-bench evaluation process (simplified)
    
    1. Apply the Agent-generated patch
    2. Run the project's test suite
    3. Check whether previously failing tests now pass
    """
    import subprocess
    
    # Check if patch can be applied
    apply_result = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch.encode(),
        cwd=repo_path,
        capture_output=True,
    )
    
    if apply_result.returncode != 0:
        return {"resolved": False, "reason": "Patch cannot be applied"}
    
    # Actually apply the patch
    subprocess.run(
        ["git", "apply", "-"],
        input=patch.encode(),
        cwd=repo_path,
    )
    
    # Run tests
    test_result = subprocess.run(
        ["python", "-m", "pytest", test_suite, "-x"],
        cwd=repo_path,
        capture_output=True,
        timeout=300,  # 5-minute timeout
    )
    
    return {
        "resolved": test_result.returncode == 0,
        "test_output": test_result.stdout.decode(),
    }
```

### Current SOTA (as of 2025)

| Agent | SWE-bench Verified | Method |
|-------|-------------------|--------|
| **DeepSWE** | 59.0% | GRPO reinforcement learning training |
| **Amazon Q Developer** | 55.2% | Closed-source commercial product |
| **Claude-3.5 Sonnet** | ~49% | Direct inference |
| **SWE-Agent** | ~33% | Open-source framework |

---

## 6. LLM-as-Judge Evaluation Method

For open-ended tasks without standard answers, **LLM-as-Judge** [7] is the most commonly used evaluation method — using a powerful LLM to judge the output of another LLM.

### Three Judgment Modes

```python
class LLMJudge:
    """LLM-as-Judge evaluator"""
    
    def __init__(self, judge_model: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        self.judge = ChatOpenAI(model=judge_model, temperature=0)
    
    def pointwise_scoring(
        self, 
        question: str, 
        answer: str, 
        rubric: str,
    ) -> dict:
        """
        Mode 1: Pointwise scoring
        Evaluates the absolute quality of a single answer
        """
        prompt = f"""Please evaluate the answer quality according to the following rubric.

Question: {question}
Answer: {answer}

Rubric:
{rubric}

Please score from 1–10 and explain your reasoning.
Output JSON: {{"score": <score>, "reasoning": "<reasoning>"}}"""
        
        response = self.judge.invoke(prompt)
        return json.loads(response.content)
    
    def pairwise_comparison(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> dict:
        """
        Mode 2: Pairwise comparison
        Determines which of two answers is better
        
        Used to calculate Win Rate and ELO scores
        """
        prompt = f"""Please compare the following two answers and determine which is better.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Please choose: A is better / B is better / Tie
And explain your reasoning.
Output JSON: {{"winner": "A"/"B"/"tie", "reasoning": "<reasoning>"}}"""
        
        response = self.judge.invoke(prompt)
        return json.loads(response.content)
    
    def reference_based(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> dict:
        """
        Mode 3: Reference-based comparison
        Compare with a standard answer
        """
        prompt = f"""Please evaluate the consistency between the answer and the reference answer.

Question: {question}
Reference answer: {reference}
Answer to evaluate: {answer}

Scoring criteria:
- 1.0: Completely consistent with the reference answer
- 0.8: Mostly consistent, with minor differences
- 0.5: Partially correct
- 0.2: Mostly incorrect
- 0.0: Completely incorrect

Output JSON: {{"score": <score>, "reasoning": "<reasoning>"}}"""
        
        response = self.judge.invoke(prompt)
        return json.loads(response.content)
```

### Win Rate Calculation

```python
def compute_win_rate(
    judge: LLMJudge,
    questions: list[str],
    answers_a: list[str],  # Agent A's answers
    answers_b: list[str],  # Agent B's answers
) -> dict:
    """Calculate the Win Rate of two Agents"""
    wins_a, wins_b, ties = 0, 0, 0
    
    for q, a, b in zip(questions, answers_a, answers_b):
        # Forward evaluation
        result1 = judge.pairwise_comparison(q, a, b)
        # Reverse evaluation (swap positions to eliminate position bias)
        result2 = judge.pairwise_comparison(q, b, a)
        
        # Combine both evaluations
        if result1["winner"] == "A" and result2["winner"] == "B":
            wins_a += 1  # Both agree A is better
        elif result1["winner"] == "B" and result2["winner"] == "A":
            wins_b += 1  # Both agree B is better
        else:
            ties += 1    # Inconsistent results, count as tie
    
    total = len(questions)
    return {
        "agent_a_win_rate": wins_a / total,
        "agent_b_win_rate": wins_b / total,
        "tie_rate": ties / total,
    }
```

### Known Biases in LLM-as-Judge [7]

| Bias Type | Description | Mitigation |
|-----------|-------------|-----------|
| **Position bias** | Tends to prefer the first (or last) answer | Swap positions and evaluate twice |
| **Verbosity bias** | Tends to prefer longer answers | Explicitly state "conciseness is not penalized" in the rubric |
| **Self-preference** | Tends to prefer its own generated answers | Use a different model as the Judge |
| **Format bias** | Tends to prefer better-formatted answers | Unify formatting before evaluation |

---

## Designing Your Own Evaluation System

### Complete Evaluation Framework

```python
import json
import time
from dataclasses import dataclass, field

@dataclass
class AgentMetrics:
    """Agent evaluation metric set"""
    
    # Quality metrics
    accuracy: float = 0.0            # Accuracy rate
    f1_score: float = 0.0            # F1 score
    hallucination_rate: float = 0.0  # Hallucination rate
    
    # Efficiency metrics
    avg_latency: float = 0.0         # Average response time (seconds)
    avg_steps: float = 0.0           # Average number of execution steps
    avg_tokens: float = 0.0          # Average token consumption
    avg_cost: float = 0.0            # Average cost (USD)
    
    # Reliability metrics
    success_rate: float = 0.0        # Task success rate
    error_rate: float = 0.0          # Error rate
    timeout_rate: float = 0.0        # Timeout rate
    
    # Safety metrics
    safety_violation_rate: float = 0.0  # Safety violation rate
    pii_leak_rate: float = 0.0          # Privacy leakage rate


class AgentBenchmarkRunner:
    """Agent benchmark test runner"""
    
    def __init__(self, agent_func, test_cases: list[dict]):
        self.agent_func = agent_func
        self.test_cases = test_cases
        self.results = []
    
    def run(self) -> AgentMetrics:
        """Run all test cases"""
        metrics = AgentMetrics()
        
        latencies = []
        step_counts = []
        token_counts = []
        successes = 0
        errors = 0
        timeouts = 0
        correct = 0
        
        for case in self.test_cases:
            try:
                start = time.time()
                result = self.agent_func(
                    case["input"],
                    timeout=case.get("timeout", 30)
                )
                elapsed = time.time() - start
                
                latencies.append(elapsed)
                step_counts.append(result.get("steps", 0))
                token_counts.append(result.get("tokens", 0))
                
                if self._check_answer(
                    result.get("answer", ""),
                    case["expected"]
                ):
                    correct += 1
                
                successes += 1
                
            except TimeoutError:
                timeouts += 1
            except Exception:
                errors += 1
            
            self.results.append({
                "case": case["input"],
                "status": "success" if successes else "error"
            })
        
        total = len(self.test_cases)
        
        metrics.accuracy = correct / total if total else 0
        metrics.success_rate = successes / total if total else 0
        metrics.error_rate = errors / total if total else 0
        metrics.timeout_rate = timeouts / total if total else 0
        metrics.avg_latency = (
            sum(latencies) / len(latencies) if latencies else 0
        )
        metrics.avg_steps = (
            sum(step_counts) / len(step_counts) if step_counts else 0
        )
        metrics.avg_tokens = (
            sum(token_counts) / len(token_counts) if token_counts else 0
        )
        
        return metrics
    
    def _check_answer(self, actual: str, expected) -> bool:
        """Check if the answer is correct (supports multiple matching methods)"""
        if isinstance(expected, str):
            return actual.strip().lower() == expected.strip().lower()
        elif isinstance(expected, list):
            return any(kw.lower() in actual.lower() for kw in expected)
        elif callable(expected):
            return expected(actual)
        return False
```

### Recommended Evaluation Combination Strategy

| Agent Type | Recommended Benchmarks | Custom Evaluation Focus |
|-----------|----------------------|------------------------|
| **General assistant** | GAIA + MMLU | Knowledge accuracy + multi-step reasoning |
| **Code Agent** | SWE-bench + HumanEval | Test pass rate + code quality |
| **Tool-calling Agent** | BFCL + ToolBench | AST match accuracy + parameter correctness |
| **Web Agent** | WebArena + ClawBench | Task completion rate + operation efficiency + live-site robustness |
| **Customer service Agent** | Custom | LLM-as-Judge + manual spot checks |

> **🏭 Production Practice**
>
> - **Build your own evaluation set**: General benchmarks only reflect a model's general capabilities; production requires building test sets tailored to your own business scenarios (typically 100–500 cases)
> - **Three-layer evaluation system**: Automated rules (fast) → LLM Judge (batch) → manual spot checks (precise), three progressive layers
> - **Evaluation frequency**: Run a full evaluation after every model upgrade or prompt change; integrate automated evaluation into CI/CD
> - **Watch Metrics**: In production, focus on monitoring these three metrics: P95 latency, tool call success rate, and user rating distribution

---

## Regression Testing: Ensuring Improvements Don't Introduce New Problems

```python
class RegressionTracker:
    """Regression test tracker"""
    
    def __init__(self, history_file: str = "eval_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> list:
        try:
            with open(self.history_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def record(self, version: str, metrics: AgentMetrics):
        """Record an evaluation result"""
        entry = {
            "version": version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": metrics.accuracy,
            "success_rate": metrics.success_rate,
            "avg_latency": metrics.avg_latency,
            "avg_tokens": metrics.avg_tokens
        }
        self.history.append(entry)
        
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def check_regression(
        self,
        current: AgentMetrics,
        threshold: float = 0.05
    ) -> list[str]:
        """Check if any metric has regressed beyond the threshold"""
        if not self.history:
            return []
        
        previous = self.history[-1]
        warnings = []
        
        if previous["accuracy"] - current.accuracy > threshold:
            warnings.append(
                f"⚠️ Accuracy dropped: "
                f"{previous['accuracy']:.1%} → {current.accuracy:.1%}"
            )
        
        if previous["success_rate"] - current.success_rate > threshold:
            warnings.append(
                f"⚠️ Success rate dropped: "
                f"{previous['success_rate']:.1%} → {current.success_rate:.1%}"
            )
        
        if (current.avg_latency > previous["avg_latency"] * 1.5 
            and previous["avg_latency"] > 0):
            warnings.append(
                f"⚠️ Latency increased: "
                f"{previous['avg_latency']:.2f}s → {current.avg_latency:.2f}s"
            )
        
        return warnings
```

---

## Summary

| Benchmark | Core Capability | Evaluation Method | Current SOTA |
|-----------|----------------|------------------|-------------|
| **BFCL** | Tool calling | AST matching algorithm | GPT-4o ~90% |
| **GAIA** | General reasoning | Quasi-exact match | GPT-4o ~75% (L1) |
| **AgentBench** | Comprehensive Agent | Task success rate | GPT-4o ~42% |
| **WebArena** | Web operations | State checking | GPT-4o ~35% |
| **SWE-bench** | Software engineering | Test pass rate | DeepSWE 59% |

> **Preview of next section**: Now that we've mastered evaluation methods, let's learn how to improve Agent performance through prompt tuning.

---

## References

[1] LIU X, YU H, ZHANG H, et al. AgentBench: Evaluating LLMs as agents[C]//ICLR. 2024.

[2] YAN F, MIAO H, ZHONG C, et al. Berkeley function calling leaderboard[EB/OL]. 2024. https://gorilla.cs.berkeley.edu/leaderboard.html.

[3] MIALON G, FOURRIER C, SWIFT C, et al. GAIA: A benchmark for general AI assistants[C]//ICLR. 2024.

[4] LIU X, YU H, ZHANG H, et al. AgentBench: Evaluating LLMs as agents[C]//ICLR. 2024.

[5] ZHOU S, XU F F, ZHU H, et al. WebArena: A realistic web environment for building autonomous agents[C]//ICLR. 2024.

[6] JIMENEZ C E, YANG J, WETTIG A, et al. SWE-bench: Can language models resolve real-world GitHub issues?[C]//ICLR. 2024.

[7] ZHENG L, CHIANG W L, SHENG Y, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena[C]//NeurIPS. 2023.

[8] ZHANG Y, WANG Y, ZHU Y, et al. ClawBench: Can AI agents complete everyday online tasks?[EB/OL]. arXiv:2604.08523, 2026. https://arxiv.org/abs/2604.08523.

---

[Next section: Prompt Tuning Strategies →](./03_prompt_tuning.md)
