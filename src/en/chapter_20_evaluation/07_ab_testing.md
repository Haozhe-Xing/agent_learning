# 20.7 A/B Testing and Regression Test Automation

> **Section Goal**: Master A/B testing and regression testing methods for Agents, build an automated prompt-variant testing framework, and integrate evaluation into the CI/CD pipeline.

---

## Why Do Agents Need A/B Testing?

When you change a prompt, adjust a tool description, or switch model versions — how do you know the change actually made things better?

Traditional software has unit tests: you change the code and run the tests to find out. But an Agent's output is non-deterministic, so "running it once" cannot give a statistically reliable conclusion. We need **rigorous A/B testing** to verify the effect of every change.

### Common Scenarios

| Change Type | Risk | Value of A/B Testing |
|------------|------|----------------------|
| System prompt change | May improve some scenarios but break others | Quantify and compare the effects of different prompts |
| Model version change | The new model may regress | Compare old and new models on real tasks |
| Tool description change | May cause tool misuse | Detect changes in tool-selection accuracy |
| Adding a new tool | May interfere with existing tool calls | Evaluate the impact on existing capabilities |
| Temperature change | Affects output diversity and quality | Balance creativity against stability |

---

## A/B Test Framework Design

### Core Concepts

The core idea of A/B testing: run **two Agent versions** on the **same test set**, then **compare their performance statistically**.

```python
"""
Agent A/B testing framework
Supports: prompt-variant comparison, statistical significance testing, automated reporting
"""
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

from langchain_openai import ChatOpenAI
from scipy import stats
import numpy as np


class TestStatus(Enum):
    """Test status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TestCase:
    """Test case"""
    id: str
    query: str                           # user input
    expected_output: Optional[str] = None # expected output (optional)
    expected_tools: Optional[list[str]] = None  # expected tools to call
    category: str = "default"            # category label
    difficulty: str = "medium"           # easy / medium / hard
    metadata: dict = field(default_factory=dict)


@dataclass
class TestRun:
    """Result of a single test run"""
    test_case_id: str
    variant: str              # "A" or "B"
    output: str               # agent output
    tools_called: list[str] = field(default_factory=list)  # tools called
    steps: int = 0            # number of execution steps
    tokens: int = 0           # token usage
    latency: float = 0.0      # response time (seconds)
    error: Optional[str] = None  # error message
    judge_score: Optional[float] = None  # judge score


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    name: str
    description: str = ""
    # variant A (control group)
    variant_a_config: dict = field(default_factory=dict)
    # variant B (treatment group)
    variant_b_config: dict = field(default_factory=dict)
    # test parameters
    confidence_level: float = 0.95      # confidence level
    min_sample_size: int = 30           # minimum sample size
    max_sample_size: int = 200          # maximum sample size
    metrics: list[str] = field(default_factory=lambda: [
        "quality_score", "tool_accuracy", "latency", "token_usage"
    ])
```

### A/B Test Framework Implementation

> ⚠️ **Honest note**: The `ABTestFramework` below is a **scaffold** — it is not tied to any specific Agent or fixed test set. To actually run it, you need to provide `agent_factory_a/b` (functions that produce real Agents) and a fixed `test_cases`. This book ships a **real, runnable** evaluation harness at `reference-agent/evaluation/harness.py` (based on a fixed JSONL task set, reproducible offline). We recommend implementing the A/B ideas here on top of that harness, rather than building another set of code that stays at the class-definition level.

```python
class ABTestFramework:
    """Agent A/B testing framework"""

    def __init__(
        self,
        agent_factory_a: Callable,
        agent_factory_b: Callable,
        judge_model: str = "gpt-4.1"
    ):
        """
        Args:
            agent_factory_a: factory function that creates the variant A Agent
            agent_factory_b: factory function that creates the variant B Agent
            judge_model: the judge model used for evaluation
        """
        self.agent_factory_a = agent_factory_a
        self.agent_factory_b = agent_factory_b
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    def run_test(
        self,
        test_cases: list[TestCase],
        config: ABTestConfig,
        progress_callback: Callable = None
    ) -> dict:
        """Run the full A/B test"""

        results_a: list[TestRun] = []
        results_b: list[TestRun] = []

        total = len(test_cases)

        for i, case in enumerate(test_cases):
            if progress_callback:
                progress_callback(i, total, case.id)

            # run variant A
            run_a = self._run_single(case, self.agent_factory_a, "A")
            results_a.append(run_a)

            # run variant B
            run_b = self._run_single(case, self.agent_factory_b, "B")
            results_b.append(run_b)

            # always evaluate both variants' outputs with the judge
            self._judge_outputs(case, run_a, run_b)

        # statistical analysis
        analysis = self._analyze_results(results_a, results_b, config)

        return {
            "config": {
                "name": config.name,
                "variant_a": config.variant_a_config,
                "variant_b": config.variant_b_config,
                "sample_size": len(test_cases)
            },
            "results_a": results_a,
            "results_b": results_b,
            "analysis": analysis
        }

    def _run_single(
        self,
        case: TestCase,
        agent_factory: Callable,
        variant: str
    ) -> TestRun:
        """Run a single test"""
        start_time = time.time()
        try:
            agent = agent_factory()
            result = agent.invoke(case.query)

            # extract result info
            output = ""
            tools_called = []
            steps = 0
            tokens = 0

            if isinstance(result, dict):
                output = result.get("output", str(result))
                tools_called = result.get("tools_called", [])
                steps = result.get("steps", 1)
                tokens = result.get("tokens", 0)
            else:
                output = str(result)

            latency = time.time() - start_time

            return TestRun(
                test_case_id=case.id,
                variant=variant,
                output=output,
                tools_called=tools_called,
                steps=steps,
                tokens=tokens,
                latency=latency
            )

        except Exception as e:
            return TestRun(
                test_case_id=case.id,
                variant=variant,
                output="",
                latency=time.time() - start_time,
                error=str(e)
            )

    def _judge_outputs(
        self,
        case: TestCase,
        run_a: TestRun,
        run_b: TestRun
    ):
        """Evaluate both outputs using an LLM judge"""

        # check tool-call accuracy
        if case.expected_tools:
            run_a_tools = set(run_a.tools_called)
            run_b_tools = set(run_b.tools_called)
            expected = set(case.expected_tools)

            # compute tool accuracy (Jaccard similarity)
            if expected:
                run_a.judge_score = len(run_a_tools & expected) / len(expected)
                run_b.judge_score = len(run_b_tools & expected) / len(expected)
                return

        # evaluate output quality with an LLM judge
        prompt = f"""You are a professional AI output quality reviewer. Please evaluate the quality of the following two Agent answers.

User question: {case.query}
{'Expected output: ' + case.expected_output if case.expected_output else ''}

Answer A:
{run_a.output}

Answer B:
{run_b.output}

Please score each answer from 0 to 10 and reply in JSON format:
{{
    "score_a": <0-10>,
    "score_b": <0-10>,
    "reasoning": "brief explanation of the scoring"
}}"""

        response = self.judge_llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            run_a.judge_score = result.get("score_a", 0) / 10.0
            run_b.judge_score = result.get("score_b", 0) / 10.0
        except json.JSONDecodeError:
            run_a.judge_score = 0.5
            run_b.judge_score = 0.5

    def _analyze_results(
        self,
        results_a: list[TestRun],
        results_b: list[TestRun],
        config: ABTestConfig
    ) -> dict:
        """Statistically analyze the test results"""

        # extract each metric's data
        scores_a = [r.judge_score for r in results_a if r.judge_score is not None]
        scores_b = [r.judge_score for r in results_b if r.judge_score is not None]

        latencies_a = [r.latency for r in results_a if r.error is None]
        latencies_b = [r.latency for r in results_b if r.error is None]

        tokens_a = [r.tokens for r in results_a if r.error is None]
        tokens_b = [r.tokens for r in results_b if r.error is None]

        errors_a = sum(1 for r in results_a if r.error is not None)
        errors_b = sum(1 for r in results_b if r.error is not None)

        analysis = {
            "quality": self._compare_groups(scores_a, scores_b, config),
            "latency": self._compare_groups(latencies_a, latencies_b, config),
            "token_usage": self._compare_groups(tokens_a, tokens_b, config),
            "error_rates": {
                "variant_a": errors_a / len(results_a) if results_a else 0,
                "variant_b": errors_b / len(results_b) if results_b else 0,
            },
            "summary": {
                "variant_a_avg_score": np.mean(scores_a) if scores_a else 0,
                "variant_b_avg_score": np.mean(scores_b) if scores_b else 0,
                "winner": self._determine_winner(scores_a, scores_b)
            }
        }

        return analysis

    def _compare_groups(
        self,
        group_a: list[float],
        group_b: list[float],
        config: ABTestConfig
    ) -> dict:
        """Compare the statistical significance of the two groups"""
        if len(group_a) < 2 or len(group_b) < 2:
            return {
                "significant": False,
                "p_value": None,
                "note": "insufficient sample size"
            }

        # use Welch's t-test (does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.std(group_a, ddof=1)**2 + np.std(group_b, ddof=1)**2) / 2
        )
        cohens_d = (
            (np.mean(group_a) - np.mean(group_b)) / pooled_std
            if pooled_std > 0 else 0
        )

        alpha = 1 - config.confidence_level

        return {
            "mean_a": float(np.mean(group_a)),
            "mean_b": float(np.mean(group_b)),
            "std_a": float(np.std(group_a, ddof=1)),
            "std_b": float(np.std(group_b, ddof=1)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "cohens_d": float(cohens_d),
            "effect_size": (
                "large" if abs(cohens_d) >= 0.8
                else "medium" if abs(cohens_d) >= 0.5
                else "small" if abs(cohens_d) >= 0.2
                else "negligible"
            )
        }

    @staticmethod
    def _determine_winner(
        scores_a: list[float],
        scores_b: list[float]
    ) -> str:
        """Determine the winning variant"""
        if not scores_a or not scores_b:
            return "undetermined"
        avg_a = np.mean(scores_a)
        avg_b = np.mean(scores_b)
        diff = abs(avg_a - avg_b)
        if diff < 0.05:
            return "tie (no significant difference)"
        return "A" if avg_a > avg_b else "B"
```

### Example: Comparing Two Prompt Variants

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# define two prompt variants
PROMPT_A = """You are a customer-service assistant. Answer the user's questions. Be concise and clear."""

PROMPT_B = """You are a professional customer-service assistant. When answering the user, follow these principles:
1. First confirm you understand the user's question.
2. Give an accurate, complete answer.
3. If the answer involves steps, list them with numbers.
4. Finally, ask whether there is anything else you can help with."""


def create_agent_a():
    """Create the variant A Agent"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_A),
        ("human", "{input}")
    ])
    chain = prompt | llm

    def agent_func(query: str) -> dict:
        response = chain.invoke({"input": query})
        return {
            "output": response.content,
            "steps": 1,
            "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        }

    return agent_func


def create_agent_b():
    """Create the variant B Agent"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_B),
        ("human", "{input}")
    ])
    chain = prompt | llm

    def agent_func(query: str) -> dict:
        response = chain.invoke({"input": query})
        return {
            "output": response.content,
            "steps": 1,
            "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        }

    return agent_func


# prepare test cases
test_cases = [
    TestCase(
        id="cs_001",
        query="My order hasn't arrived yet; it's already 3 days past the estimated delivery time.",
        category="logistics inquiry",
        difficulty="easy"
    ),
    TestCase(
        id="cs_002",
        query="I want to return an item, but it's already been opened. Can I still return it?",
        category="returns and exchanges",
        difficulty="medium"
    ),
    TestCase(
        id="cs_003",
        query="What are your membership program and points rules? What extra benefits does VIP get?",
        category="membership service",
        difficulty="hard"
    ),
]

# configure the A/B test
config = ABTestConfig(
    name="Customer-service prompt optimization test",
    description="Compare the customer-service quality of the concise vs. detailed prompt",
    variant_a_config={"prompt": "concise", "model": "gpt-4.1-mini"},
    variant_b_config={"prompt": "detailed", "model": "gpt-4.1-mini"},
    confidence_level=0.95
)

# run the test
framework = ABTestFramework(
    agent_factory_a=create_agent_a,
    agent_factory_b=create_agent_b,
    judge_model="gpt-4.1"
)

result = framework.run_test(test_cases, config)

# output results
print(f"Variant A average score: {result['analysis']['summary']['variant_a_avg_score']:.2f}")
print(f"Variant B average score: {result['analysis']['summary']['variant_b_avg_score']:.2f}")
print(f"Winner: {result['analysis']['summary']['winner']}")
print(f"Quality difference significant: {result['analysis']['quality']['significant']}")
print(f"Effect size: {result['analysis']['quality']['effect_size']}")
```

### A/B Testing Caveats

| Issue | Description | Mitigation |
|------|------|----------|
| Insufficient sample size | Statistical tests need enough samples | At least 30 test cases per variant |
| Randomness interference | LLM output randomness may affect results | Run each case 3-5 times and average |
| Judge bias | The evaluation model may favor a certain style | Average multiple judges |
| Test-set overfitting | Repeated optimization overfits the test set | Keep a hold-out test set |
| Multiple comparisons | Testing many metrics at once raises false positives | Use Bonferroni correction |

> 💡 **Best practice**: Before running an A/B test, use a power analysis to compute the required sample size. "Significant" results from small samples are often unreliable.

---

## Regression Testing: Preventing Prompt Changes from Breaking Existing Capabilities

### What Is Agent Regression Testing?

The core goal of regression testing: **make sure new changes don't break already-correct behavior**. In Agent development, this means that after every change to a prompt, tool description, or model parameter, you must verify that key scenarios still work correctly.

### Regression Testing Strategies

| Strategy | Description | Use Case |
|------|------|----------|
| Snapshot test | Record the expected output and check if the new output deviates | Scenarios with relatively fixed output |
| Behavior test | Check whether key behaviors are correct (e.g., tool calls) | Tool-usage scenarios |
| Semantic test | Use an LLM judge to check semantic equivalence | Scenarios where output changes but meaning stays the same |
| Boundary test | Test extreme inputs and edge cases | Robustness validation |

### Regression Test Framework Implementation

> ⚠️ **Honest note**: `RegressionTestSuite` is also a **scaffold** — it defines the interfaces for "register a case / save a baseline / compare for regression", but the `expected_*` checks inside `register_test` and the semantic-similarity comparison require you to wire up a real Agent and a judgment backend to actually work. This book's `reference-agent/evaluation/harness.py` already provides a lighter, immediately `pytest`-reproducible judgment implementation (`expect_contains`), which you can use as a starting point for extending semantic/snapshot judging.

```python
class RegressionTestSuite:
    """Agent regression test suite"""

    def __init__(
        self,
        agent_factory: Callable,
        judge_model: str = "gpt-4.1"
    ):
        self.agent_factory = agent_factory
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)
        self.baselines: dict[str, dict] = {}  # baseline results
        self.test_cases: list[dict] = []

    def register_test(
        self,
        name: str,
        query: str,
        test_type: str = "semantic",  # snapshot / behavior / semantic / boundary
        expected_tools: list[str] = None,
        expected_keywords: list[str] = None,
        expected_snapshot: str = None,
        category: str = "default",
        tolerance: float = 0.1  # semantic similarity tolerance
    ):
        """Register a regression test case"""
        self.test_cases.append({
            "name": name,
            "query": query,
            "type": test_type,
            "expected_tools": expected_tools or [],
            "expected_keywords": expected_keywords or [],
            "expected_snapshot": expected_snapshot,
            "category": category,
            "tolerance": tolerance
        })

    def save_baseline(self, output_path: str = "baseline.json"):
        """Save the current results as the baseline"""
        baseline_results = {}
        agent = self.agent_factory()

        for case in self.test_cases:
            result = agent(case["query"])
            baseline_results[case["name"]] = {
                "output": result if isinstance(result, str) else str(result),
                "query": case["query"],
                "timestamp": time.time()
            }

        with open(output_path, "w") as f:
            json.dump(baseline_results, f, ensure_ascii=False, indent=2)

        self.baselines = baseline_results
        print(f"Baseline saved to {output_path}, {len(baseline_results)} cases in total")

    def load_baseline(self, input_path: str = "baseline.json"):
        """Load an existing baseline"""
        with open(input_path, "r") as f:
            self.baselines = json.load(f)
        print(f"Loaded baseline with {len(self.baselines)} cases")

    def run_regression(self) -> dict:
        """Run the regression tests"""
        agent = self.agent_factory()
        results = []

        for case in self.test_cases:
            # execute the test
            actual_output = agent(case["query"])
            actual_str = actual_output if isinstance(actual_output, str) else str(actual_output)

            # check based on type
            if case["type"] == "snapshot":
                passed = self._check_snapshot(case, actual_str)
            elif case["type"] == "behavior":
                passed = self._check_behavior(case, actual_output)
            elif case["type"] == "semantic":
                passed = self._check_semantic(case, actual_str)
            elif case["type"] == "boundary":
                passed = self._check_boundary(case, actual_str)
            else:
                passed = True  # unknown type passes by default

            results.append({
                "name": case["name"],
                "category": case["category"],
                "type": case["type"],
                "passed": passed,
                "query": case["query"]
            })

        # summarize
        total = len(results)
        passed = sum(1 for r in results if r["passed"])

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "details": results
        }

    def _check_snapshot(self, case: dict, actual: str) -> bool:
        """Snapshot test: check whether the output exactly matches the baseline"""
        baseline = self.baselines.get(case["name"])
        if not baseline:
            return True  # skip if no baseline

        expected = baseline["output"]
        # allow whitespace differences
        return actual.strip() == expected.strip()

    def _check_behavior(self, case: dict, actual_output) -> bool:
        """Behavior test: check whether key behaviors are correct"""
        checks_passed = 0
        total_checks = 0

        # check tool calls
        if case["expected_tools"]:
            total_checks += 1
            actual_tools = set()
            if isinstance(actual_output, dict):
                actual_tools = set(actual_output.get("tools_called", []))

            expected_tools = set(case["expected_tools"])
            if expected_tools.issubset(actual_tools):
                checks_passed += 1

        # check keywords
        if case["expected_keywords"]:
            total_checks += 1
            actual_str = str(actual_output)
            if all(kw in actual_str for kw in case["expected_keywords"]):
                checks_passed += 1

        if total_checks == 0:
            return True

        return checks_passed == total_checks

    def _check_semantic(self, case: dict, actual: str) -> bool:
        """Semantic test: use an LLM judge to check semantic equivalence"""
        baseline = self.baselines.get(case["name"])
        if not baseline:
            return True

        expected = baseline["output"]

        prompt = f"""Determine whether the following two answers are semantically equivalent.

Question: {case['query']}

Answer A (baseline):
{expected}

Answer B (current):
{actual}

Reply with JSON only: {{"equivalent": true/false, "confidence": 0.0-1.0}}"""

        response = self.judge_llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return result.get("equivalent", False) and result.get("confidence", 0) >= (1 - case["tolerance"])
        except json.JSONDecodeError:
            return False

    def _check_boundary(self, case: dict, actual: str) -> bool:
        """Boundary test: check whether the output is reasonable (no crash, empty output, etc.)"""
        # basic checks
        if not actual or len(actual.strip()) < 5:
            return False

        # check for error markers
        error_markers = ["error", "exception", "traceback", "I cannot", "无法完成"]
        actual_lower = actual.lower()
        for marker in error_markers:
            if marker in actual_lower:
                return False

        return True
```

### Hands-on: Building a Regression Test for a Customer-Service Agent

```python
# create the regression test suite
agent_factory = create_agent_a  # use your agent factory function

suite = RegressionTestSuite(
    agent_factory=agent_factory,
    judge_model="gpt-4.1"
)

# register test cases
suite.register_test(
    name="Order inquiry",
    query="Check the status of my order",
    test_type="behavior",
    expected_tools=["query_order"],
    expected_keywords=["order"],
    category="core feature"
)

suite.register_test(
    name="Return process",
    query="I want to return an item. How do I do that?",
    test_type="semantic",
    category="core feature"
)

suite.register_test(
    name="Empty input handling",
    query="",
    test_type="boundary",
    category="boundary handling"
)

suite.register_test(
    name="Overly long input handling",
    query="Please help me process " + "very urgent" * 100,
    test_type="boundary",
    category="boundary handling"
)

# first run: save the baseline
suite.save_baseline("regression_baseline.json")

# later runs: load the baseline and run regression
suite.load_baseline("regression_baseline.json")
results = suite.run_regression()

print(f"Regression test result: {results['passed']}/{results['total']} passed")
print(f"Pass rate: {results['pass_rate']:.1%}")

# print failed cases
for detail in results["details"]:
    if not detail["passed"]:
        print(f"  ❌ {detail['name']} ({detail['type']}): {detail['query']}")
```

---

## CI/CD Integration: Automated Evaluation Pipeline

### Why CI/CD Integration?

Running tests manually is easy to overlook. Integrating Agent evaluation into the CI/CD pipeline runs evaluation automatically on every code change, catching problems early.

### GitHub Actions Automated Evaluation Configuration

```yaml
# .github/workflows/agent_eval.yml
name: Agent Evaluation

on:
  pull_request:
    paths:
      - 'agent/**'
      - 'prompts/**'
      - 'tests/**'
  schedule:
    # run the full evaluation every day at 2 AM
    - cron: '0 2 * * *'

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest scipy numpy

      - name: Run Regression Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m pytest tests/regression/ -v --tb=short

      - name: Run Eval (quality gate)
        run: |
          cd reference-agent
          pip install -e .
          python -m pytest -q

      - name: Check Quality Gate
        run: |
          # in CI, the eval's pass_rate is the quality gate;
          # when the pytest above fails, the pipeline aborts, which is equivalent to the gate failing.
          echo "Quality gate enforced by pytest exit code above."

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: results/
```

> 💡 **Relationship between CI and the scaffolds above**: The workflow above runs this book's bundled `reference-agent` evaluation directly (`pip install -e . && pytest`); its `pass_rate` is the quality gate — when pytest fails, the pipeline aborts. The `ABTestFramework` / `RegressionTestSuite` / `QualityGate` from the previous sections are **conceptual scaffolds**: when you want to do variant comparison or finer semantic regression in your own repo, just wire them to a real Agent and a fixed dataset — you don't need to design an evaluation paradigm from scratch.

### Quality Gate Script

```python
"""
Quality gate check script
Used in CI/CD pipelines to decide whether a merge is allowed based on evaluation results.
"""
import json
import sys
from dataclasses import dataclass


@dataclass
class QualityGateConfig:
    """Quality gate configuration"""
    min_pass_rate: float = 0.85       # minimum regression test pass rate
    max_regression: float = 0.05      # maximum allowed regression
    min_ab_score: float = 0.7         # minimum A/B test score
    max_latency_increase: float = 0.2 # maximum latency increase ratio
    max_token_increase: float = 0.2   # maximum token increase ratio


class QualityGate:
    """Quality gate check"""

    def __init__(self, config: QualityGateConfig = None):
        self.config = config or QualityGateConfig()

    def check(
        self,
        regression_results: dict = None,
        ab_test_results: dict = None
    ) -> dict:
        """Run the quality gate checks"""
        checks = []

        # check 1: regression test pass rate
        if regression_results:
            pass_rate = regression_results.get("pass_rate", 0)
            checks.append({
                "name": "Regression test pass rate",
                "value": pass_rate,
                "threshold": self.config.min_pass_rate,
                "passed": pass_rate >= self.config.min_pass_rate,
                "message": (
                    f"Pass rate {pass_rate:.1%} >= {self.config.min_pass_rate:.1%}"
                    if pass_rate >= self.config.min_pass_rate
                    else f"Pass rate {pass_rate:.1%} < {self.config.min_pass_rate:.1%}"
                )
            })

        # check 2: A/B test regression
        if ab_test_results:
            analysis = ab_test_results.get("analysis", {})
            quality = analysis.get("quality", {})
            score_a = quality.get("mean_a", 0)
            score_b = quality.get("mean_b", 0)

            # B is the new version, A is the baseline
            regression = max(0, score_a - score_b)

            checks.append({
                "name": "Quality regression",
                "value": regression,
                "threshold": self.config.max_regression,
                "passed": regression <= self.config.max_regression,
                "message": (
                    f"Regression {regression:.3f} <= {self.config.max_regression}"
                    if regression <= self.config.max_regression
                    else f"Regression {regression:.3f} > {self.config.max_regression}"
                )
            })

            # check 3: latency increase
            latency = analysis.get("latency", {})
            lat_a = latency.get("mean_a", 0)
            lat_b = latency.get("mean_b", 0)
            if lat_a > 0:
                latency_increase = (lat_b - lat_a) / lat_a
                checks.append({
                    "name": "Latency increase ratio",
                    "value": latency_increase,
                    "threshold": self.config.max_latency_increase,
                    "passed": latency_increase <= self.config.max_latency_increase,
                    "message": (
                        f"Latency increase {latency_increase:.1%} <= {self.config.max_latency_increase:.1%}"
                        if latency_increase <= self.config.max_latency_increase
                        else f"Latency increase {latency_increase:.1%} > {self.config.max_latency_increase:.1%}"
                    )
                })

            # check 4: token increase ratio
            token_usage = analysis.get("token_usage", {})
            tok_a = token_usage.get("mean_a", 0)
            tok_b = token_usage.get("mean_b", 0)
            if tok_a > 0:
                token_increase = (tok_b - tok_a) / tok_a
                checks.append({
                    "name": "Token increase ratio",
                    "value": token_increase,
                    "threshold": self.config.max_token_increase,
                    "passed": token_increase <= self.config.max_token_increase,
                    "message": (
                        f"Token increase {token_increase:.1%} <= {self.config.max_token_increase:.1%}"
                        if token_increase <= self.config.max_token_increase
                        else f"Token increase {token_increase:.1%} > {self.config.max_token_increase:.1%}"
                    )
                })

        all_passed = all(c["passed"] for c in checks)

        return {
            "passed": all_passed,
            "checks": checks,
            "summary": (
                "All quality gate checks passed ✅"
                if all_passed
                else f"{sum(1 for c in checks if not c['passed'])} checks failed ❌"
            )
        }


def main():
    """CI/CD entry point"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", help="regression test result file")
    parser.add_argument("--ab-test", help="A/B test result file")
    parser.add_argument("--min-pass-rate", type=float, default=0.85)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()

    # load results
    regression_results = None
    ab_test_results = None

    if args.regression:
        with open(args.regression) as f:
            regression_results = json.load(f)

    if args.ab_test:
        with open(args.ab_test) as f:
            ab_test_results = json.load(f)

    # run the quality gate
    config = QualityGateConfig(
        min_pass_rate=args.min_pass_rate,
        max_regression=args.max_regression
    )
    gate = QualityGate(config)
    result = gate.check(regression_results, ab_test_results)

    print(result["summary"])
    for check in result["checks"]:
        status = "✅" if check["passed"] else "❌"
        print(f"  {status} {check['name']}: {check['message']}")

    # a non-zero exit code means the checks failed
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
```

### Test Pipeline Architecture

```
Code change → CI/CD triggered
    │
    ├── 1. Regression test (fast, < 5 min)
    │   ├── Snapshot test
    │   ├── Behavior test
    │   └── Boundary test
    │
    ├── 2. A/B test (medium, 10-30 min)
    │   ├── Run baseline version
    │   ├── Run new version
    │   └── LLM judge evaluation
    │
    └── 3. Quality gate check
        ├── Regression pass rate ≥ 85%
        ├── Quality regression ≤ 5%
        ├── Latency increase ≤ 20%
        └── Token increase ≤ 20%
```

### CI/CD Integration Caveats

| Issue | Description | Mitigation |
|------|------|----------|
| API cost | Running evaluation on every commit is expensive | Run only regression tests on PRs; run the full evaluation on a schedule |
| Execution time | LLM evaluation is time-consuming | Parallelization + result caching |
| Non-determinism | The same code may yield different results across runs | Run multiple times and average + tolerance threshold |
| False positives | Occasional quality fluctuations get flagged as regressions | Set a reasonable tolerance and review manually |
| Secret safety | API keys must not be hard-coded | Use GitHub Secrets |

> 💡 **Best practice**: Set different evaluation strategies for different types of changes. Prompt changes run the full A/B test; code changes run only regression tests; documentation changes skip evaluation.

---

## Summary

| Concept | Description |
|---------|-------------|
| A/B testing | Compare two Agent variants on the same test set and test difference significance statistically |
| Statistical significance | Use Welch's t-test to judge whether the difference is caused by random fluctuation |
| Effect size | Cohen's d measures the practical meaning of the difference (small/medium/large) |
| Regression testing | Ensure changes don't break existing capabilities; 4 strategies: snapshot / behavior / semantic / boundary |
| Quality gate | Automated check in CI/CD that blocks the merge if it fails |
| CI/CD integration | GitHub Actions runs evaluation automatically; results become a PR merge condition |

> **Next section preview**: We will learn about model routing evaluation — how to intelligently select a model based on task complexity, finding the best balance between cost and quality.

---

## References

[1] KOHAVI R, TANG D, XU Y. Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing[M]. Cambridge University Press, 2020.

[2] ZHENG L, CHIANG W L, SHENG Y, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena[C]//NeurIPS. 2023.

[3] TAMARIT J, SNOEK J, METZE F. A/B Testing for LLMs: Practical Considerations and Pitfalls[J]. arXiv preprint, 2024.

---

[20.8 Model Routing Evaluation](./08_model_routing.md)
