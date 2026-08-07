# 18.8 Model Routing Evaluation

> **Goal of this section**: Understand the core problem of model routing, master the cost-quality trade-off analysis method, and be able to implement and evaluate an intelligent router that selects the optimal model for each task in a multi-model environment.

---

## Why Model Routing Matters

In an Agent system, not every task needs the strongest model. Simple questions can be solved with a small model; only complex reasoning and planning require a large model. **Model routing** dynamically selects the most suitable model based on task characteristics, finding the optimal balance between cost and quality.

### The Cost Reality

| Model | Input Price (/1M tokens) | Output Price (/1M tokens) | Reasoning Ability | Speed |
|------|----------------------|----------------------|----------|------|
| gpt-4.1 | $2.00 | $8.00 | Strong | Medium |
| gpt-4.1-mini | $0.40 | $1.60 | Medium | Fast |
| gpt-4.1-nano | $0.10 | $0.40 | Basic | Fastest |

Suppose an Agent processes 10,000 requests per day:

- **All on gpt-4.1**: about $100/day, or $3,000/month
- **Intelligent routing (70% small model + 30% large model)**: about $40/day, or $1,200/month
- **Savings**: about $1,800/month, or $21,600/year

> 💡 **Key insight**: In production, most requests are simple tasks (FAQ, format conversion, information extraction), and only a few require deep reasoning. Intelligent routing can "save the large model for the scenarios that truly need it."

---

## When to Use Large vs Small Models

### Decision Framework

```
Incoming task
    │
    ├─ Task classification
    │   ├── Simple (fact lookup, format conversion, simple summarization) → small model
    │   ├── Medium (multi-step reasoning, tool calling, needs context understanding) → medium model
    │   └── Complex (creative writing, complex planning, multi-constraint optimization) → large model
    │
    ├─ Risk assessment
    │   ├── Low risk (internal tools, not user-facing) → small model is acceptable
    │   └── High risk (user-facing, involves decisions) → prefer the large model
    │
    └─ Cost budget
        ├── Generous → lean toward the large model
        └── Tight → lean toward the small model + human review
```

### Task Complexity Classification Criteria

| Dimension | Simple (small model) | Medium (medium model) | Complex (large model) |
|------|---------------|---------------|---------------|
| Reasoning steps | 1 step | 2-3 steps | 4+ steps |
| Tool calls | None | 1-2 | 3+ |
| Input length | < 500 tokens | 500-2000 tokens | 2000+ tokens |
| Output requirement | Fixed format | Semi-structured | Open-ended |
| Fault tolerance | High (errors are OK) | Medium | Low (must be accurate) |
| Typical tasks | Intent classification, keyword extraction | RAG QA, simple tool calling | Complex planning, multi-turn dialogue |

---

## Cost-Quality Trade-Off Analysis

### The Relationship Between Quality and Cost

```python
"""
Cost-quality trade-off analysis tool
"""
import json
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI


@dataclass
class ModelProfile:
    """Model configuration"""
    name: str
    input_cost_per_mtok: float     # Cost per million input tokens
    output_cost_per_mtok: float    # Cost per million output tokens
    avg_latency_ms: float          # Average latency (milliseconds)
    quality_score: float           # Quality score (0-1, based on benchmarks)


@dataclass
class TaskProfile:
    """Task configuration"""
    name: str
    avg_input_tokens: int          # Average input token count
    avg_output_tokens: int         # Average output token count
    daily_volume: int              # Daily request volume
    quality_requirement: float     # Minimum quality requirement (0-1)


# Define the model profiles
MODELS = {
    "gpt-4.1": ModelProfile(
        name="gpt-4.1",
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
        avg_latency_ms=1500,
        quality_score=0.95
    ),
    "gpt-4.1-mini": ModelProfile(
        name="gpt-4.1-mini",
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
        avg_latency_ms=500,
        quality_score=0.85
    ),
    "gpt-4.1-nano": ModelProfile(
        name="gpt-4.1-nano",
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
        avg_latency_ms=200,
        quality_score=0.72
    ),
}


class CostQualityAnalyzer:
    """Cost-quality trade-off analyzer"""

    def __init__(self, models: dict[str, ModelProfile] = None):
        self.models = models or MODELS

    def calculate_cost(
        self,
        model: ModelProfile,
        task: TaskProfile
    ) -> float:
        """Calculate the daily cost"""
        input_cost = (
            task.avg_input_tokens / 1_000_000
            * model.input_cost_per_mtok
            * task.daily_volume
        )
        output_cost = (
            task.avg_output_tokens / 1_000_000
            * model.output_cost_per_mtok
            * task.daily_volume
        )
        return input_cost + output_cost

    def analyze(
        self,
        task: TaskProfile
    ) -> dict:
        """Analyze the cost and quality of all models"""
        results = []

        for name, model in self.models.items():
            cost = self.calculate_cost(model, task)
            meets_quality = model.quality_score >= task.quality_requirement

            results.append({
                "model": name,
                "daily_cost": cost,
                "monthly_cost": cost * 30,
                "quality_score": model.quality_score,
                "meets_quality": meets_quality,
                "avg_latency_ms": model.avg_latency_ms,
                "cost_per_quality_point": cost / model.quality_score if model.quality_score > 0 else float("inf")
            })

        # Sort: among quality-passing models, pick the cheapest
        valid = [r for r in results if r["meets_quality"]]
        if valid:
            best = min(valid, key=lambda x: x["daily_cost"])
        else:
            best = max(results, key=lambda x: x["quality_score"])

        return {
            "task": task.name,
            "models": results,
            "recommended": best["model"],
            "reason": (
                f"Meets quality ({best['quality_score']:.2f} >= {task.quality_requirement}) "
                f"and is the cheapest (${best['daily_cost']:.2f}/day)"
                if best["meets_quality"]
                else f"No model meets the quality bar; recommending the highest-quality {best['model']} ({best['quality_score']:.2f})"
            )
        }

    def analyze_routing(
        self,
        tasks: list[TaskProfile],
        routing_ratios: dict[str, float]
    ) -> dict:
        """Analyze the total cost and quality of a routing strategy"""
        total_cost = 0
        weighted_quality = 0
        total_volume = sum(t.daily_volume for t in tasks)

        for task in tasks:
            task_volume_ratio = task.daily_volume / total_volume

            for model_name, ratio in routing_ratios.items():
                model = self.models[model_name]
                volume = task.daily_volume * ratio
                adjusted_task = TaskProfile(
                    name=task.name,
                    avg_input_tokens=task.avg_input_tokens,
                    avg_output_tokens=task.avg_output_tokens,
                    daily_volume=int(volume),
                    quality_requirement=task.quality_requirement
                )
                cost = self.calculate_cost(model, adjusted_task)
                total_cost += cost
                weighted_quality += model.quality_score * volume

        weighted_quality /= total_volume if total_volume > 0 else 1

        return {
            "daily_cost": total_cost,
            "monthly_cost": total_cost * 30,
            "weighted_quality": weighted_quality,
            "routing_ratios": routing_ratios
        }


# Usage example
analyzer = CostQualityAnalyzer()

# Analyze a single task
task = TaskProfile(
    name="Customer service Q&A",
    avg_input_tokens=800,
    avg_output_tokens=300,
    daily_volume=5000,
    quality_requirement=0.80
)

result = analyzer.analyze(task)
print(f"Recommended model: {result['recommended']}")
print(f"Reason: {result['reason']}")

# Compare all models
print("\nModel comparison:")
for m in result["models"]:
    status = "✅" if m["meets_quality"] else "❌"
    print(f"  {status} {m['model']}: quality {m['quality_score']:.2f}, "
          f"daily cost ${m['daily_cost']:.2f}, latency {m['avg_latency_ms']}ms")
```

### Multi-Task Routing Strategy Comparison

```python
# Define several business tasks
tasks = [
    TaskProfile("FAQ answer", 200, 100, 3000, 0.70),
    TaskProfile("RAG QA", 1500, 400, 2000, 0.85),
    TaskProfile("Complex planning", 2000, 800, 500, 0.92),
]

# Strategy 1: all large model
strategy_all_large = {"gpt-4.1": 1.0}

# Strategy 2: all medium model
strategy_all_medium = {"gpt-4.1-mini": 1.0}

# Strategy 3: intelligent routing
strategy_smart = {"gpt-4.1-nano": 0.4, "gpt-4.1-mini": 0.4, "gpt-4.1": 0.2}

strategies = {
    "All large model": strategy_all_large,
    "All medium model": strategy_all_medium,
    "Intelligent routing": strategy_smart,
}

print("Routing strategy comparison:")
print(f"{'Strategy':<12} {'Monthly cost':<12} {'Weighted quality':<12} {'Cost-eff.'}")
print("-" * 55)

for name, ratios in strategies.items():
    result = analyzer.analyze_routing(tasks, ratios)
    cost_eff = result["weighted_quality"] / (result["monthly_cost"] / 1000)
    print(f"{name:<12} ${result['monthly_cost']:<11,.0f} {result['weighted_quality']:<13.2f} {cost_eff:.2f}")
```

| Strategy | Monthly cost | Weighted quality | Cost-efficiency |
|------|--------|----------|--------|
| All large model | ~$3,600 | 0.95 | 0.26 |
| All medium model | ~$720 | 0.85 | 1.18 |
| Intelligent routing | ~$1,080 | 0.86 | 0.80 |

> ⚠️ **Note**: Intelligent routing's quality is slightly below using all large models, but the cost drops by about 70%. The key is to find the balance point where "quality loss is acceptable and cost savings are significant."

---

## Router Model Training and Evaluation

### Core Task of the Router Model

The router model needs to solve a classification problem: given an input, predict which model it should be routed to.

### Method 1: Rule-Based Static Routing

The simplest method — hardcode routing rules based on input features:

```python
class StaticRouter:
    """Rule-based static router"""

    def __init__(self, rules: list[dict] = None):
        self.rules = rules or self._default_rules()

    def _default_rules(self) -> list[dict]:
        """Default routing rules"""
        return [
            {
                "name": "Simple task",
                "condition": lambda query: (
                    len(query) < 50
                    and any(kw in query for kw in ["what is", "how many", "when"])
                ),
                "model": "gpt-4.1-nano"
            },
            {
                "name": "Medium task",
                "condition": lambda query: (
                    len(query) < 200
                    or any(kw in query for kw in ["analyze", "compare", "summarize"])
                ),
                "model": "gpt-4.1-mini"
            },
            {
                "name": "Complex task",
                "condition": lambda query: (
                    len(query) >= 200
                    or any(kw in query for kw in ["plan", "design", "optimize"])
                ),
                "model": "gpt-4.1"
            },
        ]

    def route(self, query: str) -> str:
        """Routing decision"""
        for rule in self.rules:
            if rule["condition"](query):
                return rule["model"]
        return "gpt-4.1-mini"  # default medium model
```

**Pros**: Zero cost, deterministic, interpretable. **Cons**: Hard to maintain rules, cannot handle edge cases.

### Method 2: LLM-Based Dynamic Routing

Use a small LLM to judge task complexity:

```python
class LLMRouter:
    """LLM-based dynamic router"""

    def __init__(self, router_model: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=router_model, temperature=0)
        self.route_options = {
            "simple": "gpt-4.1-nano",
            "medium": "gpt-4.1-mini",
            "complex": "gpt-4.1"
        }

    def route(self, query: str, context: dict = None) -> dict:
        """Routing decision"""
        context_info = ""
        if context:
            context_info = f"\nAdditional context: {json.dumps(context, ensure_ascii=False)}"

        prompt = f"""You are a task complexity classifier. Judge the complexity of the following user request.

User request: {query}{context_info}

Complexity definitions:
- simple: simple fact lookup, keyword extraction, format conversion, completed in 1 step
- medium: requires reasoning, search, tool calling, completed in 2-3 steps
- complex: requires deep reasoning, multi-step planning, creative thinking, completed in 4+ steps

Reply with JSON only: {{"complexity": "simple/medium/complex", "confidence": 0.0-1.0, "reasoning": "brief reason"}}"""

        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            complexity = result.get("complexity", "medium")
            model = self.route_options.get(complexity, "gpt-4.1-mini")
            return {
                "model": model,
                "complexity": complexity,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "router_cost": self._estimate_router_cost(query)
            }
        except json.JSONDecodeError:
            return {
                "model": "gpt-4.1-mini",
                "complexity": "medium",
                "confidence": 0.0,
                "reasoning": "Routing parse failed, using default model",
                "router_cost": 0
            }

    def _estimate_router_cost(self, query: str) -> float:
        """Estimate routing cost (based on gpt-4.1-mini pricing)"""
        input_tokens = len(query) // 4 + 150  # rough estimate
        output_tokens = 50
        return (
            input_tokens / 1_000_000 * 0.4
            + output_tokens / 1_000_000 * 1.6
        )
```

**Pros**: Flexible, understands semantics. **Cons**: Extra cost and latency, and it can itself make mistakes.

### Method 3: Train a Dedicated Router Model

The most economical method — train a small classification model to make routing decisions:

```python
"""
Train a dedicated router model
Use labeled data to train a lightweight classifier
"""
import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


@dataclass
class RoutingExample:
    """Routed labeled sample"""
    query: str
    optimal_model: str       # Optimal model
    complexity: str          # simple / medium / complex
    quality_scores: dict     # Quality score of each model {model_name: score}


class RouterTrainingDataGenerator:
    """Generate training data for the router model"""

    def __init__(self, judge_model: str = "gpt-4.1"):
        self.llm = ChatOpenAI(model=judge_model, temperature=0)

    def generate_labels(
        self,
        queries: list[str],
        models: list[str] = None
    ) -> list[RoutingExample]:
        """Generate optimal-model labels for a batch of queries"""
        models = models or ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]

        labeled_data = []
        for query in queries:
            # Let the Judge model evaluate each model's fit for this query
            model_scores = {}
            for model in models:
                score = self._evaluate_model_fit(query, model)
                model_scores[model] = score

            # Select the highest-scoring model (accounting for cost)
            optimal = self._select_optimal_model(model_scores, models)

            # Determine complexity
            complexity = self._classify_complexity(query)

            labeled_data.append(RoutingExample(
                query=query,
                optimal_model=optimal,
                complexity=complexity,
                quality_scores=model_scores
            ))

        return labeled_data

    def _evaluate_model_fit(self, query: str, model: str) -> float:
        """Evaluate how well a model fits the query"""
        prompt = f"""Evaluate how well the following model fits the given query.

Query: {query}
Model: {model}

Rate the quality (0-10) of this model handling the query, considering:
- Is its reasoning ability sufficient
- Does its knowledge cover the topic
- Expected output quality

Reply with a single number from 0-10."""

        response = self.llm.invoke(prompt)
        try:
            return float(response.content.strip()) / 10.0
        except ValueError:
            return 0.5

    def _select_optimal_model(
        self,
        scores: dict[str, float],
        models: list[str]
    ) -> str:
        """Select the optimal model (balancing quality and cost)"""
        # Cost weights: cheaper models tolerate slightly lower quality
        cost_weights = {
            "gpt-4.1-nano": 1.0,    # Cheapest, least quality discount
            "gpt-4.1-mini": 0.85,   # Medium
            "gpt-4.1": 0.65,        # Most expensive, most quality discount
        }

        adjusted = {}
        for model, score in scores.items():
            weight = cost_weights.get(model, 0.8)
            adjusted[model] = score * weight

        return max(adjusted, key=adjusted.get)

    def _classify_complexity(self, query: str) -> str:
        """Classify query complexity"""
        prompt = f"""Judge the complexity of the following query.

Query: {query}

Reply with only: simple / medium / complex"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()
        if result in ("simple", "medium", "complex"):
            return result
        return "medium"

    def export_training_data(
        self,
        data: list[RoutingExample],
        output_path: str
    ):
        """Export training data in JSONL format"""
        with open(output_path, "w") as f:
            for example in data:
                record = {
                    "query": example.query,
                    "label": example.optimal_model,
                    "complexity": example.complexity,
                    "scores": example.quality_scores
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Exported {len(data)} training samples to {output_path}")
```

### Method Comparison

| Method | Cost | Accuracy | Latency | Maintainability |
|------|------|--------|------|----------|
| Static rules | Zero | 60-70% | 0ms | Low (rule bloat) |
| LLM routing | $0.001/call | 85-90% | 200-500ms | High |
| Trained router model | Training cost | 80-88% | <10ms | Medium (needs periodic retraining) |

---

## Router Implementation: Deduplication and Cross-References

> ⚠️ **Honest note**: An older version of this section gave the complete code of `SmartRouter` / `CascadeRouter` (400+ lines) in two subsections ("Complete Intelligent Router Implementation" and "Cascade Routing Implementation"). But **the real, runnable router implementation is already given in Chapter 20 (Deployment and Ops), section 06_model_serving** (`StaticRouter` / `LLMRouter` / `CascadeRouter`, including fallback and cost tracking). Writing the same set of code in two chapters is both redundant and prone to version drift — it's just "padding to pad the page."

> This section is about model routing **evaluation**, and its focus is **how to judge whether a routing strategy is good**, not re-implementing a router. Therefore we keep only the **strategy selection comparison** and the **evaluation metric system** here; for the actual router code, reuse Chapter 20 directly.

### Strategy Selection Comparison for the Three Routing Approaches (Evaluation Perspective)

| Strategy | Implementation location | Suitable evaluation scenario | Main evaluation metrics |
|------|----------|--------------|--------------|
| Static rule routing | Chapter 20 `StaticRouter` | Enumerable task types, cost-sensitive | Coverage rate, mis-routing rate |
| LLM dynamic routing | Chapter 20 `LLMRouter` | Hard-to-predict task complexity | Routing accuracy, extra latency |
| Cascade routing (small first, then large) | Chapter 20 `CascadeRouter` | Simple tasks dominate | Escalation rate, cost saving ratio |

### When to Evaluate Routing vs. Just Use a Large Model Directly

- When **simple tasks account for > 60%** and quality tolerance is high, routing (especially cascade) can significantly cut costs and is worth evaluating;
- When tasks are generally complex, or the routing's own latency/cost is non-negligible, simply standardizing on a medium/large model is actually easier — the marginal benefit of routing is limited.

> To see the real, runnable router code and fallback logic: jump to **Chapter 20, Section 6 (Model Serving and Routing)**. The second half of this section ("Routing Evaluation Metrics") tells you which numbers to look at once you have the routing decision logs.

---

## Routing Evaluation Metrics

### Core Metrics

```python
@dataclass
class RouterEvaluationMetrics:
    """Router evaluation metrics"""
    # Routing accuracy
    routing_accuracy: float           # Share of requests routed to the optimal model
    over_routing_rate: float          # Over-routing (should use small but used large) ratio
    under_routing_rate: float         # Under-routing (should use large but used small) ratio

    # Cost metrics
    total_cost: float                 # Total cost
    cost_vs_all_large: float          # Cost ratio vs. using all large models
    cost_vs_optimal: float            # Cost ratio vs. the theoretically optimal routing

    # Quality metrics
    avg_quality: float                # Average output quality
    quality_vs_all_large: float       # Quality ratio vs. using all large models

    # Efficiency metrics
    avg_routing_latency_ms: float     # Average routing decision latency
    avg_total_latency_ms: float       # Average total latency (routing + model call)
    router_cost_per_request: float    # Routing cost per request
```

### Methods for Evaluating a Router

```python
class RouterEvaluator:
    """Router evaluator"""

    def __init__(
        self,
        router: SmartRouter,
        test_cases: list[dict],    # {query, optimal_model, quality_requirements}
        judge_model: str = "gpt-4.1"
    ):
        self.router = router
        self.test_cases = test_cases
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    def evaluate(self) -> RouterEvaluationMetrics:
        """Full evaluation"""
        total = len(self.test_cases)

        correct_routes = 0
        over_routes = 0
        under_routes = 0

        # Tier ordering for model cost and quality
        model_tier = {
            "gpt-4.1-nano": 1,
            "gpt-4.1-mini": 2,
            "gpt-4.1": 3
        }
        model_cost = {
            "gpt-4.1-nano": 0.5e-6,
            "gpt-4.1-mini": 2e-6,
            "gpt-4.1": 10e-6
        }

        total_cost = 0
        optimal_cost = 0
        all_large_cost = 0

        quality_scores = []
        all_large_quality = 0.95  # Baseline quality of the large model

        for case in self.test_cases:
            decision = self.router.route(case["query"])
            selected = decision.selected_model
            optimal = case["optimal_model"]

            # Count routing accuracy
            if selected == optimal:
                correct_routes += 1
            elif model_tier.get(selected, 2) > model_tier.get(optimal, 2):
                over_routes += 1
            else:
                under_routes += 1

            # Compute cost
            tokens = case.get("avg_tokens", 500)
            total_cost += tokens * model_cost.get(selected, 2e-6)
            optimal_cost += tokens * model_cost.get(optimal, 2e-6)
            all_large_cost += tokens * model_cost["gpt-4.1"]

            # Estimate quality
            model_quality = {"gpt-4.1-nano": 0.72, "gpt-4.1-mini": 0.85, "gpt-4.1": 0.95}
            quality_scores.append(model_quality.get(selected, 0.85))

        avg_quality = sum(quality_scores) / total if total > 0 else 0

        return RouterEvaluationMetrics(
            routing_accuracy=correct_routes / total,
            over_routing_rate=over_routes / total,
            under_routing_rate=under_routes / total,
            total_cost=total_cost,
            cost_vs_all_large=total_cost / all_large_cost if all_large_cost > 0 else 0,
            cost_vs_optimal=total_cost / optimal_cost if optimal_cost > 0 else 0,
            avg_quality=avg_quality,
            quality_vs_all_large=avg_quality / all_large_quality,
            avg_routing_latency_ms=sum(
                r.decision.latency_ms for r in self.router.history[-total:]
            ) / total if total > 0 else 0,
            avg_total_latency_ms=0,  # needs actual measurement
            router_cost_per_request=sum(
                r.decision.router_cost for r in self.router.history[-total:]
            ) / total if total > 0 else 0
        )
```

### Interpreting the Evaluation Metrics

| Metric | Good range | Warning range | Explanation |
|------|----------|----------|------|
| routing_accuracy | > 0.80 | < 0.60 | Too-low routing accuracy means wasted cost or sacrificed quality |
| over_routing_rate | < 0.10 | > 0.25 | Over-routing wastes cost |
| under_routing_rate | < 0.05 | > 0.15 | Under-routing sacrifices quality |
| cost_vs_all_large | < 0.50 | > 0.70 | Cost savings are not significant |
| quality_vs_all_large | > 0.90 | < 0.80 | Quality loss is too large |

---

## Case Study: Cost Modeling for a Customer-Service Routing System

> ⚠️ **Honest note**: An older version of this section ran a "simulated routing" with the removed `SmartRouter` and presented a cost comparison table that looked like real results (monthly cost $2,880 / $780, etc.). Those numbers were actually **illustrative assumptions**, not measurements from a real system. Below we instead use the `CostQualityAnalyzer` retained in this section for cost modeling; the numbers are also **extrapolated from the given rates and task distribution** — treat it as a demonstration of "how to model", not a measurement of some product.

### Scenario and Task Distribution (Assumed)

An e-commerce customer-service system with 10,000 daily requests, with the following task distribution and rate assumptions (the rate table `MODELS` is defined earlier):

```python
from dataclasses import dataclass

@dataclass
class TaskProfile:
    name: str
    avg_input_tokens: int
    avg_output_tokens: int
    daily_volume: int
    quality_requirement: float

tasks = [
    TaskProfile("FAQ", 150, 80, 4000, 0.70),
    TaskProfile("Order query", 200, 100, 2500, 0.80),
    TaskProfile("Returns/exchanges", 500, 300, 1500, 0.85),
    TaskProfile("Complaint", 800, 500, 1200, 0.90),
    TaskProfile("Tech support", 1000, 600, 800, 0.92),
]
```

### Estimating the Cost of Each Routing Strategy with CostQualityAnalyzer

```python
# Reuse the CostQualityAnalyzer and MODELS rate table from earlier in this section
# (rates and quality scores for gpt-4.1 / gpt-4.1-mini / gpt-4.1-nano are given earlier)

# Single-model strategy comparison
strategies = {
    "All gpt-4.1":      {"gpt-4.1": 1.0},
    "All gpt-4.1-mini": {"gpt-4.1-mini": 1.0},
    "All gpt-4.1-nano": {"gpt-4.1-nano": 1.0},
}
for name, ratio in strategies.items():
    rep = analyzer.analyze_routing(tasks, ratio)
    print(f"{name}: monthly cost ${rep['monthly_cost']:.0f}, weighted quality {rep['weighted_quality']:.2f}")

# Routing strategy: simple tasks go to nano/mini, complex tasks go to 4.1 (illustrative ratios)
routed = analyzer.analyze_routing(tasks, {
    "gpt-4.1-nano": 0.40,
    "gpt-4.1-mini": 0.35,
    "gpt-4.1": 0.25,
})
print(f"Routing strategy: monthly cost ${routed['monthly_cost']:.0f}, weighted quality {routed['weighted_quality']:.2f}")
```

> Note: the `analyze_routing` above splits each task across different models by ratio and weights the quality — it measures "total cost and weighted quality under a given routing ratio", **not** the online decision of a real router. To get real routing results, you need to connect the router from Chapter 20 to production, record `RoutingRecord` (see the evaluation metrics in the next section), and recompute against real traffic.

### Modeling Conclusion (Illustrative)

Under the assumption that "simple tasks dominate", a routing strategy typically saves 60–80% of cost compared to "all large models", at the cost of weighted quality dropping from ~0.95 to ~0.85–0.90. Whether it pays off depends on your quality tolerance — which is exactly what the evaluation metrics in the next section answer.

---

## Summary

| Concept | Explanation |
|------|------|
| Model routing | Dynamically select the optimal model by task characteristics, balancing cost and quality |
| Decision framework | Three-layer decision by task complexity, risk level, and cost budget |
| Static routing | Rule-based, zero cost, but limited accuracy |
| LLM routing | Use an LLM to judge complexity, flexible but with extra cost |
| Trained router model | Dedicated classifier, low latency, needs periodic retraining |
| Cascade routing | Escalate from small to large model step by step, suited for scenarios where simple tasks dominate |
| Cost-quality trade-off | Small models save cost but may sacrifice quality; needs quantitative analysis |
| Evaluation metrics | Routing accuracy, over/under-routing rate, cost ratio, quality ratio |

---

## 📝 Exercises for This Chapter

After reading this chapter, close the book and answer the questions in your own words first, then expand the reference answers to check.

**Exercise 1 (Concept)**: At the start of this chapter it says "evaluating an Agent is much harder than evaluating traditional software." Name at least 3 reasons why Agent evaluation is hard, and explain why production recommends a three-layer combination of "rule-based evaluation → LLM-as-Judge → human evaluation" rather than using only one of them.

<details>
<summary>Reference answer</summary>

**Why Agent evaluation is hard** (see 17.1):

1. **Non-deterministic output**: the same input can yield different answers from an LLM each time, so you can't do "input A always gives output B" like in traditional unit tests.
2. **Diverse behavior paths**: to finish the same task, an Agent may use different tool combinations and different step orders, so it's hard to say which path is the "standard answer".
3. **Subjective quality**: whether an answer is "good" often needs human judgment, e.g. empathy, clarity — there is no objective score.
4. **Long chain**: from the user's question to the final answer, there are multiple steps in between — intent recognition, tool calling, reasoning — and when something goes wrong it's not easy to locate which step.

**Why the three-layer combination**: because each method has its strengths and weaknesses, trading off among "speed, cost, accuracy":

| Method | Speed | Cost | Consistency | Weakness |
|------|------|------|--------|------|
| Rule-based evaluation | Fastest | Lowest | Fully consistent | Only checks format/keywords, can't understand semantics |
| LLM-as-Judge | Fast | Medium | Higher | Has position bias, verbosity bias, occasionally wrong |
| Human evaluation | Slowest | Highest | Varies by person | Expensive, slow, can't scale |

The reasonable approach is **layered filtering**: first use cheap rules to quickly filter out obviously failing cases (e.g. wrong format, no cited source); then use LLM-as-Judge to score the rest in batches; finally hand only the most critical, highest-risk few cases to humans for final confirmation. This keeps coverage and speed while spending expensive human effort where it matters — exactly the engineering idea of "handle the majority with cheap means, handle the minority with expensive means".

</details>

**Exercise 2 (Distinguish)**: When BFCL evaluates tool calling, why does it insist on **AST matching** instead of direct string matching? Give an example where string matching would "misjudge". Also, what is "position bias" in LLM-as-Judge, and what method does the book use to eliminate it?

<details>
<summary>Reference answer</summary>

**Why not string matching** (see 17.2): whether a function call is "correct" should be judged by **semantics**, not by whether the characters are identical. For the same call, swapping the argument order changes the string but keeps the semantics identical. For example:

```python
ground_truth = 'get_weather(city="Beijing", unit="celsius")'
prediction   = 'get_weather(unit="celsius", city="Beijing")'
```

These two calls have exactly the same effect, but character-by-character string comparison judges them "not equal", thus marking a correct call as wrong — that's a misjudgment.

**How AST matching solves it**: parse the call into an Abstract Syntax Tree (AST) and compare the "function name" and the "argument set" separately. Arguments are compared as a dict/set, which naturally ignores order, so the two lines above are correctly judged equal. BFCL also does "type-aware matching", e.g. treating integer `1` and float `1.0` as equal, to avoid meaningless type differences causing misjudgment.

**Position bias** (see the LLM-as-Judge bias table in 17.2): the Judge model, when comparing two answers pairwise, tends to prefer the one listed first (or last) rather than truly judging by quality.

**Elimination method**: evaluate twice with swapped positions — first compare in order (A, B), then in order (B, A). Only when both results agree (both think the same one is better) is it considered a real win; otherwise it's a tie. The book's `compute_win_rate` does exactly this, using consistency to cancel out the bias from position.

</details>

**Exercise 3 (Hands-on)**: A customer-service system gets 10,000 requests/day, of which 70% are simple tasks and 30% are complex tasks. You want to use "cascade routing": try the cheap small model first; the small model can answer simple tasks correctly 90% of the time (no escalation needed), but complex tasks must always escalate to the large model. Assume the small model costs $0.001/call and the large model costs $0.01/call. Write a function that estimates how much cascade routing saves versus "all large models", and say when cascade routing is actually not worthwhile.

<details>
<summary>Reference answer</summary>

Idea: in cascade routing, **an escalated request is called twice** (small first, then large), so you must count the "wasted" small-model call into the cost.

```python
def estimate_cascade_cost(
    daily_volume: int,
    simple_ratio: float,      # Share of simple tasks
    small_success_on_simple: float,  # Success rate of small model on simple tasks
    small_cost: float,        # Cost per small-model call
    large_cost: float,        # Cost per large-model call
) -> dict:
    """Estimate cascade routing vs. all-large-model cost"""
    simple_volume = daily_volume * simple_ratio
    complex_volume = daily_volume * (1 - simple_ratio)

    # Simple tasks: all go through the small model first (cost small_cost)
    #   the successful ones stop here; the failed ones call the large model once more
    simple_small_cost = simple_volume * small_cost
    simple_escalate = simple_volume * (1 - small_success_on_simple)
    simple_large_cost = simple_escalate * large_cost

    # Complex tasks: go through the small model first (wasted once), then escalate to large
    complex_small_cost = complex_volume * small_cost
    complex_large_cost = complex_volume * large_cost

    cascade_cost = (
        simple_small_cost + simple_large_cost
        + complex_small_cost + complex_large_cost
    )
    all_large_cost = daily_volume * large_cost

    return {
        "cascade_daily": cascade_cost,
        "all_large_daily": all_large_cost,
        "saved_daily": all_large_cost - cascade_cost,
        "saved_ratio": 1 - cascade_cost / all_large_cost,
    }


r = estimate_cascade_cost(
    daily_volume=10000,
    simple_ratio=0.7,
    small_success_on_simple=0.9,
    small_cost=0.001,
    large_cost=0.01,
)
print(f"Cascade daily cost: ${r['cascade_daily']:.2f}")
print(f"All-large daily cost: ${r['all_large_daily']:.2f}")
print(f"Saved per day: ${r['saved_daily']:.2f} ({r['saved_ratio']:.0%})")
```

Let's compute:
- Simple tasks 7000 calls: small model 7000×$0.001 = $7; of which 10% (700) escalate, large model 700×$0.01 = $7
- Complex tasks 3000 calls: small model wasted 3000×$0.001 = $3; large model 3000×$0.01 = $30
- Cascade total = 7+7+3+30 = **$47/day**
- All large models = 10000×$0.01 = **$100/day**
- Saved $53/day, about **53%**

**When cascade is actually not worthwhile**: if complex tasks make up a high share (few simple tasks), then a large number of requests will "waste one small-model call before escalating", and that single small-model call is pure waste — stacked up it may even cost more than just using the large model directly. So as this chapter says — **cascade routing fits scenarios where "simple tasks dominate"**; when complex tasks dominate, it's better to route directly to the large model, or switch to an LLM router that judges complexity in one shot.

</details>

---

> **Next-section preview**: This chapter ends here. Through 8 subsections of study, you have mastered the complete methodology of Agent evaluation — from basic evaluation methods, benchmark testing, Prompt tuning, cost optimization, observability, to Agent-specific evaluation, A/B testing, and model routing. Next, we will move on to the Security and Reliability chapter.

---

## References

[1] DING S, WANG W, et al. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing[J]. arXiv preprint arXiv:2404.14618, 2024.

[2] CHEN J, GAO Y, et al. RouteLLM: Learning to Route LLMs with Preference Data[J]. arXiv preprint arXiv:2406.18665, 2024.

[3] SHENG Y, CAO S, et al. FlexLLM: A Flexible and Efficient Approach to LLM Serving[J]. arXiv preprint, 2024.
