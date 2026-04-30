# 17.8 模型路由评估

> **本节目标**：理解模型路由的核心问题，掌握成本-质量权衡分析方法，能够实现和评估智能路由器，在多模型环境下为每个任务选择最优模型。

---

## 为什么需要模型路由？

在 Agent 系统中，并非每个任务都需要最强的模型。简单的问题用小模型就能解决，只有复杂的推理和规划才需要大模型。**模型路由**（Model Routing）就是根据任务特征，动态选择最合适的模型，在成本和质量之间找到最优平衡。

### 成本差异的现实

| 模型 | 输入价格 (/1M tokens) | 输出价格 (/1M tokens) | 推理能力 | 速度 |
|------|----------------------|----------------------|----------|------|
| gpt-4.1 | $2.00 | $8.00 | 强 | 中 |
| gpt-4.1-mini | $0.40 | $1.60 | 中 | 快 |
| gpt-4.1-nano | $0.10 | $0.40 | 基础 | 最快 |

假设一个 Agent 每天处理 10,000 次请求：

- **全部用 gpt-4.1**：约 $100/天，月成本 $3,000
- **智能路由（70% 小模型 + 30% 大模型）**：约 $40/天，月成本 $1,200
- **节省**：每月 $1,800，年节省 $21,600

> 💡 **关键洞察**：生产环境中，大部分请求是简单任务（FAQ、格式转换、信息提取），只有少数需要深度推理。智能路由可以把大模型"留给真正需要它的场景"。

---

## 何时用大模型、何时用小模型？

### 决策框架

```
任务进入
    │
    ├─ 任务分类
    │   ├── 简单（事实查询、格式转换、简单摘要）→ 小模型
    │   ├── 中等（多步推理、工具调用、需要上下文理解）→ 中等模型
    │   └── 复杂（创造性写作、复杂规划、多约束优化）→ 大模型
    │
    ├─ 风险评估
    │   ├── 低风险（内部工具、非面向用户）→ 可以用小模型
    │   └── 高风险（面向用户、涉及决策）→ 倾向大模型
    │
    └─ 成本预算
        ├── 宽裕 → 偏向大模型
        └── 紧张 → 偏向小模型 + 人工复核
```

### 任务复杂度分类标准

| 维度 | 简单（小模型） | 中等（中模型） | 复杂（大模型） |
|------|---------------|---------------|---------------|
| 推理步数 | 1 步 | 2-3 步 | 4+ 步 |
| 工具调用 | 无 | 1-2 个 | 3+ 个 |
| 输入长度 | < 500 tokens | 500-2000 tokens | 2000+ tokens |
| 输出要求 | 固定格式 | 半结构化 | 开放式 |
| 容错要求 | 高（出错无妨） | 中 | 低（必须准确） |
| 典型任务 | 意图分类、关键词提取 | RAG 问答、简单工具调用 | 复杂规划、多轮对话 |

---

## 成本-质量权衡分析

### 质量与成本的关系

```python
"""
成本-质量权衡分析工具
"""
import json
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI


@dataclass
class ModelProfile:
    """模型配置"""
    name: str
    input_cost_per_mtok: float     # 每百万输入 Token 的成本
    output_cost_per_mtok: float    # 每百万输出 Token 的成本
    avg_latency_ms: float          # 平均延迟（毫秒）
    quality_score: float           # 质量评分（0-1，基于基准测试）


@dataclass
class TaskProfile:
    """任务配置"""
    name: str
    avg_input_tokens: int          # 平均输入 Token 数
    avg_output_tokens: int         # 平均输出 Token 数
    daily_volume: int              # 日请求量
    quality_requirement: float     # 最低质量要求（0-1）


# 定义模型档案
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
    """成本-质量权衡分析器"""

    def __init__(self, models: dict[str, ModelProfile] = None):
        self.models = models or MODELS

    def calculate_cost(
        self,
        model: ModelProfile,
        task: TaskProfile
    ) -> float:
        """计算单日成本"""
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
        """分析所有模型的成本和质量"""
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

        # 排序：质量达标的模型中选成本最低的
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
                f"质量达标（{best['quality_score']:.2f} >= {task.quality_requirement}）"
                f"且成本最低（${best['daily_cost']:.2f}/天）"
                if best["meets_quality"]
                else f"无模型达标，推荐质量最高的 {best['model']}（{best['quality_score']:.2f}）"
            )
        }

    def analyze_routing(
        self,
        tasks: list[TaskProfile],
        routing_ratios: dict[str, float]
    ) -> dict:
        """分析路由策略的总成本和质量"""
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


# 使用示例
analyzer = CostQualityAnalyzer()

# 分析单个任务
task = TaskProfile(
    name="客服问答",
    avg_input_tokens=800,
    avg_output_tokens=300,
    daily_volume=5000,
    quality_requirement=0.80
)

result = analyzer.analyze(task)
print(f"推荐模型：{result['recommended']}")
print(f"原因：{result['reason']}")

# 对比所有模型
print("\n各模型对比：")
for m in result["models"]:
    status = "✅" if m["meets_quality"] else "❌"
    print(f"  {status} {m['model']}: 质量 {m['quality_score']:.2f}, "
          f"日成本 ${m['daily_cost']:.2f}, 延迟 {m['avg_latency_ms']}ms")
```

### 多任务路由策略对比

```python
# 定义多种业务任务
tasks = [
    TaskProfile("FAQ回答", 200, 100, 3000, 0.70),
    TaskProfile("RAG问答", 1500, 400, 2000, 0.85),
    TaskProfile("复杂规划", 2000, 800, 500, 0.92),
]

# 策略 1：全部使用大模型
strategy_all_large = {"gpt-4.1": 1.0}

# 策略 2：全部使用中模型
strategy_all_medium = {"gpt-4.1-mini": 1.0}

# 策略 3：智能路由
strategy_smart = {"gpt-4.1-nano": 0.4, "gpt-4.1-mini": 0.4, "gpt-4.1": 0.2}

strategies = {
    "全部大模型": strategy_all_large,
    "全部中模型": strategy_all_medium,
    "智能路由": strategy_smart,
}

print("路由策略对比：")
print(f"{'策略':<12} {'月成本':<12} {'加权质量':<12} {'性价比'}")
print("-" * 55)

for name, ratios in strategies.items():
    result = analyzer.analyze_routing(tasks, ratios)
    cost_eff = result["weighted_quality"] / (result["monthly_cost"] / 1000)
    print(f"{name:<12} ${result['monthly_cost']:<11,.0f} {result['weighted_quality']:<12.2f} {cost_eff:.2f}")
```

| 策略 | 月成本 | 加权质量 | 性价比 |
|------|--------|----------|--------|
| 全部大模型 | ~$3,600 | 0.95 | 0.26 |
| 全部中模型 | ~$720 | 0.85 | 1.18 |
| 智能路由 | ~$1,080 | 0.86 | 0.80 |

> ⚠️ **注意**：智能路由的质量略低于全部用大模型，但成本降低约 70%。关键在于找到"质量损失可接受、成本节省显著"的平衡点。

---

## 路由模型（Router Model）训练与评估

### 路由模型的核心任务

路由模型需要解决一个分类问题：给定一个输入，预测应该路由到哪个模型。

### 方法 1：基于规则的静态路由

最简单的方法——根据输入特征硬编码路由规则：

```python
class StaticRouter:
    """基于规则的静态路由器"""

    def __init__(self, rules: list[dict] = None):
        self.rules = rules or self._default_rules()

    def _default_rules(self) -> list[dict]:
        """默认路由规则"""
        return [
            {
                "name": "简单任务",
                "condition": lambda query: (
                    len(query) < 50
                    and any(kw in query for kw in ["什么是", "多少", "什么时候"])
                ),
                "model": "gpt-4.1-nano"
            },
            {
                "name": "中等任务",
                "condition": lambda query: (
                    len(query) < 200
                    or any(kw in query for kw in ["分析", "对比", "总结"])
                ),
                "model": "gpt-4.1-mini"
            },
            {
                "name": "复杂任务",
                "condition": lambda query: (
                    len(query) >= 200
                    or any(kw in query for kw in ["规划", "设计", "优化"])
                ),
                "model": "gpt-4.1"
            },
        ]

    def route(self, query: str) -> str:
        """路由决策"""
        for rule in self.rules:
            if rule["condition"](query):
                return rule["model"]
        return "gpt-4.1-mini"  # 默认中等模型
```

**优点**：零成本、确定性、可解释。**缺点**：规则维护困难、无法处理边界情况。

### 方法 2：基于 LLM 的动态路由

用一个小的 LLM 来判断任务复杂度：

```python
class LLMRouter:
    """基于 LLM 的动态路由器"""

    def __init__(self, router_model: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=router_model, temperature=0)
        self.route_options = {
            "simple": "gpt-4.1-nano",
            "medium": "gpt-4.1-mini",
            "complex": "gpt-4.1"
        }

    def route(self, query: str, context: dict = None) -> dict:
        """路由决策"""
        context_info = ""
        if context:
            context_info = f"\n额外上下文：{json.dumps(context, ensure_ascii=False)}"

        prompt = f"""你是一个任务复杂度分类器。请判断以下用户请求的复杂度。

用户请求：{query}{context_info}

复杂度定义：
- simple：简单事实查询、关键词提取、格式转换，1步即可完成
- medium：需要推理、搜索、工具调用，2-3步完成
- complex：需要深度推理、多步规划、创造性思维，4+步完成

只回复 JSON：{{"complexity": "simple/medium/complex", "confidence": 0.0-1.0, "reasoning": "简短理由"}}"""

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
                "reasoning": "路由解析失败，使用默认模型",
                "router_cost": 0
            }

    def _estimate_router_cost(self, query: str) -> float:
        """估算路由成本（基于 gpt-4.1-mini 价格）"""
        input_tokens = len(query) // 4 + 150  # 粗略估算
        output_tokens = 50
        return (
            input_tokens / 1_000_000 * 0.4
            + output_tokens / 1_000_000 * 1.6
        )
```

**优点**：灵活、能理解语义。**缺点**：有额外成本和延迟，自身可能出错。

### 方法 3：训练专用路由模型

最经济的方法——训练一个小型分类模型来做路由决策：

```python
"""
训练专用路由模型
使用标注数据训练一个轻量级分类器
"""
import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


@dataclass
class RoutingExample:
    """路由标注样本"""
    query: str
    optimal_model: str       # 最优模型
    complexity: str          # simple / medium / complex
    quality_scores: dict     # 各模型的质量得分 {model_name: score}


class RouterTrainingDataGenerator:
    """生成路由模型的训练数据"""

    def __init__(self, judge_model: str = "gpt-4.1"):
        self.llm = ChatOpenAI(model=judge_model, temperature=0)

    def generate_labels(
        self,
        queries: list[str],
        models: list[str] = None
    ) -> list[RoutingExample]:
        """为一批查询生成最优模型标注"""
        models = models or ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]

        labeled_data = []
        for query in queries:
            # 让 Judge 模型评估每个模型对该查询的适配度
            model_scores = {}
            for model in models:
                score = self._evaluate_model_fit(query, model)
                model_scores[model] = score

            # 选择得分最高的模型（考虑成本）
            optimal = self._select_optimal_model(model_scores, models)

            # 判断复杂度
            complexity = self._classify_complexity(query)

            labeled_data.append(RoutingExample(
                query=query,
                optimal_model=optimal,
                complexity=complexity,
                quality_scores=model_scores
            ))

        return labeled_data

    def _evaluate_model_fit(self, query: str, model: str) -> float:
        """评估模型对查询的适配度"""
        prompt = f"""评估以下模型对给定查询的适配度。

查询：{query}
模型：{model}

请评估该模型处理此查询的质量（0-10分），考虑：
- 推理能力是否足够
- 知识是否覆盖
- 输出质量预期

只回复一个 0-10 的数字。"""

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
        """选择最优模型（平衡质量和成本）"""
        # 成本权重：小模型成本更低，可以容忍略低的质量
        cost_weights = {
            "gpt-4.1-nano": 1.0,    # 最便宜，质量折扣少
            "gpt-4.1-mini": 0.85,   # 中等
            "gpt-4.1": 0.65,        # 最贵，质量折扣多
        }

        adjusted = {}
        for model, score in scores.items():
            weight = cost_weights.get(model, 0.8)
            adjusted[model] = score * weight

        return max(adjusted, key=adjusted.get)

    def _classify_complexity(self, query: str) -> str:
        """分类查询复杂度"""
        prompt = f"""判断以下查询的复杂度。

查询：{query}

只回复：simple / medium / complex"""

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
        """导出训练数据为 JSONL 格式"""
        with open(output_path, "w") as f:
            for example in data:
                record = {
                    "query": example.query,
                    "label": example.optimal_model,
                    "complexity": example.complexity,
                    "scores": example.quality_scores
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"已导出 {len(data)} 条训练数据到 {output_path}")
```

### 方法对比

| 方法 | 成本 | 准确率 | 延迟 | 可维护性 |
|------|------|--------|------|----------|
| 静态规则 | 零 | 60-70% | 0ms | 低（规则膨胀） |
| LLM 路由 | $0.001/次 | 85-90% | 200-500ms | 高 |
| 训练路由模型 | 训练成本 | 80-88% | <10ms | 中（需定期重训） |

---

## 智能路由器完整实现

```python
"""
智能模型路由器
支持：多策略路由、降级机制、成本追踪、评估报告
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

from langchain_openai import ChatOpenAI


class RoutingStrategy(Enum):
    """路由策略"""
    STATIC = "static"        # 基于规则
    LLM = "llm"              # 基于 LLM
    CONFIDENCE = "confidence" # 基于置信度
    CASCADE = "cascade"       # 级联（先小后大）


@dataclass
class RoutingDecision:
    """路由决策"""
    selected_model: str
    strategy: RoutingStrategy
    confidence: float
    reasoning: str
    latency_ms: float
    router_cost: float


@dataclass
class RoutingRecord:
    """路由记录（用于评估和优化）"""
    timestamp: float
    query: str
    decision: RoutingDecision
    actual_quality: Optional[float] = None  # 事后评估
    actual_cost: Optional[float] = None     # 实际成本
    was_correct: Optional[bool] = None      # 路由是否正确


class SmartRouter:
    """智能模型路由器"""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LLM,
        router_model: str = "gpt-4.1-mini",
        fallback_model: str = "gpt-4.1",
        min_confidence: float = 0.6
    ):
        self.strategy = strategy
        self.router_model = router_model
        self.fallback_model = fallback_model
        self.min_confidence = min_confidence
        self.llm = ChatOpenAI(model=router_model, temperature=0)

        # 模型配置
        self.models = {
            "gpt-4.1": {
                "cost_per_token": 10e-6,
                "quality_tier": "high",
                "max_tokens": 16384
            },
            "gpt-4.1-mini": {
                "cost_per_token": 2e-6,
                "quality_tier": "medium",
                "max_tokens": 16384
            },
            "gpt-4.1-nano": {
                "cost_per_token": 0.5e-6,
                "quality_tier": "basic",
                "max_tokens": 16384
            },
        }

        # 路由历史
        self.history: list[RoutingRecord] = []

    def route(
        self,
        query: str,
        context: dict = None,
        force_model: str = None
    ) -> RoutingDecision:
        """执行路由决策"""

        # 强制指定模型
        if force_model and force_model in self.models:
            return RoutingDecision(
                selected_model=force_model,
                strategy=RoutingStrategy.STATIC,
                confidence=1.0,
                reasoning="强制指定模型",
                latency_ms=0,
                router_cost=0
            )

        start_time = time.time()

        # 根据策略路由
        if self.strategy == RoutingStrategy.STATIC:
            decision = self._route_static(query)
        elif self.strategy == RoutingStrategy.LLM:
            decision = self._route_llm(query, context)
        elif self.strategy == RoutingStrategy.CASCADE:
            decision = self._route_cascade(query)
        else:
            decision = self._route_llm(query, context)

        decision.latency_ms = (time.time() - start_time) * 1000

        # 低置信度时降级到大模型
        if decision.confidence < self.min_confidence:
            decision.reasoning += "（置信度不足，降级到大模型）"
            decision.selected_model = self.fallback_model

        # 记录路由决策
        self.history.append(RoutingRecord(
            timestamp=time.time(),
            query=query,
            decision=decision
        ))

        return decision

    def _route_static(self, query: str) -> RoutingDecision:
        """静态规则路由"""
        query_len = len(query)

        # 简单关键词 + 长度规则
        simple_keywords = ["什么是", "多少", "定义", "翻译", "格式化"]
        complex_keywords = ["规划", "设计", "优化", "分析", "对比", "评估"]

        if any(kw in query for kw in simple_keywords) and query_len < 100:
            return RoutingDecision(
                selected_model="gpt-4.1-nano",
                strategy=RoutingStrategy.STATIC,
                confidence=0.7,
                reasoning="简单查询，短文本",
                latency_ms=0,
                router_cost=0
            )
        elif any(kw in query for kw in complex_keywords) or query_len > 500:
            return RoutingDecision(
                selected_model="gpt-4.1",
                strategy=RoutingStrategy.STATIC,
                confidence=0.6,
                reasoning="复杂查询或长文本",
                latency_ms=0,
                router_cost=0
            )
        else:
            return RoutingDecision(
                selected_model="gpt-4.1-mini",
                strategy=RoutingStrategy.STATIC,
                confidence=0.7,
                reasoning="中等复杂度查询",
                latency_ms=0,
                router_cost=0
            )

    def _route_llm(self, query: str, context: dict = None) -> RoutingDecision:
        """LLM 动态路由"""
        context_text = ""
        if context:
            context_text = f"\n上下文：{json.dumps(context, ensure_ascii=False)}"

        prompt = f"""你是一个模型路由器。请判断处理以下查询应该使用哪个模型。

可用模型：
- gpt-4.1-nano：适合简单查询（事实查询、关键词提取、格式转换），成本最低
- gpt-4.1-mini：适合中等查询（推理、搜索、工具调用），成本适中
- gpt-4.1：适合复杂查询（深度推理、多步规划、创造性任务），成本最高

查询：{query}{context_text}

只回复 JSON：
{{"model": "gpt-4.1-nano/gpt-4.1-mini/gpt-4.1", "confidence": 0.0-1.0, "reasoning": "简短理由"}}"""

        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            model = result.get("model", "gpt-4.1-mini")
            if model not in self.models:
                model = "gpt-4.1-mini"

            # 估算路由成本
            router_input_tokens = len(query) // 4 + 200
            router_output_tokens = 50
            router_cost = (
                router_input_tokens * 0.4 / 1_000_000
                + router_output_tokens * 1.6 / 1_000_000
            )

            return RoutingDecision(
                selected_model=model,
                strategy=RoutingStrategy.LLM,
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                latency_ms=0,
                router_cost=router_cost
            )
        except json.JSONDecodeError:
            return RoutingDecision(
                selected_model="gpt-4.1-mini",
                strategy=RoutingStrategy.LLM,
                confidence=0.0,
                reasoning="路由解析失败",
                latency_ms=0,
                router_cost=0
            )

    def _route_cascade(self, query: str) -> RoutingDecision:
        """级联路由：先用小模型，不够再升级"""
        # 级联策略默认从最小的模型开始
        return RoutingDecision(
            selected_model="gpt-4.1-nano",
            strategy=RoutingStrategy.CASCADE,
            confidence=0.5,
            reasoning="级联策略，从小模型开始",
            latency_ms=0,
            router_cost=0
        )

    def evaluate_routing(self) -> dict:
        """评估路由效果"""
        if not self.history:
            return {"message": "无路由记录"}

        total = len(self.history)
        evaluated = [r for r in self.history if r.was_correct is not None]

        # 模型分布
        model_counts = {}
        total_router_cost = 0
        for record in self.history:
            model = record.decision.selected_model
            model_counts[model] = model_counts.get(model, 0) + 1
            total_router_cost += record.decision.router_cost

        # 准确率
        accuracy = 0
        if evaluated:
            accuracy = sum(1 for r in evaluated if r.was_correct) / len(evaluated)

        # 平均置信度
        avg_confidence = sum(r.decision.confidence for r in self.history) / total

        # 低置信度降级次数
        fallback_count = sum(
            1 for r in self.history
            if r.decision.confidence < self.min_confidence
        )

        return {
            "total_routing_decisions": total,
            "model_distribution": {
                model: {"count": count, "percentage": count / total}
                for model, count in model_counts.items()
            },
            "routing_accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "fallback_count": fallback_count,
            "total_router_cost": total_router_cost,
            "avg_router_latency_ms": sum(
                r.decision.latency_ms for r in self.history
            ) / total
        }

    def get_report(self) -> str:
        """生成路由评估报告"""
        metrics = self.evaluate_routing()

        lines = ["# 模型路由评估报告\n"]

        lines.append("## 总体指标\n")
        lines.append(f"- 总路由决策数：{metrics['total_routing_decisions']}")
        lines.append(f"- 路由准确率：{metrics['routing_accuracy']:.1%}")
        lines.append(f"- 平均置信度：{metrics['avg_confidence']:.2f}")
        lines.append(f"- 低置信度降级次数：{metrics['fallback_count']}")
        lines.append(f"- 总路由成本：${metrics['total_router_cost']:.4f}")

        lines.append("\n## 模型分布\n")
        lines.append("| 模型 | 次数 | 占比 |")
        lines.append("|------|------|------|")
        for model, info in metrics.get("model_distribution", {}).items():
            lines.append(
                f"| {model} | {info['count']} | {info['percentage']:.1%} |"
            )

        return "\n".join(lines)


# ============================================================
# 使用示例
# ============================================================

def demo_router():
    """演示智能路由器"""
    router = SmartRouter(
        strategy=RoutingStrategy.LLM,
        router_model="gpt-4.1-mini",
        fallback_model="gpt-4.1",
        min_confidence=0.6
    )

    test_queries = [
        "什么是机器学习？",                           # 简单
        "请分析 Python 和 Go 在微服务架构中的优劣势",    # 中等
        "设计一个多 Agent 协作系统，要求支持动态角色分配、"
        "任务依赖管理和冲突解决机制",                    # 复杂
        "翻译：Hello World",                           # 简单
        "总结这篇论文的核心贡献",                       # 中等
    ]

    for query in test_queries:
        decision = router.route(query)
        print(f"查询：{query[:30]}...")
        print(f"  → 模型：{decision.selected_model}")
        print(f"  → 置信度：{decision.confidence:.2f}")
        print(f"  → 理由：{decision.reasoning}")
        print()

    # 生成评估报告
    print(router.get_report())


if __name__ == "__main__":
    demo_router()
```

### 级联路由实现

级联路由是另一种常用策略：先用小模型处理，如果小模型信心不足或质量不达标，再升级到大模型。

```python
class CascadeRouter:
    """级联路由器：先小后大，逐级升级"""

    def __init__(
        self,
        model_chain: list[str] = None,
        quality_threshold: float = 0.7,
        confidence_key: str = "confidence"
    ):
        """
        Args:
            model_chain: 模型链，从小到大排列
            quality_threshold: 质量阈值，低于此值升级模型
            confidence_key: 输出中表示置信度的字段
        """
        self.model_chain = model_chain or [
            "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"
        ]
        self.quality_threshold = quality_threshold
        self.confidence_key = confidence_key
        self.llms = {
            model: ChatOpenAI(model=model, temperature=0)
            for model in self.model_chain
        }

    def route(self, query: str) -> dict:
        """级联路由执行"""
        for i, model in enumerate(self.model_chain):
            llm = self.llms[model]
            response = llm.invoke(query)

            # 评估质量/置信度
            quality = self._assess_quality(query, response.content)

            result = {
                "output": response.content,
                "model": model,
                "model_index": i,
                "quality": quality,
                "escalated": quality < self.quality_threshold and i < len(self.model_chain) - 1
            }

            # 质量达标，返回结果
            if quality >= self.quality_threshold:
                return result

            # 质量不达标，升级到下一级模型
            if i < len(self.model_chain) - 1:
                continue

        # 所有模型都尝试过了，返回最后一个
        return result

    def _assess_quality(self, query: str, output: str) -> float:
        """快速评估输出质量（启发式）"""
        # 基本检查
        if not output or len(output.strip()) < 10:
            return 0.0

        # 检查是否包含"我不确定"等低置信度标记
        low_confidence_markers = [
            "我不确定", "无法确定", "可能", "也许",
            "I'm not sure", "I cannot", "作为AI"
        ]
        for marker in low_confidence_markers:
            if marker in output:
                return 0.5

        # 检查输出长度与查询复杂度的匹配
        if len(query) > 200 and len(output) < 100:
            return 0.4

        return 0.8  # 默认高质量


# 使用示例
cascade = CascadeRouter(
    model_chain=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"],
    quality_threshold=0.7
)

result = cascade.route("设计一个分布式系统的容错方案")
print(f"最终模型：{result['model']}")
print(f"是否升级：{result['escalated']}")
print(f"输出质量：{result['quality']:.2f}")
```

### 级联路由的成本分析

| 场景 | 小模型成功率 | 平均调用次数 | 相对成本 |
|------|-------------|-------------|----------|
| 70% 简单 + 30% 复杂 | 70% | 1.3x | 35% |
| 50% 简单 + 50% 复杂 | 50% | 1.5x | 55% |
| 30% 简单 + 70% 复杂 | 30% | 1.7x | 75% |
| 全部大模型 | - | 1.0x | 100% |

> 💡 **最佳实践**：级联路由最适合简单任务占多数的场景。如果大部分任务都是复杂的，级联反而增加成本（小模型的调用浪费了）。

---

## 路由评估指标体系

### 核心指标

```python
@dataclass
class RouterEvaluationMetrics:
    """路由器评估指标"""
    # 路由准确率
    routing_accuracy: float           # 路由到最优模型的比例
    over_routing_rate: float          # 过度路由（该用小模型却用大模型）比例
    under_routing_rate: float         # 路由不足（该用大模型却用小模型）比例

    # 成本指标
    total_cost: float                 # 总成本
    cost_vs_all_large: float          # 相比全部用大模型的成本比
    cost_vs_optimal: float            # 相比理论最优路由的成本比

    # 质量指标
    avg_quality: float                # 平均输出质量
    quality_vs_all_large: float       # 相比全部用大模型的质量比

    # 效率指标
    avg_routing_latency_ms: float     # 平均路由决策延迟
    avg_total_latency_ms: float       # 平均总延迟（含路由+模型调用）
    router_cost_per_request: float    # 每次请求的路由成本
```

### 评估路由器的方法

```python
class RouterEvaluator:
    """路由器评估器"""

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
        """完整评估"""
        total = len(self.test_cases)

        correct_routes = 0
        over_routes = 0
        under_routes = 0

        # 模型成本和质量的层级排序
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
        all_large_quality = 0.95  # 大模型的基准质量

        for case in self.test_cases:
            decision = self.router.route(case["query"])
            selected = decision.selected_model
            optimal = case["optimal_model"]

            # 统计路由准确率
            if selected == optimal:
                correct_routes += 1
            elif model_tier.get(selected, 2) > model_tier.get(optimal, 2):
                over_routes += 1
            else:
                under_routes += 1

            # 计算成本
            tokens = case.get("avg_tokens", 500)
            total_cost += tokens * model_cost.get(selected, 2e-6)
            optimal_cost += tokens * model_cost.get(optimal, 2e-6)
            all_large_cost += tokens * model_cost["gpt-4.1"]

            # 估算质量
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
            avg_total_latency_ms=0,  # 需要实际测量
            router_cost_per_request=sum(
                r.decision.router_cost for r in self.router.history[-total:]
            ) / total if total > 0 else 0
        )
```

### 评估指标解读

| 指标 | 好的范围 | 警告范围 | 说明 |
|------|----------|----------|------|
| routing_accuracy | > 0.80 | < 0.60 | 路由准确率过低意味着浪费成本或牺牲质量 |
| over_routing_rate | < 0.10 | > 0.25 | 过度路由浪费成本 |
| under_routing_rate | < 0.05 | > 0.15 | 路由不足牺牲质量 |
| cost_vs_all_large | < 0.50 | > 0.70 | 成本节省不明显 |
| quality_vs_all_large | > 0.90 | < 0.80 | 质量损失过大 |

---

## 实战案例：客服系统的模型路由优化

### 场景描述

一个电商客服系统，每天处理 10,000 次请求，需要优化模型路由策略。

```python
# 定义业务任务分布
customer_service_tasks = [
    {"type": "FAQ", "volume": 4000, "complexity": "simple",
     "avg_input_tokens": 150, "avg_output_tokens": 80},
    {"type": "订单查询", "volume": 2500, "complexity": "simple",
     "avg_input_tokens": 200, "avg_output_tokens": 100},
    {"type": "退换货处理", "volume": 1500, "complexity": "medium",
     "avg_input_tokens": 500, "avg_output_tokens": 300},
    {"type": "投诉处理", "volume": 1200, "complexity": "complex",
     "avg_input_tokens": 800, "avg_output_tokens": 500},
    {"type": "技术支持", "volume": 800, "complexity": "complex",
     "avg_input_tokens": 1000, "avg_output_tokens": 600},
]

# 创建路由器
router = SmartRouter(
    strategy=RoutingStrategy.LLM,
    router_model="gpt-4.1-mini"
)

# 模拟路由决策
total_cost = 0
model_usage = {"gpt-4.1-nano": 0, "gpt-4.1-mini": 0, "gpt-4.1": 0}

for task in customer_service_tasks:
    # 根据复杂度选择查询模板
    sample_queries = {
        "simple": "我的订单什么时候发货？",
        "medium": "我收到的商品有质量问题，想退货但已经过了7天",
        "complex": "我对你们的服务非常不满，要求经理给我回电解释为什么三次投诉都没解决"
    }

    query = sample_queries[task["complexity"]]
    decision = router.route(query)
    model_usage[decision.selected_model] += task["volume"]

    # 计算成本
    model_costs = {
        "gpt-4.1-nano": 0.5e-6,
        "gpt-4.1-mini": 2e-6,
        "gpt-4.1": 10e-6
    }
    tokens = task["avg_input_tokens"] + task["avg_output_tokens"]
    cost = tokens * model_costs[decision.selected_model] * task["volume"]
    total_cost += cost

print(f"日成本：${total_cost:.2f}")
print(f"月成本：${total_cost * 30:.2f}")
print(f"模型分布：{model_usage}")
```

### 优化结果对比

| 策略 | 月成本 | 平均质量 | 节省比例 |
|------|--------|----------|----------|
| 全部 gpt-4.1 | $2,880 | 0.95 | 基线 |
| 全部 gpt-4.1-mini | $576 | 0.85 | 80% |
| 静态规则路由 | $920 | 0.87 | 68% |
| LLM 智能路由 | $780 | 0.89 | 73% |
| 级联路由 | $650 | 0.86 | 77% |

---

## 小结

| 概念 | 说明 |
|------|------|
| 模型路由 | 根据任务特征动态选择最优模型，平衡成本与质量 |
| 决策框架 | 按任务复杂度、风险等级、成本预算三层决策 |
| 静态路由 | 基于规则，零成本，但准确率有限 |
| LLM 路由 | 用 LLM 判断复杂度，灵活但额外成本 |
| 训练路由模型 | 专用分类器，低延迟，需定期重训 |
| 级联路由 | 先小后大逐级升级，适合简单任务占多数的场景 |
| 成本-质量权衡 | 小模型节省成本但可能牺牲质量，需量化分析 |
| 评估指标 | 路由准确率、过度/不足路由率、成本比、质量比 |

> **下一节预告**：本章到此结束。通过 8 个小节的学习，你已经掌握了 Agent 评估的完整方法论——从基本评估方法、基准测试、Prompt 调优、成本优化、可观测性，到 Agent 专项评估、A/B 测试和模型路由。接下来，我们将进入安全与可靠性章节。

---

## 参考文献

[1] DING S, WANG W, et al. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing[J]. arXiv preprint arXiv:2404.14618, 2024.

[2] CHEN J, GAO Y, et al. RouteLLM: Learning to Route LLMs with Preference Data[J]. arXiv preprint arXiv:2406.18665, 2024.

[3] SHENG Y, CAO S, et al. FlexLLM: A Flexible and Efficient Approach to LLM Serving[J]. arXiv preprint, 2024.
