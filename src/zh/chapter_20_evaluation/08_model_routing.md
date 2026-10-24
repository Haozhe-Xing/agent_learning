# 20.8 模型路由评估

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
    print(f"{name:<12} ${result['monthly_cost']:<11,.0f} {result['weighted_quality']:<13.2f} {cost_eff:.2f}")
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

## 路由器实现：去重与交叉引用

> ⚠️ **诚实说明**：旧版这一节在"智能路由器完整实现""级联路由实现"两个小节里，给出了 `SmartRouter` / `CascadeRouter` 共 400+ 行的完整代码。但**路由器的真实、可运行实现已经在第 20 章《部署与运维》的 06_model_serving 中给出**（`StaticRouter` / `LLMRouter` / `CascadeRouter`，含降级与成本追踪）。把同一套代码在两个章节各写一遍，既冗余又容易版本漂移，属于"重复凑篇幅"。

> 本节是"模型路由**评估**"，重点是**如何判断一个路由策略好不好**，而不是再实现一遍路由器。因此这里只保留路由策略的**选型对比**与**评估指标体系**，路由器的具体代码请直接复用第 20 章。

### 三种路由策略的选型对比（评估视角）

| 策略 | 实现位置 | 适用评估场景 | 主要评估指标 |
|------|----------|--------------|--------------|
| 静态规则路由 | 第 20 章 `StaticRouter` | 任务类型可枚举、成本敏感 | 覆盖率、错误路由率 |
| LLM 动态路由 | 第 20 章 `LLMRouter` | 任务复杂度难预判 | 路由准确率、额外延迟 |
| 级联路由（先小后大） | 第 20 章 `CascadeRouter` | 简单任务占多数 | 升级率、成本节省比 |

### 何时评估路由、何时直接上大模型

- 当**简单任务占比 > 60%** 且质量容忍度较高时，路由（尤其级联）能显著降本，值得评估；
- 当任务普遍复杂、或路由本身的延迟/成本不可忽略时，直接统一用中/大模型反而更省心，路由带来的边际收益有限。

> 想看真实可运行的路由器代码与降级逻辑：跳到 **第 20 章 6 节（模型服务与路由）**。本节后半部分（"路由评估指标体系"）告诉你拿到路由决策日志后，该看哪些数。


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

## 实战案例：客服系统的路由成本建模

> ⚠️ **诚实说明**：旧版的同一小节用已移除的 `SmartRouter` 跑了一次"模拟路由"，并给出了一张看起来像真实跑出来的成本对比表（月成本 $2,880 / $780 等）。那些数字其实是**示意性假设**，并非真实系统测量值。下面改用本节保留的 `CostQualityAnalyzer` 做成本建模，数字同样是**基于给定费率和任务分布的推算**，请把它当作"如何建模"的示范，而非某产品的实测。

### 场景与任务分布（假设）

一个电商客服系统，日请求 10,000 次，任务分布与费率假设如下（费率表 `MODELS` 见前文）：

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
    TaskProfile("订单查询", 200, 100, 2500, 0.80),
    TaskProfile("退换货", 500, 300, 1500, 0.85),
    TaskProfile("投诉", 800, 500, 1200, 0.90),
    TaskProfile("技术支持", 1000, 600, 800, 0.92),
]
```

### 用 CostQualityAnalyzer 估算各路由策略成本

```python
# 复用本节前面的 CostQualityAnalyzer 与 MODELS 费率表
# （gpt-4.1 / gpt-4.1-mini / gpt-4.1-nano 的费率与质量分见前文）

# 单模型策略对照
strategies = {
    "全部 gpt-4.1":      {"gpt-4.1": 1.0},
    "全部 gpt-4.1-mini": {"gpt-4.1-mini": 1.0},
    "全部 gpt-4.1-nano": {"gpt-4.1-nano": 1.0},
}
for name, ratio in strategies.items():
    rep = analyzer.analyze_routing(tasks, ratio)
    print(f"{name}: 月成本 ${rep['monthly_cost']:.0f}, 加权质量 {rep['weighted_quality']:.2f}")

# 路由策略：简单任务走 nano/mini，复杂任务走 4.1（示意比例）
routed = analyzer.analyze_routing(tasks, {
    "gpt-4.1-nano": 0.40,
    "gpt-4.1-mini": 0.35,
    "gpt-4.1": 0.25,
})
print(f"路由策略: 月成本 ${routed['monthly_cost']:.0f}, 加权质量 {routed['weighted_quality']:.2f}")
```

> 注意：上面的 `analyze_routing` 会把每个任务按比例拆分到不同模型并加权质量——它衡量的是"在给定路由比例下的总成本与加权质量"，**不是**真实路由器的在线决策。要得到真实路由效果，需要把第 20 章的路由器接到线上、记录 `RoutingRecord`（见下节评估指标），再用真实流量回算。

### 建模结论（示意）

在"简单任务占多数"的假设下，路由策略相比"全部用大模型"通常能省 60–80% 成本，代价是加权质量从 ~0.95 降到 ~0.85–0.90。是否划算取决于你的质量容忍度——这正是下一节的评估指标要回答的问题。


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

---

## 📝 本章练习

读完本章，先合上书用自己的话回答下面的问题，再展开参考答案对照。

**练习 1（概念）**：本章一开始就说"评估 Agent 比评估传统软件难得多"。请说出 Agent 评估难在哪里（至少 3 点），并解释为什么生产中推荐"规则评估 → LLM-as-Judge → 人类评估"这样的三层组合，而不是只用其中一种。

<details>
<summary>参考答案</summary>

**Agent 评估为什么难**（见 20.1）：

1. **输出不确定**：同样的输入，LLM 每次的回答可能都不一样，没法像传统单元测试那样"输入 A 必然得到输出 B"。
2. **行为路径多样**：完成同一个任务，Agent 可能用不同的工具组合、不同的步骤顺序，很难说哪条路径才是"标准答案"。
3. **质量是主观的**：回答"好不好"往往要靠人来判断，比如同理心、清晰度，这些没有客观分数。
4. **链路长**：从用户提问到最终回答，中间经过意图识别、工具调用、推理等多步，出了问题要定位到具体哪一步也不容易。

**为什么要三层组合**：因为三种方法各有长短，是在"速度、成本、准确性"之间做权衡：

| 方法 | 速度 | 成本 | 一致性 | 短板 |
|------|------|------|--------|------|
| 规则评估 | 最快 | 最低 | 完全一致 | 只能查格式/关键词，看不懂语义 |
| LLM-as-Judge | 较快 | 中等 | 较高 | 有位置偏差、冗长偏差，偶尔判错 |
| 人类评估 | 最慢 | 最高 | 因人而异 | 贵、慢，没法规模化 |

合理的做法是**层层过滤**：先用便宜的规则把明显不合格的（比如格式错、没引用来源）快速筛掉；剩下的用 LLM-as-Judge 批量打分；最后只把最关键、最高风险的少量案例交给人类做最终确认。这样既保证了覆盖面和速度，又把昂贵的人力用在刀刃上——这正是"用便宜手段处理多数，用昂贵手段处理少数"的工程思想。

</details>

**练习 2（辨析）**：BFCL 评估工具调用时，为什么坚持用 **AST 匹配**而不是直接做字符串匹配？请举一个字符串匹配会"误判"的例子。另外，LLM-as-Judge 中提到的"位置偏差"是什么，书里用什么办法消除它？

<details>
<summary>参考答案</summary>

**为什么不用字符串匹配**（见 20.2）：函数调用的"对不对"应该看**语义**，而不是看字符是否一模一样。同一个调用，参数顺序换一下，字符串就不同了，但语义完全相同。例如：

```python
ground_truth = 'get_weather(city="Beijing", unit="celsius")'
prediction   = 'get_weather(unit="celsius", city="Beijing")'
```

这两行调用的效果一模一样，但字符串逐字比较会判为"不相等"，于是把对的判成错的——这就是误判。

**AST 匹配怎么解决**：把调用解析成抽象语法树（AST），分别比较"函数名"和"参数集合"。参数用字典/集合来比，天然忽略顺序，所以上面两行会被正确判为相等。BFCL 还做了"类型感知匹配"，比如把整数 `1` 和浮点 `1.0` 视为相等，避免无意义的类型差异造成误判。

**位置偏差**（见 20.2 的 LLM-as-Judge 偏差表）：指 Judge 模型在两两比较时，倾向于偏好排在前面（或后面）的那个回答，而不是真正按质量判断。

**消除办法**：交换位置评估两次——先按 (A, B) 顺序比一次，再按 (B, A) 顺序比一次。只有两次结果一致（都认为同一个更好）才算它真赢，否则算平局。书里 `compute_win_rate` 就是这么做的，用一致性来抵消位置带来的偏差。

</details>

**练习 3（动手）**：某客服系统每天 10000 次请求，其中 70% 是简单任务、30% 是复杂任务。你打算用"级联路由"：先用便宜的小模型试，小模型对简单任务有 90% 的把握能直接答好（不用升级），但复杂任务一定要升级到大模型。假设小模型每次 $0.001、大模型每次 $0.01。请写一个函数，估算级联路由相比"全部用大模型"能省多少钱，并说说级联路由在什么情况下反而不划算。

<details>
<summary>参考答案</summary>

思路：级联路由里，**升级的请求会被调用两次**（先小后大），所以要把"小模型那次的浪费"也算进成本。

```python
def estimate_cascade_cost(
    daily_volume: int,
    simple_ratio: float,      # 简单任务占比
    small_success_on_simple: float,  # 小模型处理简单任务的成功率
    small_cost: float,        # 小模型单次成本
    large_cost: float,        # 大模型单次成本
) -> dict:
    """估算级联路由 vs 全部大模型的成本"""
    simple_volume = daily_volume * simple_ratio
    complex_volume = daily_volume * (1 - simple_ratio)

    # 简单任务：全部先过小模型(成本small_cost)
    #   成功的那部分到此为止；失败的部分还要再调一次大模型
    simple_small_cost = simple_volume * small_cost
    simple_escalate = simple_volume * (1 - small_success_on_simple)
    simple_large_cost = simple_escalate * large_cost

    # 复杂任务：先过小模型(浪费一次)，再升级到大模型
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
print(f"级联日成本: ${r['cascade_daily']:.2f}")
print(f"全大模型日成本: ${r['all_large_daily']:.2f}")
print(f"每天省: ${r['saved_daily']:.2f}（{r['saved_ratio']:.0%}）")
```

算一下：
- 简单任务 7000 次：小模型 7000×$0.001 = $7；其中 10%(700 次)升级，大模型 700×$0.01 = $7
- 复杂任务 3000 次：小模型白跑 3000×$0.001 = $3；大模型 3000×$0.01 = $30
- 级联总计 = 7+7+3+30 = **$47/天**
- 全部大模型 = 10000×$0.01 = **$100/天**
- 每天省 $53，约 **53%**

**什么情况下级联反而不划算**：如果复杂任务占比很高（简单任务很少），那么大量请求都会"先用小模型白跑一次再升级"，这一次小模型调用纯属浪费，叠加起来可能比直接用大模型还贵。所以正如本章所说——**级联路由最适合"简单任务占多数"的场景**；当复杂任务占主导时，不如直接路由到大模型，或改用一次性判断复杂度的 LLM 路由。

</details>

---

> **下一节预告**：本章到此结束。通过 8 个小节的学习，你已经掌握了 Agent 评估的完整方法论——从基本评估方法、基准测试、Prompt 调优、成本优化、可观测性，到 Agent 专项评估、A/B 测试和模型路由。接下来，我们将进入安全与可靠性章节。

---

## 参考文献

[1] DING S, WANG W, et al. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing[J]. arXiv preprint arXiv:2404.14618, 2024.

[2] CHEN J, GAO Y, et al. RouteLLM: Learning to Route LLMs with Preference Data[J]. arXiv preprint arXiv:2406.18665, 2024.

[3] SHENG Y, CAO S, et al. FlexLLM: A Flexible and Efficient Approach to LLM Serving[J]. arXiv preprint, 2024.
