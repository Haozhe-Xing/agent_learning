# 17.7 A/B 测试与回归测试自动化

> **本节目标**：掌握 Agent 的 A/B 测试与回归测试方法，能够构建自动化的 Prompt 变体测试框架，并将评估集成到 CI/CD 流水线中。

---

## 为什么 Agent 需要 A/B 测试？

当你修改了一个 Prompt、调整了工具描述、或者换了模型版本——你怎么知道改得更好了？

传统软件有单元测试，改了代码跑一下就知道了。但 Agent 的输出不确定，"跑一下"无法给出统计意义上可靠的结论。我们需要**严格的 A/B 测试**来验证每次变更的效果。

### 常见场景

| 变更类型 | 风险 | A/B 测试的价值 |
|----------|------|----------------|
| 修改系统 Prompt | 可能改善某些场景但破坏其他场景 | 量化对比不同 Prompt 的效果 |
| 更换模型版本 | 新模型可能退步 | 在真实任务上对比新旧模型 |
| 调整工具描述 | 可能导致工具误用 | 检测工具选择准确率变化 |
| 增加新工具 | 可能干扰原有工具调用 | 评估对已有能力的影响 |
| 修改 Temperature | 影响输出多样性和质量 | 权衡创造性和稳定性 |

---

## A/B 测试框架设计

### 核心概念

A/B 测试的核心思想：在**相同的测试集**上，让**两个版本的 Agent** 分别执行，然后**统计对比**它们的性能差异。

```python
"""
Agent A/B 测试框架
支持：Prompt 变体对比、统计显著性检验、自动化报告
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
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TestCase:
    """测试用例"""
    id: str
    query: str                           # 用户输入
    expected_output: Optional[str] = None # 期望输出（可选）
    expected_tools: Optional[list[str]] = None  # 期望调用的工具
    category: str = "default"            # 分类标签
    difficulty: str = "medium"           # easy / medium / hard
    metadata: dict = field(default_factory=dict)


@dataclass
class TestRun:
    """单次测试运行结果"""
    test_case_id: str
    variant: str              # "A" 或 "B"
    output: str               # Agent 输出
    tools_called: list[str] = field(default_factory=list)  # 调用的工具
    steps: int = 0            # 执行步骤数
    tokens: int = 0           # Token 使用量
    latency: float = 0.0      # 响应时间（秒）
    error: Optional[str] = None  # 错误信息
    judge_score: Optional[float] = None  # Judge 评分


@dataclass
class ABTestConfig:
    """A/B 测试配置"""
    name: str
    description: str = ""
    # 变体 A（对照组）
    variant_a_config: dict = field(default_factory=dict)
    # 变体 B（实验组）
    variant_b_config: dict = field(default_factory=dict)
    # 测试参数
    confidence_level: float = 0.95      # 置信水平
    min_sample_size: int = 30           # 最小样本量
    max_sample_size: int = 200          # 最大样本量
    metrics: list[str] = field(default_factory=lambda: [
        "quality_score", "tool_accuracy", "latency", "token_usage"
    ])
```

### A/B 测试框架实现

```python
class ABTestFramework:
    """Agent A/B 测试框架"""

    def __init__(
        self,
        agent_factory_a: Callable,
        agent_factory_b: Callable,
        judge_model: str = "gpt-4.1"
    ):
        """
        Args:
            agent_factory_a: 创建变体 A Agent 的工厂函数
            agent_factory_b: 创建变体 B Agent 的工厂函数
            judge_model: 用于评估的 Judge 模型
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
        """运行完整的 A/B 测试"""

        results_a: list[TestRun] = []
        results_b: list[TestRun] = []

        total = len(test_cases)

        for i, case in enumerate(test_cases):
            if progress_callback:
                progress_callback(i, total, case.id)

            # 运行变体 A
            run_a = self._run_single(case, self.agent_factory_a, "A")
            results_a.append(run_a)

            # 运行变体 B
            run_b = self._run_single(case, self.agent_factory_b, "B")
            results_b.append(run_b)

            # 用 Judge 评估两个输出
            if case.expected_output or True:  # 始终用 Judge 评估
                self._judge_outputs(case, run_a, run_b)

        # 统计分析
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
        """运行单次测试"""
        start_time = time.time()
        try:
            agent = agent_factory()
            result = agent.invoke(case.query)

            # 提取结果信息
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
        """用 LLM Judge 评估两个输出"""

        # 检查工具调用准确性
        if case.expected_tools:
            run_a_tools = set(run_a.tools_called)
            run_b_tools = set(run_b.tools_called)
            expected = set(case.expected_tools)

            # 计算工具准确率（Jaccard 相似度）
            if expected:
                run_a.judge_score = len(run_a_tools & expected) / len(expected)
                run_b.judge_score = len(run_b_tools & expected) / len(expected)
                return

        # 用 LLM Judge 评估输出质量
        prompt = f"""你是一个专业的 AI 输出质量评审员。请评估以下两个 Agent 回答的质量。

用户问题：{case.query}
{'期望输出：' + case.expected_output if case.expected_output else ''}

回答 A：
{run_a.output}

回答 B：
{run_b.output}

请分别对两个回答打分（0-10），以 JSON 格式回复：
{{
    "score_a": <0-10>,
    "score_b": <0-10>,
    "reasoning": "简要说明评分理由"
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
        """统计分析测试结果"""

        # 提取各指标数据
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
        """对比两组数据的统计显著性"""
        if len(group_a) < 2 or len(group_b) < 2:
            return {
                "significant": False,
                "p_value": None,
                "note": "样本量不足"
            }

        # 使用 Welch t-test（不假设等方差）
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

        # Cohen's d 效应量
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
                "大" if abs(cohens_d) >= 0.8
                else "中" if abs(cohens_d) >= 0.5
                else "小" if abs(cohens_d) >= 0.2
                else "可忽略"
            )
        }

    @staticmethod
    def _determine_winner(
        scores_a: list[float],
        scores_b: list[float]
    ) -> str:
        """判定获胜变体"""
        if not scores_a or not scores_b:
            return "无法判定"
        avg_a = np.mean(scores_a)
        avg_b = np.mean(scores_b)
        diff = abs(avg_a - avg_b)
        if diff < 0.05:
            return "平局（差异不显著）"
        return "A" if avg_a > avg_b else "B"
```

### 使用示例：对比两个 Prompt 变体

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 定义两个 Prompt 变体
PROMPT_A = """你是一个客服助手。请回答用户的问题。要简洁明了。"""

PROMPT_B = """你是一个专业的客服助手。回答用户问题时请遵循以下原则：
1. 先确认理解了用户的问题
2. 给出准确、完整的回答
3. 如果涉及操作步骤，请按编号列出
4. 最后询问是否还有其他问题"""


def create_agent_a():
    """创建变体 A 的 Agent"""
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
    """创建变体 B 的 Agent"""
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


# 准备测试用例
test_cases = [
    TestCase(
        id="cs_001",
        query="我的订单还没到，已经超过预计时间3天了",
        category="物流查询",
        difficulty="easy"
    ),
    TestCase(
        id="cs_002",
        query="我想退货，但商品已经拆封了，还能退吗？",
        category="退换货",
        difficulty="medium"
    ),
    TestCase(
        id="cs_003",
        query="你们的会员制度和积分规则是什么？VIP有什么额外权益？",
        category="会员服务",
        difficulty="hard"
    ),
]

# 配置 A/B 测试
config = ABTestConfig(
    name="客服 Prompt 优化测试",
    description="对比简洁版和详细版 Prompt 的客服质量",
    variant_a_config={"prompt": "简洁版", "model": "gpt-4.1-mini"},
    variant_b_config={"prompt": "详细版", "model": "gpt-4.1-mini"},
    confidence_level=0.95
)

# 运行测试
framework = ABTestFramework(
    agent_factory_a=create_agent_a,
    agent_factory_b=create_agent_b,
    judge_model="gpt-4.1"
)

result = framework.run_test(test_cases, config)

# 输出结果
print(f"变体 A 平均分：{result['analysis']['summary']['variant_a_avg_score']:.2f}")
print(f"变体 B 平均分：{result['analysis']['summary']['variant_b_avg_score']:.2f}")
print(f"获胜者：{result['analysis']['summary']['winner']}")
print(f"质量差异显著：{result['analysis']['quality']['significant']}")
print(f"效应量：{result['analysis']['quality']['effect_size']}")
```

### A/B 测试注意事项

| 问题 | 说明 | 应对策略 |
|------|------|----------|
| 样本量不足 | 统计检验需要足够的样本 | 每个变体至少 30 个测试用例 |
| 随机性干扰 | LLM 输出的随机性可能影响结果 | 每个用例运行 3-5 次取平均 |
| Judge 偏差 | 评估模型可能偏好某种风格 | 使用多个 Judge 取平均 |
| 过拟合测试集 | 反复优化导致对测试集过拟合 | 保留 hold-out 测试集 |
| 多重比较 | 同时测试多个指标增加假阳性 | 使用 Bonferroni 校正 |

> 💡 **最佳实践**：在运行 A/B 测试前，先用功效分析（Power Analysis）计算所需样本量。小样本的"显著"结果往往不可靠。

---

## 回归测试：防止 Prompt 修改破坏已有能力

### 什么是 Agent 回归测试？

回归测试的核心目标：**确保新的修改不会破坏已有的正确行为**。在 Agent 开发中，这意味着每次修改 Prompt、工具描述或模型参数后，都要验证关键场景仍然正常工作。

### 回归测试策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 快照测试 | 记录期望输出，检查新输出是否偏离 | 输出相对固定的场景 |
| 行为测试 | 检查关键行为是否正确（如工具调用） | 工具使用场景 |
| 语义测试 | 用 LLM Judge 检查语义等价性 | 输出变化但语义不变的场景 |
| 边界测试 | 测试极端输入和边界情况 | 稳健性验证 |

### 回归测试框架实现

```python
class RegressionTestSuite:
    """Agent 回归测试套件"""

    def __init__(
        self,
        agent_factory: Callable,
        judge_model: str = "gpt-4.1"
    ):
        self.agent_factory = agent_factory
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)
        self.baselines: dict[str, dict] = {}  # 基线结果
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
        tolerance: float = 0.1  # 语义相似度容差
    ):
        """注册回归测试用例"""
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
        """保存当前结果作为基线"""
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
        print(f"基线已保存到 {output_path}，共 {len(baseline_results)} 个用例")

    def load_baseline(self, input_path: str = "baseline.json"):
        """加载已有基线"""
        with open(input_path, "r") as f:
            self.baselines = json.load(f)
        print(f"已加载基线，共 {len(self.baselines)} 个用例")

    def run_regression(self) -> dict:
        """运行回归测试"""
        agent = self.agent_factory()
        results = []

        for case in self.test_cases:
            # 执行测试
            actual_output = agent(case["query"])
            actual_str = actual_output if isinstance(actual_output, str) else str(actual_output)

            # 根据类型检查
            if case["type"] == "snapshot":
                passed = self._check_snapshot(case, actual_str)
            elif case["type"] == "behavior":
                passed = self._check_behavior(case, actual_output)
            elif case["type"] == "semantic":
                passed = self._check_semantic(case, actual_str)
            elif case["type"] == "boundary":
                passed = self._check_boundary(case, actual_str)
            else:
                passed = True  # 未知类型默认通过

            results.append({
                "name": case["name"],
                "category": case["category"],
                "type": case["type"],
                "passed": passed,
                "query": case["query"]
            })

        # 汇总
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
        """快照测试：检查输出是否与基线完全匹配"""
        baseline = self.baselines.get(case["name"])
        if not baseline:
            return True  # 无基线则跳过

        expected = baseline["output"]
        # 允许空白差异
        return actual.strip() == expected.strip()

    def _check_behavior(self, case: dict, actual_output) -> bool:
        """行为测试：检查关键行为是否正确"""
        checks_passed = 0
        total_checks = 0

        # 检查工具调用
        if case["expected_tools"]:
            total_checks += 1
            actual_tools = set()
            if isinstance(actual_output, dict):
                actual_tools = set(actual_output.get("tools_called", []))

            expected_tools = set(case["expected_tools"])
            if expected_tools.issubset(actual_tools):
                checks_passed += 1

        # 检查关键词
        if case["expected_keywords"]:
            total_checks += 1
            actual_str = str(actual_output)
            if all(kw in actual_str for kw in case["expected_keywords"]):
                checks_passed += 1

        if total_checks == 0:
            return True

        return checks_passed == total_checks

    def _check_semantic(self, case: dict, actual: str) -> bool:
        """语义测试：用 LLM Judge 检查语义等价性"""
        baseline = self.baselines.get(case["name"])
        if not baseline:
            return True

        expected = baseline["output"]

        prompt = f"""请判断以下两个回答在语义上是否等价。

问题：{case['query']}

回答 A（基线）：
{expected}

回答 B（当前）：
{actual}

只回复 JSON：{{"equivalent": true/false, "confidence": 0.0-1.0}}"""

        response = self.judge_llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return result.get("equivalent", False) and result.get("confidence", 0) >= (1 - case["tolerance"])
        except json.JSONDecodeError:
            return False

    def _check_boundary(self, case: dict, actual: str) -> bool:
        """边界测试：检查输出是否合理（没有崩溃、空输出等）"""
        # 基本检查
        if not actual or len(actual.strip()) < 5:
            return False

        # 检查是否有异常标记
        error_markers = ["error", "exception", "traceback", "I cannot", "无法完成"]
        actual_lower = actual.lower()
        for marker in error_markers:
            if marker in actual_lower:
                return False

        return True
```

### 实战：为客服 Agent 建立回归测试

```python
# 创建回归测试套件
agent_factory = create_agent_a  # 使用你的 Agent 工厂函数

suite = RegressionTestSuite(
    agent_factory=agent_factory,
    judge_model="gpt-4.1"
)

# 注册测试用例
suite.register_test(
    name="订单查询",
    query="查询我的订单状态",
    test_type="behavior",
    expected_tools=["query_order"],
    expected_keywords=["订单"],
    category="核心功能"
)

suite.register_test(
    name="退货流程",
    query="我想退货，请问怎么操作？",
    test_type="semantic",
    category="核心功能"
)

suite.register_test(
    name="空输入处理",
    query="",
    test_type="boundary",
    category="边界处理"
)

suite.register_test(
    name="超长输入处理",
    query="请帮我处理" + "非常紧急" * 100,
    test_type="boundary",
    category="边界处理"
)

# 首次运行：保存基线
suite.save_baseline("regression_baseline.json")

# 后续运行：加载基线并回归测试
suite.load_baseline("regression_baseline.json")
results = suite.run_regression()

print(f"回归测试结果：{results['passed']}/{results['total']} 通过")
print(f"通过率：{results['pass_rate']:.1%}")

# 输出失败用例
for detail in results["details"]:
    if not detail["passed"]:
        print(f"  ❌ {detail['name']} ({detail['type']}): {detail['query']}")
```

---

## CI/CD 集成：自动化评估流水线

### 为什么需要 CI/CD 集成？

手动运行测试容易遗漏。将 Agent 评估集成到 CI/CD 流水线中，可以在每次代码变更时自动运行评估，及时发现问题。

### GitHub Actions 自动评估配置

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
    # 每天凌晨 2 点运行完整评估
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

      - name: Run Agent A/B Test
        if: github.event_name == 'pull_request'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/run_ab_test.py \
            --baseline main \
            --variant ${{ github.head_ref }} \
            --output results/ab_test.json

      - name: Check Quality Gate
        run: |
          python scripts/check_quality_gate.py \
            --results results/ab_test.json \
            --min-pass-rate 0.85 \
            --max-regression 0.05

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: results/
```

### 质量门禁脚本

```python
"""
质量门禁检查脚本
用于 CI/CD 流水线，根据评估结果决定是否允许合并
"""
import json
import sys
from dataclasses import dataclass


@dataclass
class QualityGateConfig:
    """质量门禁配置"""
    min_pass_rate: float = 0.85       # 最低回归测试通过率
    max_regression: float = 0.05      # 最大允许退步幅度
    min_ab_score: float = 0.7         # A/B 测试最低得分
    max_latency_increase: float = 0.2 # 最大延迟增加比例
    max_token_increase: float = 0.2   # 最大 Token 增加比例


class QualityGate:
    """质量门禁检查"""

    def __init__(self, config: QualityGateConfig = None):
        self.config = config or QualityGateConfig()

    def check(
        self,
        regression_results: dict = None,
        ab_test_results: dict = None
    ) -> dict:
        """执行质量门禁检查"""
        checks = []

        # 检查 1：回归测试通过率
        if regression_results:
            pass_rate = regression_results.get("pass_rate", 0)
            checks.append({
                "name": "回归测试通过率",
                "value": pass_rate,
                "threshold": self.config.min_pass_rate,
                "passed": pass_rate >= self.config.min_pass_rate,
                "message": (
                    f"通过率 {pass_rate:.1%} >= {self.config.min_pass_rate:.1%}"
                    if pass_rate >= self.config.min_pass_rate
                    else f"通过率 {pass_rate:.1%} < {self.config.min_pass_rate:.1%}"
                )
            })

        # 检查 2：A/B 测试退步幅度
        if ab_test_results:
            analysis = ab_test_results.get("analysis", {})
            quality = analysis.get("quality", {})
            score_a = quality.get("mean_a", 0)
            score_b = quality.get("mean_b", 0)

            # B 是新版本，A 是基线
            regression = max(0, score_a - score_b)

            checks.append({
                "name": "质量退步幅度",
                "value": regression,
                "threshold": self.config.max_regression,
                "passed": regression <= self.config.max_regression,
                "message": (
                    f"退步 {regression:.3f} <= {self.config.max_regression}"
                    if regression <= self.config.max_regression
                    else f"退步 {regression:.3f} > {self.config.max_regression}"
                )
            })

            # 检查 3：延迟增加
            latency = analysis.get("latency", {})
            lat_a = latency.get("mean_a", 0)
            lat_b = latency.get("mean_b", 0)
            if lat_a > 0:
                latency_increase = (lat_b - lat_a) / lat_a
                checks.append({
                    "name": "延迟增加比例",
                    "value": latency_increase,
                    "threshold": self.config.max_latency_increase,
                    "passed": latency_increase <= self.config.max_latency_increase,
                    "message": (
                        f"延迟增加 {latency_increase:.1%} <= {self.config.max_latency_increase:.1%}"
                        if latency_increase <= self.config.max_latency_increase
                        else f"延迟增加 {latency_increase:.1%} > {self.config.max_latency_increase:.1%}"
                    )
                })

            # 检查 4：Token 增加比例
            token_usage = analysis.get("token_usage", {})
            tok_a = token_usage.get("mean_a", 0)
            tok_b = token_usage.get("mean_b", 0)
            if tok_a > 0:
                token_increase = (tok_b - tok_a) / tok_a
                checks.append({
                    "name": "Token 增加比例",
                    "value": token_increase,
                    "threshold": self.config.max_token_increase,
                    "passed": token_increase <= self.config.max_token_increase,
                    "message": (
                        f"Token 增加 {token_increase:.1%} <= {self.config.max_token_increase:.1%}"
                        if token_increase <= self.config.max_token_increase
                        else f"Token 增加 {token_increase:.1%} > {self.config.max_token_increase:.1%}"
                    )
                })

        all_passed = all(c["passed"] for c in checks)

        return {
            "passed": all_passed,
            "checks": checks,
            "summary": (
                "所有质量门禁检查通过 ✅"
                if all_passed
                else f"{sum(1 for c in checks if not c['passed'])} 项检查未通过 ❌"
            )
        }


def main():
    """CI/CD 入口"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", help="回归测试结果文件")
    parser.add_argument("--ab-test", help="A/B 测试结果文件")
    parser.add_argument("--min-pass-rate", type=float, default=0.85)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()

    # 加载结果
    regression_results = None
    ab_test_results = None

    if args.regression:
        with open(args.regression) as f:
            regression_results = json.load(f)

    if args.ab_test:
        with open(args.ab_test) as f:
            ab_test_results = json.load(f)

    # 运行质量门禁
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

    # 非零退出码表示检查未通过
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
```

### 测试流水线架构

```
代码变更 → CI/CD 触发
    │
    ├── 1. 回归测试（快速，< 5 分钟）
    │   ├── 快照测试
    │   ├── 行为测试
    │   └── 边界测试
    │
    ├── 2. A/B 测试（中等，10-30 分钟）
    │   ├── 运行基线版本
    │   ├── 运行新版本
    │   └── LLM Judge 评估
    │
    └── 3. 质量门禁检查
        ├── 回归通过率 ≥ 85%
        ├── 质量退步 ≤ 5%
        ├── 延迟增加 ≤ 20%
        └── Token 增加 ≤ 20%
```

### CI/CD 集成注意事项

| 问题 | 说明 | 应对策略 |
|------|------|----------|
| API 成本 | 每次提交都跑评估，API 费用高 | PR 只跑回归测试，定时跑完整评估 |
| 执行时间 | LLM 评估耗时较长 | 并行化 + 结果缓存 |
| 非确定性 | 同一代码两次评估可能结果不同 | 多次运行取平均 + 容差阈值 |
| 误报 | 偶尔的质量波动被标记为退步 | 设置合理的容差，人工复查 |
| 秘钥安全 | API Key 不能硬编码 | 使用 GitHub Secrets |

> 💡 **最佳实践**：为不同类型的变更设置不同的评估策略。Prompt 变更跑完整 A/B 测试，代码变更只跑回归测试，文档变更跳过评估。

---

## 小结

| 概念 | 说明 |
|------|------|
| A/B 测试 | 在相同测试集上对比两个 Agent 变体，统计检验差异显著性 |
| 统计显著性 | 用 Welch t-test 判断差异是否由随机波动导致 |
| 效应量 | Cohen's d 衡量差异的实际意义（小/中/大） |
| 回归测试 | 确保修改不会破坏已有能力，4 种策略：快照/行为/语义/边界 |
| 质量门禁 | CI/CD 中的自动化检查，不达标则阻止合并 |
| CI/CD 集成 | GitHub Actions 自动运行评估，结果作为 PR 合并条件 |

> **下一节预告**：我们将学习模型路由评估，了解如何根据任务复杂度智能选择模型，在成本和质量之间找到最佳平衡。

---

## 参考文献

[1] KOHAVI R, TANG D, XU Y. Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing[M]. Cambridge University Press, 2020.

[2] ZHENG L, CHIANG W L, SHENG Y, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena[C]//NeurIPS. 2023.

[3] TAMARIT J, SNOEK J, METZE F. A/B Testing for LLMs: Practical Considerations and Pitfalls[J]. arXiv preprint, 2024.

---

[17.8 模型路由评估](./08_model_routing.md)
