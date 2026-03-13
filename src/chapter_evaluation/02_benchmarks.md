# 基准测试与评估指标

> **本节目标**：了解业界常用的 Agent 基准测试，掌握定量评估指标的设计与使用。

---

## 为什么需要基准测试？

想象你在面试两个候选人，如果不给同样的题目，怎么比较谁更优秀？基准测试就是给 Agent 出的"标准考题"——用统一的任务、数据和评分标准来衡量不同 Agent 的表现。

---

## 常见的 Agent 基准测试

### 1. HumanEval —— 代码生成能力

由 OpenAI 发布，包含 164 道 Python 编程题，每道题都有单元测试来验证正确性。

```python
# HumanEval 评估示例
def evaluate_humaneval(agent_func, problems: list[dict]) -> dict:
    """在 HumanEval 数据集上评估 Agent"""
    
    results = {"pass": 0, "fail": 0, "error": 0}
    
    for problem in problems:
        prompt = problem["prompt"]
        test_cases = problem["test"]
        
        try:
            # Agent 生成代码
            generated_code = agent_func(prompt)
            
            # 运行测试用例
            exec_globals = {}
            exec(generated_code + "\n" + test_cases, exec_globals)
            results["pass"] += 1
        except AssertionError:
            results["fail"] += 1
        except Exception:
            results["error"] += 1
    
    total = len(problems)
    results["pass@1"] = results["pass"] / total
    return results
```

> ⚠️ **注意**：上面的 `exec()` 仅用于演示评估逻辑。在生产环境中，务必在沙箱环境中执行不受信任的代码。

### 2. MMLU —— 知识广度

Massive Multitask Language Understanding，涵盖 57 个学科、14000+ 道选择题，测试模型的知识广度。

### 3. GSM8K —— 数学推理

Grade School Math 8K，包含 8500 道小学数学题，测试模型的逐步推理能力。

### 4. WebArena —— Web 操作能力

在真实网站环境中评估 Agent 完成任务的能力，如"在电商网站上找到最便宜的蓝色T恤并加入购物车"。

### 5. SWE-bench —— 软件工程

评估 Agent 解决真实 GitHub Issue 的能力，需要理解代码库、定位 Bug 并提交修复。

---

## 设计自己的评估指标

在实际项目中，通用基准测试往往不够用，你需要设计针对自己 Agent 的评估指标：

```python
import json
import time
from dataclasses import dataclass, field

@dataclass
class AgentMetrics:
    """Agent 评估指标集"""
    
    # 质量指标
    accuracy: float = 0.0          # 准确率
    f1_score: float = 0.0          # F1 分数
    hallucination_rate: float = 0.0  # 幻觉率
    
    # 效率指标
    avg_latency: float = 0.0       # 平均响应时间（秒）
    avg_steps: float = 0.0         # 平均执行步骤数
    avg_tokens: float = 0.0        # 平均 Token 消耗
    avg_cost: float = 0.0          # 平均成本（美元）
    
    # 可靠性指标
    success_rate: float = 0.0      # 任务成功率
    error_rate: float = 0.0        # 错误率
    timeout_rate: float = 0.0      # 超时率
    
    # 安全指标
    safety_violation_rate: float = 0.0  # 安全违规率
    pii_leak_rate: float = 0.0          # 隐私泄露率


class AgentBenchmark:
    """Agent 基准测试框架"""
    
    def __init__(self, agent_func, test_cases: list[dict]):
        self.agent_func = agent_func
        self.test_cases = test_cases
        self.results = []
    
    def run(self) -> AgentMetrics:
        """运行所有测试用例"""
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
                
                # 检查正确性
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
        
        # 计算指标
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
        """检查回答是否正确（支持多种匹配方式）"""
        if isinstance(expected, str):
            # 精确匹配（忽略大小写和首尾空白）
            return actual.strip().lower() == expected.strip().lower()
        elif isinstance(expected, list):
            # 包含任一关键词即可
            return any(kw.lower() in actual.lower() for kw in expected)
        elif callable(expected):
            # 自定义检查函数
            return expected(actual)
        return False
    
    def report(self, metrics: AgentMetrics) -> str:
        """生成评估报告"""
        return f"""
╔══════════════════════════════════════╗
║        Agent 评估报告                ║
╠══════════════════════════════════════╣
║ 📊 质量指标                          ║
║   准确率:     {metrics.accuracy:.1%}              ║
║   成功率:     {metrics.success_rate:.1%}              ║
║   错误率:     {metrics.error_rate:.1%}              ║
╠══════════════════════════════════════╣
║ ⚡ 效率指标                          ║
║   平均延迟:   {metrics.avg_latency:.2f}s             ║
║   平均步骤:   {metrics.avg_steps:.1f}               ║
║   平均Token:  {metrics.avg_tokens:.0f}              ║
╠══════════════════════════════════════╣
║ 🔒 可靠性                            ║
║   超时率:     {metrics.timeout_rate:.1%}              ║
╚══════════════════════════════════════╝
"""
```

---

## 回归测试：确保改进不会引入新问题

每次修改 Agent 后，都要跑一遍基准测试，确保没有退步：

```python
class RegressionTracker:
    """回归测试追踪器"""
    
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
        """记录一次评估结果"""
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
        """检查是否有指标退步超过阈值"""
        if not self.history:
            return []
        
        previous = self.history[-1]
        warnings = []
        
        if previous["accuracy"] - current.accuracy > threshold:
            warnings.append(
                f"⚠️ 准确率下降: "
                f"{previous['accuracy']:.1%} → {current.accuracy:.1%}"
            )
        
        if previous["success_rate"] - current.success_rate > threshold:
            warnings.append(
                f"⚠️ 成功率下降: "
                f"{previous['success_rate']:.1%} → {current.success_rate:.1%}"
            )
        
        # 延迟增加超过 50% 也要警告
        if (current.avg_latency > previous["avg_latency"] * 1.5 
            and previous["avg_latency"] > 0):
            warnings.append(
                f"⚠️ 延迟增加: "
                f"{previous['avg_latency']:.2f}s → {current.avg_latency:.2f}s"
            )
        
        return warnings
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 基准测试 | 用统一标准比较不同 Agent 的表现 |
| HumanEval | 测试代码生成能力（pass@k 指标） |
| MMLU | 测试知识广度（57 个学科） |
| 自定义指标 | 针对业务场景设计的评估体系 |
| 回归测试 | 确保每次改进不引入新问题 |

> **下一节预告**：掌握了评估方法后，我们来学习如何通过 Prompt 调优来提升 Agent 的表现。

---

[下一节：13.3 Prompt 调优策略 →](./03_prompt_tuning.md)
