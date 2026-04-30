# 17.6 Agent 专项评估框架

> **本节目标**：掌握 Agent 专项评估的前沿方法，包括 Agent-as-Judge 范式、τ-bench / OSWorld / SWE-bench 等基准测试，并能实现完整的 Agent-as-Judge 评估器。

---

## 从 LLM-as-Judge 到 Agent-as-Judge

在 18.1 中我们介绍了 LLM-as-Judge——用一个 LLM 来评判另一个 LLM 的输出质量。但 Agent 不同于普通的聊天模型：Agent 会调用工具、执行多步操作、与环境交互。仅仅评估最终输出是不够的，我们需要评估 Agent 的**整个行为轨迹**。

这就是 **Agent-as-Judge** 的核心思想：用一个 Agent（而不仅仅是一个 LLM）来评估另一个 Agent 的完整执行过程 [1]。

### LLM-as-Judge vs Agent-as-Judge

| 维度 | LLM-as-Judge | Agent-as-Judge |
|------|---------------|----------------|
| 评估对象 | 单轮文本输出 | 完整执行轨迹（多步、多工具） |
| 评估方式 | 一次性打分 | 逐步审查 + 交互验证 |
| 上下文理解 | 仅看输入和输出 | 理解工具调用、中间状态、错误恢复 |
| 评估深度 | 语义质量 | 决策质量 + 执行效率 + 错误处理 |
| 成本 | 较低 | 较高（需要多轮推理） |
| 一致性 | 较高 | 中等（评估过程更复杂） |

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class TrajectoryAspect(Enum):
    """Agent 行为轨迹的评估方面"""
    GOAL_ACHIEVEMENT = "目标达成"       # 是否完成了用户目标
    TOOL_SELECTION = "工具选择"         # 选择的工具是否合理
    TOOL_USAGE = "工具使用"             # 工具参数是否正确
    ERROR_RECOVERY = "错误恢复"         # 出错后能否自我修正
    EFFICIENCY = "执行效率"             # 是否走了弯路
    REASONING_QUALITY = "推理质量"       # 思考过程是否合理

@dataclass
class AgentTrace:
    """Agent 执行轨迹"""
    task_id: str
    user_query: str
    steps: list[dict] = field(default_factory=list)  # 每一步的详细记录
    final_output: str = ""
    success: bool = False
    total_tokens: int = 0
    total_time: float = 0.0

@dataclass
class TraceEvaluation:
    """轨迹评估结果"""
    task_id: str
    aspect: TrajectoryAspect
    score: float            # 0.0 - 1.0
    reasoning: str
    evidence: list[str] = field(default_factory=list)  # 从轨迹中提取的证据
```

---

## Agent-as-Judge 评估方法论

### 核心流程

Agent-as-Judge 的评估流程分为三个阶段：

1. **轨迹收集**：记录被评估 Agent 的完整执行过程
2. **逐步审查**：评估 Agent 对每一步进行审查
3. **综合评判**：汇总各步骤评估，给出总体评价

```python
import json
from langchain_openai import ChatOpenAI

class AgentAsJudge:
    """用 Agent 评估 Agent 的完整执行轨迹"""

    def __init__(self, model: str = "gpt-4.1"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_trace(self, trace: AgentTrace) -> dict:
        """评估完整的 Agent 执行轨迹"""

        # 阶段 1：格式化轨迹信息
        trajectory_text = self._format_trajectory(trace)

        # 阶段 2：逐步审查
        step_evaluations = self._review_steps(trajectory_text, trace.user_query)

        # 阶段 3：综合评判
        overall_evaluation = self._synthesize_evaluation(
            trace, step_evaluations
        )

        return {
            "task_id": trace.task_id,
            "step_evaluations": step_evaluations,
            "overall": overall_evaluation
        }

    def _format_trajectory(self, trace: AgentTrace) -> str:
        """将执行轨迹格式化为可读文本"""
        lines = [f"用户请求：{trace.user_query}\n"]

        for i, step in enumerate(trace.steps, 1):
            lines.append(f"--- 第 {i} 步 ---")
            if "thought" in step:
                lines.append(f"思考：{step['thought']}")
            if "action" in step:
                lines.append(f"动作：{step['action']}")
            if "tool" in step:
                lines.append(f"工具：{step['tool']}")
            if "tool_input" in step:
                lines.append(f"工具输入：{json.dumps(step['tool_input'], ensure_ascii=False)}")
            if "observation" in step:
                lines.append(f"观察：{step['observation']}")
            lines.append("")

        lines.append(f"最终输出：{trace.final_output}")
        lines.append(f"执行成功：{'是' if trace.success else '否'}")
        return "\n".join(lines)

    def _review_steps(self, trajectory: str, query: str) -> list[dict]:
        """逐步审查 Agent 行为"""
        prompt = f"""你是一个专业的 Agent 行为评审员。请逐步审查以下 Agent 的执行轨迹。

{trajectory}

请对每一步进行审查，分析：
1. 这一步的思考是否合理？
2. 选择的工具/动作是否恰当？
3. 工具参数是否正确？
4. 对观察结果的理解是否准确？

以 JSON 格式回复：
{{
    "steps": [
        {{
            "step_number": 1,
            "thought_quality": "<好/中/差>",
            "action_appropriateness": "<好/中/差>",
            "parameter_correctness": "<好/中/差>",
            "observation_understanding": "<好/中/差>",
            "issues": ["问题1", "问题2"],
            "improvement": "改进建议"
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
        """综合评判，给出总体评价"""
        prompt = f"""基于以下信息，对 Agent 的整体表现进行综合评判。

任务：{trace.user_query}
执行步骤数：{len(trace.steps)}
是否成功：{'是' if trace.success else '否'}
总耗时：{trace.total_time:.1f}s
总 Token：{trace.total_tokens}

逐步审查结果：
{json.dumps(step_evals, ensure_ascii=False, indent=2)}

请从以下维度评分（0-10）并给出总体评价：
1. 目标达成度：是否完成了用户目标
2. 决策质量：每一步的决策是否合理
3. 执行效率：是否有不必要的步骤
4. 错误处理：出错时的应对能力
5. 输出质量：最终回答的质量

以 JSON 格式回复：
{{
    "goal_achievement": <0-10>,
    "decision_quality": <0-10>,
    "execution_efficiency": <0-10>,
    "error_handling": <0-10>,
    "output_quality": <0-10>,
    "overall_score": <0-10>,
    "summary": "总体评价（2-3句话）",
    "key_strengths": ["优点1", "优点2"],
    "key_weaknesses": ["不足1", "不足2"],
    "recommendations": ["建议1", "建议2"]
}}"""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"overall_score": 0, "summary": "评估解析失败"}
```

### 实战案例：评估一个搜索 Agent

```python
# 构造一个待评估的 Agent 轨迹
sample_trace = AgentTrace(
    task_id="search_eval_001",
    user_query="帮我对比 Python 和 Rust 在 Web 后端开发中的优劣势",
    steps=[
        {
            "thought": "用户想要对比两种语言，我需要分别搜索它们的优劣势",
            "action": "调用搜索工具",
            "tool": "web_search",
            "tool_input": {"query": "Python Web 后端开发 优劣势 2025"},
            "observation": "搜索到 5 条结果：1. Python 优势：生态丰富... 2. Django/Flask..."
        },
        {
            "thought": "现在搜索 Rust 的信息",
            "action": "调用搜索工具",
            "tool": "web_search",
            "tool_input": {"query": "Rust Web 后端开发 优劣势 2025"},
            "observation": "搜索到 5 条结果：1. Rust 优势：高性能... 2. Actix/Axum..."
        },
        {
            "thought": "我已有足够信息，可以综合对比了",
            "action": "生成最终回答",
            "tool": None,
            "tool_input": None,
            "observation": None
        }
    ],
    final_output="Python 和 Rust 在 Web 后端开发中各有优势...\nPython：生态丰富、开发速度快...\nRust：高性能、内存安全...",
    success=True,
    total_tokens=3200,
    total_time=8.5
)

# 运行 Agent-as-Judge 评估
judge = AgentAsJudge(model="gpt-4.1")
result = judge.evaluate_trace(sample_trace)
print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
```

### 注意事项

| 问题 | 说明 | 应对策略 |
|------|------|----------|
| 评估偏差 | Judge Agent 可能对特定风格有偏好 | 使用多个 Judge Agent 投票 |
| 评估成本 | 每次评估需要多轮 LLM 调用 | 对简单任务用规则+LLM 混合评估 |
| 一致性问题 | 同一轨迹多次评估结果可能不同 | temperature=0 + 多次评估取平均 |
| 评估能力上限 | Judge Agent 的评估能力受自身模型能力限制 | 使用强于被评估 Agent 的模型做 Judge |
| 轨迹格式化 | 过长的轨迹可能超出上下文窗口 | 对轨迹做摘要或分段评估 |

> 💡 **最佳实践**：Agent-as-Judge 的 Judge 模型应当比被评估的 Agent 使用更强的模型。例如，用 gpt-4.1 评估 gpt-4.1-mini 驱动的 Agent，避免"学生给自己打分"的问题。

---

## τ-bench：面向工具使用的基准测试

### 什么是 τ-bench？

τ-bench（tau-bench）是 2024 年提出的专门评估 LLM Agent 工具使用能力的基准测试 [2]。与传统基准不同，τ-bench 关注 Agent 在**真实环境**中使用工具的能力，而不仅仅是选择正确的工具。

### τ-bench 的核心设计

| 特性 | 说明 |
|------|------|
| 评估维度 | 工具选择、参数填充、多步推理、错误处理 |
| 环境 | 模拟真实 API 环境（航班查询、酒店预订等） |
| 难度级别 | 简单单工具 → 复杂多工具协作 |
| 评估方式 | 端到端结果匹配 + 轨迹审查 |
| 核心创新 | 引入"用户模拟器"，模拟真实用户的多轮对话行为 |

### τ-bench 评估维度详解

```python
@dataclass
class TauBenchResult:
    """τ-bench 评估结果"""
    task_id: str
    # 核心指标
    tool_selection_accuracy: float   # 工具选择准确率
    param_fill_accuracy: float       # 参数填充准确率
    multi_step_success_rate: float   # 多步任务成功率
    error_recovery_rate: float       # 错误恢复率
    # 辅助指标
    avg_steps_per_task: float        # 平均步骤数
    avg_redundant_steps: float       # 平均冗余步骤数
    total_token_usage: int           # 总 Token 使用量

class TauBenchEvaluator:
    """τ-bench 风格的评估器"""

    def __init__(self, agent_func, user_simulator, env):
        self.agent_func = agent_func       # 待评估的 Agent
        self.user_simulator = user_simulator  # 用户模拟器
        self.env = env                       # 模拟环境

    def evaluate_task(self, task: dict) -> TauBenchResult:
        """评估单个任务"""
        steps = []
        tool_calls_correct = 0
        tool_calls_total = 0
        params_correct = 0
        params_total = 0
        errors_encountered = 0
        errors_recovered = 0

        # 模拟多轮对话
        conversation = [{"role": "user", "content": task["initial_query"]}]

        for step_idx in range(20):  # 最多 20 步
            agent_response = self.agent_func(conversation)

            # 提取工具调用
            if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
                for tc in agent_response.tool_calls:
                    tool_calls_total += 1
                    params_total += len(tc["args"])

                    # 检查工具选择是否正确
                    expected_tools = task.get("expected_tool_sequence", [])
                    if step_idx < len(expected_tools):
                        if tc["name"] == expected_tools[step_idx]:
                            tool_calls_correct += 1

                        # 检查参数
                        expected_args = task.get("expected_args", {}).get(step_idx, {})
                        for key, expected_val in expected_args.items():
                            if key in tc["args"] and tc["args"][key] == expected_val:
                                params_correct += 1

                    # 执行工具并获取结果
                    try:
                        observation = self.env.execute(tc["name"], tc["args"])
                    except Exception as e:
                        errors_encountered += 1
                        observation = f"错误：{str(e)}"

                    steps.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                        "observation": observation,
                        "is_error": "错误" in str(observation)
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
                # Agent 给出最终回答
                break

        # 评估最终结果是否正确
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
        """检查最终结果是否符合预期"""
        expected_results = task.get("expected_results", {})
        if not expected_results:
            return len(steps) > 0

        # 简化：检查关键工具是否被调用且成功
        for required_tool in expected_results.get("required_tools", []):
            found = any(s["tool"] == required_tool and not s["is_error"] for s in steps)
            if not found:
                return False
        return True

    def _count_redundant_steps(self, steps: list) -> int:
        """计算冗余步骤数（重复调用同一工具且参数相同）"""
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

## OSWorld 与 VisualWebArena：多模态 Agent 基准

### OSWorld：真实桌面环境中的 Agent 评估

OSWorld [3] 是 2024 年提出的首个在**真实操作系统环境**中评估多模态 Agent 的基准。与之前基于模拟环境的基准不同，OSWorld 让 Agent 在真实的 Ubuntu / Windows / macOS 桌面环境中完成任务。

| 特性 | 说明 |
|------|------|
| 环境 | 真实 OS（Ubuntu 22.04、Windows 11、macOS） |
| 任务类型 | 文件操作、应用使用、网页浏览、多应用协作 |
| 交互方式 | 截图 + 可访问性树（Accessibility Tree） |
| 任务数量 | 369 个真实任务 |
| 评估方式 | 基于执行结果的函数验证（非字符串匹配） |

### VisualWebArena：网页环境中的多模态 Agent 基准

VisualWebArena [4] 专注于**网页环境**中的多模态 Agent 评估，要求 Agent 通过视觉理解和操作网页：

| 特性 | 说明 |
|------|------|
| 环境 | 自托管的 Web 应用（电商、论坛、CMS） |
| 任务类型 | 信息检索、内容管理、数据操作 |
| 交互方式 | 网页截图 + DOM 操作 |
| 核心挑战 | 需要理解视觉布局、表单填写、多页面导航 |

### 多模态 Agent 基准对比

| 基准 | 环境类型 | 交互方式 | 任务数量 | 最佳成功率 |
|------|----------|----------|----------|------------|
| OSWorld | 真实桌面 OS | 截图 + 键鼠操作 | 369 | ~12.5% (2024) |
| VisualWebArena | 网页应用 | 截图 + DOM 操作 | 910 | ~14.6% (2024) |
| WebArena | 网页应用 | HTML + DOM | 812 | ~35.9% (2024) |
| τ-bench | 模拟 API | 文本 + 工具调用 | 200+ | ~68% (2024) |

> ⚠️ **注意**：OSWorld 和 VisualWebArena 的最佳成功率远低于纯文本基准，说明多模态 Agent 仍有巨大提升空间。

### 评估多模态 Agent 的关键指标

```python
@dataclass
class MultimodalEvalMetrics:
    """多模态 Agent 评估指标"""
    # 基础指标
    task_success_rate: float          # 任务完成率
    partial_success_rate: float       # 部分完成率

    # 视觉理解指标
    screenshot_understanding_acc: float  # 截图理解准确率
    element_localization_acc: float      # 元素定位准确率
    ocr_accuracy: float                  # OCR 准确率

    # 操作指标
    action_accuracy: float            # 动作选择准确率
    coordinate_accuracy: float        # 坐标定位准确率（点击任务）
    typing_accuracy: float            # 输入准确率

    # 效率指标
    avg_steps: int                    # 平均步骤数
    avg_time_per_task: float          # 平均每任务耗时
    unnecessary_actions_rate: float   # 不必要操作比例


class OSWorldStyleEvaluator:
    """OSWorld 风格的多模态 Agent 评估器"""

    def __init__(self, agent_func, environment):
        self.agent_func = agent_func
        self.env = environment

    def evaluate(self, task: dict) -> MultimodalEvalMetrics:
        """评估单个多模态任务"""
        steps_data = []
        action_correct = 0
        action_total = 0
        coord_errors = []
        typing_errors = []

        # 重置环境
        self.env.reset(task["initial_state"])

        for step_idx in range(task.get("max_steps", 15)):
            # 获取当前截图和可访问性信息
            screenshot = self.env.get_screenshot()
            accessibility_tree = self.env.get_accessibility_tree()

            # Agent 决策
            agent_action = self.agent_func(
                task["instruction"],
                screenshot,
                accessibility_tree,
                steps_data  # 之前的历史
            )

            # 记录步骤
            step_info = {
                "step": step_idx,
                "action_type": agent_action.get("type"),
                "action_params": agent_action.get("params", {}),
            }

            # 评估动作准确性
            if step_idx < len(task.get("expected_actions", [])):
                expected = task["expected_actions"][step_idx]
                action_total += 1

                if agent_action["type"] == expected["type"]:
                    action_correct += 1

                    # 评估坐标/输入准确性
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

            # 执行动作
            self.env.execute_action(agent_action)
            steps_data.append(step_info)

            # 检查是否完成
            if self.env.is_task_completed():
                break

        # 计算最终结果
        success = self.env.verify_final_state(task["expected_state"])

        return MultimodalEvalMetrics(
            task_success_rate=1.0 if success else 0.0,
            partial_success_rate=self._partial_score(task, steps_data),
            screenshot_understanding_acc=0.0,  # 需要额外评估
            element_localization_acc=0.0,       # 需要额外评估
            ocr_accuracy=0.0,                    # 需要额外评估
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
            avg_time_per_task=0.0,  # 需要实际计时
            unnecessary_actions_rate=0.0  # 需要人工标注
        )

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
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
        """计算部分完成分数"""
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

## SWE-bench Verified：代码 Agent 的黄金基准

### SWE-bench 概述

SWE-bench [5] 是评估代码 Agent 解决真实 GitHub Issue 能力的基准。2024 年推出的 **SWE-bench Verified** 版本通过人工验证，过滤掉了有问题的测试用例，使评估结果更可靠。

| 版本 | Issue 数量 | 说明 |
|------|-----------|------|
| SWE-bench Full | 2294 | 全量数据集，部分 Issue 描述不清晰 |
| SWE-bench Lite | 300 | 精选子集，但仍有质量问题 |
| SWE-bench Verified | 500 | 人工验证，每个 Issue 都确认可解决 |

### SWE-bench Verified 评估方式

```python
@dataclass
class SWEBenchResult:
    """SWE-bench 评估结果"""
    instance_id: str
    repo: str
    resolved: bool          # 是否解决了 Issue
    patch_applied: bool     # Patch 是否能应用
    tests_passed: bool      # 测试是否通过
    fail_to_pass: list[str]   # 从失败变为通过的测试
    pass_to_pass: list[str]   # 始终通过的测试
    fail_to_fail: list[str]   # 始终失败的测试

class SWEBenchEvaluator:
    """SWE-bench 风格的评估器"""

    def __init__(self, agent_func, docker_env=None):
        self.agent_func = agent_func
        self.docker_env = docker_env

    def evaluate_instance(self, instance: dict) -> SWEBenchResult:
        """评估单个 SWE-bench 实例"""
        # 1. 准备环境
        repo_path = self._setup_repo(instance)

        # 2. 让 Agent 分析问题并生成 Patch
        agent_patch = self.agent_func(
            problem_statement=instance["problem_statement"],
            repo_path=repo_path,
            hints_text=instance.get("hints_text", "")
        )

        # 3. 应用 Patch
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

        # 4. 运行测试
        test_results = self._run_tests(
            repo_path,
            instance.get("test_patch", ""),
            instance.get("fail_to_pass", []),
            instance.get("pass_to_pass", [])
        )

        # 5. 判定是否解决
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
        """设置 Git 仓库到指定版本"""
        import subprocess
        repo_dir = f"/tmp/swebench_{instance['instance_id']}"
        # 克隆并 checkout 到基础 commit
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
        """尝试应用 Patch"""
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
        """运行测试并收集结果"""
        import subprocess
        # 应用测试 Patch
        subprocess.run(
            ["git", "apply"],
            input=test_patch.encode(),
            cwd=repo_path,
            capture_output=True
        )
        # 运行测试
        result = subprocess.run(
            ["python", "-m", "pytest", "-x", "--tb=short"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300
        )
        # 解析测试结果（简化版）
        output = result.stdout + result.stderr
        return {
            "fail_to_pass_resolved": [],   # 需要解析 output
            "pass_to_pass_passed": [],
            "pass_to_pass_failed": [],
            "fail_to_fail": []
        }
```

### SWE-bench Verified 最新进展（2025—2026）

| 排名 | 方法 | 解决率 | 说明 |
|------|------|--------|------|
| OpenHands + CodeAct | ~53% | 2025 年初 | 开源最佳 |
| Devin | ~50% | 2025 年初 | 商业产品 |
| SWE-Agent + GPT-4.1 | ~48% | 2025 年 | Agent 框架 |
| AutoCodeRover | ~45% | 2024 年 | 谱分析 + LLM |
| Amazon Q Developer | ~42% | 2024 年 | Amazon 出品 |

> 💡 **趋势观察**：SWE-bench Verified 的解决率在 2025 年已突破 50%，但仍有近半数 Issue 无法自动解决。核心瓶颈在于长上下文理解、多文件修改和复杂调试推理。

---

## 完整实战：构建 Agent-as-Judge 评估系统

下面我们将实现一个完整的 Agent-as-Judge 评估系统，用于评估任意 LangChain Agent 的表现。

```python
"""
Agent-as-Judge 评估系统
支持：轨迹收集、多维度评估、报告生成
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ============================================================
# 第一部分：轨迹收集器
# ============================================================

@dataclass
class CollectedStep:
    """收集的单步执行记录"""
    step_index: int
    thought: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""
    is_error: bool = False
    timestamp: float = 0.0
    token_count: int = 0


class TraceCollector:
    """收集 Agent 执行轨迹的回调处理器"""

    def __init__(self):
        self.traces: dict[str, list[CollectedStep]] = {}
        self._current_trace: list[CollectedStep] = []
        self._step_counter: int = 0

    def start_trace(self, task_id: str):
        """开始新的轨迹收集"""
        self._current_trace = []
        self._step_counter = 0
        self.traces[task_id] = self._current_trace

    def record_step(
        self,
        thought: str = "",
        tool_name: str = "",
        tool_input: dict = None,
        tool_output: str = "",
        is_error: bool = False,
        token_count: int = 0
    ):
        """记录一步执行"""
        step = CollectedStep(
            step_index=self._step_counter,
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output,
            is_error=is_error,
            timestamp=time.time(),
            token_count=token_count
        )
        self._current_trace.append(step)
        self._step_counter += 1

    def get_trace(self, task_id: str) -> list[CollectedStep]:
        """获取指定任务的轨迹"""
        return self.traces.get(task_id, [])


# ============================================================
# 第二部分：Agent-as-Judge 评估器
# ============================================================

class AgentAsJudgeEvaluator:
    """完整的 Agent-as-Judge 评估系统"""

    def __init__(
        self,
        judge_model: str = "gpt-4.1",
        dimensions: list[str] = None
    ):
        self.llm = ChatOpenAI(model=judge_model, temperature=0)
        self.dimensions = dimensions or [
            "目标达成度",
            "工具选择合理性",
            "参数正确性",
            "错误处理能力",
            "执行效率",
            "输出质量"
        ]

    def evaluate(
        self,
        task_query: str,
        steps: list[CollectedStep],
        final_output: str,
        expected_output: str = None,
        ground_truth_steps: list[dict] = None
    ) -> dict:
        """完整评估流程"""

        # 1. 轨迹格式化
        formatted_trace = self._format_steps(steps)

        # 2. 逐步评估
        step_evals = self._evaluate_each_step(
            task_query, formatted_trace
        )

        # 3. 整体评估
        overall_eval = self._evaluate_overall(
            task_query, formatted_trace, final_output,
            expected_output, step_evals
        )

        # 4. 与 Ground Truth 对比（如果有）
        comparison = None
        if ground_truth_steps:
            comparison = self._compare_with_ground_truth(
                steps, ground_truth_steps
            )

        return {
            "query": task_query,
            "step_count": len(steps),
            "error_count": sum(1 for s in steps if s.is_error),
            "dimensions": overall_eval,
            "step_details": step_evals,
            "ground_truth_comparison": comparison,
            "final_output": final_output
        }

    def _format_steps(self, steps: list[CollectedStep]) -> str:
        """格式化执行步骤"""
        lines = []
        for s in steps:
            lines.append(f"### 步骤 {s.step_index + 1}")
            if s.thought:
                lines.append(f"思考：{s.thought}")
            if s.tool_name:
                lines.append(f"调用工具：{s.tool_name}")
                lines.append(
                    f"参数：{json.dumps(s.tool_input, ensure_ascii=False)}"
                )
            if s.tool_output:
                status = "❌ 失败" if s.is_error else "✅ 成功"
                lines.append(f"结果({status})：{s.tool_output[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _evaluate_each_step(
        self, query: str, formatted_trace: str
    ) -> list[dict]:
        """逐步评估"""
        prompt = f"""你是一个专业的 Agent 行为评审员。请逐步审查以下 Agent 执行轨迹。

用户任务：{query}

执行轨迹：
{formatted_trace}

请对每一步进行评估，输出 JSON：
{{
    "steps": [
        {{
            "step_number": 1,
            "quality": "<优/良/中/差>",
            "is_redundant": <true/false>,
            "issues": ["问题1"],
            "improvement": "改进建议"
        }}
    ]
}}"""

        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return result.get("steps", [])
        except json.JSONDecodeError:
            return []

    def _evaluate_overall(
        self,
        query: str,
        formatted_trace: str,
        final_output: str,
        expected_output: Optional[str],
        step_evals: list[dict]
    ) -> dict:
        """整体维度评估"""
        expected_section = ""
        if expected_output:
            expected_section = f"\n期望输出：\n{expected_output}\n"

        dimensions_text = "、".join(self.dimensions)

        prompt = f"""你是一个专业的 Agent 评估专家。请对以下 Agent 的整体表现进行评估。

用户任务：{query}
{expected_section}
执行轨迹：
{formatted_trace}

最终输出：
{final_output}

请从以下维度评分（0-10 分）：{dimensions_text}

以 JSON 格式回复：
{{
    "scores": {{
        "目标达成度": <0-10>,
        "工具选择合理性": <0-10>,
        "参数正确性": <0-10>,
        "错误处理能力": <0-10>,
        "执行效率": <0-10>,
        "输出质量": <0-10>
    }},
    "weighted_score": <加权总分 0-10>,
    "summary": "2-3句总结",
    "top_issue": "最需要改进的一点"
}}"""

        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"scores": {}, "weighted_score": 0, "summary": "解析失败"}

    def _compare_with_ground_truth(
        self,
        actual: list[CollectedStep],
        expected: list[dict]
    ) -> dict:
        """与 Ground Truth 对比"""
        # 工具序列匹配
        actual_tools = [s.tool_name for s in actual if s.tool_name]
        expected_tools = [s["tool"] for s in expected if "tool" in s]

        # 计算最长公共子序列比例
        lcs_len = self._lcs_length(actual_tools, expected_tools)
        tool_seq_accuracy = (
            lcs_len / len(expected_tools) if expected_tools else 1.0
        )

        return {
            "tool_sequence_accuracy": tool_seq_accuracy,
            "actual_tool_count": len(actual_tools),
            "expected_tool_count": len(expected_tools),
            "extra_steps": max(0, len(actual_tools) - len(expected_tools)),
            "missing_tools": [
                t for t in expected_tools if t not in actual_tools
            ]
        }

    @staticmethod
    def _lcs_length(s1: list, s2: list) -> int:
        """最长公共子序列长度"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]


# ============================================================
# 第三部分：批量评估与报告生成
# ============================================================

class AgentEvaluationReport:
    """生成评估报告"""

    def __init__(self, results: list[dict]):
        self.results = results

    def generate(self) -> str:
        """生成 Markdown 格式的评估报告"""
        lines = ["# Agent 评估报告\n"]

        # 总体统计
        total = len(self.results)
        avg_score = sum(
            r["dimensions"].get("weighted_score", 0)
            for r in self.results
        ) / total if total > 0 else 0

        lines.append("## 总体统计\n")
        lines.append(f"- 评估任务数：{total}")
        lines.append(f"- 平均加权得分：{avg_score:.1f} / 10")
        lines.append(f"- 平均步骤数：{sum(r['step_count'] for r in self.results) / total:.1f}")
        lines.append(f"- 平均错误数：{sum(r['error_count'] for r in self.results) / total:.1f}")

        # 维度平均分
        lines.append("\n## 各维度平均分\n")
        lines.append("| 维度 | 平均分 |")
        lines.append("|------|--------|")
        for dim in ["目标达成度", "工具选择合理性", "参数正确性",
                     "错误处理能力", "执行效率", "输出质量"]:
            scores = [
                r["dimensions"].get("scores", {}).get(dim, 0)
                for r in self.results
            ]
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"| {dim} | {avg:.1f} |")

        # 详细结果
        lines.append("\n## 详细结果\n")
        for i, result in enumerate(self.results):
            lines.append(f"### 任务 {i+1}\n")
            lines.append(f"- 查询：{result['query'][:100]}")
            lines.append(f"- 步骤数：{result['step_count']}")
            lines.append(f"- 错误数：{result['error_count']}")
            score = result["dimensions"].get("weighted_score", 0)
            lines.append(f"- 加权得分：{score:.1f} / 10")
            summary = result["dimensions"].get("summary", "")
            lines.append(f"- 评价：{summary}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 使用示例
# ============================================================

def demo_evaluation():
    """演示完整的评估流程"""

    # 1. 创建轨迹收集器
    collector = TraceCollector()

    # 2. 模拟 Agent 执行
    task_id = "demo_001"
    collector.start_trace(task_id)

    # 模拟步骤
    collector.record_step(
        thought="用户想了解北京的天气，我需要调用天气查询工具",
        tool_name="get_weather",
        tool_input={"city": "北京"},
        tool_output="北京今天晴，气温 25°C，湿度 40%",
        token_count=150
    )

    collector.record_step(
        thought="已经获取到天气信息，可以回答用户了",
        tool_name="",
        tool_input={},
        tool_output="",
        token_count=80
    )

    # 3. 运行评估
    evaluator = AgentAsJudgeEvaluator(judge_model="gpt-4.1")
    trace = collector.get_trace(task_id)

    result = evaluator.evaluate(
        task_query="北京今天天气怎么样？",
        steps=trace,
        final_output="北京今天天气晴朗，气温 25°C，湿度 40%，适合出行。",
        expected_output="北京的天气信息，包含温度和湿度"
    )

    # 4. 生成报告
    report = AgentEvaluationReport([result])
    print(report.generate())


if __name__ == "__main__":
    demo_evaluation()
```

---

## Agent 专项评估基准全景对比

单一基准无法覆盖 Agent 的全部能力。生产团队通常会按能力域组合多套基准：工具调用看 τ-bench，网页操作看 WebArena/VisualWebArena，代码修改看 SWE-bench，研究任务看 GAIA/HLE，安全看 AgentDojo/InjecAgent/ASB，记忆系统看 LoCoMo/LongMemEval。

### 按能力域选择评估基准

| 能力域 | 代表基准 | 主要评估什么 | 适合的 Agent 类型 |
|--------|----------|--------------|-------------------|
| **工具调用** | τ-bench、ToolBench、API-Bank | 工具选择、参数填充、多轮工具使用 | 工具型 Agent、客服 Agent |
| **网页操作** | WebArena、VisualWebArena、Mind2Web | 网页导航、DOM/视觉理解、表单操作 | Browser Use / Web Agent |
| **桌面操作** | OSWorld、AndroidWorld | GUI 操作、多应用协作、环境状态验证 | Computer Use Agent |
| **代码任务** | SWE-bench Verified、HumanEval、RepoBench | Issue 修复、代码生成、多文件理解 | Coding Agent |
| **深度研究** | GAIA、HLE、BrowseComp、FRAMES | 多步检索、证据整合、复杂问答 | Deep Research Agent |
| **安全鲁棒性** | AgentDojo、InjecAgent、ASB、PromptBench | 间接提示注入、工具滥用、越权行为 | Web/RAG/Tool Agent |
| **长期记忆** | LoCoMo、LongMemEval、ConvoMem | 长对话记忆、时序推理、用户画像一致性 | Memory Agent、个人助理 |

### 主流基准横向对比

| 基准 | 领域 | 核心能力 | 最佳表现 | 局限性 |
|------|------|----------|----------|--------|
| τ-bench | 工具使用 | 工具选择与参数填充 | ~68% | 环境模拟，非真实 |
| OSWorld | 桌面操作 | 多应用协作 | ~12.5% | 成功率低，成本高 |
| VisualWebArena | 网页操作 | 视觉理解+DOM操作 | ~14.6% | 仅限网页环境 |
| SWE-bench Verified | 代码修复 | 问题定位+Patch生成 | ~53% | 仅限 Python 项目 |
| WebArena | 网页导航 | 信息检索+操作 | ~35.9% | 无视觉输入版本 |
| GAIA | 通用推理 | 多步推理+工具调用 | ~45% | 任务数量有限 |
| BrowseComp | 浏览研究 | 网页检索+复杂问答 | 快速变化 | 依赖实时网页环境 |
| AgentDojo | Agent 安全 | 工具注入攻防 | 任务相关 | 偏安全专项 |
| InjecAgent | 间接注入 | 工具集成 Agent 的注入攻击 | 任务相关 | 偏攻击评估 |
| LoCoMo | 长期记忆 | 长对话、多跳记忆推理 | 方案差异大 | 主要评估记忆层 |
| LongMemEval | 长期记忆 | 长上下文记忆检索与问答 | 方案差异大 | 与具体记忆架构强相关 |
| AgentBench | 多任务 | 多种 Agent 能力 | ~35% | 评估维度不够细 |

### 如何组合成生产评估套件？

如果你要评估一个真实 Agent，不建议只报一个 benchmark 分数，而应构建“能力矩阵”：

```text
基础能力：工具调用准确率、格式合法率、任务完成率
  +
场景能力：Web / Code / Research / Memory 等专项基准
  +
安全能力：间接提示注入、越权工具调用、数据泄露测试
  +
生产指标：延迟、成本、人工接管率、回归稳定性
```

例如，一个 Deep Research Agent 的评估套件可以是：

| 评估层 | 指标 |
|--------|------|
| **任务结果** | GAIA / BrowseComp 正确率 |
| **过程质量** | 搜索轮数、来源多样性、引用有效率、冲突处理率 |
| **安全性** | 对恶意网页指令的拒绝率、外链访问审批率 |
| **生产性** | 平均耗时、Token 成本、失败重试率、人工接管率 |

这能避免一个常见误区：**Benchmark 高分不等于生产可用**。Agent 上线前必须同时满足能力、安全、成本和稳定性四个维度。

---

## 小结

| 概念 | 说明 |
|------|------|
| Agent-as-Judge | 用 Agent 评估 Agent 的完整执行轨迹，超越 LLM-as-Judge 的单轮评判 |
| τ-bench | 面向工具使用能力的专项基准，关注工具选择和参数填充 |
| WebArena / VisualWebArena | 浏览器与网页操作 Agent 的核心评估基准 |
| OSWorld | 真实桌面环境中的多模态 Agent 评估，成功率仍很低 |
| SWE-bench Verified | 代码 Agent 的黄金基准，人工验证确保评估可靠性 |
| GAIA / BrowseComp | Deep Research Agent 和多步检索推理的重要参考 |
| AgentDojo / InjecAgent / ASB | Agent 安全与间接提示注入评估基准 |
| LoCoMo / LongMemEval | 长期记忆与 Memory Governance 的评估基准 |
| 评估系统 | 轨迹收集 → 逐步审查 → 综合评判 → 报告生成 |

> **下一节预告**：我们将学习 A/B 测试与回归测试自动化，了解如何在 CI/CD 流水线中持续保障 Agent 质量。

---

## 参考文献

[1] ZHUGE M, WANG H, LIU J, et al. Agent-as-Judge: Evaluate Agents with Agents for Long Tasks[J]. arXiv preprint arXiv:2410.10934, 2024.

[2] SIYAN Z, YU G, JIAYI P, et al. τ-bench: A Benchmark for Tool-Using LLMs[J]. arXiv preprint arXiv:2406.12045, 2024.

[3] XUE Y, WU D, ZHENG Z, et al. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments[C]//NeurIPS. 2024.

[4] KOH J, LO R, JANG J, et al. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks[J]. arXiv preprint arXiv:2401.13649, 2024.

[5] JIMENEZ C E, YANG J, WETZIG A, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?[C]//ICLR. 2024.

---

[17.7 A/B 测试与回归测试自动化](./07_ab_testing.md)
