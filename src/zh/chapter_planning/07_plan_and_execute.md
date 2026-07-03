# 5.6 Plan-and-Execute 与 Test-time Compute Scaling

> **本节目标**：掌握 Plan-and-Execute 模式的架构与实现，理解 Test-time Compute Scaling 如何改变推理范式。

---

## 从 ReAct 到 Plan-and-Execute：推理范式的演进

前面介绍的 ReAct 模式（5.2 节）让 Agent "边想边做"——每一步都同时推理和行动。但在复杂任务中，这种"走一步看一步"的策略会导致两个问题：

1. **短视陷阱**：Agent 只关注下一步操作，缺乏全局视角，容易走入死胡同
2. **上下文膨胀**：每一步都包含思考+行动+观察，长链路下上下文窗口被快速消耗

**Plan-and-Execute（先规划后执行）** 模式将"规划"和"执行"解耦：

> **ReAct 模式**：思考1→行动1→观察1→思考2→行动2→观察2→...→答案（每步都做决策，容易走偏）
>
> **Plan-Execute**：规划器生成完整计划 → 执行器逐步执行 → 遇到偏差重新规划（先看全局，再执行细节）

> 📄 **背景**：Plan-and-Execute 模式最早由 LangGraph 官方在 2024 年作为推荐模式提出。它结合了早期 HuggingGPT 的"LLM 作为任务规划器"思想和 LangGraph 的状态图架构。2025-2026 年，该模式已成为生产环境 Agent 的主流架构选择。

---

## Plan-and-Execute 架构

### 核心组件

Plan-and-Execute 由两个独立的 Agent 组成：

- **规划器（Planner）**：用大模型生成完整的步骤化计划
- **执行器（Executor）**：每步单独调用，可能决定需要重新规划
- **重规划触发器（Replan Trigger）**：判断当前步骤是否严重偏离预期

```python
class PlanAndExecuteAgent:
    """规划与执行解耦：plan → execute → replan → synthesize。"""

    def __init__(self, model: str = "gpt-4.1", max_replans: int = 3):
        self.model = model
        self.max_replans = max_replans

    def run(self, task: str) -> str:
        plan = self._plan(task)
        executed, replan_count = [], 0

        for i, step in enumerate(plan):
            result = self._execute_step(step, executed)
            executed.append({"step": step, "result": result})

            # 关键：执行中检测偏差，触发重规划
            if self._should_replan(task, executed, plan[i+1:]) \
               and replan_count < self.max_replans:
                replan_count += 1
                plan = plan[:i+1] + self._replan(task, executed, plan[i+1:])

        return self._synthesize(task, executed)

    def _plan(self, task: str) -> list[dict]:
        """让 LLM 把任务拆成 3-8 步结构化计划。"""
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"""你是任务规划专家。
任务：{task}
将任务分解为 3-8 个可执行步骤，每个步骤包含 description/tool/expected_output。
以 JSON 数组返回。只生成计划，不执行。"""}],
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("steps", result.get("plan", []))

    def _execute_step(self, step: dict, history: list[dict]) -> str:
        """执行单步：把当前步骤 + 历史结果一起发给 LLM。"""
        history_text = "\n".join(
            f"- {h['step']['description']}: {h['result'][:200]}" for h in history
        )
        prompt = f"请执行以下步骤。\n步骤：{step['description']}\n工具：{step.get('tool', '通用推理')}\n预期输出：{step.get('expected_output', '')}\n已完成：\n{history_text or '（这是第一步）'}\n请直接执行并返回结果。"
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

    def _should_replan(self, task, executed, remaining) -> bool:
        """用小模型判断最近一步是否严重偏离预期。"""
        if not remaining:
            return False
        last = executed[-1]
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",   # 用小模型判断，节省成本
            messages=[{"role": "user", "content": f"""判断执行结果是否偏离预期。
步骤目标：{last['step']['description']}
预期输出：{last['step'].get('expected_output', '')}
实际结果：{last['result'][:500]}
如果严重偏离（如执行失败、获取错误数据）回答 YES，否则 NO。"""}],
            max_tokens=10
        )
        return "YES" in resp.choices[0].message.content.upper()

    def _replan(self, task, executed, old_remaining) -> list[dict]:
        """基于已执行步骤重新规划剩余部分。"""
        # 实际：把历史 + 原剩余步骤一起发给 LLM，让它生成新计划
        # 完整实现见仓库
        ...

    def _synthesize(self, task, executed) -> str:
        """综合所有步骤结果，生成最终答案。"""
        history_text = "\n".join(
            f"步骤 {i+1} - {h['step']['description']}：\n{h['result'][:300]}"
            for i, h in enumerate(executed)
        )
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"基于以下步骤结果回答：\n任务：{task}\n结果：\n{history_text}"}]
        ).choices[0].message.content
```

> 📦 **完整代码见仓库** `examples/chapter05/plan_and_execute.py`，含 `_replan` 完整实现、调试日志、错误恢复。

### Plan-and-Execute 的三个关键设计

| 决策 | 选择 | 原因 |
|---|---|---|
| **规划与执行解耦** | 必做 | 全局规划避免短视；逐步执行保证细节可控 |
| **重规划触发用小模型** | `gpt-4.1-mini` | 判断"是否偏离预期"是简单分类任务，无需大模型 |
| **`max_replans` 硬上限** | 3 次 | 防止任务失控；超过 3 次说明问题本身需要重新拆解 |

---

## LangGraph 实现 Plan-and-Execute

生产环境中，推荐用 LangGraph（第13章）实现 Plan-and-Execute——它天然支持状态图、条件路由、循环：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class PlanExecuteState(TypedDict):
    task: str
    plan: list[dict]
    current_step: int
    executed: Annotated[list, operator.add]   # 追加模式
    replan_count: int

def should_replan_or_continue(state) -> str:
    """条件路由：继续 / 重新规划 / 完成。"""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    last = state["executed"][-1] if state["executed"] else {}
    if state.get("replan_count", 0) < 3 \
       and ("失败" in last.get("result", "") or "错误" in last.get("result", "")):
        return "replan"
    return "execute"

# 节点 + 边构建图
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

**LangGraph 的优势**：状态可序列化（支持 checkpoint 和断点恢复）；图结构天然支持复杂分支；可视化调试。

---

## Test-time Compute Scaling：推理时动态扩展计算

### 核心思想

2024-2025 年，推理模型（o1/o3/DeepSeek-R1）揭示了一个深刻的发现：**在推理时投入更多计算，比训练更大的模型更有效**。

```
训练时扩展：更大的模型 + 更多数据 = 更强的能力（成本指数级增长）
推理时扩展：同一个模型 + 更多推理步数 = 更强的结果（按需投入）
```

### 三种 Test-time Compute 策略

| 策略 | 原理 | 代表实现 | 适用 |
|---|---|---|---|
| **搜索式推理** | 生成多条推理路径，搜索最优解 | Tree of Thoughts, MCTS, LATS | 有明确评估标准的决策问题 |
| **自我纠错** | 生成初稿 → 自我批评 → 修改 → 重复 | Self-Refine, CRITIC | 有客观验证的任务（代码、数学） |
| **思维链延长** | 让模型生成更长的推理链 | o1/o3, R1, Claude Extended Thinking | 复杂推理任务 |

### 实战：自适应推理深度

不是所有问题都需要深度推理。一个好的 Agent 应该**根据问题难度自动调整**：

```python
class AdaptiveReasoningAgent:
    """按难度分级：简单直接答，中等用 CoT，复杂多路径搜索。"""

    THRESHOLDS = {
        "simple": {"max_tokens": 500, "strategy": "direct"},
        "medium": {"max_tokens": 2000, "strategy": "cot"},
        "hard":   {"max_tokens": 8000, "strategy": "search"},
    }

    def run(self, question: str) -> dict:
        # 1. 快速评估难度
        difficulty = self._assess_difficulty(question)
        cfg = self.THRESHOLDS[difficulty]
        # 2. 按难度选策略
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
            messages=[{"role": "user", "content": f"""评估问题难度。
问题：{question}
标准：simple=事实问答/简单计算；medium=多步推理/综合分析；hard=创新思维/复杂证明
只回答 simple/medium/hard。"""}],
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
            messages=[{"role": "user", "content": f"{q}\n请一步步思考。"}],
            max_tokens=max_tokens
        ).choices[0].message.content

    def _multi_path_search(self, q: str, max_tokens: int) -> str:
        # 生成 3 种解法 → LLM 选最优
        paths = [
            client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": f"{q}\n用方法 {i+1} 解答"}],
                max_tokens=max_tokens // 3
            ).choices[0].message.content
            for i in range(3)
        ]
        # 综合评判（完整实现见仓库）
        ...
```

> 📦 **完整代码见仓库** `examples/chapter05/adaptive_reasoning.py`，含多路径搜索的"对比+选择"逻辑。

---

## MCTS 在 Agent 推理中的应用

**蒙特卡洛树搜索（MCTS）** 是 AlphaGo 的核心算法，被 LATS 论文引入 LLM 推理：

> 📄 **论文出处**：*"Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models"*（Zhou et al., 2024, arXiv:2310.04406）。LATS 将 LLM 的推理过程建模为搜索树——每个节点是一个"思考状态"，每条边是一个"推理步骤"。通过 MCTS 在这棵树上搜索，找到从初始状态到目标状态的最优路径。

```python
class MCTSNode:
    """搜索树节点，含 UCB1 分数。"""
    def __init__(self, state: str, parent=None):
        self.state = state; self.parent = parent
        self.children = []; self.visits = 0; self.value = 0.0
        self.action = ""

    @property
    def ucb1(self) -> float:
        """UCB1 平衡探索与利用。"""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSReasoningAgent:
    """标准 MCTS：Selection → Expansion → Simulation → Backprop。"""

    def search(self, problem: str) -> str:
        root = MCTSNode(state=problem)
        for _ in range(self.max_iterations):
            node = self._select(root)         # 选择
            if node.visits > 0 and not self._is_terminal(node):
                self._expand(node, problem)   # 扩展
                if node.children:
                    node = random.choice(node.children)
            reward = self._simulate(node, problem)   # 模拟
            self._backpropagate(node, reward)  # 回传
        return self._extract_best_path(root)  # 访问最多的路径 = 最优解

    # _select/_expand/_simulate/_backpropagate 4 个方法都是 ~15 行的标准 MCTS 流程
    # 完整代码见仓库
```

> 📦 **完整代码见仓库** `examples/chapter05/mcts_agent.py`。

---

## 推理模型时代：o1/o3 如何改变 Agent 开发

推理模型（o1/o3/DeepSeek-R1/Claude Extended Thinking）的出现，从根本上改变了 Agent 的开发方式：

| 传统 Agent | 推理模型 Agent |
|---|---|
| 需要精心设计 CoT Prompt | 简洁的指令反而效果更好 |
| 限制 thinking tokens | 让模型自己决定推理深度 |
| 上下文利用受限于模型 | 给越多上下文越好 |
| 工具调用由 LLM 自己规划 | 推理模型做规划，小模型做执行 |

**最佳实践**：
1. **简化 System Prompt**：推理模型不需要详细的 CoT 指令
2. **让模型自己决定推理深度**：设置 `max_tokens` 16000+，只在时间敏感场景限制
3. **提供丰富的上下文**：将相关文档、历史记录、约束条件全部放入 prompt
4. **用作 Agent 的"大脑"**：推理模型做规划，小模型做工具调用和简单任务

```text
架构：推理模型(规划) → 小模型(执行) → 推理模型(验证)
```

---

## 三种推理模式对比

| 维度 | ReAct | Plan-and-Execute | Test-time Compute |
|---|---|---|---|
| 核心思想 | 边想边做 | 先规划后执行 | 难题多想，简单快答 |
| 规划深度 | 单步 | 全局 | 自适应 |
| 上下文消耗 | 高（每步都冗长） | 中（计划和执行分离） | 可控（按难度调整） |
| 适用场景 | 简单交互式任务 | 多步骤复杂任务 | 难度差异大的混合任务 |
| 错误恢复 | 难（容易陷入循环） | 好（可重新规划） | 好（多路径搜索） |
| 代表实现 | LangChain Agent | LangGraph PlanExecute | o1/o3, MCTS, LATS |

> 💡 **选型建议**：
> - **快速原型**：ReAct — 实现简单，3.2 节已有完整代码
> - **生产环境**：Plan-and-Execute — 全局视角 + 灵活重规划
> - **高难度推理**：Test-time Compute + MCTS — 多路径搜索找最优解
> - **混合场景**：Adaptive Reasoning — 根据难度自动切换策略

---

## 小结

| 概念 | 说明 |
|---|---|
| Plan-and-Execute | 先生成完整计划，再逐步执行，遇到偏差重新规划 |
| Test-time Compute Scaling | 推理时投入更多计算，比训练更大模型更高效 |
| 自适应推理 | 根据问题难度自动调整推理深度和策略 |
| MCTS 推理 | 将推理建模为搜索树，通过蒙特卡洛搜索找最优路径 |
| 推理模型 | o1/o3/R1 将 CoT 内化到模型，简化了 Agent 的 Prompt 设计 |

> 📖 **延伸阅读**：
> - LangGraph. "Plan-and-Execute Agent." LangGraph Documentation, 2024.
> - Zhou et al. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models." arXiv:2310.04406, 2024.
> - Snell et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters." ICLR, 2025.
> - OpenAI. "Learning to Reason with LLMs." OpenAI Blog, 2024.

---

*下一节：[5.7 论文解读：规划与推理前沿研究](./06_paper_readings.md)*
