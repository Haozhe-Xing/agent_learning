# 5.6 Plan-and-Execute 与 Test-time Compute Scaling

> **本节目标**：掌握 Plan-and-Execute 模式的架构与实现，理解 Test-time Compute Scaling 如何改变推理范式。

---

## 从 ReAct 到 Plan-and-Execute：推理范式的演进

前面介绍的 ReAct 模式（5.2 节）让 Agent "边想边做"——每一步都同时推理和行动。但在复杂任务中，这种"走一步看一步"的策略会导致两个问题：

1. **短视陷阱**：Agent 只关注下一步操作，缺乏全局视角，容易走入死胡同
2. **上下文膨胀**：每一步都包含思考+行动+观察，长链路下上下文窗口被快速消耗

**Plan-and-Execute（先规划后执行）** 模式将"规划"和"执行"解耦，解决了这两个问题：

> **ReAct 模式**：思考1→行动1→观察1→思考2→行动2→观察2→...→答案（每步都做决策，容易走偏）
>
> **Plan-Execute**：规划器生成完整计划 → 执行器逐步执行 → 遇到偏差重新规划（先看全局，再执行细节）

> 📄 **背景**：Plan-and-Execute 模式最早由 LangGraph 官方在 2024 年作为推荐模式提出。它并非全新的学术论文，而是工程实践中的范式总结——结合了早期 HuggingGPT 的"LLM 作为任务规划器"思想和 LangGraph 的状态图架构。2025-2026 年，该模式已成为生产环境 Agent 的主流架构选择。

---

## Plan-and-Execute 架构

### 核心组件

Plan-and-Execute 由两个独立的 Agent 组成：

```python
from openai import OpenAI
import json

client = OpenAI()


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent"""
    
    def __init__(self, model: str = "gpt-4.1", max_replans: int = 3):
        self.model = model
        self.max_replans = max_replans
    
    def run(self, task: str) -> str:
        """执行 Plan-and-Execute 循环"""
        
        # 1. 规划阶段：生成完整计划
        plan = self._plan(task)
        print(f"📋 初始计划（{len(plan)} 步）：")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step['description']}")
        
        # 2. 执行阶段：逐步执行计划
        executed_steps = []
        replan_count = 0
        
        for i, step in enumerate(plan):
            print(f"\n▶️ 执行步骤 {i+1}/{len(plan)}: {step['description']}")
            
            # 执行当前步骤
            result = self._execute_step(step, executed_steps)
            executed_steps.append({
                "step": step,
                "result": result
            })
            
            print(f"  ✅ 结果: {result[:100]}...")
            
            # 3. 检查是否需要重新规划
            if self._should_replan(task, executed_steps, plan[i+1:]):
                if replan_count < self.max_replans:
                    replan_count += 1
                    print(f"\n🔄 检测到偏差，重新规划（{replan_count}/{self.max_replans}）")
                    
                    # 基于已执行的步骤重新规划剩余部分
                    remaining_plan = self._replan(
                        task, executed_steps, plan[i+1:]
                    )
                    plan = plan[:i+1] + remaining_plan
                    print(f"📋 更新后的计划：")
                    for j, s in enumerate(plan[i+1:], i+2):
                        print(f"  {j}. {s['description']}")
        
        # 4. 综合所有步骤的结果生成最终答案
        final_answer = self._synthesize(task, executed_steps)
        return final_answer
    
    def _plan(self, task: str) -> list[dict]:
        """规划器：生成完整执行计划"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""你是一个任务规划专家。请为以下任务生成一个详细的执行计划。

任务：{task}

要求：
1. 将任务分解为 3-8 个可执行的步骤
2. 每个步骤必须明确：做什么、用什么工具、预期的输出
3. 步骤之间有清晰的依赖关系
4. 不要执行任何步骤，只生成计划

以 JSON 数组格式返回：
[
  {{
    "id": 1,
    "description": "步骤描述",
    "tool": "使用的工具或方法",
    "expected_output": "预期输出",
    "depends_on": []
  }},
  ...
]"""
            }],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("steps", result.get("plan", []))
    
    def _execute_step(self, step: dict, history: list[dict]) -> str:
        """执行器：执行单个步骤"""
        history_text = "\n".join([
            f"- {h['step']['description']}: {h['result'][:200]}"
            for h in history
        ])
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""请执行以下步骤。

步骤：{step['description']}
工具：{step.get('tool', '通用推理')}
预期输出：{step.get('expected_output', '')}

已完成的步骤：
{history_text or '（这是第一个步骤）'}

请直接执行并返回结果。"""
            }]
        )
        
        return response.choices[0].message.content
    
    def _should_replan(
        self,
        task: str,
        executed: list[dict],
        remaining: list[dict]
    ) -> bool:
        """判断是否需要重新规划"""
        if not remaining:
            return False
        
        # 简化判断：检查最近执行的步骤结果是否偏离预期
        last = executed[-1]
        last_result = last["result"]
        expected = last["step"].get("expected_output", "")
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # 用小模型做判断，节省成本
            messages=[{
                "role": "user",
                "content": f"""判断执行结果是否偏离预期。

步骤目标：{last['step']['description']}
预期输出：{expected}
实际结果：{last_result[:500]}

如果实际结果严重偏离预期（如执行失败、获取到错误数据），回答 YES。
如果结果可以接受，回答 NO。

只回答 YES 或 NO。"""
            }],
            max_tokens=10
        )
        
        return "YES" in response.choices[0].message.content.upper()
    
    def _replan(
        self,
        task: str,
        executed: list[dict],
        old_remaining: list[dict]
    ) -> list[dict]:
        """重新规划剩余步骤"""
        history_text = "\n".join([
            f"- {h['step']['description']}: {h['result'][:200]}"
            for h in executed
        ])
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""基于已执行的步骤，重新规划剩余任务。

原始任务：{task}

已完成的步骤：
{history_text}

原计划的剩余步骤（可能已不适用）：
{json.dumps([s['description'] for s in old_remaining], ensure_ascii=False)}

请根据实际执行情况，生成新的剩余步骤。以 JSON 数组格式返回。"""
            }],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("steps", result.get("plan", []))
    
    def _synthesize(self, task: str, executed: list[dict]) -> str:
        """综合所有步骤结果，生成最终答案"""
        history_text = "\n".join([
            f"步骤 {i+1} - {h['step']['description']}：\n{h['result'][:300]}"
            for i, h in enumerate(executed)
        ])
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""基于以下步骤的执行结果，回答原始任务。

原始任务：{task}

执行结果：
{history_text}

请给出完整、结构化的最终答案。"""
            }]
        )
        
        return response.choices[0].message.content


# 使用示例
agent = PlanAndExecuteAgent()
result = agent.run(
    "分析2025年全球AI Agent市场规模，并与2024年对比，"
    "指出增长最快的细分领域和关键驱动因素"
)
print(f"\n📊 最终答案：\n{result}")
```

---

## LangGraph 实现 Plan-and-Execute

生产环境中，推荐用 LangGraph（第12章）实现 Plan-and-Execute，因为它天然支持状态图和条件路由：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict):
    """Plan-and-Execute 状态"""
    task: str                        # 原始任务
    plan: list[dict]                 # 当前计划
    current_step: int                # 当前执行到第几步
    executed: Annotated[list, operator.add]  # 已执行的步骤（追加模式）
    replan_count: int                # 重新规划次数


def plan_step(state: PlanExecuteState) -> dict:
    """规划节点"""
    task = state["task"]
    
    # 调用规划器生成计划
    plan = PlanAndExecuteAgent()._plan(task)
    
    return {"plan": plan, "current_step": 0, "replan_count": 0}


def execute_step(state: PlanExecuteState) -> dict:
    """执行节点"""
    plan = state["plan"]
    step_idx = state["current_step"]
    
    if step_idx >= len(plan):
        return {}
    
    step = plan[step_idx]
    result = PlanAndExecuteAgent()._execute_step(step, state.get("executed", []))
    
    return {
        "current_step": step_idx + 1,
        "executed": [{"step": step, "result": result}]
    }


def should_replan_or_continue(state: PlanExecuteState) -> str:
    """路由判断：继续执行 / 重新规划 / 完成"""
    plan = state["plan"]
    step_idx = state["current_step"]
    
    if step_idx >= len(plan):
        return "synthesize"
    
    # 检查是否需要重新规划（简化版）
    executed = state.get("executed", [])
    if executed and state.get("replan_count", 0) < 3:
        last = executed[-1]
        if "失败" in last["result"] or "错误" in last["result"]:
            return "replan"
    
    return "execute"


def replan_step(state: PlanExecuteState) -> dict:
    """重新规划节点"""
    new_plan = PlanAndExecuteAgent()._replan(
        state["task"],
        state.get("executed", []),
        state["plan"][state["current_step"]:]
    )
    
    return {
        "plan": state["plan"][:state["current_step"]] + new_plan,
        "replan_count": state.get("replan_count", 0) + 1
    }


# 构建 LangGraph 图
graph = StateGraph(PlanExecuteState)

graph.add_node("planner", plan_step)
graph.add_node("executor", execute_step)
graph.add_node("replanner", replan_step)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges(
    "executor",
    should_replan_or_continue,
    {
        "execute": "executor",
        "replan": "replanner",
        "synthesize": END,
    }
)
graph.add_edge("replanner", "executor")

app = graph.compile()

# 运行
result = app.invoke({
    "task": "调研并对比三大云平台（AWS/Azure/GCP）的 AI Agent 服务",
    "plan": [],
    "current_step": 0,
    "executed": [],
    "replan_count": 0,
})
```

---

## Test-time Compute Scaling：推理时动态扩展计算

### 核心思想

2024-2025 年，推理模型（o1/o3/DeepSeek-R1 等）揭示了一个深刻的发现：**在推理时投入更多计算，比训练更大的模型更有效。**

传统思路是"训练时花钱"——用更多 GPU、更多数据训练更大的模型。Test-time Compute Scaling 的思路是"推理时花钱"——让同一个模型花更多时间"思考"，就能获得更好的结果。

```
训练时扩展（Pre-training Scaling）：
  更大的模型 + 更多数据 = 更强的能力
  但训练成本指数级增长

推理时扩展（Test-time Scaling）：
  同一个模型 + 更多推理步数 = 更强的结果
  按需投入，只在难题上多花时间
```

### 三种 Test-time Compute 策略

```python
TEST_TIME_COMPUTE_STRATEGIES = {
    "策略一：搜索式推理（Search-based）": {
        "原理": "生成多条推理路径，搜索最优解",
        "代表": "Tree of Thoughts, MCTS, LATS",
        "计算量": "线性增长（路径数 × 每步深度）",
        "适用": "有明确评估标准的决策问题",
    },
    "策略二：自我纠错（Self-correction）": {
        "原理": "生成初稿 → 自我批评 → 修改 → 重复",
        "代表": "Self-Refine, CRITIC, 反思循环",
        "计算量": "线性增长（迭代轮数）",
        "适用": "有客观验证标准的任务（代码、数学）",
    },
    "策略三：思维链延长（Extended CoT）": {
        "原理": "让模型生成更长的推理链，在更多中间步骤中逐步逼近答案",
        "代表": "o1/o3, DeepSeek-R1, Claude Extended Thinking",
        "计算量": "线性增长（thinking tokens 数）",
        "适用": "所有复杂推理任务",
    },
}
```

### 实战：自适应推理深度

不是所有问题都需要深度推理。一个好的 Agent 应该**根据问题难度自动调整推理深度**：

```python
class AdaptiveReasoningAgent:
    """自适应推理深度的 Agent"""
    
    DIFFICULTY_THRESHOLD = {
        "simple": {"max_thinking_tokens": 500, "strategy": "direct"},
        "medium": {"max_thinking_tokens": 2000, "strategy": "cot"},
        "hard": {"max_thinking_tokens": 8000, "strategy": "search"},
    }
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
    
    def run(self, question: str) -> dict:
        """根据问题难度自适应推理"""
        
        # 1. 快速评估问题难度
        difficulty = self._assess_difficulty(question)
        config = self.DIFFICULTY_THRESHOLD[difficulty]
        
        print(f"📊 问题难度: {difficulty} → 策略: {config['strategy']}")
        
        # 2. 根据难度选择推理策略
        if config["strategy"] == "direct":
            answer = self._direct_answer(question)
        elif config["strategy"] == "cot":
            answer = self._cot_answer(question, config["max_thinking_tokens"])
        elif config["strategy"] == "search":
            answer = self._search_answer(question, config["max_thinking_tokens"])
        
        return {
            "question": question,
            "difficulty": difficulty,
            "strategy": config["strategy"],
            "answer": answer,
        }
    
    def _assess_difficulty(self, question: str) -> str:
        """评估问题难度"""
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""评估以下问题的难度等级：

问题：{question}

难度标准：
- simple: 事实性问答、简单计算、常见分类
- medium: 需要多步推理、综合分析、需要专业知识
- hard: 需要创新思维、多路径搜索、复杂数学证明

只回答 simple/medium/hard 之一。"""
            }],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result if result in self.DIFFICULTY_THRESHOLD else "medium"
    
    def _direct_answer(self, question: str) -> str:
        """简单问题：直接回答"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _cot_answer(self, question: str, max_tokens: int) -> str:
        """中等难度：思维链推理"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"{question}\n\n请一步步思考，给出详细的推理过程。"
            }],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _search_answer(self, question: str, max_tokens: int) -> str:
        """高难度：多路径搜索推理"""
        # 生成多条推理路径
        paths = []
        for i in range(3):
            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"""{question}

请用方法 {i+1} 来解决这个问题。给出详细的推理过程。
方法选项：1=代数方法, 2=图解法, 3=穷举法"""
                }],
                max_tokens=max_tokens // 3
            )
            paths.append(response.choices[0].message.content)
        
        # 选择最优路径
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""问题：{question}

以下是三种不同的解法：

解法1：{paths[0]}

解法2：{paths[1]}

解法3：{paths[2]}

请对比三种解法，选择最正确、最完整的一个，并给出最终答案。"""
            }],
            max_tokens=1000
        )
        
        return response.choices[0].message.content


# 使用示例
agent = AdaptiveReasoningAgent()

# 简单问题 → 直接回答
r1 = agent.run("中国的首都是哪里？")

# 中等问题 → CoT 推理
r2 = agent.run("一个班级40人，60%喜欢数学，75%喜欢语文，至少多少人同时喜欢？")

# 复杂问题 → 多路径搜索
r3 = agent.run(
    "证明：对任意正整数 n，1 + 2 + ... + n = n(n+1)/2。"
    "请用至少两种不同的数学方法证明。"
)
```

---

## MCTS 在 Agent 推理中的应用

**蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）** 是 AlphaGo 的核心算法，被 LATS（Language Agent Tree Search）论文引入到 LLM 推理中：

> 📄 **论文出处**：*"Tree Search for Language Model Agents"*（Zhou et al., 2024）。LATS 将 LLM 的推理过程建模为搜索树——每个节点是一个"思考状态"，每条边是一个"推理步骤"。通过 MCTS 在这棵树上搜索，找到从初始状态到目标状态的最优路径。

```python
import math
import random


class MCTSNode:
    """MCTS 搜索树节点"""
    
    def __init__(self, state: str, parent=None):
        self.state = state           # 当前推理状态
        self.parent = parent         # 父节点
        self.children = []           # 子节点
        self.visits = 0              # 访问次数
        self.value = 0.0             # 累计奖励值
        self.action = ""             # 到达此节点的推理步骤
    
    @property
    def ucb1(self) -> float:
        """UCB1 分数：平衡探索与利用"""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploit + explore
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTSReasoningAgent:
    """基于 MCTS 的推理 Agent"""
    
    def __init__(self, max_iterations: int = 20, max_depth: int = 5):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
    
    def search(self, problem: str) -> str:
        """MCTS 搜索最优推理路径"""
        
        root = MCTSNode(state=problem)
        
        for _ in range(self.max_iterations):
            # 1. 选择（Selection）：沿 UCB1 最高的路径向下
            node = self._select(root)
            
            # 2. 扩展（Expansion）：为选中节点生成子节点
            if node.visits > 0 and not self._is_terminal(node):
                self._expand(node, problem)
                if node.children:
                    node = random.choice(node.children)
            
            # 3. 模拟（Simulation）：从该节点随机推演到终点
            reward = self._simulate(node, problem)
            
            # 4. 回传（Backpropagation）：将奖励值回传到所有祖先
            self._backpropagate(node, reward)
        
        # 选择访问次数最多的路径作为最优解
        best_path = self._extract_best_path(root)
        return best_path
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择：沿 UCB1 最高的路径向下"""
        while not node.is_leaf:
            node = max(node.children, key=lambda c: c.ucb1)
        return node
    
    def _expand(self, node: MCTSNode, problem: str):
        """扩展：为节点生成候选推理步骤"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""问题：{problem}

当前推理状态：
{node.state}

请生成 3 个不同的下一步推理方向。每个方向用一句话描述。

以 JSON 格式返回：
{{"directions": ["方向1", "方向2", "方向3"]}}"""
            }],
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        for direction in result.get("directions", []):
            child = MCTSNode(
                state=f"{node.state}\n→ {direction}",
                parent=node
            )
            child.action = direction
            node.children.append(child)
    
    def _simulate(self, node: MCTSNode, problem: str) -> float:
        """模拟：从当前节点快速推演到终点，评估质量"""
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # 用小模型快速模拟
            messages=[{
                "role": "user",
                "content": f"""基于以下推理路径，快速判断是否能解决问题。

问题：{problem}
推理路径：{node.state}

请评估这条推理路径的质量（0.0-1.0）：
- 1.0：路径直接导向正确答案
- 0.5：方向正确但可能需要补充
- 0.0：方向错误或逻辑有误

只返回一个 0.0-1.0 之间的数字。"""
            }],
            max_tokens=10
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """回传：将奖励值更新到所有祖先节点"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """判断是否到达终态"""
        return "答案" in node.state or "结论" in node.state
    
    def _extract_best_path(self, root: MCTSNode) -> str:
        """提取访问次数最多的路径"""
        path = []
        node = root
        
        while node.children:
            best = max(node.children, key=lambda c: c.visits)
            path.append(best.action)
            node = best
        
        return "\n".join(f"步骤{i+1}: {step}" for i, step in enumerate(path))


# 使用示例
mcts_agent = MCTSReasoningAgent(max_iterations=15)
result = mcts_agent.search(
    "一个农夫需要带一只狼、一只羊和一棵白菜过河，"
    "但船只能载农夫和一样东西。狼会吃羊，羊会吃白菜。"
    "如何安全过河？"
)
print(f"最优推理路径：\n{result}")
```

---

## 推理模型时代：o1/o3 如何改变 Agent 开发

### 从提示工程到推理工程

推理模型（o1/o3/DeepSeek-R1/Claude Extended Thinking）的出现，从根本上改变了 Agent 的开发方式：

```python
# 传统 Agent（需要精心设计 CoT Prompt）
traditional_prompt = """
你是一个数学推理专家。请按以下步骤解题：
1. 理解问题
2. 列出已知条件
3. 制定解题策略
4. 逐步推导
5. 验证答案

问题：{question}
"""

# 推理模型 Agent（让模型自己决定思考深度）
reasoning_prompt = """
问题：{question}

请深入思考这个问题，给出详细推理过程和最终答案。
"""

# 推理模型会自动生成数百甚至数千个 thinking tokens
# 无需人类精心设计推理步骤
```

### 推理模型 + Agent 的最佳实践

```python
REASONING_MODEL_BEST_PRACTICES = {
    "1. 简化 System Prompt": {
        "说明": "推理模型不需要详细的 CoT 指令，简洁的指令反而效果更好",
        "好": "请解决这个问题",
        "差": "请先理解问题，然后分析条件，制定策略，逐步推导...",
    },
    "2. 让模型自己决定推理深度": {
        "说明": "不要限制 thinking tokens，让模型根据问题难度自行调节",
        "建议": "设置 max_tokens 为 16000+，只在时间敏感场景限制",
    },
    "3. 提供丰富的上下文": {
        "说明": "推理模型擅长整合信息，给越多上下文越好",
        "建议": "将相关文档、历史记录、约束条件全部放入 prompt",
    },
    "4. 用作 Agent 的'大脑'": {
        "说明": "推理模型做规划和复杂推理，小模型做工具调用和简单任务",
        "架构": "推理模型(规划) → 小模型(执行) → 推理模型(验证)",
    },
}
```

---

## 三种推理模式对比

| 维度 | ReAct | Plan-and-Execute | Test-time Compute |
|------|-------|------------------|-------------------|
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
|------|------|
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
