# 条件路由与循环控制

LangGraph 的强大之处在于灵活的条件路由和循环控制——这让它能够表达比简单的"调用工具"更复杂的工作流。

在上一节中，我们用 `tools_condition` 这个内置的条件函数来判断"是否需要工具"。但现实中的 Agent 往往需要更复杂的决策逻辑：根据代码审查的结果决定是通过还是打回修改、根据用户的意图路由到不同的处理流程、根据质量评分决定是否需要迭代优化……

这些都可以用**条件路由**来实现——你定义一个条件函数，它检查当前状态，返回一个字符串来标识应该走哪条路。然后用 `add_conditional_edges` 将这些字符串映射到不同的目标节点。

### 循环的力量与风险

条件路由最强大的用法是构造**循环**——让某个节点的输出可以回到之前的节点。比如"代码审查 → 修复 → 再审查 → 再修复"这样的迭代流程。但循环也带来了无限循环的风险：如果条件判断逻辑有 bug，Agent 可能永远在循环中出不来。因此，**设置最大迭代次数是必须的安全措施。**

下面我们用一个"代码审查 Agent"来演示条件路由和循环控制。这个 Agent 会分析代码、发现问题、修复代码，然后重新审查——直到代码通过审查或达到最大迭代次数。

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

llm = ChatOpenAI(model="gpt-4o-mini")

# ============================
# 带循环的代码审查 Agent
# ============================

class CodeReviewState(TypedDict):
    code: str
    review_result: Optional[str]
    issues: list
    iteration: int
    max_iterations: int
    approved: bool

def analyze_code(state: CodeReviewState) -> CodeReviewState:
    """分析代码质量"""
    response = llm.invoke([
        HumanMessage(content=f"""审查以下代码，找出所有问题（JSON格式）：
```python
{state['code']}
```
返回：{{"issues": ["问题1", "问题2"], "severity": "high/medium/low"}}""")
    ])
    
    try:
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            issues = result.get("issues", [])
        else:
            issues = []
    except:
        issues = []
    
    return {
        "issues": issues,
        "review_result": response.content,
        "iteration": state.get("iteration", 0) + 1
    }

def fix_code(state: CodeReviewState) -> CodeReviewState:
    """修复代码问题"""
    issues_text = "\n".join([f"- {issue}" for issue in state["issues"]])
    
    response = llm.invoke([
        HumanMessage(content=f"""修复以下代码中的问题：

代码：
```python
{state['code']}
```

问题列表：
{issues_text}

只返回修复后的纯 Python 代码：""")
    ])
    
    fixed_code = response.content
    if "```python" in fixed_code:
        fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
    elif "```" in fixed_code:
        fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
    
    return {"code": fixed_code}

def should_fix_or_approve(state: CodeReviewState) -> Literal["fix", "approve", "max_reached"]:
    """条件路由：决定继续修复还是通过
    
    这个函数是整个循环的核心控制逻辑：
    - 检查迭代次数上限（安全阀，防止无限循环）
    - 如果没有问题则直接通过
    - 只有存在严重问题（如 bug、安全漏洞）才触发修复循环
    """
    
    if state["iteration"] >= state["max_iterations"]:
        return "max_reached"
    
    if not state["issues"]:
        return "approve"
    
    # 只有严重问题才继续修复
    critical_keywords = ["bug", "错误", "安全漏洞", "性能问题", "语法错误"]
    has_critical = any(
        any(kw in issue.lower() for kw in critical_keywords)
        for issue in state["issues"]
    )
    
    return "fix" if has_critical else "approve"

def mark_approved(state: CodeReviewState) -> CodeReviewState:
    """标记代码已通过审查"""
    return {"approved": True}

# 构建图
graph = StateGraph(CodeReviewState)
graph.add_node("analyze", analyze_code)
graph.add_node("fix", fix_code)
graph.add_node("approve", mark_approved)

graph.add_edge(START, "analyze")
graph.add_conditional_edges(
    "analyze",
    should_fix_or_approve,
    {
        "fix": "fix",
        "approve": "approve",
        "max_reached": "approve"
    }
)
graph.add_edge("fix", "analyze")  # 修复后重新分析（循环！）
graph.add_edge("approve", END)

app = graph.compile()

# 测试
initial_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""

result = app.invoke({
    "code": initial_code,
    "review_result": None,
    "issues": [],
    "iteration": 0,
    "max_iterations": 3,
    "approved": False
})

print(f"最终代码：\n{result['code']}")
print(f"通过审查：{result['approved']}")
print(f"迭代次数：{result['iteration']}")
```

---

## 小结

条件路由的关键技巧：
- **条件函数**：返回字符串标识下一个节点
- **循环控制**：必须设置最大迭代次数，防止无限循环
- **状态跟踪**：在 State 中记录迭代次数和完成标志

---

*下一节：[9.5 Human-in-the-Loop：人机协作](./05_human_in_the_loop.md)*
