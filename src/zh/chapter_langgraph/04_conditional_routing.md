# 13.4 条件路由与循环控制

LangGraph 的强大之处，在于灵活的条件路由与循环控制——这让它能表达比"调用工具"复杂得多的流程。上一节用内置的 `tools_condition` 判断"是否要工具"；真实 Agent 往往需要更丰富的决策：根据代码审查结果决定通过还是打回、按用户意图分流到不同处理线、按质量评分决定是否迭代优化。

这些都能用**条件路由**实现：你写一个条件函数，检查当前状态，返回一个字符串指明走哪条路，再用 `add_conditional_edges` 把字符串映射到目标节点。

### 循环的力量与风险

条件路由最强大的用法是构造**循环**——让某节点的输出指回更早的节点，比如"审查 → 修复 → 再审查 → 再修复"。但循环也带来无限循环风险：条件逻辑若有 bug，Agent 可能永远出不了环。因此**设置最大迭代次数是必须的安全措施**。

下面用一个"代码审查 Agent"演示：分析代码、发现问题、修复、再审查，直到通过或达到上限。

![条件路由与循环控制——代码审查Agent示例](../svg/chapter_langgraph_04_review_loop.svg)

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4.1-mini")
# 用 Pydantic + with_structured_output 让模型直接返回结构化对象，
# 而非需要正则去抠的 JSON 文本——字段类型在编码期即可校验。
class CodeReview(BaseModel):
    issues: list[str] = Field(description="发现的问题；无则空列表")
    severity: str = Field(description="high/medium/low")
    approved: bool = Field(description="是否可直接通过")

review_llm = llm.with_structured_output(CodeReview)

class CodeReviewState(TypedDict):
    code: str
    issues: list
    iteration: int
    max_iterations: int
    approved: bool

def analyze_code(s: CodeReviewState) -> dict:
    r = review_llm.invoke([HumanMessage(f"审查代码：\n{s['code']}")])
    return {"issues": r.issues, "iteration": s.get("iteration", 0) + 1}

def fix_code(s: CodeReviewState) -> dict:
    # 把问题列表交给模型修复，只回纯代码
    return {"code": llm.invoke([HumanMessage(f"修复：\n{s['code']}\n问题：{s['issues']}")]).content}

def route(s: CodeReviewState) -> str:
    """循环控制核心：安全阀 + 达标判定"""
    if s["iteration"] >= s["max_iterations"]:
        return "approve"          # 上限兜底，强制结束
    if not s["issues"]:
        return "approve"          # 无问题直接通过
    return "fix"                  # 否则继续修复循环

graph = StateGraph(CodeReviewState)
graph.add_node("analyze", analyze_code)
graph.add_node("fix", fix_code)
graph.add_node("approve", lambda s: {"approved": True})
graph.add_edge(START, "analyze")
graph.add_conditional_edges("analyze", route, {"fix": "fix", "approve": "approve"})
graph.add_edge("fix", "analyze")     # 修复后重新分析（循环！）
graph.add_edge("approve", END)
app = graph.compile()
```

> 💡 **直觉理解**：`route` 函数是整个循环的大脑——它用"迭代次数上限"做安全阀，用"是否还有问题"决定继续还是放行。把"策略参数"（max_iterations）放进 State 而非写死，是让图可复用、可测试的常用技巧。

## 三种可复用的路由模式

| 模式 | 思路 | 典型场景 |
|------|------|---------|
| **多路分发（Fan-out）** | 按意图把请求路由到不同专业节点 | 客服按"技术/业务/投诉"分流 |
| **质量门控（Quality Gate）** | 输出前做质量检查，不达标回退重做 | 内容生成质量分 < 阈值则重写 |
| **并行汇聚（Map-Reduce）** | 用 `Send()` 动态创建并行分支，各自处理后汇聚 | 把大任务拆成子任务并行处理 |

这些模式都是"条件函数返回字符串 → 映射到节点"这一机制的变体。复杂的条件函数**先用单元测试覆盖每种输入组合**，再集成进图，能省掉大量调试时间。

## 调试路由

路由行为不符预期时，最有效的办法是**追踪每一步的路由决策**：在条件函数里打日志，或用 `app.stream()` 打印每个节点与迭代次数，确认分支走向。

---

## 小结

条件路由关键技巧：
- **条件函数**返回字符串，标识下一个节点
- **循环控制**必须设最大迭代次数，防无限循环
- **状态跟踪**在 State 中记录迭代次数与完成标志

---

*下一节：[13.5 Human-in-the-Loop：人机协作](./05_human_in_the_loop.md)*
