# 13.6 实战：工作流自动化 Agent

综合运用状态管理、条件路由、循环控制，构建一个"内容创作工作流" Agent。它天然包含**线性流程**（分析 → 大纲 → 写作）和**循环流程**（审查 → 修改 → 再审查），正好展示图结构的两大优势。

工作流步骤：①主题分析 → ②大纲生成 → ③内容撰写 → ④质量审查 → ⑤若质量不达标（< 8 分）则修改并重审，直到达标或达到最大修改次数。

> 💡 这个工作流最有力的地方就是**循环**：审查不通过时，内容在"修改 ↔ 审查"间迭代，直至达标或触顶。

![内容创作工作流](../svg/chapter_langgraph_06_content_workflow.svg)

## 状态设计原则

`ContentState` 体现一条重要原则：**状态应包含工作流每个阶段所需的数据，以及控制流程走向的元数据**。`quality_score` 与 `revision_count` 是控制循环的关键——前者决定是否需要修改，后者防止无限循环。

```python
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4.1")

class ContentReview(BaseModel):
    score: int = Field(description="质量评分，1-10")
    issues: list[str] = Field(description="发现的问题")
    suggestions: list[str] = Field(description="改进建议")

review_llm = llm.with_structured_output(ContentReview)  # 结构化输出，协议层保证格式

QUALITY_PASS_SCORE = 8   # 质量达标线（需用评测校准，非写死就够）
MAX_REVISIONS = 2

class ContentState(TypedDict):
    topic: str
    target_audience: str
    word_count: int
    outline: list
    content: str
    review_feedback: str
    quality_score: int
    revision_count: int

def analyze_topic(s): return {"outline": analyze_llm.invoke(...).key_points}
def generate_outline(s): ...          # 基于分析生成大纲
def write_content(s): ...             # 按大纲撰写正文
def review_content(s):                # 返回 score / review_feedback / revision_count+1
    r = review_llm.invoke([HumanMessage(f"审查：\n{s['content'][:800]}")])
    return {"quality_score": r.score, "review_feedback": "\n".join(r.suggestions),
            "revision_count": s.get("revision_count", 0) + 1}
def revise_content(s): ...            # 按建议修改

def route_after_review(s) -> str:
    """双重保障：达标 或 达修改上限，任一满足即结束循环"""
    if s["quality_score"] >= QUALITY_PASS_SCORE or s["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    return "revise"

graph = StateGraph(ContentState)
graph.add_node("analyze", analyze_topic)
graph.add_node("outline", generate_outline)
graph.add_node("write", write_content)
graph.add_node("review", review_content)
graph.add_node("revise", revise_content)
graph.add_node("finalize", lambda s: {"final_content": s["content"]})
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "outline")
graph.add_edge("outline", "write")
graph.add_edge("write", "review")
graph.add_conditional_edges("review", route_after_review,
                           {"finalize": "finalize", "revise": "revise"})
graph.add_edge("revise", "review")     # 修改后重新审查（循环！）
graph.add_edge("finalize", END)
app = graph.compile()
```

> 📌 完整可运行代码（含各节点实现）见本书配套仓库 `examples/`。本节重在"图如何表达线性+循环混合流程"这一套路。

## 三个值得记牢的设计点

1. **路由函数 `route_after_review` 的双重保障**：同时检查"分数 ≥ 8"与"修改次数 ≥ 2"。任一满足就结束循环——避免模型对自己的作品永远不满意导致无限迭代。
2. **节点间数据传递**：每个节点只返回需更新的状态字段（如 `write_content` 只回 `{"content": ...}`），LangGraph 自动合并进当前状态。节点无需关心 `topic`、`outline` 等无关字段。
3. **结构化输出而非正则解析**：`review_content` 用 `with_structured_output(Pydantic)` 让模型直接返回对象，省掉脆弱的"文本里抠 JSON + 兜底值"代码。但**这不等于模型不会犯错**——质量阈值（如 `QUALITY_PASS_SCORE=8`）仍需用评测校准，详见 [第20章 Agent 评估](../chapter_20_evaluation/README.md)。

> 💡 **延伸阅读**：Plan-and-Execute 模式的完整实现与 Test-time Compute Scaling 策略，详见 [5.6 Plan-and-Execute](../chapter_planning/07_plan_and_execute.md)。

---

## 小结

LangGraph 的核心价值：

| 特性 | 实现方式 |
|------|---------|
| 状态管理 | `TypedDict` 定义共享 State |
| 循环控制 | 节点可指回之前的节点 |
| 条件分支 | `add_conditional_edges` |
| 人机协作 | `interrupt_before/after` + Checkpointer |
| 持久化 | `MemorySaver` / `SqliteSaver` / `PostgresSaver` |

---

*下一章：[第14章 OpenClaw：跨平台个人 AI 助理](../chapter_openclaw/README.md)*
