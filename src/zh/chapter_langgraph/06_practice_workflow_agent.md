# 12.6 实战：工作流自动化 Agent

本节综合运用前面学到的 LangGraph 知识——状态管理、条件路由、循环控制，构建一个完整的工作流自动化 Agent。

## 场景：内容创作工作流

我们选择"内容创作"这个贴近实际的场景。这个工作流很适合用 LangGraph 来实现，因为它天然包含了**线性流程**（分析 → 大纲 → 写作）和**循环流程**（审查 → 修改 → 再审查），完美地展示了图结构的两大优势。

工作流包含以下步骤：

1. **主题分析**：解析创作需求，提取关键要素（受众、风格、结构）
2. **大纲生成**：基于分析结果，自动生成文章大纲
3. **内容撰写**：按大纲逐段撰写正文
4. **质量审查**：自动评分并给出改进建议
5. **迭代修改**：如果质量不达标（< 8分），根据建议修改内容并重新审查

这个工作流展示了 LangGraph 最强大的特性——**循环**：审查不通过时，内容会在"修改 → 审查"之间循环迭代，直到质量达标或达到最大修改次数。

### 状态设计

`ContentState` 的设计体现了一个重要原则：**状态应该包含工作流中每个阶段需要的所有数据，以及控制流程走向的元数据。** 其中 `quality_score` 和 `revision_count` 是控制循环的关键——前者决定是否需要修改，后者防止无限循环。

![内容创作工作流](../svg/chapter_langgraph_06_content_workflow.svg)

```python
# content_workflow.py
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional, List
import json

llm = ChatOpenAI(model="gpt-4.1")

class ContentState(TypedDict):
    topic: str
    target_audience: str
    word_count: int
    outline: Optional[List[str]]
    content: Optional[str]
    review_feedback: Optional[str]
    quality_score: int
    revision_count: int
    final_content: Optional[str]

def analyze_topic(state: ContentState) -> dict:
    """分析主题，提取关键要素"""
    response = llm.invoke([HumanMessage(
        content=f"""分析以下内容创作需求：
主题：{state['topic']}
受众：{state['target_audience']}
字数：{state['word_count']}

返回JSON：{{"key_points": ["要点1"], "tone": "风格", "structure": "结构类型"}}"""
    )])
    
    # 解析 LLM 返回的分析结果，提取要点作为后续大纲的参考
    import re
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            key_points = analysis.get("key_points", ["主题分析完成"])
            return {"outline": key_points}
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return {"outline": [f"主题 '{state['topic']}' 分析完成"]}

def generate_outline(state: ContentState) -> dict:
    """生成文章大纲"""
    response = llm.invoke([HumanMessage(
        content=f"为文章'{state['topic']}'生成5个章节大纲，面向{state['target_audience']}。"
                f"请直接返回章节标题列表，每行一个，不要编号。"
    )])
    # 解析 LLM 返回的大纲内容
    lines = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
    # 过滤掉空行和纯数字行，取前 5 个有效章节
    outline = [line.lstrip('0123456789.-、） ') for line in lines if len(line) > 2][:5]
    if not outline:
        # 兜底方案：如果解析失败，使用默认大纲
        outline = [f"章节{i+1}" for i in range(5)]
    return {"outline": outline}

def write_content(state: ContentState) -> dict:
    """撰写文章内容"""
    outline_text = "\n".join([f"{i+1}. {section}" for i, section in enumerate(state["outline"] or [])])
    
    response = llm.invoke([HumanMessage(
        content=f"""根据以下大纲，为'{state['target_audience']}'撰写约{state['word_count']}字的文章。

大纲：
{outline_text}

主题：{state['topic']}"""
    )])
    return {"content": response.content}

def review_content(state: ContentState) -> dict:
    """质量审查"""
    response = llm.invoke([HumanMessage(
        content=f"""审查以下文章质量（给出1-10分）：

{state.get('content', '')[:500]}...

返回JSON：{{"score": 8, "issues": ["问题1"], "suggestions": ["建议1"]}}"""
    )])
    
    try:
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = result.get("score", 7)
            feedback = "\n".join(result.get("suggestions", []))
        else:
            score, feedback = 7, "内容需要改进"
    except:
        score, feedback = 7, "审查完成"
    
    return {
        "quality_score": score,
        "review_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1
    }

def revise_content(state: ContentState) -> dict:
    """根据反馈修改内容"""
    response = llm.invoke([HumanMessage(
        content=f"""根据以下建议修改文章：

当前内容：
{state.get('content', '')[:1000]}

改进建议：
{state.get('review_feedback', '')}"""
    )])
    return {"content": response.content}

def finalize(state: ContentState) -> dict:
    """最终处理"""
    return {"final_content": state.get("content", "")}

def route_after_review(state: ContentState) -> str:
    """路由：质量分数决定是否修改"""
    score = state.get("quality_score", 0)
    revisions = state.get("revision_count", 0)
    
    if score >= 8 or revisions >= 2:
        return "finalize"
    return "revise"

# 构建图
graph = StateGraph(ContentState)
graph.add_node("analyze", analyze_topic)
graph.add_node("outline", generate_outline)
graph.add_node("write", write_content)
graph.add_node("review", review_content)
graph.add_node("revise", revise_content)
graph.add_node("finalize", finalize)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "outline")
graph.add_edge("outline", "write")
graph.add_edge("write", "review")
graph.add_conditional_edges("review", route_after_review, {
    "finalize": "finalize",
    "revise": "revise"
})
graph.add_edge("revise", "review")
graph.add_edge("finalize", END)

app = graph.compile()

# 运行
result = app.invoke({
    "topic": "Python 在人工智能开发中的应用",
    "target_audience": "Python 初学者",
    "word_count": 800,
    "outline": None,
    "content": None,
    "review_feedback": None,
    "quality_score": 0,
    "revision_count": 0,
    "final_content": None
})

print(f"质量分数：{result['quality_score']}/10")
print(f"修改次数：{result['revision_count']}")
print(f"\n最终内容（前500字）：\n{result['final_content'][:500]}")
```

### 代码解读

上面的代码有几个值得关注的设计要点：

**路由函数 `route_after_review`**：这是循环控制的核心。它同时检查两个条件：质量分数 ≥ 8 分（达标）或修改次数 ≥ 2 次（上限）。任一条件满足就结束循环。这种"双重保障"避免了模型对自己的作品永远不满意导致的无限迭代。

**节点间的数据传递**：每个节点函数只返回需要更新的状态字段，而不是整个状态。LangGraph 会自动将返回值合并到当前状态中。比如 `write_content` 只返回 `{"content": response.content}`，不需要关心 `topic`、`outline` 等其他字段。

**LLM 输出的容错解析**：`review_content` 和 `analyze_topic` 都用正则表达式从 LLM 回复中提取 JSON，并提供了兜底值。这是因为 LLM 有时不会严格按照指定格式返回（可能在 JSON 前后加上说明文字），我们需要容忍这种不确定性。

> 💡 **延伸阅读**：关于 Plan-and-Execute 模式的完整实现和 Test-time Compute Scaling 策略，详见 [5.6 Plan-and-Execute 与 Test-time Compute Scaling](../chapter_planning/07_plan_and_execute.md)。

## 本章小结

LangGraph 的核心价值：

| 特性 | 实现方式 |
|------|---------|
| 状态管理 | TypedDict 定义共享 State |
| 循环控制 | 节点可以指向之前的节点 |
| 条件分支 | `add_conditional_edges` |
| 人机协作 | `interrupt_before/after` + Checkpointer |
| 持久化 | MemorySaver / SqliteSaver |

---

*下一章：[第13章 其他主流框架概览](../chapter_frameworks/README.md)*
