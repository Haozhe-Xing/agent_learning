# 13.6 Practice: Workflow Automation Agent

This section applies all the LangGraph knowledge learned so far — state management, conditional routing, and loop control — to build a complete workflow automation Agent.

## Scenario: Content Creation Workflow

We choose "content creation" as a practical, real-world scenario. This workflow is well-suited for LangGraph because it naturally includes both **linear flow** (analyze → outline → write) and **cyclic flow** (review → revise → re-review), perfectly showcasing the two major advantages of graph structures.

The workflow includes the following steps:

1. **Topic analysis**: parse the creation requirements, extract key elements (audience, style, structure)
2. **Outline generation**: automatically generate an article outline based on the analysis
3. **Content writing**: write the body section by section following the outline
4. **Quality review**: automatically score and provide improvement suggestions
5. **Iterative revision**: if quality is below standard (< 8 points), revise the content based on suggestions and re-review

This workflow demonstrates LangGraph's most powerful feature — **loops**: when a review fails, the content cycles between "revise → review" until quality meets the standard or the maximum revision count is reached.

### State Design

The design of `ContentState` reflects an important principle: **State should contain all the data needed by each phase of the workflow, as well as metadata that controls the flow direction.** Among these, `quality_score` and `revision_count` are the keys to controlling the loop — the former determines whether revision is needed, and the latter prevents infinite loops.

![Content Creation Workflow](../svg/chapter_langgraph_06_content_workflow.svg)

```python
# content_workflow.py
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional, List
import json

llm = ChatOpenAI(model="gpt-4o")

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
    """Analyze the topic and extract key elements"""
    response = llm.invoke([HumanMessage(
        content=f"""Analyze the following content creation requirements:
Topic: {state['topic']}
Audience: {state['target_audience']}
Word count: {state['word_count']}

Return JSON: {{"key_points": ["point1"], "tone": "style", "structure": "structure type"}}"""
    )])
    
    # Parse the LLM's analysis result, extract key points as reference for the outline
    import re
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            key_points = analysis.get("key_points", ["Topic analysis complete"])
            return {"outline": key_points}
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return {"outline": [f"Analysis of topic '{state['topic']}' complete"]}

def generate_outline(state: ContentState) -> dict:
    """Generate article outline"""
    response = llm.invoke([HumanMessage(
        content=f"Generate a 5-section outline for an article on '{state['topic']}' targeting {state['target_audience']}. "
                f"Return only the section titles, one per line, without numbering."
    )])
    # Parse the LLM's outline content
    lines = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
    # Filter out empty lines and pure number lines, take the first 5 valid sections
    outline = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 2][:5]
    if not outline:
        # Fallback: use default outline if parsing fails
        outline = [f"Section {i+1}" for i in range(5)]
    return {"outline": outline}

def write_content(state: ContentState) -> dict:
    """Write article content"""
    outline_text = "\n".join([f"{i+1}. {section}" for i, section in enumerate(state["outline"] or [])])
    
    response = llm.invoke([HumanMessage(
        content=f"""Based on the following outline, write an article of approximately {state['word_count']} words for '{state['target_audience']}'.

Outline:
{outline_text}

Topic: {state['topic']}"""
    )])
    return {"content": response.content}

def review_content(state: ContentState) -> dict:
    """Quality review"""
    response = llm.invoke([HumanMessage(
        content=f"""Review the quality of the following article (score 1–10):

{state.get('content', '')[:500]}...

Return JSON: {{"score": 8, "issues": ["issue1"], "suggestions": ["suggestion1"]}}"""
    )])
    
    try:
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = result.get("score", 7)
            feedback = "\n".join(result.get("suggestions", []))
        else:
            score, feedback = 7, "Content needs improvement"
    except:
        score, feedback = 7, "Review complete"
    
    return {
        "quality_score": score,
        "review_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1
    }

def revise_content(state: ContentState) -> dict:
    """Revise content based on feedback"""
    response = llm.invoke([HumanMessage(
        content=f"""Revise the article based on the following suggestions:

Current content:
{state.get('content', '')[:1000]}

Improvement suggestions:
{state.get('review_feedback', '')}"""
    )])
    return {"content": response.content}

def finalize(state: ContentState) -> dict:
    """Final processing"""
    return {"final_content": state.get("content", "")}

def route_after_review(state: ContentState) -> str:
    """Routing: quality score determines whether to revise"""
    score = state.get("quality_score", 0)
    revisions = state.get("revision_count", 0)
    
    if score >= 8 or revisions >= 2:
        return "finalize"
    return "revise"

# Build the graph
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

# Run
result = app.invoke({
    "topic": "Applications of Python in Artificial Intelligence Development",
    "target_audience": "Python beginners",
    "word_count": 800,
    "outline": None,
    "content": None,
    "review_feedback": None,
    "quality_score": 0,
    "revision_count": 0,
    "final_content": None
})

print(f"Quality score: {result['quality_score']}/10")
print(f"Revisions: {result['revision_count']}")
print(f"\nFinal content (first 500 chars):\n{result['final_content'][:500]}")
```

### Code Walkthrough

The code above has a few design points worth noting:

**Routing function `route_after_review`**: This is the core of loop control. It checks two conditions simultaneously: quality score ≥ 8 (acceptable) or revision count ≥ 2 (limit reached). Either condition being met ends the loop. This "dual safeguard" prevents infinite iteration caused by the model never being satisfied with its own work.

**Data passing between nodes**: Each node function only returns the state fields that need to be updated, not the entire state. LangGraph automatically merges the return value into the current state. For example, `write_content` only returns `{"content": response.content}` and doesn't need to worry about other fields like `topic` or `outline`.

**Fault-tolerant parsing of LLM output**: Both `review_content` and `analyze_topic` use regular expressions to extract JSON from LLM responses, with fallback values provided. This is because LLMs sometimes don't strictly follow the specified format (they may add explanatory text before or after the JSON), and we need to tolerate this uncertainty.

## Summary

The core value of LangGraph:

| Feature | Implementation |
|---------|---------------|
| State management | TypedDict defines shared State |
| Loop control | Nodes can point to previous nodes |
| Conditional branching | `add_conditional_edges` |
| Human-AI collaboration | `interrupt_before/after` + Checkpointer |
| Persistence | MemorySaver / SqliteSaver |

---

*Next chapter: [Chapter 14: OpenClaw](../chapter_openclaw/README.md)*