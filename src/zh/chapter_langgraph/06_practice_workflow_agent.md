# 11.6 实战：工作流自动化 Agent

## 本次迭代与验收

将规划与反思变成有状态、可停止的流程。

- **成功标准**：固定审查序列：首次通过、修改一次后通过、始终不通过；检查修改次数和最终状态，不只检查是否生成正文。
- **失败用例**：始终低分时必须在两次修改后停止，并标记未达标；节点异常不得被当成质量通过。
- **安全边界**：审查分数只是代理指标，人工核对事实与引用；达到修改上限不能触发自动发布。

先保留输入、预期和实际结果，再继续扩展。评估方法见[第18章](../chapter_20_evaluation/README.md)，信任与权限设计见[第19章](../chapter_21_security/README.md)。

综合运用状态管理、条件路由、循环控制，构建一个"内容创作工作流" Agent。它天然包含**线性流程**（分析 → 大纲 → 写作）和**循环流程**（审查 → 修改 → 再审查），正好展示图结构的两大优势。

工作流步骤：①主题分析 → ②大纲生成 → ③内容撰写 → ④质量审查 → ⑤若质量不达标（< 8 分）则修改并重审，直到达标或达到最大修改次数。

> 💡 这个工作流最有力的地方就是**循环**：审查不通过时，内容在"修改 ↔ 审查"间迭代，直至达标或触顶。

![内容创作工作流](../svg/chapter_langgraph_06_content_workflow.svg)

## 状态设计原则

`ContentState` 体现一条重要原则：**状态应包含工作流每个阶段所需的数据，以及控制流程走向的元数据**。`quality_score` 与 `revision_count` 是控制循环的关键——前者决定是否需要修改，后者防止无限循环。

```python
from examples.chapter13.content_workflow import build_workflow

scores = iter([4, 9])
app = build_workflow(
    review=lambda content: next(scores),
    revise=lambda content: content + " [revised]",
    max_revisions=2,
)
result = app.invoke({"topic": "Tool use"}, config={"recursion_limit": 12})
assert result["revision_count"] == 1
assert result["accepted"] is True
```

完整状态图在 `examples/chapter13/content_workflow.py` 中，包含初稿、审查、修改与结束四个节点。默认注入固定评分序列，便于复现三条路径：首次通过、修改后通过、预算耗尽仍未通过。运行 `python examples/chapter13/content_workflow.py --always-low` 可验证最后一条路径。

初次审查不计入修改次数；只有实际完成修改才增加计数。最后同时返回 `final_content` 和 `accepted`，因此有正文不等于通过验收。评分必须是 1–10 的整数，节点异常会直接暴露。真实模型模式加 `--real`，但模型评分仍需用人工样本校准。

节点、边和编译方式依据 [LangGraph 官方 Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)。

## 控制流程为什么这样写

`review` 只更新分数，`revise` 才更新正文和修改次数。路由检查分数与剩余修改预算，`finish` 明确记录验收结果。三类状态分别承担质量判断、资源控制和交付说明，不能用“图已经结束”替代“内容已经合格”。

继续接入检查点与持久化时，应为任务生成独立会话标识，并测试恢复后不会重复发布或重复写入。这个示例不包含发布工具。

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

*下一章：[第12章 OpenClaw：跨平台个人 AI 助理](../chapter_openclaw/README.md)*
