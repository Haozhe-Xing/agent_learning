# 10.2 Chain：构建处理管道

上一节跑通了单次消息接口，本节增加输入规范化、一次分类、分支和结果汇合。任务是一个**离线客服入口练习**：只支持英文退货/退款关键词，其余请求要求澄清。固定提供者不做真实意图识别或检索；中文自然语言、否定句和同词异义都不在这条规则的质量承诺内。

使用[10.1的版本环境](./01_langchain_overview.md)，完整实现见 `reference_code/langchain_patterns/workflows.py`。

## 数据流先于链名

![LCEL的顺序、保留输入、并行与条件模式](../svg/chapter_langchain_02_chain_patterns.svg)

| 组合 | 输入 → 输出 | 本练习的责任 |
|---|---|---|
| `A | B` | A的值交给B | 清理空白后再分类 |
| `RunnablePassthrough.assign` | 字典 → 保留原字段并加字段 | 只分类一次，保存 `intent` |
| `RunnableBranch` | 同一输入 → 首个条件命中的分支或默认分支 | 已支持任务与澄清分开 |
| `RunnableParallel` | 同一输入 → 分支结果字典 | 保留原问题、意图和答案 |

这些组合是真实控制流，不应因为使用Chain就称它只能线性执行。是否改用图，取决于状态生命周期、恢复与人工决策需求，而不是出现一个 `if`。[官方Runnable入口](https://reference.langchain.com/python/langchain-core/runnables)列出了相关原语。

## 一条完整、可运行的入口流

`normalize_request`只接受非空 `question`，拒绝额外字段；`classify_fixture`是明确的英文子串规则。二者在配套文件里只有少量普通Python逻辑，先读它们再修改分类方式。

```python
from langchain_core.runnables import (
    RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch,
)
from langsmith import tracing_context
from reference_code.langchain_patterns import FIXTURE_ANSWER, build_answer_chain
from reference_code.langchain_patterns.workflows import normalize_request, classify_fixture

policy = RunnableLambda(lambda x: {
    "fact": FIXTURE_ANSWER, "question": x["question"],
}) | build_answer_chain()
route = RunnableBranch(
    (lambda x: x["intent"] == "policy", policy),
    RunnableLambda(lambda _: "Unsupported fixture question; clarify the task."),
)
flow = (
    RunnableLambda(normalize_request)
    | RunnablePassthrough.assign(intent=RunnableLambda(classify_fixture))
    | RunnableParallel(
        question=RunnableLambda(lambda x: x["question"]),
        intent=RunnableLambda(lambda x: x["intent"]), answer=route,
    )
)
with tracing_context(enabled=False):
    result = flow.invoke({"question": "  Refund policy?  "})
assert result == {"question": "Refund policy?", "intent": "policy", "answer": FIXTURE_ANSWER}
print(result)
```

输出字典保留清理后的原问题。条件函数不重复请求模型；默认分支直接说明练习不支持。若未来把分类换成模型，应把它当一次有预算且可能失败的调用，并用独立样例检查误分类；不能说保存一个 `_intent` 字段就实现了跨请求缓存。

`RunnableParallel`的各分支可以并发处理独立工作，但不能让本应依赖统计结果的报告先执行。这里没有测出时延收益，也没有真实并发模型请求。并行中的一个分支抛异常时，默认 `invoke` 不会返回一份完整成功字典；已发生的外部副作用也不会自动回滚。

## 批量调用保留每条输入的失败

```python
from langsmith import tracing_context
from reference_code.langchain_patterns import build_intake_flow

with tracing_context(enabled=False):
    results = build_intake_flow().batch(
        [{"question": "refund policy?"}, {"question": ""}],
        config={"max_concurrency": 2}, return_exceptions=True,
    )
assert results[0]["intent"] == "policy"
assert isinstance(results[1], ValueError)
print([type(item).__name__ for item in results])
```

结果中第一项是字典，第二项是 `ValueError`。`return_exceptions=True`让调用方逐条处理；它不会把错误变成可交付答案。`max_concurrency`限制这次批量工作，不是服务全局配额，也不是每秒请求率。

异步接口在脚本中可以这样调用：

```python
import asyncio
from langsmith import tracing_context
from reference_code.langchain_patterns import build_intake_flow

async def main():
    with tracing_context(enabled=False):
        return await build_intake_flow().ainvoke({"question": "refund"})

assert asyncio.run(main())["intent"] == "policy"
```

在已有事件循环的Notebook或服务里直接 `await main()`，不要再嵌套 `asyncio.run`。`ainvoke`这个名字不保证底层是原生异步I/O：某些Runnable的默认实现会把同步工作交给线程。

## 何时结束这条流程

固定提供者返回文本，是执行结果；业务验收仍需问题范围、事实出处和必要要点。本练习的 `unsupported` 是应澄清的路由，不能和“生成失败”或“已经回答”混为一谈。真实订单事实和写权限见[10.5](./05_practice_customer_service.md)。

练习：给流程输入空白问题、未知问题、额外的 `permissions` 字段。记录分别在哪一层结束，再让一个并行分支抛 `PermissionError`，确认没有把其余字段包装成完整成功。不要通过删除校验来让样例通过。

*上一节：[10.1 架构全景](./01_langchain_overview.md)*

*下一节：[10.3 使用LangChain构建Agent](./03_langchain_agents.md)*
