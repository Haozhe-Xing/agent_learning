# 10.8 LangChain 生产化模式

本节把“能调用”推进到“失败时仍能说明实际状态”。本节提供的是可运行的本地边界练习，不是一个已部署的生产服务；真实HTTP连接、跨进程状态、Redis、部署平台和线上模型均未验收。版本与入口见[10.1](./01_langchain_overview.md)。

## 先确定三份结果

| 结果 | 需要记录什么 |
|---|---|
| 执行 | 实际尝试、错误、停止原因及资源清理 |
| 交付 | 哪些内容已发出、接收端是否确认、是否需要重发 |
| 任务验收 | 冻结要求是否满足，缺少什么 |

流式结束、备用模型返回、缓存命中都不能替代任务验收。本章复用 `reference_code/langchain_patterns/` 的真实Runnable，测试在 `examples/integration_tests/test_langchain_chapter.py`。

## 首块前失败与首块后失败不同

`with_fallbacks`可在匹配异常时尝试备用Runnable，但已经发出的内容没有撤回能力。本地真实反例先输出 `partial`再抛 `TimeoutError`，备用并未执行：

```python
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from reference_code.langchain_patterns import stream_events

fallback_calls = []
def interrupted(_):
    yield "partial"
    raise TimeoutError("fixed failure after first output")

backup = RunnableLambda(lambda _: fallback_calls.append(True) or "backup")
chain = RunnableLambda(interrupted).with_fallbacks(
    [backup], exceptions_to_handle=(TimeoutError,),
)
with tracing_context(enabled=False):
    events = list(stream_events(chain, "fixture"))
assert [item["event"] for item in events] == ["delta", "error"]
assert fallback_calls == []
print(events)
```

结果是 `delta`、`error`，没有 `done`。配套 `stream_events`是进程内事件适配器；错误仅携带异常类型，不把可能含输入的错误文本直接送往客户端。`done`仅代表迭代正常耗尽，不是答案验收通过。

若错误在首块前发生，同样的fallback配置可尝试备用路径；备用还可能失败。若首块后希望重试，应用必须明确新尝试边界、重发/替换协议和幂等策略，不能悄悄拼接两份不同答案。成功片段也不能掩盖当前尝试失败。

本节没有提供SSE服务器。接HTTP服务时，应把这些事件编码为独立帧，处理断开连接、背压和取消；不能把一个JSON事件列表称作已测网络流。同步生成器也可能由服务线程调度，不能笼统宣称框架只要用了 `stream()`所有用户就无法服务。异步路由内应避免直接阻塞事件循环，并验证底层是否真正合作让出执行。

## 重试、备用与总预算一起算

10.4的示例只重试 `TimeoutError`且最多3次。配套反例还证明：`ValueError`只尝试一次，连续临时失败到上限后保留异常。规则应围绕错误语义，而不是把所有 `Exception`都当成可恢复。

| 失败类型 | 可考虑的处理 | 不应做什么 |
|---|---|---|
| 明确临时的读取/传输失败 | 总预算内退避或备用，保留每次尝试 | 无限重试或隐藏消耗 |
| 参数/模式错误 | 返回可定位问题；必要时在受限次数内修正 | 与限流一样原样重试 |
| 权限拒绝 | 停止越权动作并说明 | 换工具或改措辞绕过许可 |
| 有副作用的请求超时 | 先查询结果或使用幂等合同 | 无条件重复写入 |
| 业务验收失败 | 保留未满足项，按任务合同修订 | 仅因模型又返回文本就成功 |

SDK的重试次数、Runnable的总尝试和Agent循环会叠加。把这些层次计入一份任务预算；等待退避也消耗截止时间。10.3的模型/工具调用上限与图步骤兜底是不同计量，不要混成一个 `max_iterations` 参数。

## 异步、并发、超时各自限制什么

异步I/O可以释放等待期间的执行机会；它不会让CPU工作自动并行，也不保证同步函数能被强制取消。下面实际中止的是会响应取消的异步协程：

```python
import asyncio
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

async def slow(value):
    await asyncio.sleep(1)
    return value

async def main():
    try:
        with tracing_context(enabled=False):
            await asyncio.wait_for(RunnableLambda(slow).ainvoke("fixture"), timeout=0.01)
    except TimeoutError:
        return "deadline_exceeded"
    raise AssertionError("expected the fixture to time out")

assert asyncio.run(main()) == "deadline_exceeded"
```

配套测试另启动受控同步线程：取消 `ainvoke`的等待之后，线程仍继续，直到测试显式释放它。这不是子进程停止证明。对真实进程工具，应另设进程级超时、终止与清理；不能把 `wait_for`返回超时当成副作用已停止。

`batch`/`abatch`的 `max_concurrency`限制一批活动任务，测试观察到最多2个合作协程同时运行；它不是每秒请求率、token配额或多进程全局限制。提供者请求率、租户配额、排队和服务容量应分别设计，不为未部署的限流器声称固定吞吐。

## 先验证精确缓存，再讨论更复杂的缓存

这里用每实例 `InMemoryCache`，不更改进程的全局缓存。固定提供者按顺序有两个不同回复；相同输入第二次复用第一份结果，资料版本改变后使用下一份：

```python
from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langsmith import tracing_context
from reference_code.langchain_patterns import build_scoped_chain

model = FakeListChatModel(responses=["first", "second"], cache=InMemoryCache())
chain = build_scoped_chain(model)
request = {"scope": "team-a", "revision": "v1", "question": "fixture"}
with tracing_context(enabled=False):
    values = [chain.invoke(request), chain.invoke(request),
              chain.invoke({**request, "revision": "v2"})]
assert values == ["first", "first", "second"]
print(values)
```

`scope`与 `revision`由宿主确定，不是任意客户端可声明的授权。它们进入模型输入，用于演示不同缓存键；仅放在回调metadata里不保证改变缓存键。不要把凭据放进Prompt来区分权限。生产中可在宿主缓存层使用不敏感的权限/数据版本摘要，并按实际授权控制读取。

这个缓存不持久、不跨进程共享，也没有在本例实现TTL、失效队列或容量治理。缓存结果仍需符合当前任务与权限。即使 `temperature=0`也不能保证答案永远相同或资料永不过期；需要多样化输出的任务则可能不该复用旧答案。

Redis缓存与特定后端的语义缓存由各自集成提供，包名和构造参数需要单独核对。语义相似的问题仍可能要求不同答案；选用前要锁定具体包、后端、阈值和误命中用例。本节只验证进程内精确缓存，其他后端还需读者另行测试。

## 可观测记录不能再造成功

[10.6](./06_langsmith_integration.md)的局部记录刻意不存提示/输出，保留类型化失败。它记录Span，不能据此计算实际请求数、token和成本。真实服务需核对提供者用量、缓存命中、重试和账单，缺字段保留未知；业务结果与Trace分别验收。

对应用准备交付时，按任务补齐必要边界，而不是勾选框架名称：

- 固定用例与独立预期；正常、无答案、越权和预算失败都能定位。
- 已执行副作用、取消与清理记录一致；必要产物缺失就不标完成。
- 缓存键、数据/权限版本、过期与撤销规则一致。
- 记录与外部追踪的数据范围明确，不把输入清洗当作Prompt注入防护的充分条件。
- 若提供服务，另验认证、HTTP断开、健康检查、优雅停止与真实恢复；本章本地测试未覆盖它们。

## 运行和反例练习

```bash
python -m reference_code.langchain_patterns
python -m unittest examples.integration_tests.test_langchain_chapter -v
```

集成测试使用真实Runnable与固定提供者，拦截网络并要求零请求；覆盖结构拒绝、分支失败、选择性重试、流中断、缓存隔离、局部记录以及异步/线程取消差异。没有线上模型质量、服务可用性或性能收益结论。

练习：分别让主路径在首块前、首块后失败，记录备用是否执行和可交付状态；再更改资料版本，确认缓存行为改变。提交事件序列、实际尝试数与独立验收结果。若没有持久状态需求，可以停在这个已验证边界；需要恢复点时再读第11章。

*上一节：[10.7 LangChain生态2026](./07_langchain_ecosystem_2026.md)*

*下一章：[第11章 LangGraph](../chapter_langgraph/README.md)*
