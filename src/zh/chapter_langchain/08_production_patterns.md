# 12.8 LangChain 生产化模式

> **本节目标**：掌握 LangChain 应用从开发走向生产所需的关键工程能力——流式输出、异步执行、错误处理、缓存策略和并发控制。

---

## 从 Demo 到 Production：差距在哪里？

许多 LangChain 应用在 demo 阶段运行良好，但上线后问题频发。以下是典型的生产化挑战：

| 挑战 | Demo 阶段 | 生产环境 |
|------|----------|---------|
| **延迟** | 等几秒无所谓 | 用户期望 200ms 内看到响应 |
| **可靠性** | 偶尔报错重跑即可 | 需要 99.9% 可用性 |
| **成本** | 几次调用无所谓 | 千级 QPS 下 Token 成本指数增长 |
| **并发** | 单线程顺序执行 | 需要处理并发请求 |
| **缓存** | 不需要 | 重复查询浪费 Token 和时间 |

本节逐一解决这些问题。

---

## 流式输出（Streaming）

流式输出是提升用户体验最有效的手段——用户不需要等 Agent 跑完所有步骤，而是实时看到每一步的输出。

### 基础流式输出

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术顾问。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 同步流式
for chunk in chain.stream({"question": "解释一下什么是向量数据库"}):
    print(chunk, end="", flush=True)
print()  # 换行
```

### 异步流式输出

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术顾问。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

async def stream_response(question: str):
    """异步流式输出"""
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response("解释一下什么是向量数据库"))
```

### 完整的流式 Agent 实现

对于 Agent 应用，流式输出更复杂——你需要同时处理 LLM 的文本输出和工具调用。以下是完整的流式 Agent 实现：

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
import sys

# ============================
# 工具定义
# ============================

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    return f"知识库搜索结果：关于「{query}」的相关信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# ============================
# 构建流式 Agent
# ============================

tools = [search_knowledge, calculate]

# 流式 LLM：设置 streaming=True + 自定义 handler
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],  # 实时输出到 stdout
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手。请使用工具来回答问题。
回答时先说明思路，再给出结论。"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示中间步骤
    max_iterations=5,
)

# ============================
# 流式调用
# ============================

def run_streaming_agent(user_input: str):
    """运行流式 Agent"""
    print(f"\n🤔 用户: {user_input}")
    print("📝 助手: ", end="", flush=True)

    for chunk in agent_executor.stream({
        "input": user_input,
        "chat_history": [],
    }):
        # AgentExecutor.stream() 返回每一步的输出
        if "actions" in chunk:
            # Agent 决定调用工具
            for action in chunk["actions"]:
                print(f"\n🔧 调用工具: {action.tool}({action.tool_input})")
        elif "steps" in chunk:
            # 工具执行结果
            for step in chunk["steps"]:
                print(f"📊 工具结果: {step.observation[:100]}...")
        elif "output" in chunk:
            # 最终输出
            print(f"\n✅ 最终回答: {chunk['output']}")

run_streaming_agent("搜索一下 RAG 技术，并计算 42 * 17")
```

### FastAPI + SSE 流式 API

在生产环境中，通常通过 Server-Sent Events（SSE）将流式输出推送给前端：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

app = FastAPI()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

@app.post("/chat/stream")
async def chat_stream(question: str):
    """SSE 流式响应"""
    async def event_generator():
        async for chunk in chain.astream({"question": question}):
            # SSE 格式
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 反缓冲
        }
    )
```

> 💡 **流式输出的用户体验**：研究表明，用户对"立即开始输出"的体验评分远高于"等 3 秒后一次性输出"。即使总时间相同，流式输出的感知延迟也更低。

---

## 异步执行（Async）模式

异步是处理并发的关键。LangChain 的所有 Runnable 都支持 `ainvoke`、`astream`、`abatch` 等异步方法。

### 基础异步调用

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 异步单次调用
async def single_call():
    result = await chain.ainvoke({
        "role": "Python 专家",
        "question": "什么是 asyncio？"
    })
    print(result)

# 异步批量调用
async def batch_call():
    inputs = [
        {"role": "Python 专家", "question": "什么是装饰器？"},
        {"role": "数据分析师", "question": "什么是 pandas？"},
        {"role": "前端开发", "question": "什么是 React？"},
    ]
    results = await chain.abatch(inputs)
    for r in results:
        print(r[:50], "...")

asyncio.run(single_call())
asyncio.run(batch_call())
```

### 异步工具调用

```python
import asyncio
import aiohttp
from langchain_core.tools import tool

@tool
async def async_search(query: str) -> str:
    """异步搜索（调用远程 API）"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.example.com/search?q={query}"
        ) as resp:
            data = await resp.json()
            return data.get("results", "未找到结果")

@tool
async def async_fetch_url(url: str) -> str:
    """异步获取网页内容"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            text = await resp.text()
            return text[:500]  # 截取前 500 字符

# 在 Agent 中使用异步工具
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

tools = [async_search, async_fetch_url]
llm = ChatOpenAI(model="gpt-4.1").bind_tools(tools)

async def agent_node(state: MessagesState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()

# 异步运行
async def main():
    result = await app.ainvoke({
        "messages": [{"role": "user", "content": "搜索 LangChain 最新版本"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

> ⚠️ **异步注意事项**：
> - 异步工具必须定义 `async def _arun()` 方法（对于 `BaseTool`）或直接使用 `async def`（对于 `@tool`）
> - 在 FastAPI 等异步框架中，务必使用 `ainvoke` 而非 `invoke`，否则会阻塞事件循环
> - `abatch` 会并发执行，注意 API 的速率限制

---

## 错误处理与重试策略

LLM 应用中的错误来源比传统软件更多：网络超时、API 限流、模型幻觉、工具执行失败……

### 常见错误类型

| 错误类型 | 原因 | 处理策略 |
|---------|------|---------|
| **RateLimitError** | API 调用频率超限 | 指数退避重试 |
| **TimeoutError** | LLM 响应超时 | 重试 + 降级模型 |
| **AuthenticationError** | API Key 无效 | 配置检查 + 告警 |
| **ToolExecutionError** | 工具执行失败 | 错误回传给 Agent |
| **OutputParsingError** | 模型输出格式异常 | 重试 + 容错解析 |
| **ContextLengthExceeded** | 输入超过 Token 限制 | 截断 + 摘要 |

### 重试策略实现

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithFallbacks
import time

# 方案1：使用 with_fallbacks 链式降级
primary_llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=3)
fallback_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3)

chain = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业助手。"),
    ("human", "{question}")
]) | primary_llm | StrOutputParser()

# 设置降级链：主模型失败时自动切换到备用模型
robust_chain = chain.with_fallbacks(
    fallbacks=[
        ChatPromptTemplate.from_messages([
            ("system", "你是一个专业助手。"),
            ("human", "{question}")
        ]) | fallback_llm | StrOutputParser()
    ],
    exceptions_to_handle=(Exception,),  # 捕获所有异常
)

result = robust_chain.invoke({"question": "什么是 RAG？"})
print(result)
```

### 自定义重试逻辑

LangChain 的 `ChatOpenAI` 已内置 `max_retries`，多数场景直接用即可；`with_fallbacks` 还能在重试耗尽后切换到备用模型（见上节）。只有当你需要**非 LLM 调用**（如自写工具里的 HTTP 请求）的细粒度退避时，才需要自己写装饰器。一个最小可用的同步版本：

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """指数退避重试装饰器：第 n 次失败后等待 min(base_delay * 2**n, max_delay) 秒"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    time.sleep(min(base_delay * (2 ** attempt), max_delay))
        return wrapper
    return decorator

# 在工具中使用
from langchain_core.tools import tool

@tool
@retry_with_backoff(max_retries=2, base_delay=1.0)
def search_api(query: str) -> str:
    """调用搜索 API（带重试）"""
    import requests
    resp = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", "")
```

> 异步版本只需把 `time.sleep` 换成 `await asyncio.sleep`、函数声明为 `async def` 即可。重点是**退避而非立即重试**——连续打满请求会触发更严格的限流。

### Agent 级错误处理

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage

@tool
def risky_operation(param: str) -> str:
    """可能失败的操作"""
    import random
    if random.random() < 0.3:  # 30% 概率失败
        raise ValueError("操作失败：模拟的错误")
    return f"操作成功：{param}"

def handle_tool_error(state: MessagesState) -> dict:
    """处理工具执行错误：将错误信息回传给 Agent"""
    error = state.get("error")
    if error:
        return {
            "messages": [
                AIMessage(content=f"工具执行出错：{error}。请尝试其他方法。")
            ]
        }
    return state

# 构建带错误处理的图
tools = [risky_operation]
llm = ChatOpenAI(model="gpt-4.1").bind_tools(tools)

def agent_node(state: MessagesState):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"系统错误：{e}")]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools, handle_tool_error=True))  # 自动处理工具错误
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()
```

---

## 缓存策略

缓存是降低成本和延迟的有效手段。LangChain 提供了多种缓存实现：

### InMemoryCache

```python
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache

# 设置全局缓存
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# 注意：从 langchain-core 0.3 开始，缓存通过 set_llm_cache 设置
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# 第一次调用：实际请求 LLM
result1 = llm.invoke("什么是 Python？")
print("第一次调用完成")

# 第二次相同调用：命中缓存，不调用 LLM
result2 = llm.invoke("什么是 Python？")
print("第二次调用完成（缓存命中）")
```

### RedisCache

```python
# pip install langchain-redis

from langchain_core.globals import set_llm_cache
from langchain_redis import RedisCache

# 连接 Redis
redis_cache = RedisCache(
    redis_url="redis://localhost:6379",
    ttl=3600,  # 缓存 1 小时
)

set_llm_cache(redis_cache)

# 使用方式与 InMemoryCache 相同
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 相同的输入会命中 Redis 缓存
result = llm.invoke("什么是 Python？")
```

### SemanticCache（语义缓存）

语义缓存是 LLM 应用特有的——即使用户的提问措辞不同，只要语义相近，也能命中缓存：

```python
# pip install langchain-community

from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings

# SemanticCache 使用向量相似度判断是否命中缓存
from langchain_community.cache import SemanticCache

semantic_cache = SemanticCache(
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95,  # 相似度阈值，越高越严格
)

set_llm_cache(semantic_cache)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 第一次调用
result1 = llm.invoke("Python 是什么编程语言？")

# 语义相近的调用也会命中缓存
result2 = llm.invoke("Python 编程语言是什么？")  # 措辞不同，但语义相同
```

### 缓存策略对比

| 缓存类型 | 命中条件 | 适用场景 | 注意事项 |
|---------|---------|---------|---------|
| **InMemoryCache** | 输入完全相同 | 开发/测试 | 重启后丢失，不适合生产 |
| **RedisCache** | 输入完全相同 | 生产环境 | 需要部署 Redis |
| **SemanticCache** | 语义相似 | 高重复查询场景 | 额外 Embedding 成本，有误判风险 |

> ⚠️ **缓存注意事项**：
> - `temperature > 0` 时慎用缓存——同样的输入可能期望不同的输出
> - SemanticCache 的 `score_threshold` 需要根据实际数据调优
> - 生产环境建议用 RedisCache，语义缓存作为优化补充

---

## 并发控制与速率限制

### 速率限制器

LangChain 内置了速率限制器，防止超出 API 提供商的调用频率限制：

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# 创建速率限制器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,   # 每秒最多 2 次请求
    check_every_n_seconds=0.1,  # 检查频率
    max_bucket_size=10,      # 令牌桶大小
)

# 应用到 LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    rate_limiter=rate_limiter,  # 自动限流
)

# 批量调用时自动限流
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
chain = prompt | llm | StrOutputParser()

# 20 个请求会自动限流
questions = [{"question": f"问题 {i}"} for i in range(20)]
results = chain.batch(questions)
print(f"完成 {len(results)} 个请求")
```

### 异步并发控制

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3)
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
chain = prompt | llm | StrOutputParser()

async def process_with_semaphore(
    inputs: list[dict],
    max_concurrency: int = 5,
):
    """使用信号量控制并发数"""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_call(input_data: dict):
        async with semaphore:
            return await chain.ainvoke(input_data)

    tasks = [bounded_call(inp) for inp in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果和异常
    success = []
    failures = []
    for inp, result in zip(inputs, results):
        if isinstance(result, Exception):
            failures.append((inp, str(result)))
        else:
            success.append((inp, result))

    print(f"成功: {len(success)}, 失败: {len(failures)}")
    return success, failures

# 运行
inputs = [{"question": f"解释概念 {i}"} for i in range(20)]
asyncio.run(process_with_semaphore(inputs, max_concurrency=5))
```

---

## 生产环境 Checklist

将 LangChain 应用上线前，逐一检查以下清单：

### 可靠性

- [ ] 设置 `max_retries=3` 或自定义重试逻辑
- [ ] 配置 `with_fallbacks` 降级链（主模型 → 备用模型）
- [ ] 工具执行有超时控制（`timeout` 参数）
- [ ] Agent 有最大迭代次数限制（`max_iterations`）
- [ ] 关键路径有 try-except 错误捕获

### 性能

- [ ] 启用流式输出（`streaming=True`）
- [ ] 使用异步调用（`ainvoke` / `astream`）
- [ ] 配置缓存（RedisCache / SemanticCache）
- [ ] 批量请求使用 `abatch` 而非循环 `ainvoke`
- [ ] 设置速率限制器（`InMemoryRateLimiter`）

### 可观测性

- [ ] 启用 LangSmith / LangFuse 追踪
- [ ] 按环境隔离 Project（dev / staging / prod）
- [ ] 记录每次请求的 Token 消耗和成本
- [ ] 配置成本预算告警
- [ ] 关键业务指标接入监控（延迟 P99、错误率、缓存命中率）

### 安全

- [ ] API Key 通过环境变量注入，不硬编码
- [ ] 工具执行有沙箱隔离（参考第 18 章）
- [ ] 用户输入经过清洗，防止 Prompt 注入
- [ ] 敏感信息不出现在日志和追踪中

### 部署

- [ ] 使用 LangServe / LangGraph Platform 部署
- [ ] 健康检查端点（`/health`）
- [ ] 优雅关闭（处理进行中的请求）
- [ ] 水平扩缩容配置
- [ ] 运行评估数据集确认无回归

---

## 小结

| 生产化能力 | 关键要点 |
|-----------|---------|
| **流式输出** | `stream()` / `astream()` + SSE，大幅降低感知延迟 |
| **异步执行** | `ainvoke` / `abatch` + 信号量控制并发 |
| **错误处理** | `with_fallbacks` 降级 + 指数退避重试 |
| **缓存策略** | InMemoryCache → RedisCache → SemanticCache 逐级升级 |
| **速率限制** | `InMemoryRateLimiter` + 信号量双重保护 |
| **Checklist** | 可靠性、性能、可观测性、安全、部署五维检查 |

> 💡 **与本书其他章节的关系**：
> - [第20章 Agent 的评估与优化](../chapter_20_evaluation/README.md) 讨论了性能优化和成本控制的更多细节
> - [第21章 安全与可靠性](../chapter_21_security/README.md) 深入讲解了 Prompt 注入防御和沙箱隔离
> - [第22章 部署与生产化](../chapter_22_deployment/README.md) 涵盖了容器化、K8s、Serverless 等部署方案

---

## 📝 本章练习

读完本章，先合上书用自己的话回答下面的问题，再展开参考答案对照。

**练习 1（概念）**：本章反复出现一个词——Runnable 协议。请解释：什么是 Runnable 协议？为什么说它是 LCEL（用 `|` 管道符连接组件）能够成立的基础？再举例说明："一条链写好之后自动获得了流式、异步、批处理能力"这句话背后的原理是什么。

<details>
<summary>参考答案</summary>

**Runnable 协议是什么？**
Runnable 是 LangChain 0.2+ 引入的最核心抽象——它是一套**统一的接口约定**。LangChain 里几乎所有组件（Prompt 模板、LLM、输出解析器、工具、检索器）都实现了这个接口，因此它们都拥有完全一致的调用方法：`invoke`（同步单次）、`ainvoke`（异步）、`stream`（流式）、`batch`（批量）等。

**为什么它是 LCEL 的基础？**
LCEL 用 `|` 管道符把组件串起来，例如 `prompt | llm | parser`。这个 `|` 背后调用的其实是 Python 的 `__or__` 方法，它把两个 Runnable 组合成一个新的、同样满足 Runnable 接口的 `RunnableSequence`。正因为"每个组件都是 Runnable，组合的结果还是 Runnable"，所以才能像搭积木一样无限串联——这正是接口统一带来的可组合性。如果各组件的调用方式五花八门，`|` 就无法通用地工作。

**为什么自动获得流式/异步/批处理？**
因为这三种能力都是 Runnable 接口本身定义的方法（`stream`/`astream`、`ainvoke`、`batch`/`abatch`）。当你用 `|` 把组件组合成一条链时，这条链作为一个新的 Runnable，会把这些调用"沿管道传递"给每个子组件。例如对整条链调用 `.astream()`，LangChain 会依次让 prompt、llm、parser 都以异步流式方式工作，最终把 LLM 的 token 一块块吐出来。你不需要为每条链单独再写流式或异步逻辑——这是"面向接口编程"省下的工作量。

</details>

**练习 2（辨析）**：本章介绍了三种缓存：InMemoryCache、RedisCache、SemanticCache。有同学说："既然 SemanticCache（语义缓存）连措辞不同的问题都能命中，那它一定是最好的，应该到处都用它。" 这个说法对吗？请对比三种缓存的命中条件和代价，并说明为什么本章特别提醒"`temperature > 0` 时要慎用缓存"。

<details>
<summary>参考答案</summary>

**这个说法不全对——SemanticCache 强大但有代价和风险，不该无脑全用。**

三种缓存对比：

| 缓存 | 命中条件 | 代价 / 风险 | 适用场景 |
|------|---------|------------|---------|
| InMemoryCache | 输入**完全相同** | 重启后丢失，多进程不共享 | 本地开发 / 测试 |
| RedisCache | 输入**完全相同** | 需要部署、维护 Redis | 生产环境的主力缓存 |
| SemanticCache | 语义**相似**（向量相似度超过阈值） | 每次查询都要先算一次 Embedding（额外成本和延迟）；有**误判风险** | 高重复、措辞多变的查询 |

**SemanticCache 的两个隐患：**
1. **额外成本**：判断是否命中需要先把问题转成向量（调用 Embedding 模型），这本身要花钱和时间。如果查询重复率不高，省下的钱还不够付 Embedding 的。
2. **误判风险**：相似度阈值（`score_threshold`）定得太低，会把"看似相似实则不同"的问题误判为命中，返回错误答案。例如"Python 怎么读文件"和"Python 怎么写文件"语义很近，却可能要完全不同的答案。所以本章建议它作为优化补充，生产主力仍用 RedisCache。

**为什么 `temperature > 0` 时慎用缓存？**
缓存的前提是"同样的输入应该得到同样的输出"。但 `temperature > 0` 意味着我们**故意让模型带随机性**——比如创意写作、多样化推荐，用户每次提问可能就是想要不一样的回答。如果开了缓存，第二次相同提问会直接返回第一次的旧结果，随机性和多样性就被破坏了。因此缓存最适合 `temperature=0`（确定性输出）的场景，如固定问答、信息抽取。

</details>

**练习 3（动手）**：你要把一个 LangChain 问答链部署成生产级的 FastAPI 服务，要求满足两点：(1) 用 SSE（Server-Sent Events）流式把回答推给前端，降低用户的感知延迟；(2) 主模型用 `gpt-4.1`，当它失败时自动降级到 `gpt-4.1-mini`，保证可用性。请写出核心代码，并解释为什么在 FastAPI 里必须用 `astream` 而不是 `stream`。

<details>
<summary>参考答案</summary>

核心是把两个本章学到的能力组合起来：`with_fallbacks` 做降级 + `astream` + `StreamingResponse` 做 SSE 流式。

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术顾问。"),
    ("human", "{question}"),
])

# ── 主链：gpt-4.1 ──────────────────────────────────────
primary_chain = (
    prompt
    | ChatOpenAI(model="gpt-4.1", temperature=0, streaming=True, max_retries=3)
    | StrOutputParser()
)

# ── 备用链：gpt-4.1-mini ───────────────────────────────
fallback_chain = (
    prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0, streaming=True, max_retries=3)
    | StrOutputParser()
)

# ── 组合：主链失败自动切到备用链 ───────────────────────
robust_chain = primary_chain.with_fallbacks([fallback_chain])

@app.post("/chat/stream")
async def chat_stream(question: str):
    async def event_generator():
        # 用 astream 异步流式逐块产出
        async for chunk in robust_chain.astream({"question": question}):
            # 封装为 SSE 数据帧
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，否则流式会被攒成一坨
        },
    )
```

**为什么 FastAPI 里必须用 `astream` 而不是 `stream`？**
FastAPI 是基于**异步事件循环**（asyncio）的框架——它靠一个单线程的事件循环来同时服务大量并发请求。`stream()` 是**同步**方法，调用它时会**阻塞**当前线程，直到 LLM 返回数据。一旦阻塞，整个事件循环就被卡住，这段时间内**所有其他用户的请求都无法被处理**，并发能力直接归零。

而 `astream()` 是**异步**方法：当它在等待 LLM 返回下一个 token 时，会用 `await` 把控制权交还给事件循环，让循环去处理其他请求；数据到了再回来继续。这样单个请求的等待时间被"重叠"利用，服务才能扛住高并发。所以本章在异步注意事项里特别强调：在 FastAPI 这类异步框架中，务必使用 `ainvoke` / `astream`，否则会阻塞事件循环。

</details>

---

*上一节：[12.7 LangChain 生态 2026](./07_langchain_ecosystem_2026.md)*

*下一章：[第13章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md)*

---

## 参考文献

[1] LangChain Team. Streaming with LangChain. https://python.langchain.com/docs/how_to/streaming, 2025.

[2] LangChain Team. Caching. https://python.langchain.com/docs/how_to/caching, 2025.

[3] LangChain Team. Rate Limiting. https://python.langchain.com/docs/how_to/rate_limiting, 2025.

[4] LangChain Team. Fallbacks. https://python.langchain.com/docs/how_to/fallbacks, 2025.
