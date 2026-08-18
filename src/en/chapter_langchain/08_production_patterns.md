# 12.8 LangChain Production Patterns

> **Goal**: Master the key engineering capabilities a LangChain application needs to move from development to production — streaming output, async execution, error handling, caching strategies, and concurrency control.

---

## From Demo to Production: Where Is the Gap?

Plenty of LangChain applications run beautifully as demos and then break down constantly once they are live. Here are the typical production challenges:

| Challenge | Demo stage | Production |
|------|----------|---------|
| **Latency** | Waiting a few seconds is fine | Users expect a response within 200ms |
| **Reliability** | An occasional error just means rerunning | 99.9% availability is required |
| **Cost** | A handful of calls costs nothing | At thousand-QPS scale, Token cost grows exponentially |
| **Concurrency** | Single-threaded, sequential execution | Concurrent requests must be handled |
| **Caching** | Not needed | Repeated queries waste Tokens and time |

This section tackles these problems one by one.

---

## Streaming Output

Streaming is the single most effective way to improve user experience — users don't have to wait for the Agent to finish every step; they see each step's output in real time.

### Basic Streaming Output

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional technical consultant."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# Synchronous streaming
for chunk in chain.stream({"question": "Explain what a vector database is"}):
    print(chunk, end="", flush=True)
print()  # newline
```

### Async Streaming Output

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional technical consultant."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

async def stream_response(question: str):
    """Async streaming output"""
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response("Explain what a vector database is"))
```

### A Complete Streaming Agent Implementation

For Agent applications, streaming is more involved — you have to handle the LLM's text output *and* its tool calls at the same time. Here is a complete streaming Agent implementation:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
import sys

# ============================
# Tool definitions
# ============================

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base"""
    return f"Knowledge base search result: information related to '{query}'..."

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# ============================
# Build the streaming Agent
# ============================

tools = [search_knowledge, calculate]

# Streaming LLM: set streaming=True plus a custom handler
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],  # stream to stdout in real time
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant. Use the tools to answer questions.
Explain your reasoning first, then give the conclusion."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # show intermediate steps
    max_iterations=5,
)

# ============================
# Streaming invocation
# ============================

def run_streaming_agent(user_input: str):
    """Run the streaming Agent"""
    print(f"\n🤔 User: {user_input}")
    print("📝 Assistant: ", end="", flush=True)

    for chunk in agent_executor.stream({
        "input": user_input,
        "chat_history": [],
    }):
        # AgentExecutor.stream() yields the output of every step
        if "actions" in chunk:
            # The Agent decided to call a tool
            for action in chunk["actions"]:
                print(f"\n🔧 Calling tool: {action.tool}({action.tool_input})")
        elif "steps" in chunk:
            # Tool execution result
            for step in chunk["steps"]:
                print(f"📊 Tool result: {step.observation[:100]}...")
        elif "output" in chunk:
            # Final output
            print(f"\n✅ Final answer: {chunk['output']}")

run_streaming_agent("Search for RAG techniques, and calculate 42 * 17")
```

### FastAPI + SSE Streaming API

In production you usually push streamed output to the frontend through Server-Sent Events (SSE):

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
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

@app.post("/chat/stream")
async def chat_stream(question: str):
    """SSE streaming response"""
    async def event_generator():
        async for chunk in chain.astream({"question": question}):
            # SSE format
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        }
    )
```

> 💡 **Streaming and perceived experience**: Studies show that users rate "output starts immediately" far higher than "everything appears at once after 3 seconds". Even when the total time is identical, streaming feels much faster.

---

## Async Execution Patterns

Async is the key to handling concurrency. Every LangChain Runnable supports async methods such as `ainvoke`, `astream`, and `abatch`.

### Basic Async Calls

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# Async single call
async def single_call():
    result = await chain.ainvoke({
        "role": "Python expert",
        "question": "What is asyncio?"
    })
    print(result)

# Async batch call
async def batch_call():
    inputs = [
        {"role": "Python expert", "question": "What are decorators?"},
        {"role": "data analyst", "question": "What is pandas?"},
        {"role": "frontend developer", "question": "What is React?"},
    ]
    results = await chain.abatch(inputs)
    for r in results:
        print(r[:50], "...")

asyncio.run(single_call())
asyncio.run(batch_call())
```

### Async Tool Calls

```python
import asyncio
import aiohttp
from langchain_core.tools import tool

@tool
async def async_search(query: str) -> str:
    """Async search (calls a remote API)"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.example.com/search?q={query}"
        ) as resp:
            data = await resp.json()
            return data.get("results", "No results found")

@tool
async def async_fetch_url(url: str) -> str:
    """Async fetch of web page content"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            text = await resp.text()
            return text[:500]  # take the first 500 characters

# Using async tools inside an Agent
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

# Run asynchronously
async def main():
    result = await app.ainvoke({
        "messages": [{"role": "user", "content": "Search for the latest LangChain version"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

> ⚠️ **Async caveats**:
> - An async tool must define an `async def _arun()` method (for `BaseTool`) or simply be declared with `async def` (for `@tool`)
> - In async frameworks such as FastAPI, always use `ainvoke` rather than `invoke`, otherwise you block the event loop
> - `abatch` executes concurrently — watch out for the provider's rate limits

---

## Error Handling and Retry Strategies

LLM applications have more failure sources than traditional software: network timeouts, API rate limits, model hallucinations, tool execution failures, and so on.

### Common Error Types

| Error type | Cause | Handling strategy |
|---------|------|---------|
| **RateLimitError** | API call frequency exceeded | Exponential-backoff retry |
| **TimeoutError** | LLM response timed out | Retry + fall back to a smaller model |
| **AuthenticationError** | Invalid API key | Config check + alerting |
| **ToolExecutionError** | Tool execution failed | Feed the error back to the Agent |
| **OutputParsingError** | Malformed model output | Retry + tolerant parsing |
| **ContextLengthExceeded** | Input exceeds the Token limit | Truncate + summarize |

### Implementing a Retry Strategy

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithFallbacks
import time

# Option 1: chained downgrade with with_fallbacks
primary_llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=3)
fallback_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3)

chain = ChatPromptTemplate.from_messages([
    ("system", "You are a professional assistant."),
    ("human", "{question}")
]) | primary_llm | StrOutputParser()

# Set up the fallback chain: switch to the backup model automatically when the primary fails
robust_chain = chain.with_fallbacks(
    fallbacks=[
        ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant."),
            ("human", "{question}")
        ]) | fallback_llm | StrOutputParser()
    ],
    exceptions_to_handle=(Exception,),  # catch every exception
)

result = robust_chain.invoke({"question": "What is RAG?"})
print(result)
```

### Custom Retry Logic

```python
import time
import logging
from functools import wraps

logger = logging.getLogger("agent_retry")

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Exponential-backoff retry decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Still failing after {max_retries} retries: {e}")
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Retry #{attempt + 1}, retrying in {delay:.1f}s. Error: {e}"
                    )
                    time.sleep(delay)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Still failing after {max_retries} retries: {e}")
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Retry #{attempt + 1}, retrying in {delay:.1f}s. Error: {e}"
                    )
                    await asyncio.sleep(delay)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# Usage example
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

@retry_with_backoff(max_retries=3, base_delay=2.0)
def call_llm(prompt_text: str) -> str:
    """LLM call with retry"""
    response = llm.invoke(prompt_text)
    return response.content

# Using it inside a tool
from langchain_core.tools import tool

@tool
@retry_with_backoff(max_retries=2, base_delay=1.0)
def search_api(query: str) -> str:
    """Call the search API (with retry)"""
    import requests
    resp = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", "")
```

### Agent-Level Error Handling

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage

@tool
def risky_operation(param: str) -> str:
    """An operation that may fail"""
    import random
    if random.random() < 0.3:  # 30% chance of failure
        raise ValueError("Operation failed: simulated error")
    return f"Operation succeeded: {param}"

def handle_tool_error(state: MessagesState) -> dict:
    """Handle tool execution errors: feed the error message back to the Agent"""
    error = state.get("error")
    if error:
        return {
            "messages": [
                AIMessage(content=f"Tool execution error: {error}. Please try another approach.")
            ]
        }
    return state

# Build a graph with error handling
tools = [risky_operation]
llm = ChatOpenAI(model="gpt-4.1").bind_tools(tools)

def agent_node(state: MessagesState):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"System error: {e}")]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools, handle_tool_error=True))  # handle tool errors automatically
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()
```

---

## Caching Strategies

Caching is an effective way to cut both cost and latency. LangChain ships with several cache implementations:

### InMemoryCache

```python
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache

# Configure the global cache
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# Note: since langchain-core 0.3, the cache is configured through set_llm_cache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# First call: actually hits the LLM
result1 = llm.invoke("What is Python?")
print("First call finished")

# Second identical call: cache hit, the LLM is not called
result2 = llm.invoke("What is Python?")
print("Second call finished (cache hit)")
```

### RedisCache

```python
# pip install langchain-redis

from langchain_core.globals import set_llm_cache
from langchain_redis import RedisCache

# Connect to Redis
redis_cache = RedisCache(
    redis_url="redis://localhost:6379",
    ttl=3600,  # cache for 1 hour
)

set_llm_cache(redis_cache)

# Usage is identical to InMemoryCache
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Identical input hits the Redis cache
result = llm.invoke("What is Python?")
```

### SemanticCache

Semantic caching is unique to LLM applications — even if users phrase their question differently, a semantically close query can still hit the cache:

```python
# pip install langchain-community

from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings

# SemanticCache uses vector similarity to decide whether it is a cache hit
from langchain_community.cache import SemanticCache

semantic_cache = SemanticCache(
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95,  # similarity threshold; higher is stricter
)

set_llm_cache(semantic_cache)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# First call
result1 = llm.invoke("What kind of programming language is Python?")

# A semantically similar call also hits the cache
result2 = llm.invoke("What is the Python programming language?")  # different wording, same meaning
```

### Cache Strategy Comparison

| Cache type | Hit condition | Best for | Caveats |
|---------|---------|---------|---------|
| **InMemoryCache** | Input is exactly identical | Development / testing | Lost on restart, not suitable for production |
| **RedisCache** | Input is exactly identical | Production | Requires deploying Redis |
| **SemanticCache** | Semantically similar | Workloads with many repeated queries | Extra Embedding cost, risk of false hits |

> ⚠️ **Caching caveats**:
> - Be careful with caching when `temperature > 0` — the same input may be expected to produce different outputs
> - `score_threshold` for SemanticCache must be tuned against your actual data
> - In production, RedisCache is the recommended default, with semantic caching as a supplementary optimization

---

## Concurrency Control and Rate Limiting

### Rate Limiter

LangChain has a built-in rate limiter that keeps you under the API provider's call-frequency limits:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# Create the rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,   # at most 2 requests per second
    check_every_n_seconds=0.1,  # check frequency
    max_bucket_size=10,      # token bucket size
)

# Apply it to the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    rate_limiter=rate_limiter,  # automatic rate limiting
)

# Batch calls are rate-limited automatically
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
chain = prompt | llm | StrOutputParser()

# These 20 requests are throttled automatically
questions = [{"question": f"Question {i}"} for i in range(20)]
results = chain.batch(questions)
print(f"Completed {len(results)} requests")
```

### Async Concurrency Control

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
    """Use a semaphore to cap the concurrency level"""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_call(input_data: dict):
        async with semaphore:
            return await chain.ainvoke(input_data)

    tasks = [bounded_call(inp) for inp in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sort results from exceptions
    success = []
    failures = []
    for inp, result in zip(inputs, results):
        if isinstance(result, Exception):
            failures.append((inp, str(result)))
        else:
            success.append((inp, result))

    print(f"Succeeded: {len(success)}, Failed: {len(failures)}")
    return success, failures

# Run
inputs = [{"question": f"Explain concept {i}"} for i in range(20)]
asyncio.run(process_with_semaphore(inputs, max_concurrency=5))
```

---

## Production Checklist

Before shipping a LangChain application, walk through the following list item by item:

### Reliability

- [ ] Set `max_retries=3` or implement custom retry logic
- [ ] Configure a `with_fallbacks` chain (primary model → backup model)
- [ ] Tool execution has timeout control (the `timeout` parameter)
- [ ] The Agent has a maximum iteration limit (`max_iterations`)
- [ ] Critical paths are wrapped in try-except error handling

### Performance

- [ ] Streaming output is enabled (`streaming=True`)
- [ ] Async calls are used (`ainvoke` / `astream`)
- [ ] Caching is configured (RedisCache / SemanticCache)
- [ ] Batch requests use `abatch` instead of looping over `ainvoke`
- [ ] A rate limiter is set up (`InMemoryRateLimiter`)

### Observability

- [ ] LangSmith / LangFuse tracing is enabled
- [ ] Projects are isolated per environment (dev / staging / prod)
- [ ] Token consumption and cost are recorded for every request
- [ ] Cost budget alerts are configured
- [ ] Key business metrics are monitored (P99 latency, error rate, cache hit rate)

### Security

- [ ] API keys are injected via environment variables, never hard-coded
- [ ] Tool execution is sandboxed (see Chapter 19)
- [ ] User input is sanitized to prevent Prompt injection
- [ ] Sensitive information never appears in logs or traces

### Deployment

- [ ] Deployed with LangServe / LangGraph Platform
- [ ] Health check endpoint (`/health`)
- [ ] Graceful shutdown (drain in-flight requests)
- [ ] Horizontal autoscaling configured
- [ ] Evaluation datasets are run to confirm there are no regressions

---

## Summary

| Production capability | Key points |
|-----------|---------|
| **Streaming output** | `stream()` / `astream()` + SSE, dramatically lowers perceived latency |
| **Async execution** | `ainvoke` / `abatch` + semaphore-based concurrency control |
| **Error handling** | `with_fallbacks` downgrade + exponential-backoff retry |
| **Caching strategy** | Step up from InMemoryCache → RedisCache → SemanticCache |
| **Rate limiting** | `InMemoryRateLimiter` + semaphore as a second layer of protection |
| **Checklist** | Five dimensions: reliability, performance, observability, security, deployment |

> 💡 **How this relates to other chapters of the book**:
> - Chapter 18 [Agent Evaluation and Optimization](../chapter_20_evaluation/README.md) discusses performance optimization and cost control in more detail
> - Chapter 19 [Security and Reliability](../chapter_21_security/README.md) covers Prompt injection defense and sandbox isolation in depth
> - Chapter 20 [Deployment and Productionization](../chapter_22_deployment/README.md) covers containerization, K8s, Serverless, and other deployment options

---

## 📝 Exercises

After reading this section, close the book and answer the questions in your own words first, then expand the reference answers to check.

**Exercise 1 (Concept)**: One term shows up again and again in this section — the Runnable protocol. Explain: what is the Runnable protocol, and why is it the foundation that makes LCEL (connecting components with the `|` pipe operator) possible? Then explain, with an example, the principle behind the statement "once a chain is written, it automatically gains streaming, async, and batching capabilities".

<details>
<summary>Reference answer</summary>

**What is the Runnable protocol?**
Runnable is the single most important abstraction introduced in LangChain 0.2+ — it is a **unified interface contract**. Almost every component in LangChain (prompt templates, LLMs, output parsers, tools, retrievers) implements this interface, so they all share exactly the same invocation methods: `invoke` (synchronous, single call), `ainvoke` (async), `stream` (streaming), `batch` (batched), and so on.

**Why is it the foundation of LCEL?**
LCEL strings components together with the `|` pipe operator, for example `prompt | llm | parser`. Behind that `|` is Python's `__or__` method, which combines two Runnables into a new `RunnableSequence` that also satisfies the Runnable interface. Precisely because "every component is a Runnable, and the result of combining them is still a Runnable", you can chain them endlessly like building blocks — that is the composability a unified interface buys you. If each component had its own idiosyncratic calling convention, `|` could not work generically.

**Why do you automatically get streaming / async / batching?**
Because all three capabilities are methods defined by the Runnable interface itself (`stream`/`astream`, `ainvoke`, `batch`/`abatch`). When you combine components into a chain with `|`, that chain is a new Runnable that "passes the call down the pipeline" to each sub-component. For instance, calling `.astream()` on the whole chain makes LangChain drive the prompt, the LLM, and the parser in async streaming mode in turn, ultimately emitting the LLM's tokens chunk by chunk. You never have to write streaming or async logic separately for each chain — that is the work saved by programming against an interface.

</details>

**Exercise 2 (Distinguish)**: This section introduces three caches: InMemoryCache, RedisCache, and SemanticCache. A student says: "Since SemanticCache can even hit on questions phrased differently, it must be the best one, so we should use it everywhere." Is that right? Compare the hit conditions and costs of the three caches, and explain why this section specifically warns that you should "be careful with caching when `temperature > 0`".

<details>
<summary>Reference answer</summary>

**The claim is not entirely right — SemanticCache is powerful but comes with costs and risks, so it should not be applied blindly everywhere.**

Comparison of the three caches:

| Cache | Hit condition | Cost / risk | Best for |
|------|---------|------------|---------|
| InMemoryCache | Input is **exactly identical** | Lost on restart, not shared across processes | Local development / testing |
| RedisCache | Input is **exactly identical** | Requires deploying and maintaining Redis | The workhorse cache in production |
| SemanticCache | Semantically **similar** (vector similarity above the threshold) | Every query needs an Embedding computation first (extra cost and latency); risk of **false hits** | Highly repetitive queries with variable phrasing |

**Two hidden problems with SemanticCache:**
1. **Extra cost**: deciding whether it is a hit requires converting the question into a vector first (an Embedding model call), which costs money and time. If the query repetition rate is low, the money you save is less than what you pay for embeddings.
2. **False-hit risk**: if the similarity threshold (`score_threshold`) is set too low, questions that "look similar but actually differ" get treated as hits and return the wrong answer. For example, "How do I read a file in Python?" and "How do I write a file in Python?" are semantically close but may require completely different answers. That is why this section recommends it as a supplementary optimization, with RedisCache still doing the heavy lifting in production.

**Why be careful with caching when `temperature > 0`?**
Caching assumes "the same input should produce the same output". But `temperature > 0` means we **deliberately want randomness** from the model — in creative writing or diversified recommendations, a user asking the same question again may specifically want a different answer. If the cache is on, the second identical question returns the first stale result, destroying the randomness and diversity. So caching fits `temperature=0` (deterministic output) scenarios best, such as fixed Q&A or information extraction.

</details>

**Exercise 3 (Hands-on)**: You need to deploy a LangChain question-answering chain as a production-grade FastAPI service that satisfies two requirements: (1) push the answer to the frontend with SSE (Server-Sent Events) streaming to reduce the user's perceived latency; (2) use `gpt-4.1` as the primary model, falling back automatically to `gpt-4.1-mini` when it fails, to preserve availability. Write the core code, and explain why you must use `astream` rather than `stream` in FastAPI.

<details>
<summary>Reference answer</summary>

The core idea is to combine two capabilities from this section: `with_fallbacks` for the downgrade, plus `astream` + `StreamingResponse` for SSE streaming.

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional technical consultant."),
    ("human", "{question}"),
])

# ── Primary chain: gpt-4.1 ─────────────────────────────
primary_chain = (
    prompt
    | ChatOpenAI(model="gpt-4.1", temperature=0, streaming=True, max_retries=3)
    | StrOutputParser()
)

# ── Fallback chain: gpt-4.1-mini ───────────────────────
fallback_chain = (
    prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0, streaming=True, max_retries=3)
    | StrOutputParser()
)

# ── Combine: fall back automatically when the primary fails ──
robust_chain = primary_chain.with_fallbacks([fallback_chain])

@app.post("/chat/stream")
async def chat_stream(question: str):
    async def event_generator():
        # Use astream to emit chunks asynchronously
        async for chunk in robust_chain.astream({"question": question}):
            # Wrap each chunk as an SSE data frame
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable Nginx buffering, otherwise the stream gets clumped together
        },
    )
```

**Why must you use `astream` instead of `stream` in FastAPI?**
FastAPI is built on an **async event loop** (asyncio) — it serves a large number of concurrent requests from a single-threaded event loop. `stream()` is a **synchronous** method: calling it **blocks** the current thread until the LLM returns data. Once blocked, the entire event loop is stuck, and during that time **no other user's request can be processed** — your concurrency drops to zero.

`astream()`, on the other hand, is **asynchronous**: while it waits for the LLM's next token, it uses `await` to hand control back to the event loop so it can serve other requests, then resumes when data arrives. This way the waiting time of individual requests is "overlapped", and the service can withstand high concurrency. That is exactly why the async caveats in this section stress: in async frameworks like FastAPI, always use `ainvoke` / `astream`, otherwise you block the event loop.

</details>

---

*Previous: [12.7 LangChain Ecosystem 2026](./07_langchain_ecosystem_2026.md)*

*Next chapter: [Chapter 13 LangGraph: Building Stateful Agents](../chapter_langgraph/README.md)*

---

## References

[1] LangChain Team. Streaming with LangChain. https://python.langchain.com/docs/how_to/streaming, 2025.

[2] LangChain Team. Caching. https://python.langchain.com/docs/how_to/caching, 2025.

[3] LangChain Team. Rate Limiting. https://python.langchain.com/docs/how_to/rate_limiting, 2025.

[4] LangChain Team. Fallbacks. https://python.langchain.com/docs/how_to/fallbacks, 2025.
