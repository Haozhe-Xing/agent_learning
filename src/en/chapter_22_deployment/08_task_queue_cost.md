# 22.8 Long-Running Task Queues and Cost Governance

> **Goal**: Learn how to manage long-running Agent tasks with task queues (Celery / Temporal), master Token budget control and on-demand large/small model routing, and build a complete cost monitoring and alerting system.

---

## The Long-Running Task Problem

A single Agent request may take minutes to complete — multi-step reasoning, tool calls, subtask spawning... If it is handled with a synchronous request, you run into three problems:

1. **Timeout**: API gateways usually cap request duration (e.g., 30 seconds), while an Agent can easily exceed this limit.
2. **Resource waste**: while one Worker is blocked by a long task, other short tasks can only queue up.
3. **Unreliability**: if the service restarts, in-flight tasks are lost and cannot be resumed.

The solution is an **asynchronous task queue** — submit the Agent request as a task to the queue, and Workers consume and execute it asynchronously.

---

## Task Queue Solution Comparison

| Dimension | Celery | Temporal | Redis Queue (RQ) |
|-----------|--------|----------|-------------------|
| Positioning | General-purpose task queue | Workflow orchestration engine | Lightweight task queue |
| State persistence | Redis / DB | Built-in persistence | Redis |
| Workflow support | Canvas (chain / fan-out) | Native DAG workflows | None |
| Failure retry | ✅ Auto retry | ✅ Auto retry | ✅ Manual config |
| Task timeout | ✅ | ✅ (down to activity level) | ✅ |
| Task cancellation | Limited | ✅ (precise cancellation) | ❌ |
| Visualization | Flower | Web UI | None |
| Learning curve | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Best for | Simple task queues | Complex Agent workflows | Quick prototypes |

> 💡 **Recommendation**: If your Agent is a simple "request → execute → return" service, Celery is enough. If your Agent involves complex multi-step workflows (conditional branches, human approval, subtask orchestration), Temporal's state management and visualization far outclass Celery.

---

## Celery for Agent Scenarios

### Basic Configuration

```python
# celery_config.py
from celery import Celery
from kombu import Queue

app = Celery("agent_worker")

app.conf.update(
    # Broker (message queue)
    broker_url="redis://localhost:6379/0",
    # Result backend
    result_backend="redis://localhost:6379/1",

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Queue definitions
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("simple_tasks", routing_key="simple"),
        Queue("complex_tasks", routing_key="complex"),
        Queue("tool_calls", routing_key="tool"),
    ),

    # Default routing
    task_default_queue="default",
    task_default_routing_key="default",

    # Concurrency control
    worker_concurrency=4,
    worker_prefetch_multiplier=1,  # set to 1 for long tasks

    # Timeouts
    task_soft_time_limit=120,   # soft timeout: 2 minutes
    task_time_limit=180,        # hard timeout: 3 minutes

    # Retry policy
    task_acks_late=True,        # acknowledge only after the task finishes
    task_reject_on_worker_lost=True,  # re-enqueue if the Worker crashes

    # Result expiration
    result_expires=3600,
)
```

### Agent Task Definitions

```python
# agent_tasks.py
from celery_config import app
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)


@app.task(
    name="agent.simple_chat",
    queue="simple_tasks",
    bind=True,              # allows access to self (the task instance)
    max_retries=2,          # retry at most twice
    default_retry_delay=5,  # 5-second retry interval
)
def simple_chat(self, message: str, session_id: str = None):
    """Simple chat task: fast response with the small model"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return {
            "reply": response.choices[0].message.content,
            "model": "gpt-4.1-mini",
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }
    except Exception as exc:
        logger.error(f"simple_chat task failed: {exc}")
        raise self.retry(exc=exc)


@app.task(
    name="agent.complex_reasoning",
    queue="complex_tasks",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
    soft_time_limit=120,
    time_limit=180,
)
def complex_reasoning(self, message: str, tools: list = None,
                      session_id: str = None):
    """Complex reasoning task: multi-step inference with the large model"""
    try:
        client = OpenAI()
        messages = [
            {"role": "system", "content": "You are a senior reasoning assistant. Think carefully before answering."},
            {"role": "user", "content": message},
        ]

        kwargs = {
            "model": "gpt-4.1",
            "messages": messages,
            "temperature": 0.3,  # lower temperature for reasoning tasks
            "max_tokens": 4096,
        }

        # Enable function calling if tool definitions are provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)

        result = {
            "reply": response.choices[0].message.content,
            "model": "gpt-4.1",
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }

        # Record any tool calls
        if response.choices[0].message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in response.choices[0].message.tool_calls
            ]

        return result

    except Exception as exc:
        logger.error(f"complex_reasoning task failed: {exc}")
        raise self.retry(exc=exc)


@app.task(
    name="agent.execute_tool",
    queue="tool_calls",
    bind=True,
    max_retries=3,
    default_retry_delay=3,
)
def execute_tool(self, tool_name: str, arguments: dict):
    """Tool-call execution task"""
    try:
        # Tool registry
        tool_registry = {
            "search": _tool_search,
            "calculate": _tool_calculate,
            "query_database": _tool_query_database,
        }

        if tool_name not in tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        result = tool_registry[tool_name](**arguments)
        return {"tool": tool_name, "result": result}

    except Exception as exc:
        logger.error(f"execute_tool task failed: {exc}")
        raise self.retry(exc=exc)


# ===== Tool implementations =====

def _tool_search(query: str, limit: int = 5) -> list:
    """Search tool"""
    # Wire this to a real search API in production
    return [{"title": f"Result for {query}", "url": "https://example.com"}]


def _tool_calculate(expression: str) -> float:
    """Calculator tool"""
    # Safe calculator implementation (do NOT use eval!)
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    node = ast.parse(expression, mode="eval")
    # Simplified example; production needs stricter safety checks
    return eval(expression)  # noqa: S307 — illustration only


def _tool_query_database(sql: str) -> list:
    """Database query tool"""
    # Wire this to a real database (read-only connection) in production
    return [{"column1": "value1", "column2": "value2"}]
```

### Task Orchestration: Chaining

```python
# workflow.py
from celery import chain, group, chord
from agent_tasks import simple_chat, complex_reasoning, execute_tool


def run_agent_pipeline(message: str):
    """
    Agent execution pipeline:
    1. light classification (small model)
    2. decide which path to take based on the classification
    3. execute tool calls (if any)
    4. summarize the result (large model)
    """
    # Approach 1: chain (sequential dependency)
    pipeline = chain(
        simple_chat.s(message, session_id="classify"),
        complex_reasoning.s(tools=[
            {"type": "function", "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }}
        ]),
    )

    result = pipeline.apply_async()
    return result.id


def run_parallel_tools(tool_calls: list[dict]):
    """
    Run multiple tool calls in parallel:
    when the Agent returns several tool_calls, execute them concurrently.
    """
    # group: run multiple tasks in parallel
    tasks = group(
        execute_tool.s(tc["name"], tc["arguments"])
        for tc in tool_calls
    )
    result = tasks.apply_async()
    return result.id
```

### API Integration

```python
# api.py — FastAPI endpoint that submits tasks to the queue
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from celery.result import AsyncResult

from agent_tasks import simple_chat, complex_reasoning
from celery_config import app as celery_app

api = FastAPI(title="Agent API (Async)")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    priority: str = "normal"  # normal / high


class TaskResponse(BaseModel):
    task_id: str
    status: str


@api.post("/chat/async", response_model=TaskResponse)
async def async_chat(req: ChatRequest):
    """Submit an Agent task asynchronously"""
    if req.priority == "high":
        # High priority goes straight to the large model
        task = complex_reasoning.apply_async(
            args=[req.message],
            kwargs={"session_id": req.session_id},
            queue="complex_tasks",
            priority=0,  # lower number = higher priority
        )
    else:
        task = simple_chat.apply_async(
            args=[req.message],
            kwargs={"session_id": req.session_id},
            queue="simple_tasks",
        )

    return TaskResponse(task_id=task.id, status="pending")


@api.get("/chat/result/{task_id}")
async def get_result(task_id: str):
    """Query the result of a task"""
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"status": "pending", "result": None}
    elif result.state == "STARTED":
        return {"status": "running", "result": None}
    elif result.state == "SUCCESS":
        return {"status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    else:
        return {"status": result.state.lower(), "result": None}
```

---

## Temporal for Agent Scenarios

Temporal is a workflow orchestration engine, naturally suited to the Agent's complex execution patterns — automatic state persistence, automatic retry on failure, support for long-running execution, and visual monitoring.

### Why Temporal Fits Agent Scenarios Better

> - **Celery**: `Task1 → Task2 → Task3` (linear chain)
> - **Temporal**: a stateful DAG workflow that supports: understand intent → select tool → execute tool (auto-retry on failure) → conditional branch (needs human approval? wait for the human signal — could be hours or even days) → generate the final reply

### Temporal Workflow Implementation

```python
# temporal_workflows.py
from datetime import timedelta
from typing import Optional

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# Shared retry policy
default_retry = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=["ValueError"],
)


# ===== Activities (atomic operations) =====

@activity.defn
async def classify_intent(message: str) -> str:
    """Classify the user's intent"""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": """Classify the user intent into:
- simple_qa: simple Q&A
- analysis: data analysis
- code_gen: code generation
- multi_step: multi-step reasoning
Return only the classification name."""},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


@activity.defn
async def call_llm(prompt: str, model: str = "gpt-4.1",
                   max_tokens: int = 2048) -> dict:
    """Call the LLM"""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return {
        "content": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


@activity.defn
async def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    """Execute a tool"""
    # Reuse the previous tool registry
    tool_registry = {
        "search": lambda q: {"results": [f"Search result for: {q}"]},
        "calculate": lambda expr: {"result": eval(expr)},  # noqa
    }

    handler = tool_registry.get(tool_name)
    if not handler:
        raise ValueError(f"Unknown tool: {tool_name}")

    return handler(**arguments)


@activity.defn
async def check_token_budget(session_id: str, tokens_to_use: int) -> bool:
    """Check whether the Token budget is sufficient"""
    # Wire this to Redis or a database in production
    # Simplified illustration
    return tokens_to_use < 10000


# ===== Workflow =====

@workflow.defn
class AgentWorkflow:
    """Agent execution workflow"""

    @workflow.run
    async def run(self, message: str,
                  session_id: str = "default") -> dict:
        """Main workflow entry point"""
        total_tokens = {"input": 0, "output": 0}
        tool_results = []

        # Step 1: classify intent
        intent = await workflow.execute_activity(
            classify_intent, message,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=default_retry,
        )

        # Step 2: select the model based on intent
        model = self._select_model(intent)

        # Step 3: check the Token budget
        budget_ok = await workflow.execute_activity(
            check_token_budget, session_id, 4096,
            start_to_close_timeout=timedelta(seconds=5),
        )
        if not budget_ok:
            model = "gpt-4.1-mini"  # fall back to the small model

        # Step 4: call the LLM
        llm_result = await workflow.execute_activity(
            call_llm, message, model,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=default_retry,
        )
        total_tokens["input"] += llm_result["input_tokens"]
        total_tokens["output"] += llm_result["output_tokens"]

        # Step 5: if there are tool calls, execute them
        if "tool_calls" in llm_result:
            for tc in llm_result["tool_calls"]:
                tool_result = await workflow.execute_activity(
                    execute_tool_call,
                    tc["name"], tc["arguments"],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=default_retry,
                )
                tool_results.append(tool_result)

            # Step 6: reason once more using the tool results
            followup_prompt = (
                f"Original question: {message}\n"
                f"Tool results: {tool_results}\n"
                f"Please combine the above information into a final answer."
            )
            final_result = await workflow.execute_activity(
                call_llm, followup_prompt, model,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=default_retry,
            )
            total_tokens["input"] += final_result["input_tokens"]
            total_tokens["output"] += final_result["output_tokens"]
            llm_result = final_result

        return {
            "reply": llm_result["content"],
            "intent": intent,
            "model": model,
            "tokens": total_tokens,
            "tools_used": len(tool_results),
        }

    def _select_model(self, intent: str) -> str:
        """Select the model based on intent"""
        simple_intents = {"simple_qa", "analysis"}
        return "gpt-4.1-mini" if intent in simple_intents else "gpt-4.1"
```

### Starting the Temporal Worker

```python
# temporal_worker.py
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from temporal_workflows import (
    AgentWorkflow,
    classify_intent,
    call_llm,
    execute_tool_call,
    check_token_budget,
)


async def main():
    # Connect to the Temporal Server
    client = await Client.connect("localhost:7233")

    # Start the Worker
    worker = Worker(
        client,
        task_queue="agent-tasks",
        workflows=[AgentWorkflow],
        activities=[
            classify_intent,
            call_llm,
            execute_tool_call,
            check_token_budget,
        ],
        max_concurrent_workflow_tasks=10,
        max_concurrent_activities=20,
    )

    print("Temporal Worker started, listening on queue: agent-tasks")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

### Triggering Workflows from the API

```python
# temporal_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from temporalio.client import Client, WorkflowHandle
from datetime import timedelta
import asyncio

from temporal_workflows import AgentWorkflow

api = FastAPI(title="Agent API (Temporal)")

# Global Temporal Client
temporal_client: Client = None


@api.on_event("startup")
async def startup():
    global temporal_client
    temporal_client = await Client.connect("localhost:7233")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@api.post("/chat/async")
async def async_chat(req: ChatRequest):
    """Trigger the Agent workflow asynchronously"""
    handle = await temporal_client.start_workflow(
        AgentWorkflow.run,
        req.message,
        req.session_id,
        id=f"agent-{req.session_id}-{id(req)}",
        task_queue="agent-tasks",
        execution_timeout=timedelta(minutes=5),
    )
    return {"workflow_id": handle.id, "run_id": handle.run_id}


@api.get("/chat/result/{workflow_id}")
async def get_result(workflow_id: str):
    """Query the workflow result"""
    handle = temporal_client.get_workflow_handle(workflow_id)

    try:
        result = await handle.result()
        return {"status": "completed", "result": result}
    except Exception as e:
        # Check whether the workflow is still running
        desc = await handle.describe()
        if desc.status == 1:  # RUNNING
            return {"status": "running", "result": None}
        return {"status": "failed", "error": str(e)}
```

---

## Token Budget Control

Tokens are the core cost unit of any LLM application. An uncontrolled Agent can rack up an astronomical bill through looped calls, verbose replies, or malicious input.

### Layered Budget System

```python
# token_budget.py
import redis
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class BudgetScope(Enum):
    PER_REQUEST = "per_request"     # cap per single request
    PER_SESSION = "per_session"     # cap per single session
    PER_USER = "per_user"           # per-user daily budget
    GLOBAL = "global"               # global daily budget


@dataclass
class BudgetConfig:
    """Token budget configuration"""
    per_request_limit: int = 4096       # at most 4K tokens per request
    per_session_limit: int = 32768      # at most 32K tokens per session
    per_user_daily_limit: int = 100000  # 100K tokens per user per day
    global_daily_limit: int = 10000000  # 10M tokens globally per day
    warn_threshold: float = 0.8         # warn at 80% usage


class TokenBudgetManager:
    """Token budget manager"""

    def __init__(self, redis_client: redis.Redis,
                 config: BudgetConfig = None):
        self.redis = redis_client
        self.config = config or BudgetConfig()

    def check_and_reserve(self, user_id: str, session_id: str,
                          estimated_tokens: int) -> tuple[bool, str]:
        """
        Check and reserve the Token budget.
        Returns: (allowed, reason)
        """
        # 1. Check the per-request limit
        if estimated_tokens > self.config.per_request_limit:
            return False, (
                f"Estimated {estimated_tokens} tokens for this request "
                f"exceeds the limit {self.config.per_request_limit}"
            )

        # 2. Check the session budget
        session_key = f"budget:session:{session_id}"
        session_used = int(self.redis.get(session_key) or 0)
        if session_used + estimated_tokens > self.config.per_session_limit:
            remaining = self.config.per_session_limit - session_used
            return False, (
                f"Insufficient session Token budget: used {session_used}, "
                f"remaining {remaining}, need {estimated_tokens}"
            )

        # 3. Check the per-user daily budget
        user_key = f"budget:user:{user_id}:{datetime.now().strftime('%Y%m%d')}"
        user_used = int(self.redis.get(user_key) or 0)
        if user_used + estimated_tokens > self.config.per_user_daily_limit:
            remaining = self.config.per_user_daily_limit - user_used
            return False, f"User daily Token budget exceeded, remaining {remaining}"

        # 4. Check the global daily budget
        global_key = f"budget:global:{datetime.now().strftime('%Y%m%d')}"
        global_used = int(self.redis.get(global_key) or 0)
        if global_used + estimated_tokens > self.config.global_daily_limit:
            return False, "Global Token budget exceeded, please retry later"

        # Reserve the budget
        pipe = self.redis.pipeline()
        pipe.incrby(session_key, estimated_tokens)
        pipe.expire(session_key, 3600)  # session expires in 1 hour
        pipe.incrby(user_key, estimated_tokens)
        pipe.expire(user_key, 86400)    # user budget expires in 24 hours
        pipe.incrby(global_key, estimated_tokens)
        pipe.expire(global_key, 86400)
        pipe.execute()

        return True, "OK"

    def record_actual_usage(self, user_id: str, session_id: str,
                           actual_tokens: int, estimated_tokens: int):
        """
        Record the actual usage (which may differ from the estimate).
        If actual > estimated, deduct the difference;
        if actual < estimated, refund the difference.
        """
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return

        # Adjust each budget counter
        pipe = self.redis.pipeline()
        session_key = f"budget:session:{session_id}"
        user_key = f"budget:user:{user_id}:{datetime.now().strftime('%Y%m%d')}"
        global_key = f"budget:global:{datetime.now().strftime('%Y%m%d')}"

        pipe.incrby(session_key, diff)
        pipe.incrby(user_key, diff)
        pipe.incrby(global_key, diff)
        pipe.execute()

    def get_usage_report(self, user_id: str) -> dict:
        """Get the user's usage report"""
        user_key = f"budget:user:{user_id}:{datetime.now().strftime('%Y%m%d')}"
        used = int(self.redis.get(user_key) or 0)
        limit = self.config.per_user_daily_limit

        return {
            "user_id": user_id,
            "daily_used": used,
            "daily_limit": limit,
            "remaining": limit - used,
            "usage_percent": round(used / limit * 100, 1),
        }
```

### Token Usage Estimation

Estimate Token usage before sending the request, to avoid blindly burning the budget:

```python
# token_estimator.py
import tiktoken


class TokenEstimator:
    """Token usage estimator"""

    def __init__(self, model: str = "gpt-4.1"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_messages(self, messages: list[dict]) -> int:
        """Estimate the Token count of a list of messages"""
        total = 0
        for msg in messages:
            # ~4 tokens of formatting overhead per message
            total += 4
            total += len(self.encoding.encode(msg.get("content", "")))
            total += len(self.encoding.encode(msg.get("role", "")))

            # Extra overhead for tool definitions
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += len(self.encoding.encode(
                        json.dumps(tc.get("function", {}))
                    ))

        total += 2  # reply prefix
        return total

    def estimate_with_response(self, messages: list[dict],
                               expected_response_tokens: int = 1024) -> int:
        """Estimate the total Token count (input + expected output)"""
        input_tokens = self.estimate_messages(messages)
        return input_tokens + expected_response_tokens
```

---

## On-Demand Routing Between Large and Small Models

Combine the budget with task complexity to decide dynamically whether to use the large or the small model:

```python
# model_router.py
from openai import OpenAI
from token_budget import TokenBudgetManager, TokenEstimator
from typing import Optional
import json


class CostAwareModelRouter:
    """Cost-aware model router"""

    def __init__(self, budget_manager: TokenBudgetManager):
        self.client = OpenAI()
        self.budget = budget_manager
        self.estimator = TokenEstimator()

        # Model configuration
        self.models = {
            "small": {
                "id": "gpt-4.1-mini",
                "cost_per_1k_input": 0.0004,
                "cost_per_1k_output": 0.0016,
                "max_tokens": 16384,
            },
            "large": {
                "id": "gpt-4.1",
                "cost_per_1k_input": 0.002,
                "cost_per_1k_output": 0.008,
                "max_tokens": 16384,
            },
        }

        # Complexity classification prompt
        self.classify_prompt = """Judge the complexity of the following task:
- simple: small talk, simple Q&A, translation, format conversion
- complex: multi-step reasoning, code generation, tool calls, deep analysis

Return only 'simple' or 'complex'."""

    async def route(self, user_id: str, session_id: str,
                    messages: list[dict]) -> tuple[str, Optional[dict]]:
        """
        Route to the appropriate model.
        Returns: (model_id, budget_check_result)
        """
        user_message = messages[-1]["content"] if messages else ""

        # Step 1: quick classification (with the small model)
        classification = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": self.classify_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=10,
        ).choices[0].message.content.strip().lower()

        # Step 2: select the model
        is_complex = classification == "complex"
        model_tier = "large" if is_complex else "small"
        model_config = self.models[model_tier]

        # Step 3: estimate Tokens
        estimated = self.estimator.estimate_with_response(
            messages,
            expected_response_tokens=4096 if is_complex else 1024,
        )

        # Step 4: budget check
        allowed, reason = self.budget.check_and_reserve(
            user_id, session_id, estimated
        )

        if not allowed:
            # Budget insufficient; try downgrading to the small model
            if model_tier == "large":
                model_tier = "small"
                model_config = self.models[model_tier]
                estimated = self.estimator.estimate_with_response(
                    messages, expected_response_tokens=1024
                )
                allowed, reason = self.budget.check_and_reserve(
                    user_id, session_id, estimated
                )

            if not allowed:
                return None, {"error": reason, "estimated_tokens": estimated}

        return model_config["id"], {
            "model_tier": model_tier,
            "estimated_tokens": estimated,
            "budget_ok": allowed,
        }

    def calculate_cost(self, model_id: str, input_tokens: int,
                       output_tokens: int) -> float:
        """Calculate the cost of a single request"""
        for tier, config in self.models.items():
            if config["id"] == model_id:
                return (
                    input_tokens / 1000 * config["cost_per_1k_input"]
                    + output_tokens / 1000 * config["cost_per_1k_output"]
                )
        return 0.0
```

---

## Cost Monitoring and Alerting

### Cost Data Collection

```python
# cost_monitor.py
import time
import json
import redis
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
from dataclasses import dataclass, asdict

# Prometheus metrics
COST_COUNTER = Counter(
    "llm_cost_dollars_total",
    "Total LLM cost in USD",
    ["model", "user_tier"]
)

TOKEN_COUNTER = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "token_type"]  # token_type: input / output
)

REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120]
)

COST_RATE = Gauge(
    "llm_cost_rate_dollars_per_hour",
    "Current cost rate in USD/hour",
    ["model"]
)


@dataclass
class CostRecord:
    """Cost record"""
    timestamp: str
    user_id: str
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


class CostMonitor:
    """Cost monitoring and alerting"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

        # Alert thresholds
        self.alert_thresholds = {
            "per_request_cost": 0.5,       # alert if a single request exceeds $0.5
            "hourly_rate": 10.0,           # alert if hourly cost exceeds $10
            "daily_total": 100.0,          # alert if daily cost exceeds $100
            "user_daily": 5.0,             # alert if a single user's daily cost exceeds $5
        }

    def record(self, record: CostRecord):
        """Record the cost of one request"""
        # Update Prometheus metrics
        COST_COUNTER.labels(model=record.model, user_tier="default").inc(
            record.cost_usd
        )
        TOKEN_COUNTER.labels(model=record.model, token_type="input").inc(
            record.input_tokens
        )
        TOKEN_COUNTER.labels(model=record.model, token_type="output").inc(
            record.output_tokens
        )
        REQUEST_LATENCY.labels(model=record.model).observe(
            record.latency_seconds
        )

        # Store in Redis for alert computation
        hour_key = f"cost:hourly:{datetime.now().strftime('%Y%m%d%H')}"
        day_key = f"cost:daily:{datetime.now().strftime('%Y%m%d')}"
        user_key = f"cost:user:{record.user_id}:{datetime.now().strftime('%Y%m%d')}"

        pipe = self.redis.pipeline()
        pipe.incrbyfloat(hour_key, record.cost_usd)
        pipe.expire(hour_key, 86400)
        pipe.incrbyfloat(day_key, record.cost_usd)
        pipe.expire(day_key, 172800)
        pipe.incrbyfloat(user_key, record.cost_usd)
        pipe.expire(user_key, 172800)
        pipe.execute()

        # Check alerts
        self._check_alerts(record)

    def _check_alerts(self, record: CostRecord):
        """Check whether any alert should fire"""
        alerts = []

        # Per-request cost alert
        if record.cost_usd > self.alert_thresholds["per_request_cost"]:
            alerts.append({
                "type": "HIGH_REQUEST_COST",
                "message": (
                    f"Single-request cost ${record.cost_usd:.4f} "
                    f"exceeds threshold ${self.alert_thresholds['per_request_cost']}"
                ),
                "severity": "warning",
            })

        # Hourly cost alert
        hour_key = f"cost:hourly:{datetime.now().strftime('%Y%m%d%H')}"
        hourly_cost = float(self.redis.get(hour_key) or 0)
        if hourly_cost > self.alert_thresholds["hourly_rate"]:
            alerts.append({
                "type": "HIGH_HOURLY_COST",
                "message": (
                    f"Hourly cost ${hourly_cost:.2f} "
                    f"exceeds threshold ${self.alert_thresholds['hourly_rate']}"
                ),
                "severity": "critical",
            })

        # Per-user daily cost alert
        user_key = f"cost:user:{record.user_id}:{datetime.now().strftime('%Y%m%d')}"
        user_cost = float(self.redis.get(user_key) or 0)
        if user_cost > self.alert_thresholds["user_daily"]:
            alerts.append({
                "type": "HIGH_USER_COST",
                "message": (
                    f"User {record.user_id} daily cost ${user_cost:.2f} "
                    f"exceeds threshold ${self.alert_thresholds['user_daily']}"
                ),
                "severity": "warning",
            })

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

    def _send_alert(self, alert: dict):
        """Send an alert (integrate with Slack / DingTalk / email, etc.)"""
        # Wire this to a real alert channel in production
        # Example: push to a Redis alert queue
        alert["timestamp"] = datetime.now().isoformat()
        self.redis.lpush("alerts:queue", json.dumps(alert, ensure_ascii=False))
        print(f"[ALERT] {alert['type']}: {alert['message']}")

    def get_cost_dashboard(self) -> dict:
        """Get cost dashboard data"""
        now = datetime.now()
        hour_key = f"cost:hourly:{now.strftime('%Y%m%d%H')}"
        day_key = f"cost:daily:{now.strftime('%Y%m%d')}"

        return {
            "hourly_cost": float(self.redis.get(hour_key) or 0),
            "daily_cost": float(self.redis.get(day_key) or 0),
            "hourly_budget": self.alert_thresholds["hourly_rate"] * 24,
            "daily_budget": self.alert_thresholds["daily_total"],
        }
```

### Cost Dashboard API

```python
# cost_api.py
from fastapi import APIRouter, Depends
from cost_monitor import CostMonitor, CostRecord
from token_budget import TokenBudgetManager
import redis

router = APIRouter(prefix="/cost", tags=["Cost Management"])

redis_client = redis.Redis(host="localhost", port=6379, db=0)
cost_monitor = CostMonitor(redis_client)
budget_manager = TokenBudgetManager(redis_client)


@router.get("/dashboard")
async def cost_dashboard():
    """Cost dashboard"""
    return cost_monitor.get_cost_dashboard()


@router.get("/usage/{user_id}")
async def user_usage(user_id: str):
    """Per-user usage"""
    return budget_manager.get_usage_report(user_id)


@router.get("/alerts")
async def recent_alerts(limit: int = 20):
    """Recent alerts"""
    alerts = redis_client.lrange("alerts:queue", 0, limit - 1)
    return [json.loads(a) for a in alerts]
```

### Grafana Alert Rules

```yaml
# grafana-alerts.yaml — Grafana alert rules
apiVersion: 1
groups:
  - orgId: 1
    name: llm-cost-alerts
    rules:
      - uid: hourly-cost-alert
        title: "LLM hourly cost too high"
        condition: C
        data:
          - refId: A
            relativeTimeRange:
              from: 600
              to: 0
            datasourceUid: prometheus
            model:
              expr: increase(llm_cost_dollars_total[1h])
              instant: true
          - refId: B
            relativeTimeRange:
              from: 600
              to: 0
            datasourceUid: __expr__
            model:
              type: reduce
              expression: A
              reducer: lastNotNull
          - refId: C
            relativeTimeRange:
              from: 600
              to: 0
            datasourceUid: __expr__
            model:
              type: threshold
              expression: B
              conditions:
                - evaluator:
                    params:
                      - 10
                    type: gt
        noDataState: OK
        executionErrorState: Alerting
        for: 2m
        annotations:
          summary: "Hourly LLM cost exceeds $10"
          description: "LLM call cost over the past 1 hour exceeded the threshold"
```

---

## Caveats and Best Practices

1. **Task idempotency**: An Agent task may execute multiple times due to retries. Make sure tool calls (sending email, writing to a database) are idempotent:

```python
def send_email_idempotent(to: str, subject: str, body: str, idempotency_key: str):
    """Idempotent email sending"""
    # Check whether it was already sent
    if redis.get(f"email_sent:{idempotency_key}"):
        return {"status": "already_sent"}

    # Send the email
    result = email_client.send(to, subject, body)

    # Mark as sent
    redis.set(f"email_sent:{idempotency_key}", "1", ex=86400)
    return result
```

2. **Budget reservation vs. actual consumption**: `check_and_reserve` reserves based on an estimate; the actual consumption may differ. Always call `record_actual_usage` after the request completes to correct the difference, otherwise the budget will drift over time.

3. **Temporal's long waits**: If a workflow needs to wait for human approval, do not use `time.sleep` — Temporal supports native `workflow.wait_condition`, which can wait hours or even days without consuming compute resources.

4. **Celery result backend overhead**: If you don't need to query every task's result (e.g., pure async execution), set `task_ignore_result=True` to reduce storage overhead.

5. **Precision of cost data**: `incrbyfloat` may suffer floating-point precision issues under high concurrency. In production, store integers (e.g., 1 unit = 0.001 cents) and convert only when displaying.

6. **Alert storms**: Under high-frequency requests, avoid the same alert firing repeatedly. Use Redis `SET NX EX` to deduplicate alerts:

```python
def send_alert_dedup(alert_type: str, message: str, dedup_window: int = 300):
    """Deduplicated alert"""
    key = f"alert:dedup:{alert_type}"
    if redis.set(key, "1", nx=True, ex=dedup_window):
        # Send only once within the deduplication window
        _actual_send_alert(alert_type, message)
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Celery | Mature task queue, good for simple Agent scenarios |
| Temporal | Workflow orchestration engine, good for complex Agent workflows |
| Token budget | Layered control: per-request / per-session / per-user / global |
| Large/small model routing | Dynamically pick the model by complexity + budget; cuts cost 60%–80% |
| Cost monitoring | Prometheus metrics + Redis aggregation + multi-level alerts |
| Idempotency | Tool calls must be idempotent to prevent side effects from retries |

> 🎓 **Chapter summary**: From deployment architecture to inference serving, from K8s orchestration to Serverless GPU, from task queues to cost governance — we have completed the Agent's full evolution from "code that runs" to "a controllable production service". Deployment is not the end, but the starting point of continuous optimization — monitoring, alerting, and cost governance are the core of long-term operations.

---

## 📝 Exercises

After reading this chapter, close the book and answer the questions in your own words first, then expand the reference answers to check.

**Exercise 1 (Concept)**: Section 22.1 ("The Five Unique Challenges of Agent Deployment") lists "side effects of tool calls" as one of them. Pick that item and explain why it makes "failure retries" dangerous, and describe how this chapter (19.8) solves the problem.

<details>
<summary>Reference answer</summary>

**Why "side effects of tool calls" make retries dangerous** (see 19.1, 19.8):

Traditional web APIs are mostly "read" operations — retrying on failure is harmless, it just queries the database again. But an Agent produces **real side effects on the outside world**: sending email, transferring money, writing to a database, placing an order. These are "write" operations. If a request fails right after "the email was sent but success wasn't returned", the system retries as usual and **sends the same email again** — the user gets two copies and is charged twice. In other words, the retry mechanism itself backfires in side-effect scenarios.

**Solution: Idempotency**. The core idea is to assign a unique "idempotency key" to every side-effecting operation; before executing, check "has this key's operation already been done?" — if yes, return the previous result instead of repeating. The example in 19.8:

```python
def send_email_idempotent(to, subject, body, idempotency_key):
    # Check first whether this key was already sent
    if redis.get(f"email_sent:{idempotency_key}"):
        return {"status": "already_sent"}   # already sent, return without resending
    result = email_client.send(to, subject, body)
    redis.set(f"email_sent:{idempotency_key}", "1", ex=86400)  # mark as sent
    return result
```

No matter how many times the task is retried, the email is truly sent only once. That is why this chapter emphasizes: "tool calls must be idempotent to prevent side effects from retries".

</details>

**Exercise 2 (Distinguish)**: This chapter compares Celery and Temporal for "managing the Agent's async tasks". A student says "they're about the same, just pick either one". Point out what is wrong with that view, and give a concrete Agent scenario where "only Temporal will do".

<details>
<summary>Reference answer</summary>

This view ignores that the two have **different positioning** (see the comparison table in 19.8):

- **Celery is a "task queue"**: it excels at relatively independent, linear tasks of the form "submit a task → Worker executes → return result", at most simple chaining or fan-out via chain/group.
- **Temporal is a "workflow orchestration engine"**: it manages the entire multi-step process as a **stateful workflow** — state is persisted automatically, each step (Activity) can be retried independently on failure, conditional branches are supported, and it can "wait for a long time".

Why you can't just "pick either": if the Agent flow is complex (multi-step, branching, waiting on external signals), Celery forces you to hand-write a pile of state-saving and recovery logic — bug-prone and hard to trace.

**A scenario where "only Temporal will do"**: a **refund Agent with human approval**. The flow is: user requests a refund → Agent understands and verifies the order → when the amount exceeds 1000 yuan, **pause and wait for a human manager's approval** (which may take hours, or until the manager comes in the next day) → after approval, continue calling the refund API → notify the user.

The key here is "waiting for human approval may take hours/days". Handling this with Celery is awkward — you can't have a Worker `sleep` idle for a whole day hogging resources. Temporal natively supports `workflow.wait_condition`, which can suspend and wait for an external signal without consuming compute, then seamlessly continue when the approval signal arrives; the entire workflow state is persisted throughout, and survives a service restart. This is exactly what the chapter points out: "if your Agent involves complex multi-step workflows (conditional branches, human approval, subtask orchestration), Temporal's state management and visualization far outclass Celery".

</details>

**Exercise 3 (Hands-on)**: The Token budget in 19.8 is "layered" (per-request / per-session / per-user / global). Implement a **simplified** budget-check function `can_proceed(user_id, estimated_tokens)` that handles only the "per-user daily budget" layer: use a plain dictionary to simulate storage and reject when the daily limit is exceeded. Then explain: why should this counter use Redis rather than a Python dictionary in production?

<details>
<summary>Reference answer</summary>

Simplified implementation (only the per-user daily budget layer):

```python
from datetime import date


class SimpleUserBudget:
    """Simplified: only controls a single user's daily token budget"""

    def __init__(self, daily_limit: int = 100000):
        self.daily_limit = daily_limit
        # key = (user_id, date string), value = tokens used today
        self.usage = {}

    def can_proceed(self, user_id: str, estimated_tokens: int) -> tuple[bool, str]:
        today = date.today().isoformat()
        key = (user_id, today)
        used = self.usage.get(key, 0)

        if used + estimated_tokens > self.daily_limit:
            remaining = self.daily_limit - used
            return False, f"Daily budget exceeded: used {used}, remaining {remaining}, need {estimated_tokens}"

        # Reserve (deduct) the budget
        self.usage[key] = used + estimated_tokens
        return True, "OK"


# Test
budget = SimpleUserBudget(daily_limit=1000)
print(budget.can_proceed("user_A", 600))   # (True, 'OK')
print(budget.can_proceed("user_A", 600))   # (False, daily budget exceeded...) used 600, need 600 more → over
print(budget.can_proceed("user_B", 600))   # (True, 'OK')  different users don't interfere
```

**Why production should use Redis instead of a Python dictionary**:

1. **Shared across processes / instances**: a production Agent service usually runs multiple workers and multiple Pods (horizontal scaling, covered in 19.1 and 19.7). A Python dictionary is **in-process** — each instance keeps its own count, so instance B has no idea what instance A's user spent; the budget becomes meaningless. Redis is the **single shared** counter across all instances.
2. **Atomic operations prevent concurrency bugs**: when a user sends several requests at once, the "read used → add → write back" steps are not atomic on their own; two requests may both read the old value and both think the budget is fine, causing overspend. Redis `INCRBY` is atomic and naturally avoids this race.
3. **Automatic expiration**: the chapter uses `expire(key, 86400)` to auto-reset the counter after 24 hours, implementing the "daily" reset. A Python dictionary needs hand-written cleanup logic and is prone to memory leaks.
4. **Survives restarts**: after a restart, the Python dictionary is cleared and the budget is "reset", letting users bypass the limit; Redis persists its data, so the count survives restarts.

That is why this chapter's `TokenBudgetManager` uses a Redis pipeline + incrby + expire — precisely to satisfy these four requirements: shared, atomic, auto-expiring, and persistent.

</details>

---

[22.7 Kubernetes Orchestration and Serverless GPU](./07_k8s_serverless.md)
