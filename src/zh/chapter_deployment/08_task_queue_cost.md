# 19.8 长任务队列与成本治理

> **本节目标**：学会使用任务队列（Celery / Temporal）管理 Agent 的长时任务，掌握 Token 预算控制和大小模型按需路由策略，建立完整的成本监控与告警体系。

---

## Agent 的长任务问题

Agent 的一个请求可能需要数分钟才能完成——多步推理、工具调用、子任务派生……如果用同步请求处理，会遇到三个问题：

1. **超时**：API 网关通常限制请求时长（如 30 秒），Agent 可能远超这个限制
2. **资源浪费**：一个 Worker 被长任务阻塞时，其他短任务只能排队
3. **不可靠**：如果服务重启，进行中的任务会丢失，无法恢复

解决方案是**异步任务队列**——将 Agent 请求作为任务投递到队列，Worker 异步消费并执行。

---

## 任务队列方案对比

| 维度 | Celery | Temporal | Redis Queue (RQ) |
|------|--------|----------|-------------------|
| 定位 | 通用任务队列 | 工作流编排引擎 | 轻量级任务队列 |
| 状态持久化 | Redis / DB | 自带持久化 | Redis |
| 工作流支持 | Canvas（链式/扇出） | 原生 DAG 工作流 | 无 |
| 失败重试 | ✅ 自动重试 | ✅ 自动重试 | ✅ 手动配置 |
| 任务超时 | ✅ | ✅（精确到活动级别） | ✅ |
| 任务取消 | 有限支持 | ✅（精确取消） | ❌ |
| 可视化 | Flower | Web UI | 无 |
| 学习曲线 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 适用场景 | 简单任务队列 | 复杂 Agent 工作流 | 快速原型 |

> 💡 **选型建议**：如果你的 Agent 是简单的"请求-执行-返回"，Celery 足够。如果 Agent 涉及复杂的多步工作流（如条件分支、人工审批、子任务编排），Temporal 的状态管理和可视化能力远超 Celery。

---

## Celery 在 Agent 场景的应用

### 基础配置

```python
# celery_config.py
from celery import Celery
from kombu import Queue

app = Celery("agent_worker")

app.conf.update(
    # Broker（消息队列）
    broker_url="redis://localhost:6379/0",
    # 结果后端
    result_backend="redis://localhost:6379/1",

    # 序列化
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # 队列定义
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("simple_tasks", routing_key="simple"),
        Queue("complex_tasks", routing_key="complex"),
        Queue("tool_calls", routing_key="tool"),
    ),

    # 默认路由
    task_default_queue="default",
    task_default_routing_key="default",

    # 并发控制
    worker_concurrency=4,
    worker_prefetch_multiplier=1,  # 长任务建议设为 1

    # 超时设置
    task_soft_time_limit=120,   # 软超时 2 分钟
    task_time_limit=180,        # 硬超时 3 分钟

    # 重试策略
    task_acks_late=True,        # 任务执行完才确认
    task_reject_on_worker_lost=True,  # Worker 崩溃时重新入队

    # 结果过期
    result_expires=3600,
)
```

### Agent 任务定义

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
    bind=True,              # 允许访问 self（任务实例）
    max_retries=2,          # 最多重试 2 次
    default_retry_delay=5,  # 重试间隔 5 秒
)
def simple_chat(self, message: str, session_id: str = None):
    """简单对话任务：使用小模型快速响应"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "你是一个 AI 助手。"},
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
        logger.error(f"simple_chat 任务失败: {exc}")
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
    """复杂推理任务：使用大模型进行多步推理"""
    try:
        client = OpenAI()
        messages = [
            {"role": "system", "content": "你是一个高级推理助手。请仔细思考后再回答。"},
            {"role": "user", "content": message},
        ]

        kwargs = {
            "model": "gpt-4.1",
            "messages": messages,
            "temperature": 0.3,  # 推理任务降低温度
            "max_tokens": 4096,
        }

        # 如果有工具定义，启用 function calling
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

        # 如果有工具调用，记录
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
        logger.error(f"complex_reasoning 任务失败: {exc}")
        raise self.retry(exc=exc)


@app.task(
    name="agent.execute_tool",
    queue="tool_calls",
    bind=True,
    max_retries=3,
    default_retry_delay=3,
)
def execute_tool(self, tool_name: str, arguments: dict):
    """执行工具调用任务"""
    try:
        # 工具注册表
        tool_registry = {
            "search": _tool_search,
            "calculate": _tool_calculate,
            "query_database": _tool_query_database,
        }

        if tool_name not in tool_registry:
            raise ValueError(f"未知工具: {tool_name}")

        result = tool_registry[tool_name](**arguments)
        return {"tool": tool_name, "result": result}

    except Exception as exc:
        logger.error(f"execute_tool 任务失败: {exc}")
        raise self.retry(exc=exc)


# ===== 工具实现 =====

def _tool_search(query: str, limit: int = 5) -> list:
    """搜索工具"""
    # 实际实现对接搜索 API
    return [{"title": f"Result for {query}", "url": "https://example.com"}]


def _tool_calculate(expression: str) -> float:
    """计算器工具"""
    # 安全的计算器实现（不要用 eval！）
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    node = ast.parse(expression, mode="eval")
    # 简化示例，生产环境需要更严格的安全检查
    return eval(expression)  # noqa: S307 — 仅示意


def _tool_query_database(sql: str) -> list:
    """数据库查询工具"""
    # 实际实现对接数据库（只读连接）
    return [{"column1": "value1", "column2": "value2"}]
```

### 任务编排：链式执行

```python
# workflow.py
from celery import chain, group, chord
from agent_tasks import simple_chat, complex_reasoning, execute_tool

def run_agent_pipeline(message: str):
    """
    Agent 执行流水线：
    1. 简单分类（小模型）
    2. 根据分类结果决定走哪条路径
    3. 执行工具调用（如有）
    4. 总结结果（大模型）
    """
    # 方式一：链式执行（顺序依赖）
    pipeline = chain(
        simple_chat.s(message, session_id="classify"),
        complex_reasoning.s(tools=[
            {"type": "function", "function": {
                "name": "search",
                "description": "搜索信息",
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
    并行执行多个工具调用：
    当 Agent 返回多个 tool_call 时，可以并行执行
    """
    # group：并行执行多个任务
    tasks = group(
        execute_tool.s(tc["name"], tc["arguments"])
        for tc in tool_calls
    )
    result = tasks.apply_async()
    return result.id
```

### API 端集成

```python
# api.py — FastAPI 端提交任务到队列
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
    """异步提交 Agent 任务"""
    if req.priority == "high":
        # 高优先级直接走大模型
        task = complex_reasoning.apply_async(
            args=[req.message],
            kwargs={"session_id": req.session_id},
            queue="complex_tasks",
            priority=0,  # 数字越小优先级越高
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
    """查询任务结果"""
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

## Temporal 在 Agent 场景的应用

Temporal 是一个工作流编排引擎，天生适合 Agent 的复杂执行模式——状态自动持久化、失败自动重试、支持长时间运行、可视化监控。

### 为什么 Agent 场景更适合 Temporal？

> - **Celery**：`Task1 → Task2 → Task3`（线性链）
> - **Temporal**：有状态的 DAG Workflow，支持：理解意图 → 选择工具 → 执行工具（失败自动重试）→ 条件分支（需要人工审批？等待人工信号，可等几小时甚至几天）→ 生成最终回复

### Temporal 工作流实现

```python
# temporal_workflows.py
from datetime import timedelta
from typing import Optional

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# 通用重试策略
default_retry = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=["ValueError"],
)


# ===== Activities（原子操作）=====

@activity.defn
async def classify_intent(message: str) -> str:
    """分类用户意图"""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": """将用户意图分类为：
- simple_qa: 简单问答
- analysis: 数据分析
- code_gen: 代码生成
- multi_step: 多步推理
只返回分类名称。"""},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


@activity.defn
async def call_llm(prompt: str, model: str = "gpt-4.1",
                   max_tokens: int = 2048) -> dict:
    """调用 LLM"""
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
    """执行工具"""
    # 复用之前的工具注册表
    tool_registry = {
        "search": lambda q: {"results": [f"Search result for: {q}"]},
        "calculate": lambda expr: {"result": eval(expr)},  # noqa
    }

    handler = tool_registry.get(tool_name)
    if not handler:
        raise ValueError(f"未知工具: {tool_name}")

    return handler(**arguments)


@activity.defn
async def check_token_budget(session_id: str, tokens_to_use: int) -> bool:
    """检查 Token 预算是否充足"""
    # 实际实现对接 Redis 或数据库
    # 简化示意
    return tokens_to_use < 10000


# ===== Workflow（工作流）=====

@workflow.defn
class AgentWorkflow:
    """Agent 执行工作流"""

    @workflow.run
    async def run(self, message: str,
                  session_id: str = "default") -> dict:
        """主工作流入口"""
        total_tokens = {"input": 0, "output": 0}
        tool_results = []

        # Step 1: 分类意图
        intent = await workflow.execute_activity(
            classify_intent, message,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=default_retry,
        )

        # Step 2: 根据意图选择模型
        model = self._select_model(intent)

        # Step 3: 检查 Token 预算
        budget_ok = await workflow.execute_activity(
            check_token_budget, session_id, 4096,
            start_to_close_timeout=timedelta(seconds=5),
        )
        if not budget_ok:
            model = "gpt-4.1-mini"  # 降级到小模型

        # Step 4: 调用 LLM
        llm_result = await workflow.execute_activity(
            call_llm, message, model,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=default_retry,
        )
        total_tokens["input"] += llm_result["input_tokens"]
        total_tokens["output"] += llm_result["output_tokens"]

        # Step 5: 如果有工具调用，执行工具
        if "tool_calls" in llm_result:
            for tc in llm_result["tool_calls"]:
                tool_result = await workflow.execute_activity(
                    execute_tool_call,
                    tc["name"], tc["arguments"],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=default_retry,
                )
                tool_results.append(tool_result)

            # Step 6: 用工具结果再做一次推理
            followup_prompt = (
                f"原始问题: {message}\n"
                f"工具结果: {tool_results}\n"
                f"请综合以上信息给出最终回答。"
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
        """根据意图选择模型"""
        simple_intents = {"simple_qa", "analysis"}
        return "gpt-4.1-mini" if intent in simple_intents else "gpt-4.1"
```

### Temporal Worker 启动

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
    # 连接 Temporal Server
    client = await Client.connect("localhost:7233")

    # 启动 Worker
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

    print("Temporal Worker 启动，监听队列: agent-tasks")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 从 API 触发工作流

```python
# temporal_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from temporalio.client import Client, WorkflowHandle
from datetime import timedelta
import asyncio

from temporal_workflows import AgentWorkflow

api = FastAPI(title="Agent API (Temporal)")

# 全局 Temporal Client
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
    """异步触发 Agent 工作流"""
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
    """查询工作流结果"""
    handle = temporal_client.get_workflow_handle(workflow_id)

    try:
        result = await handle.result()
        return {"status": "completed", "result": result}
    except Exception as e:
        # 检查工作流是否还在运行
        desc = await handle.describe()
        if desc.status == 1:  # RUNNING
            return {"status": "running", "result": None}
        return {"status": "failed", "error": str(e)}
```

---

## Token 预算控制

Token 是 LLM 应用的核心成本单位。不加控制的 Agent 可能因为循环调用、冗长回复或恶意输入导致天价账单。

### 分层预算体系

```python
# token_budget.py
import redis
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class BudgetScope(Enum):
    PER_REQUEST = "per_request"     # 单次请求上限
    PER_SESSION = "per_session"     # 单次会话上限
    PER_USER = "per_user"           # 单用户日预算
    GLOBAL = "global"               # 全局日预算


@dataclass
class BudgetConfig:
    """Token 预算配置"""
    per_request_limit: int = 4096       # 单次请求最多 4K tokens
    per_session_limit: int = 32768      # 单次会话最多 32K tokens
    per_user_daily_limit: int = 100000  # 单用户每天 100K tokens
    global_daily_limit: int = 10000000  # 全局每天 10M tokens
    warn_threshold: float = 0.8         # 80% 时告警


class TokenBudgetManager:
    """Token 预算管理器"""

    def __init__(self, redis_client: redis.Redis,
                 config: BudgetConfig = None):
        self.redis = redis_client
        self.config = config or BudgetConfig()

    def check_and_reserve(self, user_id: str, session_id: str,
                          estimated_tokens: int) -> tuple[bool, str]:
        """
        检查并预留 Token 预算
        返回: (是否允许, 原因)
        """
        # 1. 检查单次请求限制
        if estimated_tokens > self.config.per_request_limit:
            return False, (
                f"单次请求预估 {estimated_tokens} tokens "
                f"超过限制 {self.config.per_request_limit}"
            )

        # 2. 检查会话预算
        session_key = f"budget:session:{session_id}"
        session_used = int(self.redis.get(session_key) or 0)
        if session_used + estimated_tokens > self.config.per_session_limit:
            remaining = self.config.per_session_limit - session_used
            return False, (
                f"会话 Token 预算不足：已用 {session_used}，"
                f"剩余 {remaining}，需要 {estimated_tokens}"
            )

        # 3. 检查用户日预算
        user_key = f"budget:user:{user_id}:{datetime.now().strftime('%Y%m%d')}"
        user_used = int(self.redis.get(user_key) or 0)
        if user_used + estimated_tokens > self.config.per_user_daily_limit:
            remaining = self.config.per_user_daily_limit - user_used
            return False, f"用户日 Token 预算不足，剩余 {remaining}"

        # 4. 检查全局日预算
        global_key = f"budget:global:{datetime.now().strftime('%Y%m%d')}"
        global_used = int(self.redis.get(global_key) or 0)
        if global_used + estimated_tokens > self.config.global_daily_limit:
            return False, "全局 Token 预算不足，请稍后重试"

        # 预留预算
        pipe = self.redis.pipeline()
        pipe.incrby(session_key, estimated_tokens)
        pipe.expire(session_key, 3600)  # 会话 1 小时过期
        pipe.incrby(user_key, estimated_tokens)
        pipe.expire(user_key, 86400)    # 用户预算 24 小时过期
        pipe.incrby(global_key, estimated_tokens)
        pipe.expire(global_key, 86400)
        pipe.execute()

        return True, "OK"

    def record_actual_usage(self, user_id: str, session_id: str,
                           actual_tokens: int, estimated_tokens: int):
        """
        记录实际使用量（与预估可能有差异）
        如果实际使用 > 预估，扣除差额；如果 < 预估，退还差额
        """
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return

        # 调整各类预算计数
        pipe = self.redis.pipeline()
        session_key = f"budget:session:{session_id}"
        user_key = f"budget:user:{user_id}:{datetime.now().strftime('%Y%m%d')}"
        global_key = f"budget:global:{datetime.now().strftime('%Y%m%d')}"

        pipe.incrby(session_key, diff)
        pipe.incrby(user_key, diff)
        pipe.incrby(global_key, diff)
        pipe.execute()

    def get_usage_report(self, user_id: str) -> dict:
        """获取用户使用报告"""
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

### Token 用量估算

在请求发出前估算 Token 用量，避免盲目消耗预算：

```python
# token_estimator.py
import tiktoken

class TokenEstimator:
    """Token 用量估算器"""

    def __init__(self, model: str = "gpt-4.1"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_messages(self, messages: list[dict]) -> int:
        """估算一组消息的 Token 数"""
        total = 0
        for msg in messages:
            # 每条消息有 ~4 tokens 的格式开销
            total += 4
            total += len(self.encoding.encode(msg.get("content", "")))
            total += len(self.encoding.encode(msg.get("role", "")))

            # 工具定义的额外开销
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += len(self.encoding.encode(
                        json.dumps(tc.get("function", {}))
                    ))

        total += 2  # 回复前缀
        return total

    def estimate_with_response(self, messages: list[dict],
                               expected_response_tokens: int = 1024) -> int:
        """估算总 Token 数（输入 + 预期输出）"""
        input_tokens = self.estimate_messages(messages)
        return input_tokens + expected_response_tokens
```

---

## 按需路由大小模型

结合预算和任务复杂度，动态决定使用大模型还是小模型：

```python
# model_router.py
from openai import OpenAI
from token_budget import TokenBudgetManager, TokenEstimator
from typing import Optional
import json


class CostAwareModelRouter:
    """成本感知的模型路由器"""

    def __init__(self, budget_manager: TokenBudgetManager):
        self.client = OpenAI()
        self.budget = budget_manager
        self.estimator = TokenEstimator()

        # 模型配置
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

        # 复杂度分类 prompt
        self.classify_prompt = """判断以下任务的复杂度：
- simple: 闲聊、简单问答、翻译、格式转换
- complex: 多步推理、代码生成、工具调用、深度分析

只返回 simple 或 complex。"""

    async def route(self, user_id: str, session_id: str,
                    messages: list[dict]) -> tuple[str, Optional[dict]]:
        """
        路由到合适的模型
        返回: (模型ID, 预算检查结果)
        """
        user_message = messages[-1]["content"] if messages else ""

        # Step 1: 快速分类（用小模型）
        classification = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": self.classify_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=10,
        ).choices[0].message.content.strip().lower()

        # Step 2: 选择模型
        is_complex = classification == "complex"
        model_tier = "large" if is_complex else "small"
        model_config = self.models[model_tier]

        # Step 3: 估算 Token
        estimated = self.estimator.estimate_with_response(
            messages,
            expected_response_tokens=4096 if is_complex else 1024,
        )

        # Step 4: 预算检查
        allowed, reason = self.budget.check_and_reserve(
            user_id, session_id, estimated
        )

        if not allowed:
            # 预算不足，尝试降级到小模型
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
        """计算单次请求成本"""
        for tier, config in self.models.items():
            if config["id"] == model_id:
                return (
                    input_tokens / 1000 * config["cost_per_1k_input"]
                    + output_tokens / 1000 * config["cost_per_1k_output"]
                )
        return 0.0
```

---

## 成本监控与告警

### 成本数据采集

```python
# cost_monitor.py
import time
import json
import redis
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
from dataclasses import dataclass, asdict

# Prometheus 指标
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
    """成本记录"""
    timestamp: str
    user_id: str
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


class CostMonitor:
    """成本监控与告警"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

        # 告警阈值
        self.alert_thresholds = {
            "per_request_cost": 0.5,       # 单次请求超过 $0.5 告警
            "hourly_rate": 10.0,           # 每小时成本超过 $10 告警
            "daily_total": 100.0,          # 日成本超过 $100 告警
            "user_daily": 5.0,             # 单用户日成本超过 $5 告警
        }

    def record(self, record: CostRecord):
        """记录一次请求的成本"""
        # 更新 Prometheus 指标
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

        # 存储到 Redis 用于告警计算
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

        # 检查告警
        self._check_alerts(record)

    def _check_alerts(self, record: CostRecord):
        """检查是否触发告警"""
        alerts = []

        # 单次请求成本告警
        if record.cost_usd > self.alert_thresholds["per_request_cost"]:
            alerts.append({
                "type": "HIGH_REQUEST_COST",
                "message": (
                    f"单次请求成本 ${record.cost_usd:.4f} "
                    f"超过阈值 ${self.alert_thresholds['per_request_cost']}"
                ),
                "severity": "warning",
            })

        # 小时级成本告警
        hour_key = f"cost:hourly:{datetime.now().strftime('%Y%m%d%H')}"
        hourly_cost = float(self.redis.get(hour_key) or 0)
        if hourly_cost > self.alert_thresholds["hourly_rate"]:
            alerts.append({
                "type": "HIGH_HOURLY_COST",
                "message": (
                    f"小时成本 ${hourly_cost:.2f} "
                    f"超过阈值 ${self.alert_thresholds['hourly_rate']}"
                ),
                "severity": "critical",
            })

        # 用户日成本告警
        user_key = f"cost:user:{record.user_id}:{datetime.now().strftime('%Y%m%d')}"
        user_cost = float(self.redis.get(user_key) or 0)
        if user_cost > self.alert_thresholds["user_daily"]:
            alerts.append({
                "type": "HIGH_USER_COST",
                "message": (
                    f"用户 {record.user_id} 日成本 ${user_cost:.2f} "
                    f"超过阈值 ${self.alert_thresholds['user_daily']}"
                ),
                "severity": "warning",
            })

        # 发送告警
        for alert in alerts:
            self._send_alert(alert)

    def _send_alert(self, alert: dict):
        """发送告警（对接 Slack / 钉钉 / 邮件等）"""
        # 这里对接实际的告警通道
        # 示例：写入 Redis 告警队列
        alert["timestamp"] = datetime.now().isoformat()
        self.redis.lpush("alerts:queue", json.dumps(alert, ensure_ascii=False))
        print(f"[ALERT] {alert['type']}: {alert['message']}")

    def get_cost_dashboard(self) -> dict:
        """获取成本看板数据"""
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

### 成本看板 API

```python
# cost_api.py
from fastapi import APIRouter, Depends
from cost_monitor import CostMonitor, CostRecord
from token_budget import TokenBudgetManager
import redis

router = APIRouter(prefix="/cost", tags=["成本管理"])

redis_client = redis.Redis(host="localhost", port=6379, db=0)
cost_monitor = CostMonitor(redis_client)
budget_manager = TokenBudgetManager(redis_client)


@router.get("/dashboard")
async def cost_dashboard():
    """成本看板"""
    return cost_monitor.get_cost_dashboard()


@router.get("/usage/{user_id}")
async def user_usage(user_id: str):
    """用户使用量"""
    return budget_manager.get_usage_report(user_id)


@router.get("/alerts")
async def recent_alerts(limit: int = 20):
    """最近告警"""
    alerts = redis_client.lrange("alerts:queue", 0, limit - 1)
    return [json.loads(a) for a in alerts]
```

### Grafana 告警规则

```yaml
# grafana-alerts.yaml — Grafana 告警规则
apiVersion: 1
groups:
  - orgId: 1
    name: llm-cost-alerts
    rules:
      - uid: hourly-cost-alert
        title: "LLM 小时成本过高"
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
          summary: "小时 LLM 成本超过 $10"
          description: "过去 1 小时的 LLM 调用成本超过阈值"
```

---

## 注意事项与最佳实践

1. **任务幂等性**：Agent 任务可能因为重试而执行多次。确保工具调用（发邮件、写数据库）是幂等的：

```python
def send_email_idempotent(to: str, subject: str, body: str, idempotency_key: str):
    """幂等的邮件发送"""
    # 检查是否已发送
    if redis.get(f"email_sent:{idempotency_key}"):
        return {"status": "already_sent"}

    # 发送邮件
    result = email_client.send(to, subject, body)

    # 标记为已发送
    redis.set(f"email_sent:{idempotency_key}", "1", ex=86400)
    return result
```

2. **预算预留 vs 实际消耗**：`check_and_reserve` 是基于估算的预留，实际消耗可能不同。务必在请求完成后调用 `record_actual_usage` 修正差异，否则预算会逐渐偏差。

3. **Temporal 的长时间等待**：Agent 工作流中如果需要等待人工审批，不要用 `time.sleep`——Temporal 支持原生的 `workflow.wait_condition`，可以等几小时甚至几天而不消耗计算资源。

4. **Celery 的结果后端开销**：如果不需要查询每个任务的结果（如纯异步执行），可以设置 `task_ignore_result=True` 减少存储开销。

5. **成本数据的精度**：`incrbyfloat` 在高并发下可能有浮点精度问题。生产环境建议用整数（以 0.001 美分 = 1 个单位），只在展示时转换。

6. **告警风暴**：高频请求场景下，避免同一告警反复触发。使用 Redis 的 `SET NX EX` 实现告警去重：

```python
def send_alert_dedup(alert_type: str, message: str, dedup_window: int = 300):
    """去重告警"""
    key = f"alert:dedup:{alert_type}"
    if redis.set(key, "1", nx=True, ex=dedup_window):
        # 只在去重窗口内发送一次
        _actual_send_alert(alert_type, message)
```

---

## 小结

| 概念 | 说明 |
|------|------|
| Celery | 成熟的任务队列，适合简单 Agent 场景 |
| Temporal | 工作流编排引擎，适合复杂 Agent 工作流 |
| Token 预算 | 分层控制：请求级 / 会话级 / 用户级 / 全局级 |
| 大小模型路由 | 按复杂度 + 预算动态选择模型，成本降 60%-80% |
| 成本监控 | Prometheus 指标 + Redis 聚合 + 多级告警 |
| 幂等性 | 工具调用必须幂等，防止重试导致副作用 |

> 🎓 **本章总结**：从部署架构到推理服务化，从 K8s 编排到 Serverless GPU，从任务队列到成本治理，我们完成了 Agent 从"能跑的代码"到"可控的生产服务"的完整进化。部署不是终点，而是持续优化的起点——监控、告警、成本治理，是长期运营的核心。

---

[19.7 Kubernetes 编排与 Serverless GPU](./07_k8s_serverless.md)
