# 19.1 Agent 应用的部署架构

> **本节目标**：理解 Agent 从开发环境到生产环境需要哪些架构变化，掌握 Agent 特有的部署挑战与解决方案。

---

## Agent 部署的独特挑战

部署一个 Agent 应用与部署传统 Web 服务有本质区别。传统 API 的请求处理时间通常在毫秒级，行为可预测；而 Agent 的请求处理更像一个"迷你程序的执行"——时间不确定、步骤不确定、资源消耗不确定。

### 五大独特挑战

**挑战一：不确定的执行时间**

一个简单问答可能 2 秒完成，而一个涉及多步推理 + 工具调用的请求可能需要 30 秒甚至数分钟。这使得传统的请求超时设置和负载均衡策略不再适用。

```python
# 传统 API：执行时间可预测
@app.get("/users/{id}")
async def get_user(id: int):  # 延迟：~50ms，标准差 ~10ms
    return db.get_user(id)

# Agent API：执行时间不可预测
@app.post("/agent/chat")
async def agent_chat(msg: str):  # 延迟：2s-120s，标准差 ~30s
    return await agent.process(msg)
```

**挑战二：工具调用的副作用**

Agent 不只是"回答问题"，它可能发邮件、写文件、调用外部 API、执行代码。这意味着一个失败的请求重试可能导致副作用重复执行（如重复发送邮件）。

**挑战三：长上下文的状态管理**

Agent 的对话可能持续数十轮，携带数万 Token 的上下文。这些状态需要在多个服务实例之间共享，且不能丢失。

**挑战四：成本不可控**

一个复杂请求可能触发多次 LLM 调用（推理 + 工具选择 + 结果总结），单次请求成本可能从 $0.01 到 $1.00 不等。如果不加限制，一个恶意用户可以快速耗尽预算。

**挑战五：可观测性困难**

传统服务的日志是线性的（请求 → 处理 → 响应），而 Agent 的执行是树状或图状的（推理 → 工具A → 子推理 → 工具B → 回溯 → 工具C → 最终回复）。标准的日志和监控工具难以捕捉这种复杂的执行路径。

---

## 三种架构模式对比

根据团队规模、流量需求和成本预算，Agent 应用的部署可以选择三种架构模式：

### 模式一：单体架构

所有组件（API、Agent 逻辑、工具、存储）运行在同一进程中。

```
[Nginx] → [FastAPI + Agent + Tools + DB]
```

**适用场景**：个人项目、原型验证、< 100 QPS

**优点**：开发简单、部署方便、调试容易

**缺点**：无法独立扩展各组件、单点故障

### 模式二：微服务架构

将 Agent 拆分为多个独立服务，通过消息队列或 RPC 通信。

> `[API Gateway]` → `[API Service]` → `[Agent Service]` → `[Tool Service]`
>
> 各服务分别连接：Redis / Vector DB / External APIs

**适用场景**：团队协作、100-10000 QPS、需要独立扩展

**优点**：各组件独立部署和扩展、故障隔离、团队并行开发

**缺点**：运维复杂度高、网络延迟增加、需要分布式追踪

### 模式三：Serverless 架构

使用云函数（AWS Lambda、Cloudflare Workers）按需执行，无需管理服务器。

> `[API Gateway]` → `[Lambda: API Handler]` → `[Lambda: Agent Logic]` → `[Managed Services]`

**适用场景**：流量波动大、按量付费、< 1000 QPS

**优点**：零运维、自动扩缩容、按实际调用付费

**缺点**：冷启动延迟（1-5s）、执行时间限制（通常 15 分钟）、调试困难

### 三种模式量化对比

| 维度 | 单体 | 微服务 | Serverless |
|------|------|--------|-----------|
| 运维复杂度 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 扩展能力 | 垂直扩展 | 水平+垂直 | 自动 |
| 冷启动延迟 | 无 | 无 | 1-5s |
| 最大请求时长 | 无限制 | 无限制 | 15min |
| 月成本（低流量） | ~$20 | ~$100+ | ~$5 |
| 月成本（高流量） | ~$200 | ~$500+ | ~$300 |
| 适合阶段 | MVP/原型 | 增长期 | 探索期/波动流量 |

> 💡 **推荐路径**：大多数 Agent 项目建议从**单体架构**起步，验证产品价值后再迁移到微服务。过早的架构拆分是常见的过度工程陷阱。

---

## Agent 的可观测性需求

传统的"请求日志 + 错误率 + 响应时间"三板斧对 Agent 应用远远不够。你还需要追踪：

| 观测维度 | 传统 API | Agent 应用 |
|---------|---------|-----------|
| 请求链路 | 单层 | 多层嵌套（推理 → 工具 → 子推理） |
| Token 用量 | 无 | 每次 LLM 调用的 input/output tokens |
| 工具调用记录 | 无 | 调用了哪些工具、参数、结果 |
| 推理质量 | 无 | Agent 是否正确理解了用户意图 |
| 成本追踪 | 固定 | 每请求动态成本 |

推荐的可观测性工具栈：

- **LangSmith**：LangChain 官方的 Agent 追踪平台，可视化完整的推理链路
- **Phoenix (Arize)**：开源的 LLM 可观测性工具，支持 Trace 和 Evaluation
- **OpenTelemetry + 自定义 Span**：在每个 Agent 步骤中埋入追踪点

```python
from opentelemetry import trace

tracer = trace.get_tracer("agent-service")

async def agent_process(question: str):
    with tracer.start_as_current_span("agent.process") as span:
        span.set_attribute("user.question", question)
        
        with tracer.start_as_current_span("agent.text_to_sql"):
            sql = await text2sql.convert(question)
            span.set_attribute("generated.sql", sql)
        
        with tracer.start_as_current_span("agent.execute_query"):
            data = db.execute_readonly(sql)
            span.set_attribute("result.row_count", len(data))
        
        # ... 后续步骤
```

---

## 开发环境 vs 生产环境

![Agent 生产部署架构](../svg/chapter_deployment_01_architecture.svg)

在开发阶段，你可能直接在终端运行一个 Python 脚本。但要服务真实用户，需要考虑更多问题：

| 维度 | 开发环境 | 生产环境 |
|------|---------|---------|
| 并发 | 单用户 | 数百/数千并发 |
| 可用性 | 随时可以重启 | 7×24 不间断运行 |
| 错误处理 | print 看日志 | 结构化日志 + 告警 |
| 状态管理 | 内存中的字典 | Redis / 数据库 |
| 密钥管理 | .env 文件 | 密钥管理服务 |
| 监控 | 手动观察 | Prometheus + Grafana |

---

## 典型的生产部署架构

<!-- 架构图见上方 SVG -->

### 核心组件说明

```python
from dataclasses import dataclass

@dataclass
class ProductionArchitecture:
    """生产架构各层说明"""
    
    layers = {
        "负载均衡": {
            "工具": "Nginx / AWS ALB / Cloudflare",
            "职责": [
                "SSL/TLS 加密终止",
                "请求分发到多个 API 实例",
                "限流和防 DDoS",
                "健康检查（自动剔除故障实例）"
            ]
        },
        "API 服务": {
            "工具": "FastAPI / Flask",
            "职责": [
                "接收和验证用户请求",
                "API Key / JWT 认证",
                "会话管理（从 Redis 读写）",
                "请求日志记录",
                "流式响应（SSE）"
            ]
        },
        "Agent 核心": {
            "工具": "LangChain / LangGraph / 自研",
            "职责": [
                "理解用户意图",
                "调用 LLM 进行推理",
                "执行工具调用",
                "管理对话上下文"
            ]
        },
        "存储层": {
            "Redis": "会话状态、缓存、限流计数器",
            "向量数据库": "长期记忆、知识库检索",
            "关系数据库": "用户信息、使用记录、审计日志"
        }
    }
```

---

## 状态管理：从内存到 Redis

开发时用内存字典存对话，生产中要用 Redis：

```python
import json
from datetime import datetime

class RedisSessionManager:
    """基于 Redis 的会话管理（生产级）"""
    
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl  # 会话过期时间（秒）
    
    def get_history(self, session_id: str) -> list[dict]:
        """获取对话历史"""
        key = f"session:{session_id}:messages"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        return []
    
    def add_message(
        self, session_id: str, role: str, content: str
    ):
        """添加消息到会话"""
        key = f"session:{session_id}:messages"
        history = self.get_history(session_id)
        
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保留最近的消息，避免无限增长
        if len(history) > 50:
            history = history[-50:]
        
        self.redis.set(key, json.dumps(history, ensure_ascii=False))
        self.redis.expire(key, self.ttl)
    
    def clear_session(self, session_id: str):
        """清除会话"""
        self.redis.delete(f"session:{session_id}:messages")
```

---

## 配置管理

生产环境的配置需要更加规范：

```python
from pydantic_settings import BaseSettings

class AgentConfig(BaseSettings):
    """Agent 生产配置（从环境变量加载）"""
    
    # API 配置
    openai_api_key: str
    openai_model: str = "gpt-4.1"
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Redis 配置
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 3600
    
    # 限流配置
    rate_limit_per_minute: int = 60
    
    # 安全配置
    api_key_header: str = "X-API-Key"
    cors_origins: list[str] = ["*"]
    
    # Agent 配置
    max_steps: int = 10
    max_tokens: int = 4096
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"  # 所有环境变量以 AGENT_ 开头

# 使用：
# 环境变量 AGENT_OPENAI_API_KEY=sk-xxx 自动加载
# config = AgentConfig()
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 分层架构 | 负载均衡 → API服务 → Agent核心 → 存储 |
| 状态外置 | 用 Redis 替代内存字典管理会话 |
| 配置管理 | 从环境变量加载配置，不硬编码 |
| 水平扩展 | 多实例部署，无状态 API 层 |

> 💡 **延伸阅读**：关于模型路由评估和 A/B 测试，详见 [17.7 A/B 测试与回归测试自动化](../chapter_evaluation/07_ab_testing.md) 和 [17.8 模型路由评估](../chapter_evaluation/08_model_routing.md)。

> **下一节预告**：接下来我们用 FastAPI 把 Agent 封装成一个完整的 API 服务。

---

[19.2 API 服务化：FastAPI / Flask 封装](./02_api_service.md)
