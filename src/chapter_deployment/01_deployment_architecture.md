# Agent 应用的部署架构

> **本节目标**：理解 Agent 从开发环境到生产环境需要哪些架构变化。

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

```python
"""
生产级 Agent 架构示意

┌─────────────────────────────────────────────────┐
│                   客户端层                        │
│   Web 浏览器 / App / API 客户端                   │
└────────────────────┬────────────────────────────┘
                     │ HTTPS
┌────────────────────▼────────────────────────────┐
│               负载均衡层 (Nginx)                  │
│   SSL 终止 / 请求路由 / 限流                      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             API 服务层 (FastAPI)                  │
│   认证 / 请求验证 / 会话管理                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ 实例 1   │  │ 实例 2   │  │ 实例 3   │      │
│   └──────────┘  └──────────┘  └──────────┘      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              Agent 核心层                        │
│   LLM 调用 / 工具执行 / 记忆管理                  │
└──────┬──────────────┬────────────┬──────────────┘
       │              │            │
  ┌────▼────┐   ┌─────▼─────┐  ┌──▼──────────┐
  │  Redis  │   │ 向量数据库 │  │ LLM API     │
  │ (会话)  │   │ (记忆)     │  │ (OpenAI等)  │
  └─────────┘   └───────────┘  └─────────────┘
"""
```

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
    openai_model: str = "gpt-4o"
    
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

> **下一节预告**：接下来我们用 FastAPI 把 Agent 封装成一个完整的 API 服务。

---

[下一节：15.2 API 服务化：FastAPI / Flask 封装 →](./02_api_service.md)
