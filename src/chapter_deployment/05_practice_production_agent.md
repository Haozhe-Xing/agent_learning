# 实战：部署一个生产级 Agent 服务

> **本节目标**：综合运用本章所学知识，完成一个 Agent 服务从开发到部署的完整流程。

---

## 项目结构

```
agent-service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口
│   ├── agent.py          # Agent 核心逻辑
│   ├── config.py         # 配置管理
│   └── middleware.py     # 中间件
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── tests/
    └── test_api.py
```

---

## 核心代码

### config.py —— 配置管理

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o"
    redis_url: str = "redis://localhost:6379"
    api_keys: str = ""  # 逗号分隔的有效 API Key
    max_concurrent: int = 50
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"

settings = Settings()
```

### agent.py —— Agent 核心

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings

class ProductionAgent:
    """生产级 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            streaming=True
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的AI助手。请准确、简洁地回答问题。"),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm
    
    async def run(self, message: str, history: list[dict] = None):
        """执行 Agent（非流式）"""
        chat_history = self._build_history(history or [])
        response = await self.chain.ainvoke({
            "input": message,
            "history": chat_history
        })
        return response.content
    
    async def stream(self, message: str, history: list[dict] = None):
        """执行 Agent（流式）"""
        chat_history = self._build_history(history or [])
        async for chunk in self.chain.astream({
            "input": message,
            "history": chat_history
        }):
            if chunk.content:
                yield chunk.content
    
    def _build_history(self, history: list[dict]):
        """构建对话历史"""
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

# 全局单例
agent = ProductionAgent()
```

### main.py —— API 入口

```python
import uuid
import json
import asyncio

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.config import settings
from app.agent import agent

app = FastAPI(title="Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 并发控制
semaphore = asyncio.Semaphore(settings.max_concurrent)

# ===== 模型 =====

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ===== 认证 =====

async def verify_key(x_api_key: str = Header(...)):
    valid = settings.api_keys.split(",")
    if x_api_key not in valid:
        raise HTTPException(401, "无效的 API Key")

# ===== 端点 =====

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _=Depends(verify_key)):
    session_id = req.session_id or str(uuid.uuid4())
    
    async with semaphore:
        try:
            reply = await asyncio.wait_for(
                agent.run(req.message),
                timeout=settings.request_timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "请求超时")
    
    return ChatResponse(reply=reply, session_id=session_id)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, _=Depends(verify_key)):
    async def generate():
        async with semaphore:
            async for token in agent.stream(req.message):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=4)
```

---

## 部署步骤

### 1. 准备环境变量

```bash
# ⚠️ 以下是 .env.example 文件的模板
# 请复制为 .env 并填入真实值：cp .env.example .env
# 🔒 安全提醒：.env 文件必须加入 .gitignore，切勿提交到版本控制！

AGENT_OPENAI_API_KEY=sk-your-key-here
AGENT_API_KEYS=key1,key2,key3
AGENT_MODEL_NAME=gpt-4o
AGENT_REDIS_URL=redis://redis:6379
```

### 2. 构建并启动

```bash
# 构建镜像并启动
docker compose up -d --build

# 检查服务状态
docker compose ps

# 验证健康检查
curl http://localhost:8000/health
```

### 3. 测试 API

![测试金字塔](../svg/chapter_deployment_05_test_pyramid.svg)

```bash
# 普通对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"message": "你好，请介绍一下 Python"}'

# 流式对话
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"message": "讲一个简短的故事"}' \
  --no-buffer
```

---

## 部署检查清单

| 检查项 | 说明 | ✅ |
|--------|------|---|
| 环境变量 | API Key 等敏感信息不硬编码 | |
| 健康检查 | /health 端点正常返回 | |
| 认证 | API Key 验证生效 | |
| 限流 | Nginx 限流配置正确 | |
| 日志 | 请求日志正常记录 | |
| 监控 | 错误率、延迟可观测 | |
| 备份 | Redis 数据持久化 | |
| SSL | HTTPS 证书配置 | |

---

## 小结

| 概念 | 说明 |
|------|------|
| 项目结构 | 清晰分层：配置、核心、API、中间件 |
| 配置管理 | Pydantic Settings + 环境变量 |
| 并发控制 | Semaphore + 超时机制 |
| 流式响应 | SSE 实时推送生成过程 |
| 容器部署 | Docker Compose 一键启动 |

> 🎓 **本章总结**：从 API 封装到容器化部署，从流式响应到并发处理，我们走完了 Agent 从"能运行的脚本"到"生产级服务"的完整路径。下一步，让我们进入综合项目篇，构建真实的 Agent 应用！

---

[下一章：第16章 项目实战：AI 编程助手 →](../chapter_coding_agent/README.md)
