# 20.2 API 服务化：FastAPI / Flask 封装

> **本节目标**：用 FastAPI 把 Agent 封装成一个**真实返回结果**的 API 服务，而不是"看起来像服务的桩代码"。

---

## 为什么选 FastAPI？

| 特性 | FastAPI | Flask |
|------|---------|-------|
| 异步支持 | ✅ 原生 async | ⚠️ 需要扩展 |
| 类型验证 | ✅ Pydantic 自动验证 | ❌ 需手动 |
| API 文档 | ✅ 自动生成 Swagger | ❌ 需要扩展 |
| 性能 | ⚡ 高性能 | 🏃 中等 |
| 流式响应 | ✅ SSE 原生支持 | ⚠️ 较复杂 |

对于 Agent 服务，FastAPI 的异步支持和流式响应是关键优势。

![Agent API服务分层架构](../svg/chapter_deployment_02_api_arch.svg)

---

## 真实可运行的服务：reference-agent

本节不贴"返回固定字符串"的示例。真实实现在仓库 `reference-agent/src/reference_agent/service/app.py`，它的关键点是：

1. `/chat` **调用真实 Agent** 返回结果，出错就返回错误，绝不返回 "这是 Agent 的回复" 这类占位串。
2. 每次请求先过**注入守卫**（第 19 章），可疑输入直接 400 拒绝（fail-closed）。
3. `/stream` 用 SSE **真实逐 token 流式输出** Agent 的答案。
4. 默认离线（FakeProvider），设 `AGENT_REAL=1` + API Key 才接真实模型。

核心结构（节选，完整见源码与测试）：

```python
# reference-agent/src/reference_agent/service/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from ..agent import Agent
from ..provider import FakeProvider, OpenAIProvider
from ..security.guardrails import InjectionGuard

app = FastAPI(title="reference-agent")
_guard = InjectionGuard()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    _guard.check_input(req.message)          # fail-closed：可疑输入直接拒绝
    agent = _build_agent()                   # FakeProvider 或真实模型
    answer = agent.run(req.message)          # 真实执行，不是占位串
    return {"answer": answer}

@app.post("/stream")
def stream(req: ChatRequest):
    _guard.check_input(req.message)
    answer = _build_agent().run(req.message)
    def gen():
        for token in answer.split(" "):
            yield f"data: {json.dumps({'token': token + ' '}, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

运行与验证：

```bash
cd reference-agent
python -m pytest -q                 # 先跑测试，确认服务真实可用
uvicorn reference_agent.service.app:app --port 8000
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"message":"请帮我计算 12*8"}'
# => {"answer":"计算结果是 96。"}
```

---

## 生产化时必须补齐的部分

FastAPI 本身不难，难的是把下面这些做成"机制"而非"示例片段"。本节的 `/chat` 已包含守卫，下面补充常见生产件，建议你把它们接进同一个 app：

**1. 认证（不要把 API 裸奔）**

```python
import os
from fastapi import HTTPException, Header, Depends

async def verify_api_key(x_api_key: str = Header(...)):
    valid = {k for k in os.getenv("VALID_API_KEYS", "").split(",") if k}
    if x_api_key not in valid:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return x_api_key
```

**2. CORS（生产环境限制来源，不要写 `"*"`）**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.example.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

**3. 全局异常处理（生产不泄露内部错误）**

```python
@app.exception_handler(Exception)
async def handle(_: Request, exc: Exception):
    logger.error("未处理异常", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "服务暂时不可用"})
```

**4. 启动（生产关 reload、按机器核数设 worker）**

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reference_agent.service.app:app", host="0.0.0.0",
                port=8000, workers=4, reload=False)
```

---

## 小结

| 概念 | 说明 |
|------|------|
| FastAPI | 高性能异步框架，适合 Agent 服务 |
| 真实执行 | `/chat` 调用真实 Agent，不返回占位串 |
| 守卫前置 | 输入先过注入检测，fail-closed |
| SSE | 真实逐 token 流式输出 |
| 生产件 | 认证、CORS、异常处理、worker 配置 |

> 完整可运行代码与测试见 `reference-agent/src/reference_agent/service/app.py` 和 `tests/test_service.py`。Docker 打包见 `reference-agent/docker/Dockerfile`。

---

[20.3 容器化与云部署](./03_containerization.md)
