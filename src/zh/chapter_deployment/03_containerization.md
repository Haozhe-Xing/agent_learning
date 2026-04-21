# 20.3 容器化与云部署

> **本节目标**：学会用 Docker 打包 Agent 服务，实现"一次构建，到处运行"。

---

## 为什么需要容器化？

![Agent 服务容器化部署流程](../svg/chapter_deployment_03_docker_flow.svg)

"在我电脑上能跑"是开发者最经典的问题。Docker 解决了这个问题——把代码、依赖、配置打包成一个镜像，在任何环境都能一致运行。

---

## 编写 Dockerfile

```dockerfile
# ===== 构建阶段 =====
FROM python:3.11-slim AS builder

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== 运行阶段 =====
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /install /usr/local

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash agent
USER agent

# 复制应用代码
COPY --chown=agent:agent . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### requirements.txt

```
fastapi==0.115.6
uvicorn[standard]==0.32.0
langchain==0.3.14
langchain-openai==0.3.6
redis==5.2.1
pydantic-settings==2.7.1
```

---

## Docker Compose 编排

一个完整的 Agent 服务通常需要多个组件协同工作：

```yaml
# docker-compose.yml
# 注意：Docker Compose V2 已废弃顶层 version 字段，直接从 services 开始即可

services:
  # Agent API 服务
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AGENT_OPENAI_API_KEY=${OPENAI_API_KEY}
      - AGENT_REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "2.0"
    restart: unless-stopped
  
  # Redis（会话存储）
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
  
  # Nginx（反向代理）
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - agent-api
    restart: unless-stopped

volumes:
  redis_data:
```

### Nginx 配置

```nginx
# nginx.conf
upstream agent_backend {
    server agent-api:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # 限流：每个 IP 每秒最多 10 个请求
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # SSE 流式响应支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
    
    location /health {
        proxy_pass http://agent_backend;
        # 健康检查不限流
    }
}
```

---

## 镜像优化最佳实践

生产环境中，Docker 镜像的体积和构建速度直接影响部署效率：

```dockerfile
# === 优化技巧 1：精确控制 COPY ===
# 使用 .dockerignore 排除无关文件
# .dockerignore 内容：
# __pycache__/
# *.pyc
# .env
# .git/
# tests/
# docs/
# *.md

# === 优化技巧 2：固定基础镜像版本 ===
# 不要用 python:3.11-slim，而是用带 hash 的精确版本
# 确保构建的可重复性
FROM python:3.11.9-slim-bookworm AS builder

# === 优化技巧 3：合并 RUN 命令减少层数 ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*
```

### 镜像体积对比

| 方案 | 镜像大小 | 说明 |
|------|---------|------|
| `python:3.11` | ~900MB | 完整版，包含编译工具 |
| `python:3.11-slim` | ~120MB | 精简版，推荐生产使用 |
| 多阶段构建 | ~80MB | 构建和运行分离 |
| Alpine 基础 | ~50MB | 最小但可能有兼容问题 |

---

## 环境变量与密钥管理

Agent 服务需要管理 API Key 等敏感配置，**绝对不能硬编码在镜像中**：

```python
# config.py — 使用 pydantic-settings 管理配置
from pydantic_settings import BaseSettings
from pydantic import Field

class AgentConfig(BaseSettings):
    """Agent 服务配置（从环境变量读取）"""
    
    # API Keys（必须从环境变量注入，不设默认值）
    openai_api_key: str = Field(..., alias="AGENT_OPENAI_API_KEY")
    
    # 服务配置
    host: str = Field(default="0.0.0.0", alias="AGENT_HOST")
    port: int = Field(default=8000, alias="AGENT_PORT")
    workers: int = Field(default=4, alias="AGENT_WORKERS")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="AGENT_REDIS_URL")
    
    # 安全配置
    max_tokens_per_request: int = Field(default=4000, alias="AGENT_MAX_TOKENS")
    rate_limit_per_minute: int = Field(default=60, alias="AGENT_RATE_LIMIT")
    
    # 日志级别
    log_level: str = Field(default="INFO", alias="AGENT_LOG_LEVEL")
    
    class Config:
        env_prefix = ""  # 不自动添加前缀，使用 alias
        case_sensitive = False

# 使用
config = AgentConfig()  # 自动从环境变量读取
```

### Docker Compose 中注入密钥

```yaml
services:
  agent-api:
    build: .
    environment:
      # 方式1：从 .env 文件读取（开发环境）
      - AGENT_OPENAI_API_KEY=${OPENAI_API_KEY}
    
    # 方式2：使用 Docker Secrets（生产环境更安全）
    # secrets:
    #   - openai_api_key
    
    # 方式3：从外部密钥管理服务读取
    # 如 AWS Secrets Manager、HashiCorp Vault
```

> ⚠️ **安全提醒**：永远不要将 `.env` 文件提交到 Git 仓库。在 `.gitignore` 中添加 `.env`。

---

## 日志与监控

容器化 Agent 服务的日志管理至关重要——你需要能够追溯每一次 Agent 的决策过程：

```python
import logging
import json
from datetime import datetime

class AgentLogger:
    """结构化日志记录器"""
    
    def __init__(self, service_name: str = "agent-api"):
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # 输出 JSON 格式日志（方便 ELK/Loki 等日志系统采集）
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "service": "agent-api",
            }
            if hasattr(record, "extra_data"):
                log_data.update(record.extra_data)
            return json.dumps(log_data, ensure_ascii=False)
    
    def log_request(self, request_id: str, user_input: str, model: str):
        self.logger.info(
            "Agent request received",
            extra={"extra_data": {
                "request_id": request_id,
                "input_length": len(user_input),
                "model": model,
                "event": "request_start",
            }}
        )
    
    def log_tool_call(self, request_id: str, tool_name: str, duration_ms: float):
        self.logger.info(
            f"Tool called: {tool_name}",
            extra={"extra_data": {
                "request_id": request_id,
                "tool": tool_name,
                "duration_ms": duration_ms,
                "event": "tool_call",
            }}
        )
    
    def log_response(self, request_id: str, tokens_used: int, duration_ms: float):
        self.logger.info(
            "Agent response sent",
            extra={"extra_data": {
                "request_id": request_id,
                "tokens_used": tokens_used,
                "duration_ms": duration_ms,
                "event": "request_complete",
            }}
        )
```

---

## 一键启动

```bash
# 构建并启动所有服务
docker compose up -d --build

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f agent-api

# 停止所有服务
docker compose down
```

---

## 云平台部署选项

| 平台 | 适用场景 | 优势 |
|------|---------|------|
| AWS ECS / Fargate | 企业级部署 | 自动扩缩容、集成 AWS 生态 |
| Google Cloud Run | 无服务器部署 | 按请求计费、自动扩缩 |
| 阿里云容器服务 | 国内部署 | 低延迟、合规 |
| Railway / Render | 快速原型 | 简单、免运维 |

---

## 小结

| 概念 | 说明 |
|------|------|
| Dockerfile | 多阶段构建，减小镜像体积 |
| Docker Compose | 编排多个服务协同工作 |
| Nginx | 反向代理、限流、SSL |
| 健康检查 | 自动检测服务状态 |

> **下一节预告**：部署好了服务，还需要处理流式响应和高并发的问题。

---

[下一节：18.4 流式响应与并发处理 →](./04_streaming_concurrency.md)
