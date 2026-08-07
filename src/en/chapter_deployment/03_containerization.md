# 20.3 Containerization and Cloud Deployment

> **Section Goal**: Learn how to package an Agent service with Docker to achieve "build once, run anywhere."

---

## Why Containerization?

![Agent Service Containerization Deployment Flow](../svg/chapter_deployment_03_docker_flow.svg)

"It works on my machine" is the most classic developer problem. Docker solves this — packaging code, dependencies, and configuration into an image that runs consistently in any environment.

---

## Writing a Dockerfile

```dockerfile
# ===== Build Stage =====
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Runtime Stage =====
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy dependencies from build stage
COPY --from=builder /install /usr/local

# Create non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash agent
USER agent

# Copy application code
COPY --chown=agent:agent . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start command
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

## Docker Compose Orchestration

A complete Agent service typically requires multiple components working together:

```yaml
# docker-compose.yml
# Note: Docker Compose V2 has deprecated the top-level version field; start directly from services

services:
  # Agent API service
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
  
  # Redis (session storage)
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
  
  # Nginx (reverse proxy)
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

### Nginx Configuration

```nginx
# nginx.conf
upstream agent_backend {
    server agent-api:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Rate limiting: max 10 requests per second per IP
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # SSE streaming response support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
    
    location /health {
        proxy_pass http://agent_backend;
        # No rate limiting for health checks
    }
}
```

---

## Image Optimization Best Practices

In production, Docker image size and build speed directly affect deployment efficiency:

```dockerfile
# === Optimization Tip 1: Precise COPY control ===
# Use .dockerignore to exclude irrelevant files
# .dockerignore contents:
# __pycache__/
# *.pyc
# .env
# .git/
# tests/
# docs/
# *.md

# === Optimization Tip 2: Pin base image versions ===
# Don't use python:3.11-slim; use an exact version with hash
# Ensures build reproducibility
FROM python:3.11.9-slim-bookworm AS builder

# === Optimization Tip 3: Merge RUN commands to reduce layers ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*
```

### Image Size Comparison

| Approach | Image Size | Notes |
|----------|-----------|-------|
| `python:3.11` | ~900MB | Full version, includes build tools |
| `python:3.11-slim` | ~120MB | Slim version, recommended for production |
| Multi-stage build | ~80MB | Build and runtime separated |
| Alpine base | ~50MB | Smallest but may have compatibility issues |

---

## Environment Variables and Secret Management

Agent services need to manage sensitive configuration like API Keys — **never hardcode them in the image**:

```python
# config.py — use pydantic-settings to manage configuration
from pydantic_settings import BaseSettings
from pydantic import Field

class AgentConfig(BaseSettings):
    """Agent service configuration (read from environment variables)"""
    
    # API Keys (must be injected via environment variables, no default value)
    openai_api_key: str = Field(..., alias="AGENT_OPENAI_API_KEY")
    
    # Service configuration
    host: str = Field(default="0.0.0.0", alias="AGENT_HOST")
    port: int = Field(default=8000, alias="AGENT_PORT")
    workers: int = Field(default=4, alias="AGENT_WORKERS")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="AGENT_REDIS_URL")
    
    # Security configuration
    max_tokens_per_request: int = Field(default=4000, alias="AGENT_MAX_TOKENS")
    rate_limit_per_minute: int = Field(default=60, alias="AGENT_RATE_LIMIT")
    
    # Log level
    log_level: str = Field(default="INFO", alias="AGENT_LOG_LEVEL")
    
    class Config:
        env_prefix = ""  # No auto prefix, use alias
        case_sensitive = False

# Usage
config = AgentConfig()  # Automatically reads from environment variables
```

### Injecting Secrets in Docker Compose

```yaml
services:
  agent-api:
    build: .
    environment:
      # Method 1: Read from .env file (development environment)
      - AGENT_OPENAI_API_KEY=${OPENAI_API_KEY}
    
    # Method 2: Use Docker Secrets (more secure for production)
    # secrets:
    #   - openai_api_key
    
    # Method 3: Read from external secret management service
    # e.g., AWS Secrets Manager, HashiCorp Vault
```

> ⚠️ **Security reminder**: Never commit `.env` files to Git repositories. Add `.env` to `.gitignore`.

---

## Logging and Monitoring

Log management for containerized Agent services is critical — you need to be able to trace every Agent decision:

```python
import logging
import json
from datetime import datetime

class AgentLogger:
    """Structured logger"""
    
    def __init__(self, service_name: str = "agent-api"):
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # Output JSON format logs (easy for ELK/Loki to collect)
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

## One-Command Startup

```bash
# Build and start all services
docker compose up -d --build

# Check service status
docker compose ps

# View logs
docker compose logs -f agent-api

# Stop all services
docker compose down
```

---

## Cloud Platform Deployment Options

| Platform | Use Case | Advantages |
|----------|----------|-----------|
| AWS ECS / Fargate | Enterprise deployment | Auto-scaling, integrated AWS ecosystem |
| Google Cloud Run | Serverless deployment | Pay per request, auto-scaling |
| Alibaba Cloud Container Service | Domestic deployment | Low latency, compliance |
| Railway / Render | Rapid prototyping | Simple, zero operations |

---

## Summary

| Concept | Description |
|---------|-------------|
| Dockerfile | Multi-stage build to reduce image size |
| Docker Compose | Orchestrate multiple services working together |
| Nginx | Reverse proxy, rate limiting, SSL |
| Health Check | Automatically detect service status |

> **Next Section Preview**: With the service deployed, we still need to handle streaming responses and high concurrency.

---

[Next: 18.4 Streaming Responses and Concurrency Handling →](./04_streaming_concurrency.md)