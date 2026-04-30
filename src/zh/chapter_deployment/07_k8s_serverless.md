# 19.7 Kubernetes 编排与 Serverless GPU

> **本节目标**：学会用 Kubernetes 编排完整的 Agent 服务栈，掌握 Serverless GPU 平台（Modal / RunPod）的使用方法，理解 GPU 工作负载的自动伸缩策略。

---

## 为什么需要 K8s 编排？

当 Agent 应用从单机部署走向生产级服务时，你面临的问题不再是"能不能跑"，而是：

1. **多组件协同**：推理服务、API 网关、Redis、向量数据库需要统一编排
2. **GPU 资源调度**：GPU 是稀缺资源，需要精准调度和共享
3. **弹性伸缩**：流量波动大，需要根据负载自动扩缩容
4. **故障恢复**：单点故障不应影响整体服务可用性

Docker Compose 适合单机开发，但生产环境需要 Kubernetes。

---

## Agent 服务的 K8s 架构

```
[Ingress / API Gateway]
         │
    ┌────┴────┐
    │         │
[API Service] [API Service]  ← 无状态，水平扩展
    │         │
    └────┬────┘
         │
   ┌─────┼──────┐
   │     │      │
[Inference] [Redis] [Vector DB]  ← 有状态，持久化
 Service    (StatefulSet)  (StatefulSet)
(GPU Pod)
```

---

## 完整的 K8s 部署清单

### 命名空间与 GPU 资源

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-prod
  labels:
    env: production
```

```yaml
# gpu-resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: agent-prod
spec:
  hard:
    requests.nvidia.com/gpu: "8"    # 最多 8 块 GPU
    limits.nvidia.com/gpu: "8"
    requests.cpu: "32"
    requests.memory: 64Gi
```

### API 服务 Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  namespace: agent-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
        - name: agent-api
          image: your-registry/agent-api:v1.2.0  # 锁定版本
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "1"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "2Gi"
          env:
            - name: AGENT_OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: openai-api-key
            - name: AGENT_MODEL_NAME
              value: "gpt-4.1"
            - name: AGENT_REDIS_URL
              value: "redis://redis:6379"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      topologySpreadConstraints:   # 跨可用区分布
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: agent-api
```

### API 服务 Service 与 HPA

```yaml
# api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-api
  namespace: agent-prod
spec:
  selector:
    app: agent-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

```yaml
# api-hpa.yaml — API 层自动伸缩
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
  namespace: agent-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容需稳定 5 分钟
      policies:
        - type: Percent
          value: 25
          periodSeconds: 120
```

### Redis StatefulSet

```yaml
# redis-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: agent-prod
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          resources:
            requests:
              cpu: "0.5"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
          volumeMounts:
            - name: redis-data
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: redis-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

### Ingress 配置

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-ingress
  namespace: agent-prod
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Content-Type-Options: nosniff";
spec:
  tls:
    - hosts:
        - agent.your-domain.com
      secretName: agent-tls
  rules:
    - host: agent.your-domain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: agent-api
                port:
                  number: 80
```

### Secret 管理

```yaml
# secrets.yaml — 使用外部密钥管理工具（如 Sealed Secrets / External Secrets Operator）
# 以下仅为示意，实际应使用加密方案
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: agent-prod
type: Opaque
stringData:
  openai-api-key: "sk-your-key-here"  # 实际中应从 Vault/Sealed Secrets 注入
```

---

## GPU 工作负载的自动伸缩

GPU Pod 的伸缩比 CPU Pod 复杂得多——GPU 设备不能被多个 Pod 共享（通常），冷启动时间长（模型加载需 30-120 秒），且费用昂贵。因此 GPU 伸缩需要更谨慎的策略。

### 基于队列长度的 GPU 伸缩

```python
"""
GPU 推理服务的自定义伸缩指标
基于请求队列长度决定是否扩缩 GPU Pod
"""
import time
from prometheus_client import Gauge
from kubernetes import client, config

# 自定义指标：等待中的请求数
PENDING_REQUESTS = Gauge(
    "inference_pending_requests",
    "Number of pending inference requests"
)

class GPUAutoscaler:
    """GPU 推理服务自定义伸缩器"""

    def __init__(self, namespace: str = "agent-prod",
                 deployment: str = "vllm-qwen72b"):
        config.load_incluster_config()
        self.apps_api = client.AppsV1Api()
        self.namespace = namespace
        self.deployment = deployment

        # 伸缩阈值
        self.scale_up_threshold = 10     # 等待请求 > 10，扩容
        self.scale_down_threshold = 2    # 等待请求 < 2，缩容
        self.min_replicas = 1
        self.max_replicas = 4
        self.cooldown_seconds = 120      # 伸缩冷却期

        self.last_scale_time = 0

    def get_current_replicas(self) -> int:
        """获取当前副本数"""
        deploy = self.apps_api.read_namespaced_deployment(
            name=self.deployment, namespace=self.namespace
        )
        return deploy.spec.replicas

    def scale(self, target_replicas: int):
        """调整副本数"""
        target_replicas = max(self.min_replicas,
                              min(self.max_replicas, target_replicas))
        current = self.get_current_replicas()

        if target_replicas == current:
            return

        # 冷却期检查
        now = time.time()
        if now - self.last_scale_time < self.cooldown_seconds:
            return

        self.apps_api.patch_namespaced_deployment(
            name=self.deployment,
            namespace=self.namespace,
            body={"spec": {"replicas": target_replicas}}
        )
        self.last_scale_time = now
        print(f"GPU 伸缩: {current} → {target_replicas} 副本")

    def reconcile(self, pending_count: int):
        """根据队列长度决策伸缩"""
        current = self.get_current_replicas()

        if pending_count > self.scale_up_threshold:
            self.scale(current + 1)
        elif pending_count < self.scale_down_threshold and current > self.min_replicas:
            self.scale(current - 1)
```

### GPU 伸缩的关键注意点

| 注意点 | 说明 | 建议 |
|--------|------|------|
| 冷启动时间 | 模型加载需 30-120s | 保持 minReplicas ≥ 1，避免缩容到 0 |
| GPU 不可共享 | 一个 Pod 独占 GPU | 使用时间分片（MPS）或多实例 GPU（MIG） |
| 缩容冷却 | 频繁伸缩浪费资源 | 缩容冷却期设为 5-10 分钟 |
| 预测性伸缩 | 流量有规律可循 | 按时间段预设副本数（CronHPA） |
| 成本控制 | GPU 按小时计费昂贵 | 低峰期切换到 CPU 推理或 Serverless |

---

## Serverless GPU 方案

如果你的 GPU 使用不是持续的（例如只在白天高峰期需要推理服务），Serverless GPU 可以大幅降低成本——只在推理时才占用 GPU，按实际使用时间计费。

### 方案对比

| 维度 | Modal | RunPod Serverless | AWS SageMaker Async |
|------|-------|-------------------|---------------------|
| 计费粒度 | 毫秒级 | 秒级 | 秒级 |
| 冷启动 | ~1s（容器缓存） | 5-30s | 30-120s |
| GPU 类型 | A10G / A100 / H100 | A100 / A6000 / RTX 4090 | 多种 |
| 最大运行时间 | 无限制 | 10 分钟 | 1 小时 |
| Python 原生 | ✅（装饰器语法） | ❌（需构建镜像） | ❌ |
| 适用场景 | 低延迟推理、批处理 | 通用 GPU 计算 | 长时间训练/推理 |
| 最低成本（A100） | ~$1.94/h | ~$1.64/h | ~$3.51/h |

### Modal 实战

Modal 的核心理念是"像写本地代码一样写云函数"——通过装饰器将函数部署到云端 GPU：

```python
# modal_app.py
import modal

# 定义 Modal 应用和 GPU 镜像
app = modal.App("agent-inference")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04")
    .pip_install(
        "vllm==0.6.3",
        "transformers==4.46.3",
    )
)

# 创建持久化的模型实例（避免冷启动重复加载）
@app.cls(
    image=image,
    gpu=modal.gpu.A100(size="80GB"),
    container_idle_timeout=300,   # 空闲 5 分钟后释放
    timeout=600,                  # 单次请求最长 10 分钟
    allow_concurrent_inputs=50,   # 允许并发请求数
)
class InferenceService:
    """部署在 Modal 上的推理服务"""

    @modal.enter()
    def load_model(self):
        """容器启动时加载模型"""
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model="Qwen/Qwen2.5-72B-Instruct-AWQ",
            quantization="awq",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2048,
        )
        print("模型加载完成")

    @modal.method()
    def generate(self, prompt: str) -> str:
        """生成推理结果"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    @modal.method()
    async def chat(self, messages: list[dict]) -> str:
        """Chat 格式推理"""
        from vllm import SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-72B-Instruct-AWQ"
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text


# 本地调用入口
@app.local_entrypoint()
def main():
    service = InferenceService()
    result = service.generate.remote("请解释什么是 PagedAttention")
    print(result)
```

### RunPod Serverless 实战

RunPod Serverless 需要先构建 Docker 镜像，然后部署为 Serverless Endpoint：

```dockerfile
# Dockerfile.runpod
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir \
    vllm==0.6.3 \
    transformers==4.46.3

# 复制 Handler 代码
COPY handler.py .

# RunPod Serverless 入口
CMD ["python", "-u", "handler.py"]
```

```python
# handler.py — RunPod Serverless Handler
import runpod
from vllm import LLM, SamplingParams

# 全局加载模型（冷启动时执行一次）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.9,
    max_model_len=16384,
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2048,
)


def handler(event: dict) -> dict:
    """RunPod Serverless 请求处理函数"""
    input_data = event["input"]
    prompt = input_data.get("prompt", "")
    messages = input_data.get("messages")

    if messages:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-AWQ"
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    return {"output": generated_text}


# 启动 Serverless Worker
runpod.serverless.start({"handler": handler})
```

### RunPod Serverless 配置

```yaml
# runpod-config.yaml — RunPod Serverless Endpoint 配置
# 通过 RunPod 控制台或 API 创建
endpoint:
  name: agent-inference
  image: your-registry/agent-inference:latest
  gpu_type: "NVIDIA A100 80GB"
  gpu_count: 1

  # 自动伸缩配置
  autoscaling:
    min_workers: 0          # 无请求时缩容到 0
    max_workers: 4          # 最多 4 个 Worker
    idle_timeout: 300       # 空闲 5 分钟释放
    scale_up_threshold: 5   # 队列 > 5 时扩容
    scale_down_threshold: 1

  # 资源配置
  resources:
    memory: 32Gi
    container_disk: 50Gi

  # 环境变量
  env:
    - name: MODEL_NAME
      value: "Qwen/Qwen2.5-7B-Instruct-AWQ"
    - name: MAX_MODEL_LEN
      value: "16384"
```

---

## 混合部署策略：自建 + Serverless

最经济的方案是混合部署：基线流量用自建 GPU 服务器（成本最低），峰值流量溢出到 Serverless GPU（弹性最强）。

```python
"""
混合路由器：自建推理服务 + Serverless 溢出
基线流量走自建服务（成本低），超出容量时溢出到 Modal/RunPod
"""
import httpx
import asyncio
from dataclasses import dataclass
from enum import Enum

class BackendType(Enum):
    SELF_HOSTED = "self_hosted"
    MODAL = "modal"
    RUNPOD = "runpod"

@dataclass
class Backend:
    type: BackendType
    base_url: str
    max_concurrent: int
    current_load: int = 0
    cost_per_1k_tokens: float = 0.0

class HybridRouter:
    """混合路由：优先自建，溢出到 Serverless"""

    def __init__(self):
        self.backends = [
            Backend(
                type=BackendType.SELF_HOSTED,
                base_url="http://vllm-service:8000",
                max_concurrent=50,
                cost_per_1k_tokens=0.0008,  # 自建成本（折算）
            ),
            Backend(
                type=BackendType.MODAL,
                base_url="https://modal-endpoint.example.com",
                max_concurrent=100,
                cost_per_1k_tokens=0.002,  # Serverless 成本（按量）
            ),
        ]

    async def route_request(self, messages: list[dict],
                            model: str = "qwen2.5-72b") -> dict:
        """路由请求到可用的后端"""
        # 按优先级（自建优先）检查可用性
        for backend in self.backends:
            if backend.current_load < backend.max_concurrent:
                backend.current_load += 1
                try:
                    result = await self._call_backend(backend, messages, model)
                    return {
                        "result": result,
                        "backend": backend.type.value,
                        "cost_estimate": backend.cost_per_1k_tokens,
                    }
                finally:
                    backend.current_load -= 1

        # 所有后端都满载，排队等待
        raise RuntimeError("所有推理后端均已满载，请稍后重试")

    async def _call_backend(self, backend: Backend,
                            messages: list[dict], model: str) -> dict:
        """调用指定后端"""
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{backend.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            return response.json()
```

### 混合部署成本估算

| 场景 | 纯自建 | 纯 Serverless | 混合部署 |
|------|--------|--------------|---------|
| 月请求量 | 100 万次 | 100 万次 | 100 万次 |
| 基线 QPS | 5 | — | 5（自建覆盖） |
| 峰值 QPS | 20 | 20 | 5 自建 + 15 Serverless |
| 月 GPU 成本 | ~$2,400（2×A100 按月） | ~$1,800（按量） | ~$1,400（1×A100 + 峰值溢出） |
| 可用性 | 峰值时可能过载 | 高（弹性） | 高 |
| 成本效率 | 低（闲置浪费） | 中 | **高** |

> 💡 **混合部署的关键**：准确预测基线流量，确保自建 GPU 覆盖 60%-80% 的日常流量，只将峰值溢出到 Serverless。

---

## K8s 部署的常见配置

### ConfigMap 管理应用配置

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: agent-prod
data:
  AGENT_MODEL_NAME: "gpt-4.1"
  AGENT_MAX_STEPS: "10"
  AGENT_MAX_TOKENS: "4096"
  AGENT_RATE_LIMIT_PER_MINUTE: "60"
  AGENT_LOG_LEVEL: "INFO"
```

### PodDisruptionBudget 保证可用性

```yaml
# pdb.yaml — 确保滚动更新时始终有足够副本在线
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: agent-api-pdb
  namespace: agent-prod
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: agent-api
```

### NetworkPolicy 网络隔离

```yaml
# networkpolicy.yaml — 只允许 API Pod 访问 Redis 和推理服务
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-network-policy
  namespace: agent-prod
spec:
  podSelector:
    matchLabels:
      app: agent-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
    - to:
        - podSelector:
            matchLabels:
              app: vllm
      ports:
        - port: 8000
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - port: 53
          protocol: UDP
```

---

## 部署流程与验证

### 使用 kubectl 部署

```bash
# 1. 创建命名空间
kubectl apply -f namespace.yaml

# 2. 创建密钥（从外部密钥管理系统注入）
kubectl create secret generic agent-secrets \
    --from-literal=openai-api-key='sk-your-key' \
    -n agent-prod

# 3. 按顺序部署各组件
kubectl apply -f configmap.yaml
kubectl apply -f redis-statefulset.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f api-service.yaml
kubectl apply -f api-hpa.yaml
kubectl apply -f ingress.yaml
kubectl apply -f pdb.yaml
kubectl apply -f networkpolicy.yaml

# 4. 验证部署状态
kubectl get pods -n agent-prod
kubectl get svc -n agent-prod
kubectl get hpa -n agent-prod

# 5. 检查 Pod 日志
kubectl logs -f deployment/agent-api -n agent-prod

# 6. 测试服务
curl -X POST https://agent.your-domain.com/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "hello"}'
```

### 滚动更新

```bash
# 更新镜像版本
kubectl set image deployment/agent-api \
    agent-api=your-registry/agent-api:v1.3.0 \
    -n agent-prod

# 查看滚动更新状态
kubectl rollout status deployment/agent-api -n agent-prod

# 如果出问题，快速回滚
kubectl rollout undo deployment/agent-api -n agent-prod
```

---

## 注意事项与最佳实践

1. **GPU 节点污点与容忍**：GPU 节点通常设置污点（taint），防止非 GPU 工作负载调度上去。推理服务 Pod 需要添加对应的容忍（toleration）：

```yaml
spec:
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
```

2. **模型缓存用 PVC**：避免每次 Pod 调度都重新下载模型（几十 GB）。使用 PersistentVolumeClaim 缓存模型文件：

```yaml
volumeMounts:
  - name: model-cache
    mountPath: /root/.cache/huggingface
volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache-pvc
```

3. **就绪探针的重要性**：推理服务加载模型需要时间，必须设置合理的 `initialDelaySeconds`，否则流量会被路由到还没准备好的 Pod。

4. **Serverless 冷启动优化**：Modal 支持 `container_idle_timeout` 参数，适当延长空闲超时（如 5 分钟）可显著减少冷启动。

5. **不要将 GPU 服务缩容到 0**：除非使用 Serverless 方案，自建 K8s 的 GPU 服务应保持 minReplicas ≥ 1。模型加载时间过长，缩容到 0 会导致首次请求超时。

6. **多可用区部署**：生产环境至少跨 2 个可用区部署，防止单可用区故障导致服务不可用。

---

## 小结

| 概念 | 说明 |
|------|------|
| K8s 编排 | 统一管理 API、推理、存储等组件 |
| GPU 伸缩 | 基于队列长度的自定义伸缩，冷启动需注意 |
| Modal | Python 原生 Serverless GPU，毫秒级计费 |
| RunPod Serverless | Docker 镜像部署，灵活度高 |
| 混合部署 | 自建覆盖基线 + Serverless 处理峰值，成本最优 |
| PDB / NetworkPolicy | 保证可用性与安全隔离 |

> **下一节预告**：服务部署好了，但 Agent 的长任务如何管理？Token 成本如何控制？我们来看任务队列与成本治理。

---

[19.8 长任务队列与成本治理](./08_task_queue_cost.md)
