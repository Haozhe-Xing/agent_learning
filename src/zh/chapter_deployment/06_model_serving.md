# 19.6 模型推理服务化

> **本节目标**：掌握大模型推理服务化的三大框架（vLLM / SGLang / TGI），了解主流量化方案（GPTQ / AWQ / GGUF），学会设计模型路由策略以实现成本与性能的最优平衡。

---

## 为什么需要推理服务化？

在前面的章节中，我们通过 OpenAI API 调用模型。但在生产环境中，你可能需要：

1. **部署开源模型**：出于数据隐私、成本控制或定制化需求，使用自托管模型
2. **降低推理延迟**：通过连续批处理（continuous batching）和 KV Cache 复用提升吞吐
3. **灵活路由请求**：根据任务复杂度，将简单请求路由到小模型，复杂请求路由到大模型

直接用 `transformers` 库加载模型服务化存在严重瓶颈——它不支持请求批处理，每次只能处理一个请求，GPU 利用率极低。推理服务化框架正是为了解决这个问题而生。

---

## 三大推理框架对比

| 维度 | vLLM | SGLang | TGI (Text Generation Inference) |
|------|------|--------|----------------------------------|
| 开发方 | UC Berkeley | UC Berkeley / LMSYS | HuggingFace |
| 核心技术 | PagedAttention | RadixAttention | FlashAttention + Continuous Batching |
| 连续批处理 | ✅ | ✅ | ✅ |
| KV Cache 复用 | ✅ PagedAttention | ✅ RadixAttention（自动前缀共享） | ✅ |
| 流式输出 | ✅ | ✅ | ✅ |
| OpenAI 兼容 API | ✅ | ✅ | ✅ |
| 多模态支持 | ✅ | ✅（实验性） | ✅ |
| 量化支持 | GPTQ / AWQ / FP8 | GPTQ / AWQ / FP8 | GPTQ / AWQ / EETQ / bitsandbytes |
| LoRA 动态加载 | ✅ | ✅ | ✅ |
| 典型吞吐 | 高 | 最高（相同前缀场景） | 高 |
| 社区活跃度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 适用场景 | 通用推理服务 | 多轮对话 / 共享前缀 | HuggingFace 生态集成 |

> 💡 **选型建议**：如果 Agent 有大量多轮对话（共享 system prompt + 历史消息），SGLang 的 RadixAttention 能显著减少重复计算；如果需要最广泛的模型兼容性和社区支持，选择 vLLM；如果已经在用 HuggingFace 生态（Inference Endpoints 等），TGI 是最顺手的。

---

## vLLM 部署实战

### 启动推理服务

```bash
# 基础启动
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --served-model-name qwen2.5-72b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching

# 启用量化模型（AWQ）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --served-model-name qwen2.5-72b-awq \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--tensor-parallel-size` | 张量并行数（跨 GPU 切分） | 等于 GPU 数量 |
| `--gpu-memory-utilization` | GPU 显存使用比例 | 0.85-0.95 |
| `--max-model-len` | 最大上下文长度 | 根据模型和显存决定 |
| `--enable-prefix-caching` | 启用前缀缓存（复用 system prompt） | Agent 场景强烈推荐 |
| `--max-num-seqs` | 最大并发序列数 | 128-256 |
| `--swap-space` | CPU 换页空间大小（GB） | 4-8 |

### 使用 OpenAI 兼容 API 调用

```python
from openai import OpenAI

# 连接自托管的 vLLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 默认不鉴权
)

response = client.chat.completions.create(
    model="qwen2.5-72b",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "解释 PagedAttention 的原理"}
    ],
    temperature=0.7,
    max_tokens=2048,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## SGLang 部署实战

SGLang 的核心优势是 RadixAttention——当多个请求共享相同的 prompt 前缀（如 system prompt），可以自动复用 KV Cache，避免重复计算。这对于 Agent 场景特别有价值，因为同一 Agent 的多次对话通常共享相同的 system prompt 和工具定义。

### 启动推理服务

```bash
# 单 GPU 启动
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --served-model-name qwen2.5-72b \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.9 \
    --context-length 32768

# 多 GPU 张量并行
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp 4 \
    --host 0.0.0.0 \
    --port 8000
```

### Agent 场景中的前缀复用

```python
import requests
import json

SGLANG_URL = "http://localhost:8000"

# Agent 的 system prompt（所有请求共享）
SYSTEM_PROMPT = """你是一个数据分析助手。你可以使用以下工具：
- search: 搜索数据
- analyze: 分析数据
- visualize: 生成可视化图表

请根据用户需求选择合适的工具。"""

def agent_chat(user_message: str, history: list[dict] = None):
    """使用 SGLang 进行 Agent 对话，自动复用共享前缀的 KV Cache"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json={
            "model": "qwen2.5-72b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        stream=False,
    )
    return response.json()["choices"][0]["message"]["content"]

# 多次调用时，system prompt 的 KV Cache 被自动复用
result1 = agent_chat("帮我搜索上季度的销售数据")
result2 = agent_chat("分析一下这些数据的趋势")
```

---

## TGI 部署实战

TGI 是 HuggingFace 推出的推理服务器，与 HuggingFace 模型生态深度集成。

### Docker 启动

```bash
# 使用 Docker 启动 TGI
docker run --gpus all -p 8000:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2.5-72B-Instruct \
    --num-shard 4 \
    --max-input-length 32000 \
    --max-total-tokens 32768 \
    --max-batch-size 128 \
    --quantize awq
```

### TGI 的水量标记（Watermark）

TGI 内置了水量标记功能，可以在模型输出中嵌入不可见的标记，用于检测 AI 生成内容：

```bash
docker run --gpus all -p 8000:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --watermark-gamma 0.5 \
    --watermark-delta 2.0
```

---

## 量化方案对比

量化是降低推理成本的核心手段——将模型权重从 FP16（16 位浮点）压缩到更低精度，牺牲极少质量换取显著的显存节省和推理加速。

| 方案 | 精度 | 显存节省 | 质量损失 | 推理加速 | 适用场景 |
|------|------|---------|---------|---------|---------|
| **GPTQ** | 4-bit | ~75% | 小 | 中 | GPU 推理，离线量化 |
| **AWQ** | 4-bit | ~75% | 极小 | 中 | GPU 推理，激活感知 |
| **GGUF** | 2-8-bit 可选 | 50%-87% | 可控 | CPU 友好 | CPU / 消费级 GPU 推理 |
| FP8 | 8-bit | ~50% | 极小 | 高 | H100 / 4090 等新硬件 |
| BitsAndBytes | 4-bit / 8-bit | 50%-75% | 小 | 低 | 动态量化，无需预量化模型 |

> ⚠️ **重要提醒**：量化不是万能的。对于需要精确数学推理或严格格式输出的场景（如 JSON 生成），4-bit 量化的错误率可能显著上升。建议在上线前对量化模型做充分的评估。

### GPTQ 量化实战

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# 准备校准数据
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calibration_data = []
for i, example in enumerate(dataset):
    if i >= 128:  # 128 条校准数据通常足够
        break
    tokens = tokenizer(example["text"], return_tensors="pt",
                       max_length=2048, truncation=True)
    calibration_data.append(tokens.input_ids)

# 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4-bit 量化
    group_size=128,      # 分组大小
    desc_act=True,       # 激活值排序（提升质量但更慢）
    damp_percent=0.01,   # 防止数值不稳定
)

# 加载模型并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

model.quantize(calibration_data)

# 保存量化模型
model.save_quantized("qwen2.5-7b-gptq-4bit")
tokenizer.save_pretrained("qwen2.5-7b-gptq-4bit")
```

### AWQ 量化实战

AWQ（Activation-aware Weight Quantization）通过感知激活值分布来保护重要权重，相比 GPTQ 质量损失更小：

```bash
# 使用 autoawq 库量化
python -m awq.entrypoint \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --w_bit 4 \
    --q_group_size 128 \
    --zero_point \
    --output_dir qwen2.5-7b-awq-4bit
```

### GGUF 量化（llama.cpp）

GGUF 是 llama.cpp 的原生格式，支持 CPU 推理和 Apple Silicon 加速：

```bash
# 使用 llama.cpp 的转换工具
python convert_hf_to_gguf.py Qwen/Qwen2.5-7B-Instruct \
    --outfile qwen2.5-7b-f16.gguf \
    --outtype f16

# 量化为 Q4_K_M（推荐的质量/体积平衡点）
./llama-quantize qwen2.5-7b-f16.gguf qwen2.5-7b-Q4_K_M.gguf Q4_K_M
```

| GGUF 量化等级 | 体积（7B 模型） | 质量评价 | 推荐用途 |
|--------------|----------------|---------|---------|
| Q8_0 | ~7.7GB | 几乎无损 | 对质量要求高 |
| Q5_K_M | ~5.3GB | 轻微损失 | 平衡选择 |
| **Q4_K_M** | **~4.4GB** | **可接受** | **推荐默认选择** |
| Q3_K_M | ~3.5GB | 有明显损失 | 极端资源受限 |
| Q2_K | ~2.8GB | 严重损失 | 不推荐 |

---

## 模型路由策略

在生产环境中，不是每个请求都需要最强的大模型。模型路由（Model Routing）通过智能分配请求到不同能力的模型，实现成本与质量的平衡。

### 策略一：基于任务复杂度的静态路由

最简单的路由方式——根据请求类型预设路由规则：

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ComplexityLevel(Enum):
    SIMPLE = "simple"       # 简单问答、格式转换
    MODERATE = "moderate"   # 一般推理、摘要
    COMPLEX = "complex"     # 多步推理、代码生成

@dataclass
class ModelEndpoint:
    name: str
    model_id: str
    base_url: str
    cost_per_1k_tokens: float  # 每千 Token 成本（美元）
    max_tokens: int

class StaticModelRouter:
    """基于任务类型的静态模型路由"""

    def __init__(self):
        self.models = {
            ComplexityLevel.SIMPLE: ModelEndpoint(
                name="fast-model",
                model_id="gpt-4.1-mini",
                base_url="https://api.openai.com/v1",
                cost_per_1k_tokens=0.0004,
                max_tokens=16384,
            ),
            ComplexityLevel.MODERATE: ModelEndpoint(
                name="balanced-model",
                model_id="gpt-4.1-mini",
                base_url="https://api.openai.com/v1",
                cost_per_1k_tokens=0.0004,
                max_tokens=16384,
            ),
            ComplexityLevel.COMPLEX: ModelEndpoint(
                name="power-model",
                model_id="gpt-4.1",
                base_url="https://api.openai.com/v1",
                cost_per_1k_tokens=0.002,
                max_tokens=16384,
            ),
        }

        # 任务类型到复杂度的映射
        self.task_mapping = {
            "summarize": ComplexityLevel.SIMPLE,
            "translate": ComplexityLevel.SIMPLE,
            "format": ComplexityLevel.SIMPLE,
            "qa": ComplexityLevel.MODERATE,
            "analyze": ComplexityLevel.MODERATE,
            "code_gen": ComplexityLevel.COMPLEX,
            "multi_step_reason": ComplexityLevel.COMPLEX,
            "tool_use": ComplexityLevel.COMPLEX,
        }

    def route(self, task_type: str) -> ModelEndpoint:
        complexity = self.task_mapping.get(task_type, ComplexityLevel.MODERATE)
        return self.models[complexity]

# 使用示例
router = StaticModelRouter()
model = router.route("code_gen")
print(f"路由到: {model.name} ({model.model_id})")
# 输出: 路由到: power-model (gpt-4.1)
```

### 策略二：基于 LLM 分类器的动态路由

让一个小模型判断请求的复杂度，再路由到合适的模型：

```python
import json
from openai import OpenAI

class DynamicModelRouter:
    """使用 LLM 分类器动态路由请求"""

    ROUTER_PROMPT = """你是一个请求分类器。根据用户的输入，判断其复杂度等级。

复杂度等级定义：
- simple: 简单问答、格式转换、翻译、摘要等，不需要深度推理
- moderate: 需要一定推理能力，如分析、比较、解释
- complex: 需要多步推理、代码生成、复杂工具调用、数学计算

请只返回一个 JSON 对象：
{"complexity": "simple" | "moderate" | "complex", "reason": "简要理由"}"""

    def __init__(self):
        self.client = OpenAI()
        self.router_model = "gpt-4.1-mini"  # 用小模型做路由
        self.target_models = {
            "simple": "gpt-4.1-mini",
            "moderate": "gpt-4.1-mini",
            "complex": "gpt-4.1",
        }

    def classify(self, user_input: str) -> dict:
        """分类请求复杂度"""
        response = self.client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": self.ROUTER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"complexity": "moderate", "reason": "parse failed"}

    def route(self, user_input: str) -> str:
        """返回应该使用的模型"""
        result = self.classify(user_input)
        complexity = result.get("complexity", "moderate")
        return self.target_models[complexity]

# 使用示例
router = DynamicModelRouter()

# 简单请求 → 小模型
model = router.route("将以下文本翻译为英文：你好世界")
print(f"使用模型: {model}")  # gpt-4.1-mini

# 复杂请求 → 大模型
model = router.route("设计一个分布式任务队列系统，支持优先级、重试和死信队列")
print(f"使用模型: {model}")  # gpt-4.1
```

### 策略三：基于置信度的回退路由

先用小模型尝试，如果置信度不够再回退到大模型：

```python
from openai import OpenAI
import json

class FallbackRouter:
    """基于置信度的回退路由：先尝试小模型，不够再升级"""

    def __init__(self):
        self.client = OpenAI()
        self.fast_model = "gpt-4.1-mini"
        self.power_model = "gpt-4.1"

    def _needs_escalation(self, response_content: str, user_input: str) -> bool:
        """判断是否需要升级到更强模型"""
        # 检查是否有明确的"无法回答"信号
        escalation_signals = [
            "我无法", "超出我的能力", "无法完成",
            "需要更专业的", "建议咨询",
        ]
        for signal in escalation_signals:
            if signal in response_content:
                return True

        # 如果回复过短，可能质量不足
        if len(response_content) < 20 and len(user_input) > 50:
            return True

        return False

    def route(self, messages: list[dict], stream: bool = False):
        """先尝试小模型，必要时回退到大模型"""
        # 第一次尝试：小模型
        response = self.client.chat.completions.create(
            model=self.fast_model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=stream,
        )

        if stream:
            return response, self.fast_model

        content = response.choices[0].message.content

        # 检查是否需要升级
        if self._needs_escalation(content, messages[-1]["content"]):
            # 回退到大模型
            response = self.client.chat.completions.create(
                model=self.power_model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=stream,
            )
            return response, self.power_model

        return response, self.fast_model

# 使用示例
router = FallbackRouter()
messages = [{"role": "user", "content": "帮我写一个 Python 快排实现"}]
response, used_model = router.route(messages)
print(f"最终使用模型: {used_model}")
```

### 三种路由策略对比

| 维度 | 静态路由 | 动态路由 | 回退路由 |
|------|---------|---------|---------|
| 实现复杂度 | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| 路由准确性 | 中 | 高 | 高 |
| 额外延迟 | 无 | 有（分类请求） | 可能（回退时） |
| 成本节省 | 中 | 高 | 高 |
| 适用场景 | 任务类型固定 | 任务类型多样 | 对质量敏感 |

> 💡 **实践建议**：从静态路由起步，收集一段时间的请求日志后，分析复杂度分布再考虑升级到动态路由。回退路由适合对质量要求极高的场景（如医疗、法律），但不适合高吞吐场景（回退会增加 2 倍延迟）。

---

## 推理服务的生产配置

### vLLM 的 K8s Deployment 配置示例

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen72b
  labels:
    app: vllm
    model: qwen2.5-72b
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
        model: qwen2.5-72b
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command:
            - python
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --model
            - Qwen/Qwen2.5-72B-Instruct-AWQ
            - --quantization
            - awq
            - --served-model-name
            - qwen2.5-72b
            - --tensor-parallel-size
            - "2"
            - --gpu-memory-utilization
            - "0.9"
            - --max-model-len
            - "32768"
            - --enable-prefix-caching
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 2
            requests:
              nvidia.com/gpu: 2
          env:
            - name: MODEL_NAME
              value: "qwen2.5-72b"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
      nodeSelector:
        gpu-type: "a100-80g"
```

### 推理服务监控指标

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义监控指标
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model", "status"]
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Request latency in seconds",
    ["model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120]
)

TOKENS_PROCESSED = Counter(
    "inference_tokens_total",
    "Total tokens processed",
    ["model", "type"]  # type: input / output
)

ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Currently processing requests",
    ["model"]
)

GPU_MEMORY_USED = Gauge(
    "inference_gpu_memory_used_bytes",
    "GPU memory used",
    ["gpu_id"]
)

class InferenceMetrics:
    """推理服务指标收集器"""

    def __init__(self, model_name: str):
        self.model = model_name

    def record_request(self, status: str, duration: float,
                       input_tokens: int, output_tokens: int):
        REQUEST_COUNT.labels(model=self.model, status=status).inc()
        REQUEST_LATENCY.labels(model=self.model).observe(duration)
        TOKENS_PROCESSED.labels(model=self.model, type="input").inc(input_tokens)
        TOKENS_PROCESSED.labels(model=self.model, type="output").inc(output_tokens)

    def set_active_requests(self, count: int):
        ACTIVE_REQUESTS.labels(model=self.model).set(count)

# 启动 Prometheus 指标服务
start_http_server(9090)
```

---

## 注意事项与最佳实践

1. **前缀缓存是 Agent 场景的杀手特性**：Agent 的 system prompt 通常很长（含工具定义），且每次请求都相同。务必启用 vLLM 的 `--enable-prefix-caching` 或使用 SGLang 的 RadixAttention。

2. **量化模型的格式输出质量会下降**：如果 Agent 依赖严格的 JSON/函数调用格式输出，4-bit 量化模型的格式错误率可能比 FP16 高 2-5 倍。建议对格式输出场景使用 8-bit 量化或 FP16。

3. **模型预热（Warm-up）**：首次推理请求的延迟会远高于后续请求（需加载权重到 GPU、编译 CUDA Kernel）。生产部署时应发送几个预热请求：

```python
import requests

def warm_up_model(base_url: str, model_name: str, warmup_rounds: int = 3):
    """预热推理服务，避免首次请求延迟过高"""
    for i in range(warmup_rounds):
        requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )
    print(f"模型 {model_name} 预热完成（{warmup_rounds} 轮）")
```

4. **GPU 显存碎片**：长文本请求和短文本请求交替出现时，PagedAttention 可能产生显存碎片。设置 `--swap-space` 参数允许将部分 KV Cache 换出到 CPU 内存。

5. **版本锁定**：推理框架更新频繁，API 可能不兼容。生产环境务必锁定 Docker 镜像版本，不要用 `latest` 标签。

---

## 小结

| 概念 | 说明 |
|------|------|
| vLLM | PagedAttention，通用性最强，社区最大 |
| SGLang | RadixAttention，多轮对话场景性能最优 |
| TGI | HuggingFace 生态集成，开箱即用 |
| GPTQ / AWQ | GPU 推理的 4-bit 量化方案，大幅降低显存需求 |
| GGUF | CPU / 消费级 GPU 友好的量化格式 |
| 模型路由 | 按任务复杂度分配大小模型，平衡成本与质量 |

> **下一节预告**：推理服务部署好了，接下来我们学习如何用 Kubernetes 编排整个 Agent 服务栈，以及 Serverless GPU 方案如何进一步降低成本。

---

[19.7 Kubernetes 编排与 Serverless GPU](./07_k8s_serverless.md)
