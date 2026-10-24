# 20.6 Model Inference Serving

> **Goal of this section**: Master the three major LLM inference serving frameworks (vLLM / SGLang / TGI), understand the mainstream quantization options (GPTQ / AWQ / GGUF), and learn to design model routing strategies that strike the best balance between cost and performance.

---

## Why Do We Need Inference Serving?

In earlier chapters we called models through the OpenAI API. In a production environment, though, you may need to:

1. **Deploy open-source models**: use self-hosted models for reasons of data privacy, cost control, or customization
2. **Reduce inference latency**: raise throughput through continuous batching and KV cache reuse
3. **Route requests flexibly**: send simple requests to a small model and complex requests to a large model, based on task complexity

Serving a model by loading it directly with the `transformers` library has a severe bottleneck — it does not support request batching, it can only handle one request at a time, and GPU utilization is extremely low. Inference serving frameworks were born precisely to solve this problem.

---

## Comparing the Three Major Inference Frameworks

| Dimension | vLLM | SGLang | TGI (Text Generation Inference) |
|------|------|--------|----------------------------------|
| Developer | UC Berkeley | UC Berkeley / LMSYS | HuggingFace |
| Core technique | PagedAttention | RadixAttention | FlashAttention + Continuous Batching |
| Continuous batching | ✅ | ✅ | ✅ |
| KV cache reuse | ✅ PagedAttention | ✅ RadixAttention (automatic prefix sharing) | ✅ |
| Streaming output | ✅ | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ | ✅ |
| Multimodal support | ✅ | ✅ (experimental) | ✅ |
| Quantization support | GPTQ / AWQ / FP8 | GPTQ / AWQ / FP8 | GPTQ / AWQ / EETQ / bitsandbytes |
| Dynamic LoRA loading | ✅ | ✅ | ✅ |
| Typical throughput | High | Highest (with shared prefixes) | High |
| Community activity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Best for | General inference serving | Multi-turn dialogue / shared prefixes | HuggingFace ecosystem integration |

> 💡 **Selection advice**: If your Agent has a lot of multi-turn dialogue (shared system prompt + history messages), SGLang's RadixAttention can cut repeated computation dramatically; if you need the broadest model compatibility and community support, choose vLLM; if you are already inside the HuggingFace ecosystem (Inference Endpoints, etc.), TGI is the smoothest fit.

---

## vLLM in Practice

### Starting the Inference Service

```bash
# Basic startup
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --served-model-name qwen2.5-72b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching

# Serving a quantized model (AWQ)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --served-model-name qwen2.5-72b-awq \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```

### Key Parameters Explained

| Parameter | Description | Recommended value |
|------|------|--------|
| `--tensor-parallel-size` | Tensor parallelism degree (splitting across GPUs) | Equal to the number of GPUs |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | 0.85–0.95 |
| `--max-model-len` | Maximum context length | Depends on the model and available memory |
| `--enable-prefix-caching` | Enable prefix caching (reuse the system prompt) | Strongly recommended for Agent scenarios |
| `--max-num-seqs` | Maximum number of concurrent sequences | 128–256 |
| `--swap-space` | CPU swap space size (GB) | 4–8 |

### Calling It Through the OpenAI-Compatible API

```python
from openai import OpenAI

# Connect to the self-hosted vLLM service
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM does not require auth by default
)

response = client.chat.completions.create(
    model="qwen2.5-72b",
    messages=[
        {"role": "system", "content": "You are a professional AI assistant."},
        {"role": "user", "content": "Explain how PagedAttention works"}
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

## SGLang in Practice

SGLang's core advantage is RadixAttention — when several requests share the same prompt prefix (such as the system prompt), the KV cache can be reused automatically, avoiding duplicate computation. This is especially valuable for Agent scenarios, because repeated conversations with the same Agent usually share the same system prompt and tool definitions.

### Starting the Inference Service

```bash
# Single-GPU startup
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --served-model-name qwen2.5-72b \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.9 \
    --context-length 32768

# Multi-GPU tensor parallelism
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp 4 \
    --host 0.0.0.0 \
    --port 8000
```

### Prefix Reuse in Agent Scenarios

```python
import requests
import json

SGLANG_URL = "http://localhost:8000"

# The Agent's system prompt (shared by all requests)
SYSTEM_PROMPT = """You are a data analysis assistant. You can use the following tools:
- search: search for data
- analyze: analyze data
- visualize: generate visualization charts

Choose the appropriate tool based on the user's needs."""

def agent_chat(user_message: str, history: list[dict] = None):
    """Run an Agent conversation with SGLang; the shared-prefix KV cache is reused automatically"""

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

# Across multiple calls, the system prompt's KV cache is reused automatically
result1 = agent_chat("Search last quarter's sales data for me")
result2 = agent_chat("Analyze the trends in this data")
```

---

## TGI in Practice

TGI is the inference server released by HuggingFace, deeply integrated with the HuggingFace model ecosystem.

### Starting It with Docker

```bash
# Start TGI with Docker
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

### TGI's Watermark

TGI has a built-in watermarking feature that embeds invisible markers into the model's output, which can be used to detect AI-generated content:

```bash
docker run --gpus all -p 8000:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --watermark-gamma 0.5 \
    --watermark-delta 2.0
```

---

## Comparing Quantization Options

Quantization is the core lever for lowering inference cost — it compresses model weights from FP16 (16-bit floating point) down to lower precision, sacrificing a tiny amount of quality in exchange for large memory savings and faster inference.

| Option | Precision | Memory saved | Quality loss | Speedup | Best for |
|------|------|---------|---------|---------|---------|
| **GPTQ** | 4-bit | ~75% | Small | Medium | GPU inference, offline quantization |
| **AWQ** | 4-bit | ~75% | Very small | Medium | GPU inference, activation-aware |
| **GGUF** | 2–8-bit, selectable | 50%–87% | Controllable | CPU-friendly | CPU / consumer-GPU inference |
| FP8 | 8-bit | ~50% | Very small | High | Newer hardware such as H100 / 4090 |
| BitsAndBytes | 4-bit / 8-bit | 50%–75% | Small | Low | Dynamic quantization, no pre-quantized model needed |

> ⚠️ **Important reminder**: Quantization is not a silver bullet. For scenarios that need precise mathematical reasoning or strictly formatted output (such as JSON generation), the error rate of 4-bit quantization can rise significantly. Evaluate a quantized model thoroughly before shipping it.

### GPTQ Quantization in Practice

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# Prepare calibration data
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calibration_data = []
for i, example in enumerate(dataset):
    if i >= 128:  # 128 calibration samples are usually enough
        break
    tokens = tokenizer(example["text"], return_tensors="pt",
                       max_length=2048, truncation=True)
    calibration_data.append(tokens.input_ids)

# Configure the quantization parameters
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4-bit quantization
    group_size=128,      # group size
    desc_act=True,       # sort by activation (better quality but slower)
    damp_percent=0.01,   # guards against numerical instability
)

# Load the model and quantize it
model = AutoGPTQForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

model.quantize(calibration_data)

# Save the quantized model
model.save_quantized("qwen2.5-7b-gptq-4bit")
tokenizer.save_pretrained("qwen2.5-7b-gptq-4bit")
```

### AWQ Quantization in Practice

AWQ (Activation-aware Weight Quantization) protects important weights by taking the activation distribution into account, so it loses less quality than GPTQ:

```bash
# Quantize with the autoawq library
python -m awq.entrypoint \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --w_bit 4 \
    --q_group_size 128 \
    --zero_point \
    --output_dir qwen2.5-7b-awq-4bit
```

### GGUF Quantization (llama.cpp)

GGUF is llama.cpp's native format; it supports CPU inference and Apple Silicon acceleration:

```bash
# Use llama.cpp's conversion tool
python convert_hf_to_gguf.py Qwen/Qwen2.5-7B-Instruct \
    --outfile qwen2.5-7b-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M (the recommended quality/size sweet spot)
./llama-quantize qwen2.5-7b-f16.gguf qwen2.5-7b-Q4_K_M.gguf Q4_K_M
```

| GGUF quantization level | Size (7B model) | Quality assessment | Recommended use |
|--------------|----------------|---------|---------|
| Q8_0 | ~7.7GB | Nearly lossless | When quality matters most |
| Q5_K_M | ~5.3GB | Slight loss | Balanced choice |
| **Q4_K_M** | **~4.4GB** | **Acceptable** | **Recommended default** |
| Q3_K_M | ~3.5GB | Noticeable loss | Extremely constrained resources |
| Q2_K | ~2.8GB | Severe loss | Not recommended |

---

## Model Routing Strategies

In production, not every request needs the strongest large model. Model routing intelligently dispatches requests to models of different capability, balancing cost against quality.

### Strategy 1: Static Routing by Task Complexity

The simplest form of routing — preset routing rules based on request type:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ComplexityLevel(Enum):
    SIMPLE = "simple"       # simple Q&A, format conversion
    MODERATE = "moderate"   # general reasoning, summarization
    COMPLEX = "complex"     # multi-step reasoning, code generation

@dataclass
class ModelEndpoint:
    name: str
    model_id: str
    base_url: str
    cost_per_1k_tokens: float  # cost per 1K tokens (USD)
    max_tokens: int

class StaticModelRouter:
    """Static model routing based on task type"""

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

        # Mapping from task type to complexity level
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

# Usage example
router = StaticModelRouter()
model = router.route("code_gen")
print(f"Routed to: {model.name} ({model.model_id})")
# Output: Routed to: power-model (gpt-4.1)
```

### Strategy 2: Dynamic Routing with an LLM Classifier

Let a small model judge the complexity of the request, then route it to the right model:

```python
import json
from openai import OpenAI

class DynamicModelRouter:
    """Route requests dynamically using an LLM classifier"""

    ROUTER_PROMPT = """You are a request classifier. Judge the complexity level of the user's input.

Complexity level definitions:
- simple: simple Q&A, format conversion, translation, summarization — no deep reasoning needed
- moderate: needs some reasoning ability, such as analysis, comparison, explanation
- complex: needs multi-step reasoning, code generation, complex tool calls, mathematical computation

Return only a JSON object:
{"complexity": "simple" | "moderate" | "complex", "reason": "brief rationale"}"""

    def __init__(self):
        self.client = OpenAI()
        self.router_model = "gpt-4.1-mini"  # use a small model for routing
        self.target_models = {
            "simple": "gpt-4.1-mini",
            "moderate": "gpt-4.1-mini",
            "complex": "gpt-4.1",
        }

    def classify(self, user_input: str) -> dict:
        """Classify the complexity of the request"""
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
        """Return the model that should be used"""
        result = self.classify(user_input)
        complexity = result.get("complexity", "moderate")
        return self.target_models[complexity]

# Usage example
router = DynamicModelRouter()

# Simple request → small model
model = router.route("Translate the following text into French: Hello, world")
print(f"Using model: {model}")  # gpt-4.1-mini

# Complex request → large model
model = router.route("Design a distributed task queue system supporting priorities, retries, and a dead-letter queue")
print(f"Using model: {model}")  # gpt-4.1
```

### Strategy 3: Confidence-Based Fallback Routing

Try the small model first, and escalate to the large model when confidence is not high enough:

```python
from openai import OpenAI
import json

class FallbackRouter:
    """Confidence-based fallback routing: try the small model first, escalate if it is not enough"""

    def __init__(self):
        self.client = OpenAI()
        self.fast_model = "gpt-4.1-mini"
        self.power_model = "gpt-4.1"

    def _needs_escalation(self, response_content: str, user_input: str) -> bool:
        """Decide whether we need to escalate to a stronger model"""
        # Check for explicit "I cannot answer" signals
        escalation_signals = [
            "I cannot", "beyond my capabilities", "unable to complete",
            "requires more specialized", "recommend consulting",
        ]
        for signal in escalation_signals:
            if signal in response_content:
                return True

        # A very short reply may indicate insufficient quality
        if len(response_content) < 20 and len(user_input) > 50:
            return True

        return False

    def route(self, messages: list[dict], stream: bool = False):
        """Try the small model first, fall back to the large model when necessary"""
        # First attempt: the small model
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

        # Check whether escalation is needed
        if self._needs_escalation(content, messages[-1]["content"]):
            # Fall back to the large model
            response = self.client.chat.completions.create(
                model=self.power_model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=stream,
            )
            return response, self.power_model

        return response, self.fast_model

# Usage example
router = FallbackRouter()
messages = [{"role": "user", "content": "Write a quicksort implementation in Python for me"}]
response, used_model = router.route(messages)
print(f"Final model used: {used_model}")
```

### Comparing the Three Routing Strategies

| Dimension | Static routing | Dynamic routing | Fallback routing |
|------|---------|---------|---------|
| Implementation complexity | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Routing accuracy | Medium | High | High |
| Extra latency | None | Yes (classification request) | Possible (when falling back) |
| Cost savings | Medium | High | High |
| Best for | Fixed task types | Diverse task types | Quality-sensitive workloads |

> 💡 **Practical advice**: Start with static routing; after collecting request logs for a while, analyze the complexity distribution before considering an upgrade to dynamic routing. Fallback routing suits scenarios with extremely high quality requirements (medicine, law), but not high-throughput scenarios (a fallback doubles the latency).

---

## Production Configuration for Inference Services

### Example vLLM Kubernetes Deployment Configuration

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

### Monitoring Metrics for Inference Services

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define the monitoring metrics
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
    """Metrics collector for the inference service"""

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

# Start the Prometheus metrics server
start_http_server(9090)
```

---

## Caveats and Best Practices

1. **Prefix caching is the killer feature for Agent scenarios**: an Agent's system prompt is usually long (it contains tool definitions) and identical on every request. Be sure to enable vLLM's `--enable-prefix-caching` or use SGLang's RadixAttention.

2. **Quantized models produce lower-quality formatted output**: if your Agent depends on strict JSON / function-call formatting, a 4-bit quantized model's format error rate can be 2–5× higher than FP16. For format-sensitive scenarios, prefer 8-bit quantization or FP16.

3. **Model warm-up**: the first inference request has far higher latency than subsequent ones (weights must be loaded onto the GPU and CUDA kernels compiled). A production deployment should send a few warm-up requests:

```python
import requests

def warm_up_model(base_url: str, model_name: str, warmup_rounds: int = 3):
    """Warm up the inference service so the first real request is not too slow"""
    for i in range(warmup_rounds):
        requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )
    print(f"Model {model_name} warm-up complete ({warmup_rounds} rounds)")
```

4. **GPU memory fragmentation**: when long-text and short-text requests alternate, PagedAttention can produce memory fragmentation. Set the `--swap-space` parameter to allow part of the KV cache to be swapped out to CPU memory.

5. **Pin your versions**: inference frameworks update frequently and APIs may break. In production, always pin the Docker image version — never use the `latest` tag.

---

## Summary

| Concept | Description |
|------|------|
| vLLM | PagedAttention; the most general-purpose, with the largest community |
| SGLang | RadixAttention; best performance for multi-turn dialogue scenarios |
| TGI | HuggingFace ecosystem integration; works out of the box |
| GPTQ / AWQ | 4-bit quantization for GPU inference; greatly reduces memory requirements |
| GGUF | Quantization format friendly to CPU / consumer GPUs |
| Model routing | Dispatch to large or small models by task complexity, balancing cost and quality |

> **Coming up next**: With the inference service deployed, we will next learn how to orchestrate the whole Agent service stack with Kubernetes, and how Serverless GPU options can lower costs even further.

---

[20.7 Kubernetes Orchestration and Serverless GPU](./07_k8s_serverless.md)
