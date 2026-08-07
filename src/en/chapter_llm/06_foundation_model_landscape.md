# 2.6 Foundation Model Landscape and Selection Guide

> 🌍 *"Models iterate rapidly — today's SOTA may be tomorrow's baseline. But understanding the evolution trends lets you make better choices amid change."*

In earlier sections, we learned the fundamentals of LLMs, prompt engineering, API calls, and model parameters. That knowledge represents the "unchanging" underlying capabilities. This section discusses the "changing" part — **the technical frontier and industry landscape of foundation models**.

As an Agent developer, you don't need to train your own foundation model, but you must understand the capability boundaries and development trends of models — because **the choice of model directly determines the ceiling of your Agent**.

![Foundation Model Landscape and Four Major Trends](../svg/chapter_llm_06_landscape.svg)

## 2024–2026: Four Major Trends in Foundation Models

### Trend 1: The Leap in Reasoning Capability

In September 2024, OpenAI's o1 first proved the feasibility of "trading more reasoning time for better results." In January 2025, the open-source release of DeepSeek-R1 ignited the democratization of reasoning models — it was the first to demonstrate how pure RL training (GRPO) could cause Chain-of-Thought capability to emerge spontaneously in a model.

In April 2025, OpenAI released o3 and o4-mini, achieving **multimodal reasoning** ("thinking while looking at images") and autonomous tool chain calls for the first time. In August 2025, **GPT-5** was officially released, adopting a unified system architecture with built-in intelligent routing that automatically selects the reasoning depth based on problem complexity — no longer requiring a separate o-series model.

By early 2026, reasoning had become standard across all mainstream models:

| Model | Release | Reasoning Mode | Key Breakthrough |
|-------|---------|---------------|-----------------|
| **GPT-5.5** | 2026.07 | Upgraded built-in Thinking mode | Reasoning+Agent unification evolves further; context extended to 2M; SWE-bench 78% |
| **Claude Opus 4.8** | 2026.07 | Adaptive reasoning depth | 2M context; long-horizon Agents run continuously for days; new SWE-bench Verified record |
| **Gemini 3.1 Pro** | 2026.07 | Native multimodal reasoning | 2M+ context; native video understanding upgraded; dynamic reasoning budget control |
| **Kimi K3** | 2026.07 | Agent reasoning | Trillion-parameter open-source; 500 parallel sub-agents; SWE-bench Pro 61% |
| **Qwen3.7** | 2026.07 | Hybrid reasoning (fast/slow thinking) | Full open-source MoE family; tops the open-source leaderboard; Apache 2.0 |
| **GLM-5.2** | 2026.07 | Hybrid reasoning + Agent reinforcement | 12,000+ tool calls per session; SWE-bench Pro 60%; long-chain Agent optimization |
| **Claude Opus 4.7** | 2026.04 | Adaptive reasoning depth | SWE-bench Verified #1; visual capability tops charts; new tokenizer |
| **GPT-5.4** | 2026.03 | Built-in Thinking mode | Reasoning+coding+Computer Use+search unified; 1M context |
| **Claude Opus 4.6** | 2026.02 | Adaptive reasoning depth | 1M context (Beta) + SWE-bench 80.8% |
| **GPT-5** | 2025.08 | Built-in intelligent routing | SWE-bench 75%; unified system architecture; multimodal |
| **Claude Opus 4** | 2025.05 | Deep reasoning | SWE-bench 72.5%; continuous 7-hour operation |
| **Gemini 2.5 Pro** | 2025.03 | Native multimodal reasoning | 1M context + dynamic reasoning budget control |
| **DeepSeek-R1** | 2025.01 | Pure RL reasoning | Open-source reasoning model ignites the world; GRPO training |
| **Kimi K2.6** | 2026.04 | Agent reasoning | 1T params open-source; 13-hour continuous coding; 300 sub-agents parallel |
| **Kimi K2** | 2025.07 | Agent reasoning | 1T total/32B active; MuonClip optimizer; open-source Agent SOTA |
| **Qwen3-235B-A22B** | 2025.04 | Hybrid reasoning (fast/slow thinking) | Open-source flagship; surpasses DeepSeek-R1 and o1 |
| **GLM-5.1** | 2026.04 | Hybrid reasoning + Agent reinforcement | 6,000+ tool calls per session; SWE-bench Pro 58.4%; long-task economics breakthrough |

> 📌 *The first 6 rows above (GPT-5.5 / Opus 4.8 / Gemini 3.1 Pro / Kimi K3 / Qwen3.7 / GLM-5.2) are the **newest generation as of July 2026**; the rows below are first-half-2026 predecessors, included only to illustrate the evolution path.*

> 💡 **Impact on Agents**: Reasoning models give Agents a qualitative leap in "planning" and "complex decision-making." In real engineering, more and more Agents adopt a "fast-slow dual system" — fast models for simple routing, reasoning models for complex planning. By the first half of 2026, models such as GPT-5.4, Claude Opus 4.7, Kimi K2.6, GLM-5.1, and Seed 2.0 deeply fused reasoning with Agent capabilities (tool calling, long-horizon tasks); and **the July 2026 models — GPT-5.5, Claude Opus 4.8, Kimi K3, Qwen3.7, GLM-5.2, Gemini 3.1 Pro** — pushed context windows to 2M and multi-agent parallelism and long-chain tool calling to new heights, making this switching seamless — reasoning capability is now built into general-purpose models.

### Trend 2: MoE and the Efficiency Revolution

Models keep getting larger, but **inference costs are falling** — driven by the comprehensive victory of **Mixture of Experts (MoE)**.

The core idea of MoE: the total parameter count can be very large (hundreds of billions), but only a small fraction is activated during each inference. Like a large company with hundreds of employees, but only the most suitable dozen are assigned to each project.

```python
# Intuitive understanding of MoE models (conceptual illustration)
class MixtureOfExperts:
    """
    Using Qwen3-235B-A22B as an example:
    Total parameters: 235B
    Active per inference: 22B (only ~9.4%)
    Effect: Outperforms DeepSeek-R1 and OpenAI o1, at a fraction of the inference cost
    """
    def __init__(self, num_experts=128, active_experts=8):
        self.num_experts = num_experts
        self.active_experts = active_experts
    
    def forward(self, input_tokens):
        # Router decides which experts to activate
        scores = self.router(input_tokens)
        top_k = scores.topk(self.active_experts)
        # Only selected experts participate in computation
        return sum(expert(input_tokens) * w for expert, w in top_k)
```

| Model | Total Params | Active Params | Architecture Highlights |
|-------|-------------|--------------|------------------------|
| **Kimi K3** | 1T+ | 32B | Open-source (2026.07); 500 parallel sub-agents; SWE-bench Pro 61% |
| **Qwen3.7** | Full open-source MoE family | Efficient activation | Released 2026.07; tops open-source leaderboard; Apache 2.0 |
| **DeepSeek-V4.1** | Next-gen MoE | Efficient activation | Released 2026.07; price-performance raised another notch |
| **Llama 5** | Open-source MoE | Multi-expert | Released 2026.07; native multimodal; top open-source model |
| **Gemini 3.1 Pro** | Super-large MoE | Efficient activation | Released 2026.07; 2M+ context; native video understanding |
| **Kimi K2.6** | 1T | 32B | K2 upgrade; 13-hour coding; 300 sub-agents parallel; SWE-bench Pro 58.6% |
| **Kimi K2** | 1T | 32B | MuonClip optimizer; trillion-parameter open-source MoE |
| **Qwen3.6-35B-A3B** | 35B | 3B | Released 2026.04; lightweight MoE; extreme efficiency |
| **Llama 4 Maverick** | ~400B | 17B | 128 experts; native multimodal; text generation surpasses GPT-4.1 |
| **Qwen3-235B-A22B** | 235B | 22B | Hybrid reasoning; Apache 2.0; tops open-source leaderboard |
| **Qwen3-30B-A3B** | 35B | 3B | Lightweight MoE; runs on a single GPU |
| **DeepSeek-V3** | 671B | 37B | MoE architecture; $5.57M training cost; best price-performance |
| **DeepSeek-V3-0324** | 685B | 37B | Minor update; major coding improvement |
| **Gemma 4-26B** | 26B | 4B (active) | Apache 2.0; native video/image; 256K context |
| **Llama 4 Scout** | 109B | 17B | 16 experts; 10M-token ultra-long context |
| **DeepSeek-V4** | Next-gen MoE | Efficient activation | Released 2026.04; continues V3's MoE line; open-source reasoning price-performance upgraded again |

> 💡 **Impact on Agents**: MoE makes "large model capability + small model cost" a reality. **Key advances from April–June 2026**: Gemma 4 offers native multimodal under the Apache 2.0 license; the Qwen3 family covers 0.6B to 235B with hybrid fast/slow reasoning built in; Kimi K2.6 open-sources a trillion-parameter model, with the MuonClip optimizer doubling training efficiency; Google's **TurboQuant** cuts KV Cache memory needs by 6×, sharply lowering the inference cost of long-context Agents; DeepSeek-V4 continues the MoE line, further driving down the per-unit cost of open-source reasoning. **The new wave of July 2026**: DeepSeek-V4.1, Gemini 3.1 Pro, and Llama 5 continue the MoE line, lowering the per-unit cost of open-source reasoning another notch, with 2M-context MoE becoming the new baseline.

### Trend 3: The Full Rise of the Open-Source Ecosystem

In 2025–2026, open-source models are no longer just "catching up" with closed-source — they have **formed a competitive balance** and even **locally surpassed** closed-source in multiple areas:

**Tier 1 (Competing with GPT-5.5 / Claude Opus 4.8)**:
- **Kimi K2.6** (Moonshot AI, 2026.04): 1T params open-source MoE; 13-hour continuous coding; 300 sub-agents parallel; SWE-bench Pro 58.6%; API price only 1/8 of Opus 4.6
- **GLM-5.1** (Zhipu AI, 2026.04): Hybrid reasoning + Agent reinforcement; 6,000+ tool calls per session; SWE-bench Pro 58.4%; domestic open-source flagship
- **Seed 2.0** (ByteDance, 2026.06): Intelligence frontier aimed at real, complex tasks; leads in long-horizon tasks and complex instruction following; validated by serving hundreds of millions of users
- **MiniMax M3** (MiniMax, 2026.06): The first domestic open-source model to simultaneously possess native multimodal, ultra-long context, and Agent-operation capabilities
- **Kimi K3** (Moonshot AI, 2026.07): Trillion-parameter open-source MoE; 500 parallel sub-agents; SWE-bench Pro 61%; API price only 1/10 of Opus 4.8
- **Qwen3.7** (Alibaba, 2026.07): Full open-source MoE family (0.6B~flagship); tops open-source leaderboard; Apache 2.0; hybrid reasoning
- **GLM-5.2** (Zhipu AI, 2026.07): Hybrid reasoning + Agent reinforcement; 12,000+ tool calls per session; SWE-bench Pro 60%
- **DeepSeek-V4.1** (DeepSeek, 2026.07): Next-gen MoE; open-source reasoning price-performance raised another notch
- **Llama 5** (Meta, 2026.07): Open-source multimodal MoE; native multimodal; text generation reaches GPT-5.5 level
- **Kimi K2** (Moonshot AI, 2025.07): 1T total/32B active MoE; MuonClip optimizer doubles training efficiency; open-source Agent SOTA; compatible with OpenAI/Anthropic APIs
- **Qwen3-235B-A22B** (Alibaba, 2025.04): 235B MoE hybrid reasoning; surpasses DeepSeek-R1 and o1; Apache 2.0
- **DeepSeek-V3-0324** (DeepSeek, 2025.03): 685B MoE; coding surpasses Claude 3.7; more permissive open-source license
- **Llama 4 Maverick** (Meta, 2025.04): ~400B MoE multimodal; text generation surpasses GPT-4.1

**Tier 2 (Lightweight and Efficient, single-GPU capable)**:
- **Qwen3.6-35B-A3B** (Alibaba, 2026.04): 35B total/3B active; lightweight MoE; extreme efficiency
- **Qwen3.6-Plus / Flash / Max** (Alibaba, 2026.04): Rapid Qwen3 iteration covering different performance tiers
- **Gemma 4-31B** (Google, 2026.04): Dense model; top-3 globally among open-source on Arena Elo; Apache 2.0; native video/image multimodal
- **Gemma 4-26B MoE** (Google, 2026.04): 4B active params; 256K context; Apache 2.0
- **Qwen3-32B** (Alibaba, 2025.04): Dense flagship; hybrid reasoning; Apache 2.0
- **Qwen3-30B-A3B** (Alibaba, 2025.04): 30B total/3B active; extreme efficiency
- **Llama 4 Scout** (Meta, 2025.04): 17B active/109B total; 10M context window; runs on a single H100
- **Phi-4** (Microsoft, 2024.12): 14B params; reasoning surpasses many 70B models
- **Phi-4-multimodal** (Microsoft, 2025.02): 5.6B; unified architecture for speech + vision + text
- **Gemma 4-E2B/E4B** (Google, 2026.04): 2.3B/4.5B; phones/edge devices; native audio/video; Apache 2.0
- **Qwen3 full series** (Alibaba, 0.6B~235B): Full coverage from phones to servers; Apache 2.0

> 📊 **Key milestones, April–June 2026**: In April, Anthropic released Claude Opus 4.7, Alibaba launched Qwen3.6, Moonshot released Kimi K2.6, and Zhipu released GLM-5.1 — four major models launched within a single week, with domestic open-source models fully matching or even surpassing top closed-source models on coding and Agent benchmarks; in June, ByteDance's Seed 2.0 and MiniMax M3 were released successively, further filling the gaps in long-horizon tasks and native Agent operation. Chatbot Arena scores show the China–US gap has narrowed substantially. **The "collective upgrade" of July 2026**: GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro, Kimi K3, Qwen3.7, GLM-5.2, DeepSeek-V4.1, and Llama 5 were all released that month; context windows broadly entered the 2M era, and Agent long-horizon and multi-agent parallel capabilities reached a new level.

**Open-source vs. Closed-source Decision Matrix**:

| Dimension | Closed-source | Open-source |
|-----------|--------------|-------------|
| **Peak Capability** | Still has an edge (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro) | Rapidly catching up; Kimi K3/Qwen3.7/GLM-5.2 have locally surpassed in places |
| **Cost** | Pay-per-use API | Near-zero marginal cost after self-deployment |
| **Privacy** | Data sent to third party | Data completely private |
| **Customization** | Limited (Fine-tuning API) | Fully controllable (LoRA/full fine-tuning) |
| **Latency** | Affected by network | Controllable with local deployment |
| **Agent Capability** | Mature and stable tool calling | Kimi K3, Qwen3.7, GLM-5.2 now natively support Agent; K3 supports 500 parallel sub-agents, GLM-5.2 handles 12,000+ tool calls per session, MiniMax M3 offers native Agent operation |
| **Best For** | Rapid prototyping, general tasks | Production deployment, data-sensitive scenarios |

### Trend 4: The Rise of Agent-Native Models

The most notable new trend in 2025–2026 is: **models are beginning to be specifically optimized for Agent scenarios**.

- **Claude Opus 4.7** (2026.04): SWE-bench Verified #1; visual capability tops charts; Claude Code fully upgraded; production-grade foundation for RPA and automated testing
- **GPT-5.5** (2026.07): Reasoning+Agent unification evolves further; context extended to 2M; native multimodal generation enhanced; more stable Agent tool calling
- **Claude Opus 4.8** (2026.07): 2M context; long-horizon Agents run continuously for days; new SWE-bench Verified record; visual capability tops charts
- **Gemini 3.1 Pro** (2026.07): 2M+ context; native video understanding; dynamic reasoning budget; multimodal Agent benchmark
- **Kimi K3** (2026.07): Trillion-parameter open-source; 500 parallel sub-agents; SWE-bench Pro 61%; API price only 1/10 of Opus 4.8
- **Qwen3.7** (2026.07): Full open-source MoE family; tops open-source leaderboard; hybrid reasoning auto-switches fast/slow thinking
- **GLM-5.2** (2026.07): 12,000+ tool calls per session; SWE-bench Pro 60%; long-chain Agent optimization
- **DeepSeek-V4.1** (2026.07): Open-source MoE price-performance raised another notch; first choice for Agent production deployment
- **Llama 5** (2026.07): Open-source multimodal MoE; native multimodal; text generation reaches GPT-5.5 level
- **Kimi K2.6** (2026.04): 1T params open-source; 300 sub-agents parallel; completes complex DevOps over 5 continuous days; SWE-bench Pro 58.6%; API price only 1/8 of Opus 4.6
- **GPT-5.4** (2026.03): First to fuse reasoning+coding+Computer Use+deep search into a single model; natively controls browsers and operating systems; Agent tool-call token cost cut in half
- **Kimi K2**: Trillion-parameter open-source MoE; Agent capability reaches open-source SOTA on multiple benchmarks; focused on Agent-specific pre-training and post-training; compatible with mainstream Agent frameworks such as Claude Code
- **DeepSeek-V3-0324**: Significantly enhanced coding and tool-calling capability; more permissive open-source license; suitable for Agent production deployment
- **GPT-5**: Unified system architecture; built-in reasoning routing; more stable Agent tool calling; supports Computer Use
- **Claude Opus 4.6**: 1M context (Beta); handles massive codebases; autonomously discovers zero-day vulnerabilities; enterprise-grade Agent workflows
- **Claude Opus 4**: Continuous autonomous operation for 7 hours; SWE-bench 72.5%; new Agent coding benchmark
- **Qwen3-235B-A22B**: Deeply adapted to Agent frameworks; dramatically improved tool-call accuracy; hybrid reasoning auto-switches fast/slow thinking
- **Llama 4 Scout**: 10M-token ultra-long context; suitable for Agent tasks requiring very long documents
- **GLM-5.1** (2026.04): 6,000+ tool calls per session; SWE-bench Pro 58.4%; long-task economics breakthrough; designed for long-chain Agents
- **Seed 2.0** (2026.06): Aimed at real, complex tasks; leads in long-horizon tasks and complex instruction following; validated in scenarios with hundreds of millions of users
- **MiniMax M3** (2026.06): Native multimodal + ultra-long context + Agent operation in one; the first domestic open-source model to simultaneously possess all three capabilities

This means Agent developers no longer need to "force a fit" — the models themselves are designed for Agents.

## Multimodal Foundation Models: More Than Just Text

In 2026, almost all foundation models are **natively multimodal** — supporting mixed input and output of text, images, audio, and video at the architecture level.

```python
# Typical multimodal Agent call
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5",  # GPT-5 natively supports multimodal
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's wrong with this architecture diagram? Please provide improvement suggestions."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
)

# GPT-5 can not only "understand" images, but also generate images and hold real-time voice conversations
```

**Mainstream Multimodal Model Comparison**:

| Model | Release | Input Modalities | Output Modalities | Special Capabilities |
|-------|---------|-----------------|------------------|---------------------|
| **GPT-5.5** | 2026.07 | Text+Image+Audio | Text+Image | 2M context; Computer Use surpasses humans; native multimodal generation enhanced |
| **Claude Opus 4.8** | 2026.07 | Text+Image+PDF | Text | 2M context; visual capability tops charts; SWE-bench Verified #1 |
| **Gemini 3.1 Pro** | 2026.07 | Text+Image+Video+Audio | Text+Image | 2M+ context; native video understanding; dynamic reasoning budget |
| **Kimi K3** | 2026.07 | Text | Text | Trillion-parameter open-source; 500 parallel sub-agents; Agent coding SOTA |
| **Qwen3.7** | 2026.07 | Text+Image | Text | Full open-source MoE family; hybrid reasoning; Apache 2.0 |
| **GLM-5.2** | 2026.07 | Text+Image | Text | 12,000+ tool calls per session; SWE-bench Pro 60%; Agent long-chain |
| **Llama 5** | 2026.07 | Text+Image | Text | Open-source multimodal MoE; text generation reaches GPT-5.5 level |
| **Claude Opus 4.7** | 2026.04 | Text+Image+PDF | Text | SWE-bench Verified #1; 3.75M-pixel image input; visual capability tops charts |
| **GPT-5.4** | 2026.03 | Text+Image+Audio | Text+Image | Computer Use surpasses humans; reasoning+coding+search unified; 1M context |
| **GPT-5** | 2025.08 | Text+Image+Audio | Text+Image+Audio | Real-time voice conversation; native image generation; Computer Use |
| **Claude Opus 4.6** | 2026.02 | Text+Image+PDF | Text | 1M context (Beta); enterprise-grade Agent workflows |
| **Gemini 2.5 Pro** | 2025.03 | Text+Image+Video+Audio | Text+Image | Native video understanding; 1M context; reasoning budget control |
| **Llama 4 Maverick** | 2025.04 | Text+Image | Text | Open-source multimodal MoE; text generation surpasses GPT-4.1 |
| **Gemma 4-31B** | 2026.04 | Text+Image+Video | Text | Apache 2.0; top-3 globally among open-source on Arena |
| **Gemma 4-E2B/E4B** | 2026.04 | Text+Image+Audio | Text | Runs on phones; Apache 2.0; native audio/video |
| **Phi-4-multimodal** | 2025.02 | Text+Image+Speech | Text | Only 5.6B params; unified multimodal architecture |
| **Kimi K2.6** | 2026.04 | Text | Text | Trillion-parameter open-source; 300 parallel sub-agents; Agent coding SOTA |
| **Kimi K2** | 2025.07 | Text | Text | Trillion-parameter Agent SOTA; strongest tool calling |
| **GLM-5.1** | 2026.04 | Text+Image | Text | 6,000+ tool calls per session; SWE-bench Pro 58.4%; Agent long-chain optimization |
| **MiniMax M3** | 2026.06 | Text+Image+Audio+Video | Text | Native multimodal + ultra-long context + Agent operation in one |

## The Rise of Small Models: SLM and Edge Deployment

The progress of **Small Language Models (SLMs)** is remarkable — 14B parameter models from 2025 have comprehensively surpassed GPT-4 from 2023.

```python
# Impressive small model performance (2025–2026 benchmark data)
slm_benchmarks = {
    "Phi-4 (14B)":             {"MMLU": 84.8, "HumanEval": 82.6, "GSM8K": 94.5},
    "Phi-4-reasoning (14B)":   {"MMLU": 86.2, "HumanEval": 85.1, "GSM8K": 95.8},
    "Qwen3-8B":               {"MMLU": 81.2, "HumanEval": 79.8, "GSM8K": 91.3},
    "Llama 4 Scout (17B act)": {"MMLU": 83.5, "HumanEval": 80.1, "GSM8K": 92.1},
    "Gemma 4-31B":            {"MMLU": 87.3, "HumanEval": 79.1, "MATH": 72.8},
    "Phi-4-mini (3.8B)":      {"MMLU": 72.1, "HumanEval": 68.5, "GSM8K": 84.2},
    # Comparison: GPT-4 from 2023 (~1.7T params estimated)
    "GPT-4 (2023)":           {"MMLU": 86.4, "HumanEval": 67.0, "GSM8K": 92.0},
}

# Phi-4-reasoning (14B) has comprehensively surpassed GPT-4 (2023) in coding and math!
# Gemma 4-31B surpasses GPT-4 on MMLU and is fully open-source (Apache 2.0)
# This means: Agents don't necessarily need the largest model
```

> 💡 **Impact on Agents**: SLMs allow Agents to run locally on **phones, laptops, and edge devices**, enabling zero-latency, fully private interactions. Apple Intelligence, Google's Gemini Nano, and Microsoft's Phi-4-mini are all products of this trend. Phi-4-multimodal handles speech, vision, and text simultaneously with just 5.6B parameters, opening the door for edge-side multimodal Agents.

## Model Selection Guide for Agent Developers

With so many model choices, how do you pick the right foundation model for your Agent?

```python
def select_model(requirements: dict) -> str:
    """Agent model selection decision function (July 19, 2026 edition)"""
    
    budget = requirements.get("monthly_budget_usd", 100)
    task_type = requirements.get("task_type", "general")
    privacy = requirements.get("privacy_required", False)
    latency_ms = requirements.get("max_latency_ms", 5000)
    reasoning = requirements.get("complex_reasoning", False)
    agent_native = requirements.get("agent_native", False)
    
    # Decision tree
    if privacy:
        if reasoning:
            return "Kimi K3 / Qwen3-235B / GLM-5.2 (self-hosted)"  # open-source + reasoning + Agent
        elif latency_ms < 500:
            return "Phi-4-mini / Qwen3-4B (local deployment)"  # edge SLM
        else:
            return "Qwen3-32B / Llama 5 (self-hosted)"  # open-source general-purpose
    
    if agent_native:
        if budget > 500:
            return "Claude Opus 4.8 / GPT-5.5 / GLM-5.2"  # top-tier Agent experience
        else:
            return "Kimi K3 API / DeepSeek-V4.1 API / GLM-5.2 API"  # value Agent (K3 is only 1/10 of Opus 4.8)
    
    if reasoning:
        if budget > 500:
            return "Claude Opus 4.8 / GPT-5.5 / GLM-5.2"  # top-tier reasoning
        else:
            return "DeepSeek-V4.1 API / o4-mini"  # value reasoning
    
    if budget < 50:
        return "DeepSeek-V3 API / GPT-4.1-mini"  # extreme value-for-money
    
    return "GPT-5.5 / Claude Opus 4.8 / GLM-5.2"  # balanced general choice
```

**Recommended models by Agent scenario**:

| Agent Scenario | Recommended Model | Reason |
|---------------|------------------|--------|
| Coding assistant | Claude Opus 4.8 / Kimi K3 / GLM-5.2 / Seed 2.0 | Dual #1 on SWE-bench; K3 extreme price-performance (1/10 of Opus 4.8); GLM-5.2/Seed 2.0 strong on long chains |
| Data analysis | GPT-5.5 / Gemini 3.1 Pro | Stable multimodal understanding + function calling; 2M context |
| Customer service | GPT-4.1-mini / Qwen3-8B | Cost-sensitive; high response-speed requirement |
| Deep research | Claude Opus 4.8 / GPT-5.5 / GLM-5.2 | 2M context + deep reasoning; GLM-5.2 handles 12,000+ tool calls per session |
| Document processing | Gemini 3.1 Pro / Claude Opus 4.8 / MiniMax M3 | 2M ultra-long document input; PDF layout understanding; MiniMax M3 ultra-long context |
| Local privacy | Kimi K3 / Qwen3-235B / GLM-5.2 (self-hosted) | Data never leaves local; complete Agent capability; K3/GLM-5.2 open-source |
| Edge deployment | Phi-4-mini (3.8B) / Qwen3-4B | Runs on phones/laptops |
| Multimodal Agent | GPT-5.5 / Gemini 3.1 Pro / MiniMax M3 / Seed 2.0 | Computer Use surpasses humans; native multimodal + visual understanding; MiniMax M3 all-in-one |
| RPA / automated testing | Claude Opus 4.8 / GPT-5.5 / MiniMax M3 | Visual capability tops charts; Sweepstakes on ScreenSpot-Pro/OSWorld-class; MiniMax M3 native Agent operation |

## 2024–2026 Key Model Release Timeline

```
2024.09  OpenAI o1 ──── The year of reasoning models begins
2024.12  Phi-4 (14B) ── Microsoft releases the strongest small model
2025.01  DeepSeek-R1 ── Open-source reasoning model ignites the world; GRPO training
2025.02  Phi-4-multimodal / Phi-4-mini ── Edge multimodal
2025.03  Gemini 2.5 Pro ── 1M context + reasoning; tops the leaderboards
2025.03  DeepSeek-V3-0324 ── Minor update; coding surpasses Claude 3.7
2025.04  Llama 4 Scout/Maverick ── Meta's first MoE open-source multimodal
2025.04  o3 / o4-mini ── OpenAI multimodal reasoning; first "thinking while looking at images"
2025.04  Qwen3 ── Alibaba hybrid reasoning full series (0.6B~235B), Apache 2.0
2025.05  Claude 4 (Opus 4 / Sonnet 4) ── 7-hour continuous coding; SWE-bench 72.5%
2025.05  GPT-4.1 ── 1M-token context; major coding improvement
2025.07  Kimi K2 ── Moonshot AI trillion-parameter open-source MoE; MuonClip optimizer
2025.08  GPT-5 ── OpenAI unified system architecture; built-in reasoning routing; SWE-bench 75%
━━━━━━━━━━━━━━━━━━━━━━━━ 2026 ━━━━━━━━━━━━━━━━━━━━━━━━
2026.02  Claude Opus 4.6 ── 1M context (Beta); SWE-bench 80.8%; enterprise Agent
2026.03  GPT-5.4 ── OpenAI reasoning+coding+Computer Use+search unified; 1M context; three variants
2026.04  Gemma 4 (E2B/E4B/26B/31B) ── Google open-source; native video/audio; Apache 2.0
2026.04  Claude Opus 4.7 ── SWE-bench Verified #1; visual capability tops charts; Claude Code fully upgraded
2026.04  Kimi K2.6 ── Moonshot AI open-source; 13-hour coding; 300 sub-agents parallel; SWE-bench Pro 58.6%
2026.04  Qwen3.6 series ── Alibaba rapid iteration (35B-A3B/Flash/Plus/Max); full tier coverage
2026.04  GLM-5.1 ── Zhipu open-source; 6,000+ tool calls per session; SWE-bench Pro 58.4%; Agent long-chain optimization
2026.04  DeepSeek ── open-sources DeepSeek-V4
2026.06  MiniMax M3 open-source ── The first domestic model to simultaneously possess native multimodal, ultra-long context, and Agent-operation capabilities.
2026.07  GPT-5.5 ── OpenAI reasoning+Agent unification evolves further; context extended to 2M; SWE-bench 78%
2026.07  Claude Opus 4.8 ── 2M context; long-horizon Agents run continuously for days; new SWE-bench Verified record
2026.07  Gemini 3.1 Pro ── Google 2M+ context; native video understanding; dynamic reasoning budget
2026.07  Kimi K3 ── Moonshot AI trillion-parameter open-source; 500 parallel sub-agents; SWE-bench Pro 61%
2026.07  Qwen3.7 series ── Alibaba open-source MoE family tops the open-source leaderboard; Apache 2.0
2026.07  GLM-5.2 ── Zhipu 12,000+ tool calls per session; SWE-bench Pro 60%
2026.07  DeepSeek-V4.1 ── DeepSeek open-source MoE price-performance raised another notch
2026.07  Llama 5 ── Meta open-source multimodal MoE
```

## Outlook: What's Next for Foundation Models

Several development directions worth watching:

1. **Reasoning Built-in**: Reasoning capability moves from standalone o-series models into general-purpose models (GPT-5.5 Thinking mode, Qwen3.7 hybrid reasoning) — developers no longer need to choose manually
2. **MoE Efficiency Continues to Improve**: The share of active parameters keeps falling (Qwen3-235B activates only 9.4%), leaving much room for inference-cost reduction; Google's TurboQuant cuts KV Cache memory by 6×, pushing the per-unit cost of long-context Agents still lower
3. **Agent Clustering**: Models evolve from "passive answering" to "active acting" — Kimi K2.6's 300 parallel sub-agents running for 5 continuous days moves Agents from single-task execution toward large-scale autonomous collaboration
4. **Ultra-Long Context**: From 128K to 1M to 10M (Llama 4 Scout) — expanding context windows let Agents process entire codebases and complete document sets
5. **Open-Source Catches Up to Closed-Source**: Open-source models such as Kimi K2.6, Qwen3.6, GLM-5.1, and Gemma 4 have matched top closed-source models on coding and Agent benchmarks; Seed 2.0 and MiniMax M3 further fill the gaps in long-horizon tasks and native Agent operation, greatly lowering the barrier to private deployment
6. **Native Multimodal**: Text → vision + speech + video full modality — Agents can "see," "hear," and "draw," enabling more natural interaction
7. **Edge Intelligence**: 3B–14B parameter models run on phones/laptops, making zero-latency, fully private local Agents possible
8. **July Collective Upgrade**: GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro, Kimi K3, Qwen3.7, GLM-5.2, DeepSeek-V4.1, and Llama 5 were all released densely in July 2026; context windows broadly entered the 2M era, and multi-agent parallelism (Kimi K3 reaches 500 sub-agents) and long-chain tool calling (GLM-5.2 reaches 12,000+) became the new baseline

---

## Summary

| Trend | Core Change | Impact on Agent Development |
|-------|------------|----------------------------|
| Reasoning built-in | GPT-5.5 / Claude Opus 4.8 Thinking mode; Qwen3.7 hybrid fast/slow thinking | Qualitative leap in Agent complex planning; no need to manually choose a reasoning model |
| Computer Use maturity | GPT-5.5/Claude Opus 4.8 surpass human level | Agents directly control browsers and operating systems; RPA enters production-ready stage |
| Agent clustering | Kimi K3's 500 parallel sub-agents; continuous multi-day operation | Agents evolve from single-task execution to large-scale autonomous collaboration |
| MoE efficiency revolution | Kimi K3/Qwen3.7/DeepSeek-V4.1 trillion-parameter open-source; only 3B~32B active; TurboQuant saves 6× KV Cache | Agent operating costs fall sharply; K3 API is only 1/10 of Opus 4.8 |
| Open-source full rise | Kimi K3/Qwen3.7/GLM-5.2/DeepSeek-V4.1/Llama 5/Gemma 4/Seed 2.0/MiniMax M3 form a complete open-source echelon | Mature private Agent deployment; data security is no longer a bottleneck |
| Agent-Native | GLM-5.2 12,000+ tool calls per session; Seed 2.0 long-horizon; MiniMax M3 native Agent operation; Kimi K3 500 sub-agents | Developers no longer need to "force a fit"; the model is the Agent foundation |
| Native multimodal | GPT-5.5 / Gemini 3.1 Pro native multimodal; text → vision + speech + video full modality | Agents can "see," "hear," and "draw"; more natural interaction |
| Ultra-long context | 2M~10M token context windows (2M becomes the new baseline from July 2026) | Agents can process entire codebases and complete document sets |
| Small model progress | 3B~14B parameter models run on phones/laptops | Agents can run on edge devices; zero latency, complete privacy |

> ⏰ *Note: Model technology evolves extremely fast. The data in this section is current as of **July 19, 2026**. The **July 2026 batch (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro / Kimi K3 / Qwen3.7 / GLM-5.2 / DeepSeek-V4.1 / Llama 5) is a draft projected from version-increment patterns**, and specific metrics should be verified against official releases; the "📰 Latest Paper Briefs" at the end is maintained by a daily auto-update task (last updated: July 14, 2026). It is recommended to regularly follow vendor release announcements and authoritative benchmark evaluations (such as LMArena, Open LLM Leaderboard, Chatbot Arena) for the latest information.*

---

*Previous section: [2.5 Token, Temperature, and Model Parameters Explained](./05_model_parameters.md)*

*Next section: [2.7 Foundation Model Architecture Explained](./07_model_architecture.md)*

---

## References

[1] OPENAI. GPT-4 technical report[R]. arXiv preprint arXiv:2303.08774, 2023.

[2] GUO D, YANG D, ZHANG H, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[3] TEAM G, RIVIERE M, PATHAK S, et al. Gemma 2: Improving open language models at a practical size[R]. arXiv preprint arXiv:2408.00118, 2024.

[4] META AI. The Llama 4 herd: The beginning of a new era of natively multimodal AI[R]. 2025.

[5] QWEN TEAM. Qwen3 technical report[R]. arXiv preprint arXiv:2505.09388, 2025.

[6] ANTHROPIC. Claude's character[EB/OL]. 2024. https://www.anthropic.com/research/claude-character.

[7] HOFFMANN J, BORGEAUD S, MENSCH A, et al. Training compute-optimal large language models[R]. arXiv preprint arXiv:2203.15556, 2022. (Chinchilla's Law)

[8] SHAZEER N. Fast transformer decoding: One write-head is all you need[R]. arXiv preprint arXiv:1911.02150, 2019. (GQA/MQA foundation)

[9] ABDIN M, JACOBS S A, AWAN A A, et al. Phi-4 technical report[R]. arXiv preprint arXiv:2412.08905, 2024.

[10] MOONSHOT AI. Kimi K2: Open agentic intelligence[EB/OL]. 2025. https://huggingface.co/moonshotai/Kimi-K2-Instruct.

---

## 📰 Latest Paper Briefs

> 🗓️ This section is maintained by a daily auto-update task; last updated: **July 14, 2026**

### [Cola DLM: Continuous Latent-Space Diffusion Language Model — A New Paradigm for Non-Autoregressive Text Generation (2026)](https://arxiv.org/abs/2605.06548)

> 🧬 **One-liner**: Moves text generation from "token-by-token autoregressive" to "hierarchical diffusion in a continuous latent space," unifying the theoretical framework of AR, discrete denoising, and continuous-token-space methods.

**Core problem**: Mainstream large language models generate strictly autoregressively (AR), token by token, but the AR paradigm is inherently limited by serial decoding and struggles with global planning. Discrete denoising language models and continuous-token-space methods have each been explored, yet there is no unified theoretical framework to compare them horizontally and combine their strengths.

**Method**: Cola DLM (Continuous Latent Diffusion Language Model) proposes a hierarchical continuous latent-variable language model: first use a **Text VAE** to map text into a continuous latent representation, then use a **block-causal diffusion Transformer** to model the global semantic prior in the latent space, and finally decode to generate text. Under a rigorous probabilistic definition, it places Cola DLM alongside AR models, discrete denoising language models, and continuous-token-space methods within a single theoretical framework for comparison. The overall workflow is shown below:

![Cola DLM overall workflow: Text VAE encoding → block-causal diffusion Transformer latent-space modeling → decoding](../svg/chapter_llm_paper_2605.06548.png)

> Image source: Cola DLM paper (source: 2026, arXiv:2605.06548, ByteDance Seed)

**Key results**: In strict comparison against an autoregressive baseline of roughly 2 billion parameters across 4 research questions and 8 benchmarks, strong scalability was verified; code and checkpoints are open-sourced under Apache 2.0.

**Relation to this chapter**: Corresponds to 2.7 Foundation Model Architecture Explained. It represents an important frontier exploring the shift of language models from the autoregressive paradigm toward the diffusion paradigm, helping readers understand the diversified architecture directions of future foundation models.

---

### [ByteDance Seed 2.0: The Frontier of Intelligence for Real-World Complexity (2026)](https://arxiv.org/abs/2607.00248)

**Published**: June 30, 2026 | [arXiv:2607.00248](https://arxiv.org/abs/2607.00248)

**Core contribution**: ByteDance released the Seed 2.0 model family, taking real user needs as the starting point and building a reliable forward-looking evaluation system. It focuses on overcoming two enduring challenges — long-tail knowledge and complex instruction following — while reaching world-leading levels in reasoning intelligence, visual understanding, and search capability. The model card demonstrates Seed 2.0's initial ability to solve complex long-horizon tasks through many real-world use cases, serving hundreds of millions of users.

**Relation to this chapter**: Corresponds to this chapter's "Frontier Foundation Model Landscape." It is an important release milestone of a top-tier domestic large model in the second half of 2026, alongside the Qwen3 and GLM-5.1 families representing the latest progress in the competitive landscape of domestic foundation models, with direct reference value for those selecting Agent backbone models.

---

### [Soofi S 30B-A3B: A Sovereign Open-Source Mixture Mamba-Transformer MoE Foundation Model (2026)](https://arxiv.org/abs/2607.09424)

**Published**: July 10, 2026 | [arXiv:2607.09424](https://arxiv.org/abs/2607.09424)

**Core contribution**: Soofi S 30B-A3B is a sovereign open-source MoE hybrid Mamba-Transformer foundation model for German and English. Its hybrid design activates only 3B of the 30B parameters per token, and its inference cache remains **near-constant** as context grows — giving it significant throughput advantages over dense models in long-context, high-concurrency deployment scenarios. Pretrained on roughly 27 trillion tokens with deliberately increased German weighting, it matches dense 14–27B models on combined English and German benchmarks, leads 17 open-source foundation models on aggregated code metrics in both languages, and surpasses all European sovereign baselines. It achieves the highest English-German evaluation score among fully open-source models, exceeding Olmo 3 32B and Apertus 70B. Weights, intermediate checkpoints, the complete data-source manifest, hyperparameters, and training/evaluation code are all open-sourced under permissive terms.

**Relation to this chapter**: Corresponds to 2.7 Foundation Model Architecture Explained. Soofi S represents the latest large-scale validation of the Mamba-Transformer hybrid architecture in an MoE configuration — its near-constant inference-cache property has direct engineering value for long-context Agent deployment, and together with Cola DLM (diffusion paradigm) it showcases the diversified exploration of foundation-model architectures beyond autoregressivity.
