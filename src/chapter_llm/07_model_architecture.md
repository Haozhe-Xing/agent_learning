# 基座模型架构详解

> 🏗️ *"了解模型的工作原理让你做出更好的判断，而了解模型的架构演进，能让你理解整个行业在朝哪里走。"*

在 3.1 节中，我们用直觉理解了 LLM 的基本原理——Transformer、注意力机制、Token 预测。本节将深入一层，带你看看**模型的"骨架"长什么样**——从 Decoder-Only 架构到注意力机制的变体、归一化方案、位置编码、MoE 路由，再到 2024—2026 年各大开源模型的具体架构选择。

这些知识不是"学术装饰"——当你要为 Agent 选择部署模型、评估推理成本、或理解为什么某个模型在长文本上表现更好时，**对架构的理解就是你的底层判断力**。

## 现代 LLM 的标准"骨架"：Decoder-Only Transformer

2023 年以来，几乎所有主流 LLM 都采用了 **Decoder-Only** 架构。这与最初 Transformer 的 Encoder-Decoder 结构不同——它只保留了解码器部分，通过**因果注意力掩码（Causal Mask）**保证每个 Token 只能"看到"它左边的内容。

```python
# 因果注意力掩码的直觉
# 在生成 "我 喜欢 吃 苹果" 时：
#
#        我  喜欢  吃  苹果
#  我    [✓]  [✗]  [✗]  [✗]     ← "我" 只能看到自己
#  喜欢  [✓]  [✓]  [✗]  [✗]     ← "喜欢" 能看到 "我" 和自己
#  吃    [✓]  [✓]  [✓]  [✗]     ← "吃" 能看到前面所有词
#  苹果  [✓]  [✓]  [✓]  [✓]     ← "苹果" 能看到整个序列
#
# 对角线以上的 [✗] 就是因果掩码——防止"偷看未来"
```

为什么选 Decoder-Only？

| 架构 | 代表模型 | 适合任务 | 为何 LLM 不用 |
|------|---------|---------|-------------|
| Encoder-Only | BERT | 理解类（分类、NER） | 不能自回归生成 |
| Encoder-Decoder | T5, BART | 翻译、摘要 | 复杂度高，不利于超大规模扩展 |
| **Decoder-Only** | **GPT, Llama, DeepSeek** | **所有生成任务** | ✅ 架构简洁，易扩展，训练高效 |

一个标准的 Decoder-Only Transformer 层长这样：

![Decoder-Only Transformer一层的标准结构](../svg/chapter_llm_07_decoder_layer.svg)

```python
class TransformerDecoderLayer:
    """现代 LLM 一层的标准结构（2024+ 共识版）"""
    
    def forward(self, x):
        # 1. Pre-Norm + 注意力
        residual = x
        x = self.norm1(x)                  # RMSNorm (Pre-Normalization)
        x = self.attention(x)              # 因果自注意力 (GQA/MLA)
        x = residual + x                   # 残差连接
        
        # 2. Pre-Norm + FFN
        residual = x
        x = self.norm2(x)                  # RMSNorm
        x = self.ffn(x)                    # SwiGLU 前馈网络
        x = residual + x                   # 残差连接
        
        return x
```

接下来，我们逐个拆解每个组件的技术演进。

## 注意力机制的演进：MHA → GQA → MLA

![注意力机制演进：MHA→GQA→MLA](../svg/chapter_llm_07_attention_evolution.svg)

注意力机制是 Transformer 的"心脏"。从 2017 到 2025 年，它经历了三代关键变体——驱动力是**推理效率**，尤其是 **KV-Cache 的显存压力**。

### MHA：经典多头注意力

最初 Transformer 使用的多头注意力（Multi-Head Attention），每个头都有独立的 Query、Key、Value 投影：

```python
class MultiHeadAttention:
    """MHA: 每个头都有独立的 Q、K、V"""
    def __init__(self, d_model=4096, n_heads=32):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 128
        
        # 每个头独立的 Q、K、V 投影
        self.wq = Linear(d_model, n_heads * self.head_dim)  # 32 组 Q
        self.wk = Linear(d_model, n_heads * self.head_dim)  # 32 组 K ← 这很多！
        self.wv = Linear(d_model, n_heads * self.head_dim)  # 32 组 V ← 这也很多！
    
    # KV Cache 大小 = n_layers × n_heads × seq_len × head_dim × 2
    # 对于 Llama-2-70B (80层, 64头, 128维): 
    # 每 1K token 的 KV Cache ≈ 2.5 GB！
```

**问题**：推理时要缓存所有层、所有头的 K 和 V——当序列长度增长时，显存爆炸。

### GQA：分组查询注意力

Llama 2（2023）引入了 **Grouped-Query Attention**——让多个 Query 头共享一组 Key-Value：

```python
class GroupedQueryAttention:
    """GQA: 多个 Q 头共享一组 KV，大幅减少 KV Cache"""
    def __init__(self, d_model=4096, n_q_heads=32, n_kv_heads=8):
        self.n_q_heads = n_q_heads     # 32 个 Query 头
        self.n_kv_heads = n_kv_heads   # 8 个 KV 头（每 4 个 Q 共享 1 个 KV）
        self.head_dim = d_model // n_q_heads
        
        self.wq = Linear(d_model, n_q_heads * self.head_dim)   # 32 组 Q
        self.wk = Linear(d_model, n_kv_heads * self.head_dim)  # 只有 8 组 K！
        self.wv = Linear(d_model, n_kv_heads * self.head_dim)  # 只有 8 组 V！
    
    # KV Cache 缩小为 MHA 的 1/4（32→8 头）
    # Llama 3 70B: KV Cache 从 2.5GB/1K → ~0.6GB/1K
```

GQA 几乎**不损失模型质量**（大量消融实验验证），却将 KV Cache 减少了 4~8 倍。这就是为什么 2023 年之后几乎所有主流模型都采用了 GQA。

**哪些模型用了 GQA？**
- Llama 2/3/4、Qwen 2/2.5/3、Gemma 2/3、Mistral/Mixtral、Phi-3/4

### MLA：多头潜在注意力（DeepSeek 创新）

DeepSeek-V2（2024）提出了更激进的方案——**Multi-head Latent Attention**。它不是减少 KV 头的数量，而是**把整个 KV 压缩到一个低维潜在空间**：

```python
class MultiHeadLatentAttention:
    """
    MLA: DeepSeek 的核心创新
    不是"共享头"，而是"压缩 KV 到低维空间"
    """
    def __init__(self, d_model=7168, n_heads=128, kv_lora_rank=512):
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank  # KV 压缩到 512 维
        
        # KV 先下投影到低维空间
        self.kv_down_proj = Linear(d_model, kv_lora_rank)     # 7168 → 512
        # 推理时按需上投影恢复
        self.kv_up_proj = Linear(kv_lora_rank, n_heads * 128 * 2)  # 512 → 全尺寸 KV
    
    def forward(self, x, kv_cache=None):
        # 压缩 KV：只缓存 512 维的潜在向量！
        compressed_kv = self.kv_down_proj(x)  # [batch, seq, 512]
        
        # 推理时存入缓存的是压缩后的向量
        if kv_cache is not None:
            kv_cache.store(compressed_kv)  # 只存 512 维！
        
        # 计算注意力时实时解压
        full_kv = self.kv_up_proj(compressed_kv)
        k, v = full_kv.chunk(2, dim=-1)
        # ... 正常注意力计算
```

**效果有多惊人？**

| 注意力类型 | KV Cache / Token | 对比 MHA |
|-----------|-----------------|---------|
| MHA（Llama 2 级别） | ~2.5 GB / 1K tokens | 基准 |
| GQA（Llama 3 级别） | ~0.6 GB / 1K tokens | 减少 75% |
| **MLA（DeepSeek-V3）** | ~0.04 GB / 1K tokens | **减少 98.6%** |

MLA 让 DeepSeek-V3（671B 参数）可以在相对有限的硬件上处理极长上下文——这是 GQA 无法做到的。

### 三代注意力机制对比

```
MHA:    Q₁ → K₁,V₁    Q₂ → K₂,V₂    Q₃ → K₃,V₃    Q₄ → K₄,V₄
        每个 Q 对应独立 KV                 → KV Cache 最大

GQA:    Q₁ ─┐          Q₃ ─┐
        Q₂ ─┤→ K₁,V₁   Q₄ ─┤→ K₂,V₂
        多个 Q 共享一组 KV                  → KV Cache 缩小 4~8x

MLA:    Q₁ ─┐
        Q₂ ─┤→ [压缩向量 c] → 实时解压 → K,V
        Q₃ ─┤   (512 维)
        Q₄ ─┘
        所有 KV 压缩为低维潜在表示           → KV Cache 缩小 ~70x
```

## 归一化的演进：LayerNorm → RMSNorm + Pre-Norm

### 从 Post-Norm 到 Pre-Norm

最初的 Transformer 使用 **Post-Normalization**——先计算注意力/FFN，再做归一化。GPT-2（2019）发现把归一化放在注意力/FFN**之前**（Pre-Normalization）能显著改善深层网络的训练稳定性：

```python
# Post-Norm (原始 Transformer，已淘汰)
x = x + Attention(x)
x = LayerNorm(x)        # 归一化在后面

# Pre-Norm (现代标准)
x = x + Attention(RMSNorm(x))  # 归一化在前面
# 梯度可以通过残差连接直接回传，不会被归一化层"阻断"
```

### 从 LayerNorm 到 RMSNorm

标准 LayerNorm 需要计算均值和方差：

```python
# LayerNorm: 减均值，除标准差
def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return gamma * (x - mean) / sqrt(var + eps) + beta

# RMSNorm: 只除 RMS（均方根），去掉均值中心化
def rms_norm(x, gamma):
    rms = sqrt(mean(x ** 2) + eps)
    return gamma * x / rms
    # 没有 mean 计算，没有 beta 偏置 → 更快！
```

RMSNorm 的优势：
- **更快**：省去了均值计算和偏置参数
- **效果相当**：大量实验证明在 LLM 训练中与 LayerNorm 表现持平
- **硬件友好**：更简单的计算 → 更好的 GPU 核优化

> 📊 **行业共识**：在 53 个被分析的 Transformer 模型中，**77.4%** 采用了 RMSNorm。2023 年后发布的主流模型几乎 100% 使用 Pre-Norm + RMSNorm。

## 位置编码的演进：绝对编码 → RoPE

Transformer 架构本身对 Token 的顺序是"无感"的——它不知道"苹果"在"吃"的前面还是后面。位置编码就是告诉模型"Token 在序列中的位置"。

### RoPE：旋转位置编码

2024—2026 年的事实标准是 **RoPE（Rotary Position Embeddings）**，由 Su 等人在 2021 年提出：

```python
# RoPE 的核心思想：用旋转矩阵编码位置信息
# 
# 关键洞察：两个向量的点积（注意力分数）
# 在旋转后只依赖于它们的"相对位置差"
#
# 数学表达（简化版）：
# q_m · k_n = f(q, k, m-n)  
#             ↑ 只依赖相对位置 m-n

def apply_rope(x, position_ids, dim):
    """对 Q 和 K 应用旋转位置编码"""
    # 生成频率
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    # 计算角度 = 位置 × 频率
    angles = position_ids.unsqueeze(-1) * freqs
    
    # 将向量拆成对，每对做二维旋转
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.cat([
        x1 * cos - x2 * sin,  # 旋转变换
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated
```

**RoPE 的优势**：
1. **相对位置感知**：注意力分数自然编码了 Token 之间的相对距离
2. **无需学习参数**：纯数学变换，不增加模型参数量
3. **可外推**：通过调整频率基数（base），可以扩展到训练时未见过的更长序列
4. **与 FlashAttention 兼容**：易于集成到高效注意力内核中

### 上下文扩展：YaRN 与 NTK-aware 缩放

RoPE 的一个关键实践问题是**如何将模型推理到训练时没见过的更长序列**：

```python
# 模型训练时 max_seq_len = 8192
# 但你想在 128K 甚至 1M 上下文下使用它

# 方法 1: NTK-aware 缩放（调整频率基数）
def ntk_scaled_rope(dim, max_position, base=10000, scaling_factor=16):
    """NTK-aware 缩放：提高 base，让高频分量保持精度"""
    new_base = base * (scaling_factor ** (dim / (dim - 2)))
    freqs = 1.0 / (new_base ** (torch.arange(0, dim, 2) / dim))
    return freqs

# 方法 2: YaRN（Yet another RoPE extensioN）
# 结合了 NTK 缩放 + 注意力分数温度修正
# Llama 4 Scout 用 YaRN 实现了 10M token 上下文！
```

> 📊 **行业共识**：在被分析的 53 个模型中，**69.8%** 采用了 RoPE。2022 年后的 Decoder-Only LLM 中，RoPE 是绝对的主流选择。

## 激活函数与 FFN：SwiGLU 的统治

Transformer 中每一层除了注意力之外，还有一个**前馈网络（FFN/MLP）**。它的激活函数经历了显著演进：

```python
# 经典 FFN：两层线性变换 + ReLU
class ClassicFFN:
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))
        # 参数量: 2 × d_model × d_ff (通常 d_ff = 4 × d_model)

# 现代 FFN：SwiGLU (Swish-Gated Linear Unit)
class SwiGLU_FFN:
    def forward(self, x):
        gate = F.silu(self.w_gate(x))    # Swish 激活 = x * sigmoid(x)
        up = self.w_up(x)                 # 上投影
        return self.w_down(gate * up)      # 门控 × 上投影 → 下投影
        # 参数量: 3 × d_model × d_ff (多了一个 gate 投影)
        # 但通常 d_ff 从 4d 缩减到 ~2.67d 来保持总参数量
```

SwiGLU 的核心是**门控机制**——它让网络自己决定"哪些信息通过、哪些被抑制"，比简单的 ReLU 有更强的表达能力。

```
ReLU:     max(0, x)              → 简单截断
GeLU:     x · Φ(x)              → 概率性门控
SwiGLU:   Swish(Wx) ⊙ (Vx)     → 学习到的门控 × 内容
```

> 📊 **行业共识**：**71.7%** 的被分析模型使用 SwiGLU 或 GeGLU。LLaMA 之后，这几乎成了不成文的标准。

## MoE 架构：稀疏的力量

我们在 3.6 节从"趋势"角度介绍了 MoE。这里深入看看它的**架构细节**。

### MoE 的基本结构

MoE 将标准 FFN 层替换为多个"专家"网络 + 一个"路由器"：

```python
class MoELayer:
    """混合专家层：替代标准 FFN"""
    def __init__(self, d_model, n_experts=64, n_active=8):
        # 64 个专家，但每个 token 只激活 8 个
        self.experts = [SwiGLU_FFN(d_model) for _ in range(n_experts)]
        self.router = Linear(d_model, n_experts)  # 路由器：决定激活哪些专家
    
    def forward(self, x):
        # 1. 路由决策：每个 token 独立选择专家
        router_logits = self.router(x)              # [batch, seq, n_experts]
        weights, indices = router_logits.topk(k=8)  # 选 top-8 专家
        weights = F.softmax(weights, dim=-1)         # 归一化权重
        
        # 2. 专家计算：只激活被选中的专家
        output = 0
        for i, (expert_idx, w) in enumerate(zip(indices, weights)):
            output += w * self.experts[expert_idx](x)
        
        return output
```

### 不同模型的 MoE 配置差异很大

| 模型 | 总专家数 | 激活专家数 | 共享专家 | 路由方式 | 负载均衡 |
|------|---------|----------|---------|---------|---------| 
| **Mixtral 8×22B** | 8 | 2 | 无 | Top-2 softmax | 辅助损失 |
| **DeepSeek-V3** | 256 | 8 | 1 个共享 | Top-8 sigmoid | **无辅助损失**（偏差项） |
| **DeepSeek V4** | 256 | 8 | 1 个共享 | Top-8 sigmoid | 无辅助损失 + **mHC 超连接** |
| **Kimi K2** | 128+ | ~8 | 有 | Top-K | MuonClip 优化器稳定训练 |
| **Llama 4 Scout** | 16 | 1 | 无 | Top-1 | 辅助损失 |
| **Llama 4 Maverick** | 128 | 1 | 无 | Token-choice | 辅助损失 |
| **Qwen 3 (235B)** | 128 | 8 | 有 | Top-8 | 辅助损失 |
| **Qwen3.5-Plus** | 128 | 8 | 有 | Top-8 | 优化辅助损失 |
| **MiniMax M2.5** | — | — | — | — | Lightning Attention 混合 |

### DeepSeek 的两个关键创新

**1. 共享专家（Shared Expert）**

DeepSeek 指定一部分专家为"始终激活"，提供稳定的通用知识基底：

```python
class DeepSeekMoE:
    """DeepSeek 的 MoE：共享专家 + 路由专家"""
    def __init__(self):
        self.shared_expert = SwiGLU_FFN()     # 始终参与计算
        self.routed_experts = [SwiGLU_FFN() for _ in range(256)]
        self.router = Linear(d_model, 256)
    
    def forward(self, x):
        # 共享专家：所有 token 都经过
        shared_out = self.shared_expert(x)
        
        # 路由专家：每个 token 选 top-8
        indices, weights = self.route(x)
        routed_out = weighted_sum(self.routed_experts, indices, weights)
        
        return shared_out + routed_out
```

**2. 无辅助损失的负载均衡**

传统 MoE 的一个难题是"路由坍塌"——所有 token 都涌向少数几个专家。通常的解决方案是添加辅助损失函数来惩罚不均衡，但这会干扰主要训练目标。

DeepSeek-V3 引入了一个简洁的替代方案——**给每个专家添加一个可学习的偏差项**：

```python
# 传统方式：辅助损失（会干扰主训练目标）
loss = main_loss + alpha * load_balance_loss

# DeepSeek 方式：偏差项（不干扰主训练目标）
router_logits = self.router(x) + self.expert_bias
# expert_bias 不参与梯度更新，而是通过规则调整：
# 如果某专家负载过高 → 降低其 bias
# 如果某专家负载过低 → 提高其 bias
```

## 开源模型架构全景对比

现在我们把所有技术模块放在一起，看看 2024—2026 年主流开源模型的完整架构选择：

| 架构组件 | Llama 3 (2024) | Llama 4 (2025) | DeepSeek-V3 | DeepSeek V4 | Qwen 3 | Qwen3.5 | Kimi K2 | Kimi K2.5 |
|---------|----------------|----------------|-------------|-------------|--------|---------|---------|-----------|
| **基本架构** | Dense | MoE | MoE | MoE | Dense/MoE | MoE | MoE | MoE |
| **注意力** | GQA | GQA | **MLA** | **MLA** + DSA 2.0 | GQA | **Gated DeltaNet 混合** | GQA | **Kimi Linear 混合** |
| **归一化** | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| **残差连接** | 标准加性 | 标准加性 | 标准加性 | **mHC 超连接** | 标准加性 | 标准加性 | 标准加性 | **Attention Residuals** |
| **位置编码** | RoPE | RoPE+YaRN | RoPE | RoPE | RoPE+YaRN | RoPE+YaRN | RoPE | RoPE |
| **激活函数** | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| **优化器** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | **MuonClip** | **MuonClip** |
| **MoE 专家数** | — | 16/128 | 256+1 | 256+1 | 128 | 128 | 128+ | — |
| **总参/激活** | 8B~405B | 109B~400B | 671B/~37B | 671B/~37B | 0.6B~235B | 397B/17B | **1T/32B** | 48B/3B |
| **上下文** | 128K | 10M | 128K | **1M+** | 32K~128K | 262K | 128K | 256K |

### 一个关键观察：架构正在"分化"

如果说 2024—2025 年的主题是架构**收敛**（共识堆栈），那 2026 年的主题是架构**分化**——在共识堆栈的基础上，各大模型开始探索截然不同的创新路径：

```
"共识堆栈" (2024—2025，仍然是基础):
├── Decoder-Only 架构
├── Pre-Normalization + RMSNorm
├── RoPE 位置编码
├── SwiGLU 激活函数
├── GQA 或 MLA 注意力
├── 无偏置 (No Bias)
└── 大规模模型 → MoE

"分化前沿" (2026 新突破):
├── 混合注意力 ── Gated DeltaNet (Qwen3.5) / Kimi Linear / Lightning Attention (MiniMax)
├── 残差连接重写 ── Attention Residuals (Kimi K2.5) / mHC 超连接 (DeepSeek V4)
├── 优化器革新 ── MuonClip 替代 AdamW (Kimi K2/K2.5)
├── 知识-推理分离 ── Engram 内存架构 (DeepSeek V4)
└── 多 Token 预测 ── 同时预测多个 token (DeepSeek V4 / Qwen3.5)
```

差异化的竞争正从"训练数据和规模"向**架构创新**回归：
1. **混合注意力设计**（线性注意力 + 全注意力的混合比例和方式）
2. **信息流优化**（残差连接、超连接等层间信息传递机制）
3. **训练效率**（优化器创新、多 Token 预测等）
4. **推理效率**（知识卸载、稀疏注意力、KV-Cache 优化）
5. **MoE 的具体设计**（专家数量、路由策略、负载均衡）
6. **长上下文扩展技术**（YaRN、NTK 缩放、线性注意力）

## FlashAttention：让长上下文成为可能的硬件魔法

以上都是"模型架构"层面的创新。但有一个**计算层面**的技术突破对 LLM 的实际能力影响巨大——**FlashAttention**。

标准注意力的问题在于需要**实例化整个注意力矩阵**（N×N），当 N 到百万级时，显存直接爆炸：

```python
# 标准注意力：O(N²) 内存
def standard_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d)  # [N, N] ← N=1M 时需要 1TB 内存！
    weights = softmax(scores)
    return weights @ V

# FlashAttention：分块计算，O(N) 内存
def flash_attention(Q, K, V, block_size=256):
    """
    核心思想：不实例化完整的 N×N 矩阵
    而是分块计算 + 在线 softmax 更新
    """
    output = zeros_like(Q)
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            q_block = Q[i:i+block_size]
            k_block = K[j:j+block_size]
            v_block = V[j:j+block_size]
            # 只计算这个小块的注意力
            block_score = q_block @ k_block.T / sqrt(d)
            # 在线更新 softmax（无需完整矩阵）
            output[i:i+block_size] = online_softmax_update(
                output[i:i+block_size], block_score, v_block
            )
    return output
    # 内存从 O(N²) 降到 O(N)
    # 速度提升 2~4 倍（更好的 GPU 内存层次利用）
```

FlashAttention 的三代演进：

| 版本 | 年份 | 关键改进 |
|------|------|---------|
| FlashAttention-1 | 2022 | IO-aware 分块计算，O(N²) → O(N) 内存 |
| FlashAttention-2 | 2023 | 更好的并行化，速度再提升 2x |
| FlashAttention-3 | 2024 | 利用 Tensor Core 异步执行，接近硬件理论峰值 |

> 💡 **对 Agent 的影响**：FlashAttention 是支撑百万级上下文窗口的底层功臣。没有它，Gemini 2.5 Pro 的 2M 上下文和 Llama 4 Scout 的 10M 上下文都不可能实现。作为 Agent 开发者，你不需要自己实现它（各大推理框架已经内置），但了解它有助于理解模型的能力边界。

## 2026 年架构新突破

2025 年底到 2026 年初，基座模型架构迎来了一波重要创新——打破了此前"架构已结晶"的判断，多个组件被重新设计。以下是最值得关注的四个方向。

### 混合注意力：线性 + 全注意力

2026 年最重要的架构趋势是**混合注意力**——用线性复杂度的注意力变体替代大部分全注意力层，仅保留少量全注意力层处理需要全局信息的场景。

```python
# 混合注意力的核心思想
class HybridAttentionBlock:
    """
    2026 年主流设计：每 4 层中 3 层用线性注意力，1 层用全注意力
    
    Qwen3.5:  Gated DeltaNet : Gated Attention = 3:1
    Kimi K2.5: KDA (Kimi Delta Attention) : Full Attention = 3:1
    MiniMax M2.5: Lightning Attention : Full Attention = 混合
    """
    def __init__(self, layer_idx, d_model):
        if layer_idx % 4 == 3:  # 每 4 层一个全注意力层
            self.attn = FullAttention(d_model)      # O(N²) 但保留全局建模能力
        else:
            self.attn = GatedDeltaNet(d_model)       # O(N) 线性复杂度
    
    def forward(self, x):
        return self.attn(x)
```

**Gated DeltaNet（Qwen3.5 采用）**：结合了 Delta Rule（增量学习规则）和门控机制，既有线性注意力的 O(N) 复杂度，又通过门控保留了对重要信息的选择性记忆：

```python
class GatedDeltaNet:
    """
    Gated DeltaNet：Qwen3.5 的线性注意力变体
    核心思想：用"增量更新"替代"全局注意力矩阵"
    
    对比：
    - 全注意力：每个 token 都和所有 token 计算注意力 → O(N²)
    - Gated DeltaNet：维护一个压缩状态，增量更新 → O(N)
    """
    def forward(self, x):
        # 1. 计算查询、键、值
        q, k, v = self.qkv_proj(x).split(3)
        
        # 2. 门控：决定"记住多少旧信息，接收多少新信息"
        gate = torch.sigmoid(self.gate_proj(x))  # 门控信号
        
        # 3. Delta Rule 增量更新状态矩阵
        # S_{t} = gate * S_{t-1} + (1 - gate) * k_t ⊗ v_t
        state = gate * prev_state + (1 - gate) * torch.outer(k, v)
        
        # 4. 用查询向量从状态中提取信息
        output = q @ state
        return output
    
    # 关键优势：
    # - 推理时不需要 KV-Cache（状态矩阵固定大小）
    # - 128K~1M 上下文下，解码速度提升 5~6 倍
    # - 通过门控保留了对重要信息的选择性注意
```

**Kimi Linear（Kimi K2.5 采用）**：Moonshot AI 提出的 KDA（Kimi Delta Attention），以 3:1 比例混合线性注意力和全局注意力，在 128K~1M 范围内实现 5~6 倍解码加速。

**效果对比**：

| 注意力类型 | 复杂度 | 128K 解码速度 | 1M 解码速度 | 质量损失 |
|-----------|--------|-------------|------------|---------|
| 全注意力（标准 Transformer） | O(N²) | 基准 | 基准 | — |
| GQA | O(N²)（KV 更小） | ~1.2x | ~1.2x | 几乎无 |
| Gated DeltaNet 混合 3:1 | O(N)（大部分层） | ~4x | **~5x** | 极低 |
| Kimi Linear 混合 3:1 | O(N)（大部分层） | ~5x | **~6x** | 极低 |

> 💡 **对 Agent 的影响**：混合注意力让**长上下文 Agent 变得经济可行**。之前在 1M 上下文下运行 Agent 的推理成本极高，现在推理延迟降低 5~6 倍意味着成本也大幅降低。这对需要处理整个代码仓库、长文档的 Agent 场景至关重要。

### Attention Residuals：重写残差连接

Kimi K2.5 在 GTC 2026 上提出了一个大胆的架构修改——**Attention Residuals（AttnRes）**，重写了自 ResNet 以来沿用 10 年的标准残差连接。

```python
# 标准残差连接（2015 年至今的默认设计）
class StandardResidual:
    """
    x_{l+1} = x_l + F_l(x_l)
    所有前序层的输出以固定权重 1 累加 → 深层网络中信号会"稀释"
    """
    def forward(self, x, layer_output):
        return x + layer_output  # 简单加法，权重固定为 1

# Attention Residuals（Kimi K2.5 提出）
class AttentionResiduals:
    """
    用 Softmax 注意力替代固定权重的残差累加
    每一层可以"主动选择"从哪些前序层获取信息
    
    效果：等效于 1.25 倍计算量的标准训练，但几乎零额外开销
    """
    def forward(self, x, all_previous_outputs):
        # 计算当前层对所有前序层输出的注意力权重
        # （而不是固定权重 1 的累加）
        scores = self.query(x) @ self.key(all_previous_outputs).T
        weights = F.softmax(scores, dim=-1)
        
        # 有选择地组合前序层的表示
        aggregated = weights @ all_previous_outputs
        return aggregated

# Block AttnRes（实用变体，减少内存开销）
class BlockAttentionResiduals:
    """
    将层划分为块，在块级别进行注意力聚合
    结合缓存流水线通信，几乎零额外开销
    """
    pass
```

**为什么重要？** 标准残差连接的"加法累积"会导致深层网络的隐藏状态不可控增长，稀释每一层的贡献。AttnRes 让每一层用**学习到的、依赖于输入的权重**有选择地组合前序信息，训练更稳定，下游任务表现更好。

### MuonClip：优化器革新

Kimi K2 引入的 **MuonClip 优化器**是 2025—2026 年训练层面最重要的创新。它挑战了 AdamW 长达 11 年的统治地位：

```python
# AdamW（2014 年至今的行业标准）
# 基于一阶梯度 + 动量 + 自适应学习率

# MuonClip（Kimi K2 提出）
# 基于 Muon 动量 + Newton-Schulz 迭代 + QK-Clip 稳定机制
class MuonClipOptimizer:
    """
    核心创新：
    1. 将 Muon 优化器扩展到万亿参数规模
    2. Newton-Schulz 迭代 + QK-Clip 解决 logits 爆炸
    3. 分布式 Muon 适配大规模 GPU 集群
    
    效果：token 训练效率比 AdamW 提升 2 倍
    含义：同等算力预算下，模型能力翻倍
    """
    def __init__(self, params, lr, max_logit=100):
        self.max_logit = max_logit  # QK-Clip：限制最大 logits
    
    def step(self):
        # 1. Muon 动量更新
        momentum = self.compute_muon_momentum()
        
        # 2. Newton-Schulz 迭代（解决大规模训练不稳定性）
        update = self.newton_schulz_iterate(momentum)
        
        # 3. QK-Clip：将 logits 严格限制在 100 以内
        # 防止万亿参数训练中的 logits 爆炸
        update = self.clip_qk(update, self.max_logit)
        
        # 4. 应用更新
        self.apply_update(update)
```

**影响**：MuonClip 的成功意味着 AdamW 不再是唯一选择。如果这种训练效率提升能泛化到其他架构，可能从根本上改变整个行业的训练经济学——用一半的计算量达到相同的模型能力。

### Engram 内存架构（DeepSeek V4）

DeepSeek V4 提出了一个全新概念——**Engram 内存**，将知识存储与推理计算解耦：

```python
class EngramMemory:
    """
    DeepSeek V4 的 Engram 内存架构
    核心思想：静态知识不应该占用昂贵的 GPU 显存
    
    传统方式：所有知识编码在模型参数中 → 全部加载到 GPU
    Engram 方式：静态知识存储在 CPU 内存 → GPU 专注推理计算
    """
    def __init__(self, vocab_size, embedding_dim):
        # N-gram 嵌入存储在 CPU 内存中
        self.ngram_embeddings = CPUStorage(vocab_size, embedding_dim)
        # O(1) 哈希查找，不占用 GPU 显存
        self.hash_table = HashIndex()
    
    def lookup(self, input_tokens):
        """从 CPU 内存中 O(1) 查找知识嵌入"""
        hashed = self.hash_table(input_tokens)
        knowledge = self.ngram_embeddings[hashed]  # CPU → GPU 传输
        return knowledge
    
    def forward(self, x, input_tokens):
        # 1. 从 Engram 获取静态知识
        knowledge = self.lookup(input_tokens)
        
        # 2. GPU 上进行推理计算
        reasoning_output = self.transformer_layers(x + knowledge)
        
        return reasoning_output
    
    # 效果：
    # - 释放 GPU 显存用于推理 → 更长上下文、更大批次
    # - 知识基准测试显著提升
    # - 推理和知识存储可独立扩展
```

**mHC（Manifold-Constrained Hyper-Connections）**是 DeepSeek V4 的另一个创新——用 Sinkhorn-Knopp 算法约束残差混合矩阵，在仅增加 6.7% 训练开销的情况下，维持数百层网络的信号稳定性。

> 💡 **对 Agent 的影响**：Engram 内存的"知识-推理分离"范式特别适合 Agent 场景——Agent 需要大量的领域知识（存在 CPU 内存中），同时需要强大的推理能力（GPU 专注计算）。这让在受限硬件上运行知识密集型 Agent 成为可能。

---

## 本节小结

| 架构组件 | 演进方向 | 现代共识 | 前沿突破（2026） |
|---------|---------|---------|---------|
| **整体架构** | Encoder-Decoder → Decoder-Only | Decoder-Only | MoE 成为大模型标配 |
| **注意力机制** | MHA → GQA → MLA | GQA / MLA | **混合注意力**：Gated DeltaNet / Kimi Linear（延迟降 5~6x） |
| **归一化** | Post-Norm → Pre-Norm + RMSNorm | Pre-Norm + RMSNorm | 收敛完成，几乎无争议 |
| **残差连接** | 固定加性残差 | 标准残差 | **Attention Residuals**（Kimi K2.5）/ **mHC**（DeepSeek V4） |
| **位置编码** | 绝对编码 → RoPE | RoPE | YaRN/NTK 扩展到 10M+ |
| **激活函数** | ReLU → GeLU → SwiGLU | SwiGLU | 门控机制成为标准 |
| **MoE** | 密集 → 稀疏混合专家 | Top-K 路由 + 共享专家 | 万亿参数级开源 MoE（Kimi K2） |
| **优化器** | SGD → Adam → AdamW | AdamW | **MuonClip**（训练效率翻倍） |
| **知识存储** | 全部编码在参数中 | 参数化存储 | **Engram 内存**（知识-推理分离） |
| **推理加速** | 标准注意力 → FlashAttention | FA-2/3 | 分块 + IO 优化接近硬件极限 |

> 📖 *理解这些架构组件不是为了让你去训练模型——而是为了在模型选型、推理优化、成本估算时有底层判断力。当有人说"这个模型用了 Gated DeltaNet 混合注意力"时，你就知道它在长文本场景下的推理延迟会非常低；当有人说"用了 Engram 内存"时，你就知道它可以在更小的 GPU 上处理知识密集型任务。2026 年，架构创新重新回到了竞争的前沿。*

---

*下一节：[3.8 SFT 与强化学习训练数据准备](./08_training_data.md)*
