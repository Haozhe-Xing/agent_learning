# 10.6 实战：完整 Agentic-RL 训练 Pipeline

## 项目概述与实验设计

本节将从零构建一个完整的 Agentic-RL 训练项目，验证前四节介绍的所有理论与方法。

![Agentic-RL 完整训练 Pipeline](../svg/chapter_agentic_rl_06_pipeline.svg)

> **实验目标**：训练一个能够使用计算器工具解决数学推理问题的 Agent 模型
>
> **基座模型**：`Qwen/Qwen2.5-1.5B-Instruct`（消费级 GPU 可训练）
>
> **数据集**：GSM8K [1]（8,500 条小学数学应用题，含标准答案）
>
> **训练流程**：数据准备 → SFT（格式学习）→ GRPO（推理优化）→ 评估对比

### 为什么选择 GSM8K？

GSM8K 是验证 Agentic-RL 效果的理想基准数据集，具备以下关键特性：

| 特性 | 说明 | 对训练的意义 |
|------|------|------------|
| **客观可验证** | 每题有唯一正确的数值答案 | 可自动计算准确率，无需人工标注奖励 |
| **多步推理** | 平均需要 3–5 个推理步骤 | 能充分测试 Agent 的链式推理能力 |
| **规模适中** | 7,473 训练题 + 1,319 测试题 | 训练成本可控，结果具有统计显著性 |
| **社区基准** | 广泛用于 LLM 评估 | 有大量公开基准数据可供对比 |

### 硬件需求与预期训练时间

| 配置 | SFT 阶段 | GRPO 阶段 | 备注 |
|------|---------|----------|------|
| **最低配置** | 1× RTX 3090 24GB | 1× RTX 3090 24GB | 需开启 QLoRA 4-bit 量化 |
| **推荐配置** | 1× A100 40GB | 1× A100 40GB | 全精度 bfloat16 训练 |
| **训练时间（最低配置）** | 约 2–4 小时 | 约 4–8 小时 | 1.5B 模型，3 个 epoch |

---

## Step 1：环境搭建

```bash
# 创建项目目录与虚拟环境
mkdir -p agent-rl-training && cd agent-rl-training
python -m venv venv && source venv/bin/activate

# 安装核心依赖（版本经过兼容性验证）
pip install torch>=2.1.0
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install trl>=0.12.0
pip install datasets accelerate bitsandbytes
pip install wandb tensorboard          # 实验追踪（强烈推荐）
```

---

## Step 2：数据准备

```python
"""
step2_prepare_data.py

将 GSM8K 原始数据转换为 Agent 格式的 SFT 训练数据。

GSM8K 原始格式：
  question: "Natalia sold clips to 48 of her friends..."
  answer:   "Natalia sold 48/2 = <<48/2=24>>24 clips... #### 72"

目标格式（Agent 轨迹）：
  <think>推理过程</think>
  <tool_call>calculator(expression="...")</tool_call>
"""

import re
from datasets import load_dataset, Dataset


def extract_final_answer(solution: str) -> str:
    """从 GSM8K 解答中提取 '#### 数字' 格式的最终答案"""
    match = re.search(r'####\s*(.+)', solution)
    return match.group(1).strip().replace(",", "") if match else ""


def extract_calculations(solution: str) -> list[str]:
    """提取 GSM8K 解答中的计算表达式（格式：<<expr=result>>）"""
    return re.findall(r'<<(.+?)=.+?>>', solution)


def convert_to_agent_format(example: dict) -> dict:
    """
    将 GSM8K 样本转换为 Agent SFT 训练格式
    
    转换策略：
    1. 提取推理步骤作为 <think> 内容
    2. 提取最后一个计算表达式作为 <tool_call> 参数
    3. 构建完整的 ChatML 格式对话
    """
    question = example["question"]
    solution = example["answer"]
    final_answer = extract_final_answer(solution)
    calculations = extract_calculations(solution)

    # 提取推理步骤（去除 #### 行和计算标注）
    steps = [
        re.sub(r'<<.+?>>', '', line).strip()
        for line in solution.split("\n")
        if line.strip() and "####" not in line
    ]
    think_content = "\n".join(steps)

    # 构建 Agent 格式回答
    if calculations:
        # 使用最后一个计算表达式（通常是最终计算步骤）
        final_expr = calculations[-1]
        agent_response = (
            f"<think>\n{think_content}\n"
            f"最终需要计算：{final_expr}\n</think>\n\n"
            f"<tool_call>\ncalculator(expression=\"{final_expr}\")\n</tool_call>"
        )
    else:
        agent_response = (
            f"<think>\n{think_content}\n</think>\n\n"
            f"最终答案是 **{final_answer}**。"
        )

    # 构建 ChatML 格式对话
    conversation = (
        "<|im_start|>system\n"
        "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
        "需要精确计算时使用 calculator 工具。\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{agent_response}\n<|im_end|>"
    )

    return {
        "text": conversation,
        "question": question,
        "answer": final_answer,
    }


# ── 加载并转换数据集 ──────────────────────────────────────────────────────
print("📦 加载 GSM8K 数据集...")
dataset = load_dataset("openai/gsm8k", "main")

print("🔄 转换为 Agent 格式...")
sft_train = dataset["train"].map(convert_to_agent_format, remove_columns=dataset["train"].column_names)
sft_test  = dataset["test"].map(convert_to_agent_format, remove_columns=dataset["test"].column_names)

print(f"✅ 训练集：{len(sft_train)} 条 | 测试集：{len(sft_test)} 条")

# 数据质量验证
valid_train = sft_train.filter(lambda x: "<think>" in x["text"] and x["answer"] != "")
print(f"📊 格式验证通过率：{len(valid_train) / len(sft_train):.1%}")

sft_train.save_to_disk("./data/sft_train")
sft_test.save_to_disk("./data/sft_test")
```

---

## Step 3：SFT 训练

```python
"""
step3_sft_training.py

SFT 阶段：通过模仿学习让模型习得 Agent 行为格式。
目标：将基座模型的格式符合率从 ~5% 提升至 ~85%+。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk

# ── 数据加载 ──────────────────────────────────────────────────────────────
train_dataset = load_from_disk("./data/sft_train")
eval_dataset  = load_from_disk("./data/sft_test")

# ── 模型加载（QLoRA 配置）────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA 配置 ─────────────────────────────────────────────────────────────
# 1.5B 模型使用 r=16 即可，参数量约 8M（占总参数量 ~0.5%）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# ── 训练配置 ──────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./checkpoints/sft",

    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # 有效 batch size = 16
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",

    bf16=True,
    gradient_checkpointing=True,

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,        # 自动加载验证集最优检查点

    max_seq_length=1024,
    dataset_text_field="text",
    report_to="tensorboard",
)

# ── 训练执行 ──────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset.select(range(200)),
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("🚀 开始 SFT 训练...")
print(f"   模型：{model_name} | LoRA r={lora_config.r} | 训练数据：{len(train_dataset)} 条")
trainer.train()

trainer.save_model("./checkpoints/sft-final")
tokenizer.save_pretrained("./checkpoints/sft-final")
print("✅ SFT 训练完成！")
```

---

## Step 4：GRPO 强化学习训练

```python
"""
step4_grpo_training.py

GRPO 阶段：通过强化学习信号引导模型探索超越 SFT 数据质量的推理策略。
目标：在 SFT 基础上，将准确率进一步提升 10–20 个百分点。
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk

# ── 加载 SFT 模型（合并 LoRA 权重）──────────────────────────────────────
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/sft-final")
model = model.merge_and_unload()   # 合并 LoRA 权重，恢复标准模型结构

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ── 准备 GRPO 训练数据（需要 "prompt" 字段）──────────────────────────────
train_dataset = load_from_disk("./data/sft_train")

def prepare_grpo_prompt(example: dict) -> dict:
    """将训练样本转换为 GRPO 所需的 prompt 格式"""
    return {
        "prompt": (
            "<|im_start|>system\n"
            "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
            "需要精确计算时使用 calculator 工具。\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "answer": example["answer"],
    }

grpo_dataset = train_dataset.map(prepare_grpo_prompt)

# ── 奖励函数：数学 Agent 综合评估 ────────────────────────────────────────
def math_agent_reward(completions: list[str], **kwargs) -> list[float]:
    """
    数学 Agent 综合奖励函数
    
    奖励维度及权重：
    - 准确率（0.50）：最终数值是否正确（允许 1% 相对误差）
    - 格式（0.20）：<think>/<tool_call> 标签是否规范
    - 推理质量（0.20）：推理步骤是否充分、包含计算过程
    - 简洁性（0.10）：输出长度是否合理
    """
    rewards = []
    answers = kwargs.get("answer", [""] * len(completions))

    for completion, answer in zip(completions, answers):
        reward = 0.0

        # ── 维度 1：准确率（权重 0.50）────────────────────────────────────
        try:
            numbers = re.findall(r'-?[\d,]+\.?\d*', completion)
            if numbers:
                pred = float(numbers[-1].replace(",", ""))
                true_val = float(str(answer).replace(",", ""))
                if abs(pred - true_val) / (abs(true_val) + 1e-8) < 0.01:
                    reward += 0.50
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # ── 维度 2：格式正确性（权重 0.20）───────────────────────────────
        has_think = "<think>" in completion and "</think>" in completion
        if has_think:
            reward += 0.10
            think = completion.split("<think>")[1].split("</think>")[0].strip()
            if len(think) > 20:
                reward += 0.10   # 有实质性推理内容

        # ── 维度 3：推理质量（权重 0.20）─────────────────────────────────
        if has_think:
            think = completion.split("<think>")[1].split("</think>")[0]
            lines = [l.strip() for l in think.split("\n") if l.strip()]
            if len(lines) >= 2:
                reward += 0.10   # 多步推理
            if re.search(r'[\d+\-*/=]', think):
                reward += 0.10   # 包含数学计算过程

        # ── 维度 4：简洁性（权重 0.10）───────────────────────────────────
        token_count = len(completion.split())
        if token_count <= 300:
            reward += 0.10
        elif token_count > 800:
            reward -= 0.05   # 过长惩罚

        rewards.append(max(0.0, reward))

    return rewards

# ── GRPO 训练配置 ─────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir="./checkpoints/grpo",

    num_generations=8,               # G=8：每题生成 8 个回答用于组内比较
    num_train_epochs=1,              # GRPO 通常 1–2 个 epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,              # RL 学习率 ≈ SFT 学习率的 1/40
    warmup_ratio=0.1,
    max_grad_norm=0.5,               # 梯度裁剪，防止 RL 训练中的梯度爆炸

    max_new_tokens=512,
    temperature=0.7,                 # 保证 G 个回答的多样性

    kl_coef=0.05,                    # KL 散度惩罚系数

    bf16=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
)

# ── 训练执行 ──────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
    reward_funcs=math_agent_reward,
)

print("🚀 开始 GRPO 训练...")
print(f"   组大小 G={grpo_config.num_generations} | 学习率={grpo_config.learning_rate} | KL 系数={grpo_config.kl_coef}")
trainer.train()
trainer.save_model("./checkpoints/grpo-final")
print("✅ GRPO 训练完成！")
```

---

## Step 5：系统性评估与对比分析

```python
"""
step5_evaluation.py

对比评估三个阶段的模型性能：
  基座模型（Baseline）→ SFT 模型 → GRPO 模型

评估指标：
  - 准确率（Accuracy）：最终答案正确率
  - 格式符合率（Format Compliance）：<think> 标签使用率
  - 平均输出长度（Avg. Length）：Token 数
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


def evaluate_model(
    model_path: str,
    test_data,
    num_samples: int = 200,
    device: str = "cuda",
) -> dict:
    """
    在 GSM8K 测试集上评估模型性能
    
    Args:
        model_path:  模型路径（HuggingFace 格式）
        test_data:   测试数据集
        num_samples: 评估样本数（完整评估用 1319）
    
    Returns:
        包含各项指标的评估结果字典
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    correct = 0
    format_ok = 0
    total_tokens = 0
    total = 0

    for example in test_data.select(range(num_samples)):
        prompt = (
            "<|im_start|>system\n"
            "你是一个数学助手。解题时请先在 <think> 标签中写出完整的推理过程，"
            "需要精确计算时使用 calculator 工具。\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # 准确率评估
        try:
            true_val = float(example["answer"].replace(",", ""))
            numbers = re.findall(r'-?[\d,]+\.?\d*', response)
            if numbers:
                pred = float(numbers[-1].replace(",", ""))
                if abs(pred - true_val) / (abs(true_val) + 1e-8) < 0.01:
                    correct += 1
        except (ValueError, ZeroDivisionError):
            pass

        # 格式符合率
        if "<think>" in response and "</think>" in response:
            format_ok += 1

        total_tokens += len(response.split())
        total += 1

    del model   # 释放显存，为下一个模型腾出空间

    return {
        "accuracy":          correct / total,
        "format_compliance": format_ok / total,
        "avg_length":        total_tokens / total,
        "total_samples":     total,
    }


# ── 评估三个阶段的模型 ────────────────────────────────────────────────────
test_data = load_from_disk("./data/sft_test")

models_to_eval = [
    ("🔵 基座模型",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("🟡 SFT 模型",  "./checkpoints/sft-merged"),
    ("🟢 GRPO 模型", "./checkpoints/grpo-final"),
]

results = {}
for name, path in models_to_eval:
    print(f"\n{name} 评估中...")
    results[name] = evaluate_model(path, test_data, num_samples=200)

# ── 结果展示 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📈 Agentic-RL 训练效果对比（GSM8K 测试集，n=200）")
print("=" * 65)
print(f"{'指标':<20} {'基座模型':>12} {'SFT':>12} {'GRPO':>12}")
print("-" * 65)

metrics = [
    ("准确率",     "accuracy",          ".1%"),
    ("格式符合率", "format_compliance", ".1%"),
    ("平均输出长度", "avg_length",      ".0f"),
]

for label, key, fmt in metrics:
    row = f"{label:<20}"
    for name, _ in models_to_eval:
        val = results[name][key]
        row += f" {val:>11{fmt}}"
    print(row)

print("=" * 65)
print("\n📌 预期结果参考（Qwen2.5-1.5B）：")
print("   基座模型：准确率 ~35–45%，格式符合率 ~5%")
print("   SFT 后：  准确率 ~45–55%，格式符合率 ~85%")
print("   GRPO 后： 准确率 ~55–65%，格式符合率 ~90%")
```

---

## Step 6：模型导出与部署

```python
"""
step6_export.py

将训练好的模型导出为生产可用的格式。
支持 HuggingFace 格式（用于 vLLM/TGI 部署）和 GGUF 格式（用于本地部署）。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 加载最终模型 ──────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/grpo-final",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/grpo-final")

# ── 方式 1：HuggingFace 格式（推荐用于服务端部署）────────────────────────
# 兼容 vLLM、Text Generation Inference (TGI)、Ollama 等推理框架
model.save_pretrained("./export/hf-model", safe_serialization=True)
tokenizer.save_pretrained("./export/hf-model")
print("✅ HuggingFace 格式已导出至 ./export/hf-model")

# ── 方式 2：GGUF 格式（用于 llama.cpp / Ollama 本地部署）────────────────
# 需要安装 llama.cpp 并使用其转换脚本
# python llama.cpp/convert_hf_to_gguf.py ./export/hf-model \
#     --outtype q4_k_m \
#     --outfile ./export/model-q4_k_m.gguf
print("💡 GGUF 格式转换命令：")
print("   python llama.cpp/convert_hf_to_gguf.py ./export/hf-model \\")
print("       --outtype q4_k_m --outfile ./export/model-q4_k_m.gguf")
```

---

## 完整项目结构

```
agent-rl-training/
├── data/
│   ├── sft_train/              # SFT 训练数据（7,473 条 Agent 格式轨迹）
│   └── sft_test/               # 评估数据（1,319 条）
├── checkpoints/
│   ├── sft/                    # SFT 训练检查点（含 TensorBoard 日志）
│   ├── sft-final/              # SFT 最终 LoRA 适配器权重
│   ├── sft-merged/             # SFT 合并后的完整模型（用于 GRPO 初始化）
│   ├── grpo/                   # GRPO 训练检查点
│   └── grpo-final/             # GRPO 最终模型（用于评估和部署）
├── export/
│   ├── hf-model/               # HuggingFace 格式（服务端部署）
│   └── model-q4_k_m.gguf       # GGUF 格式（本地部署，可选）
├── step2_prepare_data.py
├── step3_sft_training.py
├── step4_grpo_training.py
├── step5_evaluation.py
├── step6_export.py
└── requirements.txt
```

> **📌 工程实践要点**
>
> - **实验追踪**：强烈建议使用 wandb 或 MLflow 记录每次训练的超参数、损失曲线和评估指标，便于复现和对比
> - **数据增强**：可用 GPT-4 对 GSM8K 题目进行改写，生成更多样化的训练数据，通常能带来 2–5% 的准确率提升
> - **课程学习（Curriculum Learning）**：先用简单题（1–2 步推理）训练，再逐步引入复杂题（4–5 步推理），收敛速度通常更快
> - **模型规模效应**：本教程使用 1.5B 模型作为教学演示；实际生产中，7B–14B 模型的 GRPO 提升幅度更显著（通常 15–25%）
> - **成本估算**：A100 40GB 上，1.5B 模型完整训练约需 6–12 小时；7B 模型约需 24–48 小时；14B 模型约需 48–96 小时

---

## 本章总结

通过本章的系统学习，你已掌握了 Agentic-RL 训练的完整知识体系：

| 章节 | 核心知识点 | 关键结论 |
|------|-----------|---------|
| **10.1 概述** | MDP 建模、两阶段范式 | RL 训练可涌现出超越训练数据的推理策略 |
| **10.2 SFT + LoRA** | 监督微调、参数高效训练 | LoRA 以 <1% 的参数量实现接近全参数微调的效果 |
| **10.3 PPO** | 策略梯度、重要性采样、优势函数、Clip 机制 | PPO 是 RLHF 的经典算法，但 Critic 导致显存占用 ≈ 3× |
| **10.4 DPO** | 隐式奖励、Bradley-Terry 模型、闭式解 | DPO 将 RL 转化为监督学习，极简但无法在线探索 |
| **10.5 GRPO + 奖励设计** | 组内比较、多维度奖励、奖励黑客防御 | GRPO 将显存需求从 3× 降至 1.5×；奖励函数是 RL 训练效果的决定性因素 |
| **10.6 实战** | 完整 Pipeline、评估对比 | GSM8K 上：基座 ~40% → SFT ~50% → GRPO ~60% |

Agentic-RL 代表了 LLM 应用的一个重要发展方向：**从"提示工程"到"训练优化"的范式转变**。随着算法的持续演进和计算成本的降低，这一技术将在越来越多的高价值 Agent 场景中发挥关键作用。

---

## 参考文献

[1] COBBE K, KOSARAJU V, BAVARIAN M, et al. Training verifiers to solve math word problems[R]. arXiv preprint arXiv:2110.14168, 2021.

[2] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[3] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-rank adaptation of large language models[C]//International Conference on Learning Representations (ICLR). 2022.

[4] SHAO Z, WANG P, ZHU Q, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models[R]. arXiv preprint arXiv:2402.03300, 2024.

[5] BENGIO Y, LOURADOUR J, COLLOBERT R, et al. Curriculum learning[C]//International Conference on Machine Learning (ICML). 2009.
