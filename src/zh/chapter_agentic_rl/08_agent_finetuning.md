# 11.8 专为 Agent 的微调：让模型真正会用工具

> 🔧 *"通用 SFT 教会模型说话；Agent SFT 教会模型做事。两者看起来相似，目标却截然不同。"*

第 11.2 节讲了通用的 SFT + LoRA 方法。但当你的目标是训练一个**能可靠使用工具、遵循 Agent 行为格式、减少幻觉调用**的模型时，数据构建和训练策略都需要专门设计。

本节聚焦：**如何微调出一个真正会用工具的 Agent 模型**。

---

## 通用 SFT 的局限：为什么不够用？

用通用对话数据训练出的模型，在 Agent 任务上往往出现以下问题：

```
❌ 幻觉工具调用：调用根本不存在的工具
   模型输出: {"tool": "search_google", "args": {"q": "天气"}}
   实际工具: ["get_weather", "web_search", "calculator"]  ← 没有 search_google

❌ 格式不一致：同样的工具，调用格式每次不同
   第1次: {"function": "get_weather", "location": "北京"}
   第2次: get_weather(city="北京")
   第3次: [tool: get_weather] 参数: 北京

❌ 多步推理断链：第3步忘记了第1步的结果
   步骤1: 查到股价 = 150元  
   步骤3: "我需要先查一下股价"  ← 忘了已经查过

❌ 滥用工具：简单问题也要调用工具，浪费延迟和成本
   用户: "3 + 5 等于多少？"
   模型: 调用 calculator(expr="3+5")  ← 直接回答不更好吗？
```

这些问题的根源是：**通用训练数据中工具调用场景稀少、格式混乱**。解决方法是用专门构建的 Agent 轨迹数据重新微调。

---

## Agent 微调数据的三种来源

### 来源一：人工专家标注（质量最高，成本最高）

最直接的方式：让专家演示正确的 Agent 行为轨迹。

```python
# 一条完整的 Agent SFT 训练样本（多轮轨迹格式）
training_sample = {
    "id": "agent_001",
    "messages": [
        {
            "role": "system",
            "content": """你是一个智能助手，可以使用以下工具：
            
工具列表：
1. get_stock_price(ticker: str) -> float  # 获取股票价格
2. calculate(expr: str) -> float          # 数学计算
3. search_news(query: str) -> list[str]   # 搜索新闻

规则：
- 只使用列表中的工具，不要调用不存在的工具
- 每次只调用一个工具
- 工具调用格式必须是合法的 JSON"""
        },
        {
            "role": "user", 
            "content": "帮我查一下苹果公司（AAPL）最近的股价，并计算如果买100股需要多少钱？"
        },
        {
            "role": "assistant",
            # 第1步：推理 + 工具调用
            "content": "我需要先查询 AAPL 的当前股价，然后计算总费用。",
            "tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": "213.45"  # 工具返回结果
        },
        {
            "role": "assistant",
            # 第2步：根据结果继续推理
            "content": "AAPL 当前股价为 213.45 美元，现在计算 100 股的总费用。",
            "tool_calls": [{"name": "calculate", "arguments": {"expr": "213.45 * 100"}}]
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": "21345.0"
        },
        {
            "role": "assistant",
            # 第3步：整合结果，给出最终答案
            "content": "苹果公司（AAPL）当前股价为 **213.45 美元**。购买 100 股需要 **21,345 美元**（不含交易费用）。"
        }
    ]
}
```

**关键点**：每条样本都是完整的多轮轨迹，包含：
1. 工具列表的清晰定义
2. 每步的推理过程（内心独白）
3. 正确格式的工具调用
4. 工具返回结果的正确解读
5. 最终整合性回答

### 来源二：基于强模型的自动合成（效率最高）

用 GPT-4.1 / Claude Sonnet 4.5 等强模型批量生成轨迹数据，再经过过滤：

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

SYNTHESIS_SYSTEM_PROMPT = """你是一个数据合成专家。
给定工具定义和用户任务，生成一条正确的 Agent 轨迹数据。

要求：
1. 推理过程要清晰可见（每步说明为什么要这样做）
2. 工具调用格式严格遵循 JSON Schema
3. 遇到工具返回错误时要正确处理
4. 不要调用工具列表以外的工具
5. 简单问题直接回答，不要过度使用工具

输出格式：JSON 格式的完整对话轨迹"""

async def synthesize_trajectory(
    tools: list[dict],
    task: str,
    model: str = "gpt-4.1"
) -> dict | None:
    """用强模型合成一条 Agent 轨迹"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
工具列表：{tools}

用户任务：{task}

请生成一条完整的 Agent 轨迹，包含推理过程和工具调用。
"""}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # 低温度，保证格式一致
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Synthesis failed for task '{task}': {e}")
        return None


async def batch_synthesize(
    tool_sets: list[list[dict]],
    task_pool: list[str],
    n_samples: int = 1000,
    concurrency: int = 20
) -> list[dict]:
    """批量合成训练数据"""
    import random
    
    semaphore = asyncio.Semaphore(concurrency)  # 控制并发
    
    async def bounded_synthesize(tools, task):
        async with semaphore:
            return await synthesize_trajectory(tools, task)
    
    # 随机组合工具集和任务
    pairs = [
        (random.choice(tool_sets), random.choice(task_pool))
        for _ in range(n_samples)
    ]
    
    results = await asyncio.gather(*[
        bounded_synthesize(tools, task) 
        for tools, task in pairs
    ])
    
    # 过滤 None 和格式错误的样本
    valid = [r for r in results if r is not None and validate_trajectory(r)]
    print(f"合成成功率: {len(valid)}/{n_samples} = {len(valid)/n_samples:.1%}")
    return valid


def validate_trajectory(trajectory: dict) -> bool:
    """验证轨迹格式是否合法"""
    try:
        messages = trajectory.get("messages", [])
        # 必须有 system、user、至少一个 assistant
        roles = [m["role"] for m in messages]
        if "system" not in roles or "user" not in roles:
            return False
        if roles.count("assistant") < 1:
            return False
        # 检查工具调用格式
        for msg in messages:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    if not all(k in call for k in ["name", "arguments"]):
                        return False
                    if not isinstance(call["arguments"], dict):
                        return False
        return True
    except (KeyError, TypeError):
        return False
```

### 来源三：真实用户交互过滤（最贴近生产）

从线上 Agent 系统收集真实的用户交互，经过质量过滤作为训练数据：

```python
class TrajectoryCollector:
    """从线上 Agent 收集训练轨迹"""
    
    def __init__(self, quality_threshold: float = 0.8):
        self.threshold = quality_threshold
    
    def collect_from_production(self, 
                                 raw_logs: list[dict]) -> list[dict]:
        """从线上日志过滤高质量轨迹"""
        high_quality = []
        
        for log in raw_logs:
            score = self._quality_score(log)
            if score >= self.threshold:
                # 清洗 PII、截断过长序列
                cleaned = self._clean_trajectory(log)
                high_quality.append(cleaned)
        
        return high_quality
    
    def _quality_score(self, log: dict) -> float:
        """
        质量评分维度：
        - 用户满意度（点赞/评分/继续对话）
        - 任务是否成功完成（有最终回答）
        - 工具调用是否成功（无错误重试）
        - 轨迹长度是否合理（非空转）
        """
        score = 0.0
        
        # 维度1: 用户显式满意度
        if log.get("user_rating", 0) >= 4:
            score += 0.3
        elif log.get("conversation_continued"):  # 用户继续对话 = 隐式满意
            score += 0.15
        
        # 维度2: 任务完成
        messages = log.get("messages", [])
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "assistant" and len(last_msg.get("content", "")) > 50:
            score += 0.3
        
        # 维度3: 工具调用成功率
        tool_calls = sum(1 for m in messages if m.get("role") == "tool")
        tool_errors = sum(1 for m in messages 
                         if m.get("role") == "tool" and "error" in str(m.get("content", "")).lower())
        if tool_calls > 0:
            success_rate = 1 - tool_errors / tool_calls
            score += 0.2 * success_rate
        
        # 维度4: 轨迹效率（不空转）
        n_turns = len([m for m in messages if m["role"] == "assistant"])
        if 1 <= n_turns <= 8:  # 合理的轨迹长度
            score += 0.2
        
        return score
    
    def _clean_trajectory(self, log: dict) -> dict:
        """清洗轨迹：脱敏、格式标准化"""
        import re
        cleaned = log.copy()
        
        for msg in cleaned.get("messages", []):
            content = str(msg.get("content", ""))
            # 脱敏：手机号、邮箱、身份证
            content = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE]', content)
            content = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', content)
            msg["content"] = content
        
        return cleaned
```

---

## 专为 Agent 设计的训练数据格式

不同于通用对话，Agent 微调数据需要覆盖更多特殊场景：

### 场景一：工具选择决策（重中之重）

```python
# ✅ 正确：正确选择工具
{
    "input": "用户: 2 + 2 等于多少？",
    "output": "4",  # ← 直接回答，不用工具
    "annotation": "简单数学，直接回答更快"
}

{
    "input": "用户: 北京今天天气怎么样？",
    "tool_call": {"name": "get_weather", "args": {"city": "北京"}},
    "annotation": "需要实时信息，调用天气工具"
}

# ❌ 错误样本（也要包含，用于对比学习）
{
    "input": "用户: 你叫什么名字？",
    "wrong_output": {"tool_call": "search_database(query='AI名字')"},
    "correct_output": "我是一个 AI 助手，没有固定名字。",
    "error_type": "unnecessary_tool_use"
}
```

### 场景二：错误恢复（提升鲁棒性）

```python
# 工具调用失败时的正确处理方式
error_recovery_sample = {
    "messages": [
        {"role": "user", "content": "帮我查 TSLA 股价"},
        {"role": "assistant", 
         "content": "查询 TSLA 股价...",
         "tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}]},
        {"role": "tool", 
         "content": "ERROR: API rate limit exceeded"},  # ← 模拟失败
        {"role": "assistant", 
         # ✅ 正确：承认失败，提供替代方案
         "content": "抱歉，股价查询服务暂时不可用（超出 API 限制）。"
                    "您可以直接访问 Yahoo Finance 或 Google 搜索 'TSLA 股价' 获取实时数据。"}
    ]
}
```

### 场景三：多工具协作（长链推理）

```python
# 需要多步工具调用的复杂任务
multi_tool_sample = {
    "task": "分析 AAPL 最近新闻对股价的影响",
    "trajectory": [
        # 步骤1: 搜索新闻
        {"thought": "先获取最近的 AAPL 相关新闻"},
        {"tool_call": {"name": "search_news", "args": {"query": "AAPL Apple 2026"}}},
        {"tool_result": ["苹果Q2财报超预期...", "iPhone 18系列发布..."]},
        
        # 步骤2: 获取股价
        {"thought": "获取当前股价和历史价格作对比"},
        {"tool_call": {"name": "get_stock_price", "args": {"ticker": "AAPL"}}},
        {"tool_result": "213.45"},
        
        # 步骤3: 分析（不需要工具，直接推理）
        {"thought": "现在有了新闻和股价，可以进行分析了"},
        {"final_answer": "根据最近新闻分析：..."}
    ]
}
```

---

## 三大开源 Agent 微调数据集

业界已有现成的高质量数据集，可直接用于微调：

### 1. Gorilla（Function Calling 专项）

```python
# Gorilla 项目：专为 API / Function Calling 训练的数据集
# 来源：UC Berkeley，包含 1600+ 真实 API 的调用示例

from datasets import load_dataset

gorilla_data = load_dataset("gorilla-llm/APIBench", split="train")
# 格式：用户意图 → API 调用 → 执行结果

# 示例样本
sample = {
    "instruction": "What are the symptoms of diabetes?",
    "api_call": 'requests.get("https://api.medlineplus.gov/v2/spellcheck", params={"terms": "diabetes"})',
    "provider": "medlineplus"
}
```

### 2. ToolLLM / ToolBench（覆盖最广）

```python
# ToolBench：16000+ 真实 API 的工具调用数据
# 包含单工具和多工具场景，有完整的思维链

toolbench_data = load_dataset("ToolBench/ToolBench", split="train")
# 平均每条样本包含 3-8 轮工具调用

# 结构特点：
# - instruction: 用户意图
# - tools: 可用工具列表（动态变化）  ← 训练模型适应不同工具集
# - conversations: 完整的多轮轨迹（含 CoT）
```

### 3. AgentInstruct（微软，质量最高）

```python
# AgentInstruct（微软 2024）：
# - 25M+ 合成 Agent 轨迹
# - 覆盖代码生成、RAG、多模态、浏览器操作等场景
# - 用于训练 Orca 3 / Phi-3 系列模型

# 关键创新：
# 1. 从种子任务自动生成复杂变体（提升难度多样性）
# 2. 用奖励模型对生成轨迹打分过滤
# 3. 分领域专项训练后合并

# 训练 Phi-3 Mini 的效果：
# AgentBench 上比基础模型提升 40%+
# Function Calling 准确率从 52% → 78%
```

---

## 专为 Agent 的训练配置

通用 SFT 和 Agent SFT 在训练配置上有几个关键差异：

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Agent SFT 专用配置
agent_training_args = TrainingArguments(
    output_dir="./agent-sft-output",
    
    # ① 批次大小：Agent 轨迹通常更长，需要减小 batch size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # 等效 batch_size=16
    
    # ② 学习率：Agent 任务通常需要更小的学习率
    learning_rate=5e-5,             # 比通用 SFT 低一些
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # ③ 序列长度：Agent 轨迹通常比对话长
    max_seq_length=8192,            # 确保能容纳完整轨迹
    
    # ④ 训练轮次：Agent 数据通常更少，避免过拟合
    num_train_epochs=2,             # 2-3 轮通常足够
    
    # ⑤ 仅对 assistant 回复计算损失（关键！）
    # 不要让模型"学习"用户输入和工具输出的格式
)

# 关键设置：response_template 确保只训练 assistant 部分
trainer = SFTTrainer(
    model=model,
    args=agent_training_args,
    train_dataset=agent_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        # 只计算 assistant token 的损失
        # user/system/tool 部分的 loss mask = 0
        label_pad_token_id=-100,
    ),
    formatting_func=format_trajectory_for_training,
)
```

```python
def format_trajectory_for_training(sample: dict) -> str:
    """
    将 Agent 轨迹格式化为训练文本，
    并正确设置 loss mask（只训练 assistant 部分）
    """
    messages = sample["messages"]
    
    # 使用 ChatML 格式（大多数模型支持）
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        # 工具调用转为文本表示
        if "tool_calls" in msg:
            tool_call_str = json.dumps(msg["tool_calls"], ensure_ascii=False)
            content = f"{content}\n<tool_call>{tool_call_str}</tool_call>"
        
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return formatted
```

---

## 评估专为 Agent 微调模型的效果

```python
class AgentEvaluator:
    """评估 Agent 微调模型的专项指标"""
    
    def evaluate(self, model, test_cases: list[dict]) -> dict:
        results = {
            "tool_selection_accuracy": 0,   # 工具选择正确率
            "argument_accuracy": 0,          # 参数填充正确率  
            "format_validity": 0,            # 格式合法率
            "task_completion_rate": 0,       # 任务完成率
            "unnecessary_tool_rate": 0,      # 不必要工具调用率
        }
        
        for case in test_cases:
            prediction = model.generate(case["input"])
            
            # 1. 工具选择：是否选对了工具名
            results["tool_selection_accuracy"] += (
                self._check_tool_selection(prediction, case["expected_tool"])
            )
            
            # 2. 参数准确性：关键参数是否正确提取
            results["argument_accuracy"] += (
                self._check_arguments(prediction, case["expected_args"])
            )
            
            # 3. 格式合法：能否被解析为合法 JSON
            try:
                json.loads(extract_tool_call(prediction))
                results["format_validity"] += 1
            except json.JSONDecodeError:
                pass
        
        n = len(test_cases)
        return {k: v/n for k, v in results.items()}
```

---

## 实战建议：从哪里开始

```
阶段一（1-2周）：数据构建
├── 定义你的工具集（10-50个工具是合适的起点）
├── 用 GPT-4.1 合成 5000-10000 条基础轨迹
├── 过滤格式错误和明显错误的样本（保留 70-80%）
└── 人工抽查 100 条，评估质量分布

阶段二（1周）：训练
├── 基于 Llama 3.1 8B 或 Qwen2.5 7B 做 LoRA 微调
├── 训练 2 轮，监控 validation loss
└── 每 500 步检查点：定量评估工具选择准确率

阶段三（持续）：迭代
├── 线上收集真实失败案例 → 加入训练集
├── 每 2 周微调一次新版本
└── A/B 测试：新模型 vs 旧模型在工具调用成功率上的对比
```

> 💡 **经验法则**：  
> - 5000 条高质量 Agent 轨迹 > 50000 条低质量通用数据  
> - 覆盖"失败恢复"的数据价值是"成功轨迹"的 2-3 倍  
> - 工具列表随机化（每次训练看到不同的工具组合）能大幅提升模型的工具泛化能力

---

## 本节小结

| 维度 | 通用 SFT | Agent SFT |
|------|---------|----------|
| **数据格式** | 单轮对话 | 多轮轨迹（含工具调用+结果） |
| **关键能力** | 语言生成 | 工具选择、参数提取、错误恢复 |
| **数据量** | 10K-1M | 1K-50K（质量优先） |
| **损失计算** | 全序列 | 仅 assistant token |
| **评估指标** | BLEU/ROUGE | 工具调用准确率、任务完成率 |
| **核心难点** | 语言多样性 | 格式一致性 + 工具泛化 |

> 🔗 **与下一节的关系**：本节解决的是"如何一次性微调出一个好的 Agent 模型"。第 11.9 节将介绍如何让这个过程持续循环改进——**Agentic 数据飞轮**。

---

## 参考文献

1. Patil et al. "Gorilla: Large Language Model Connected with Massive APIs." NeurIPS 2023.
2. Qin et al. "ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs." ICLR 2024.
3. Mitra et al. "AgentInstruct: Toward Generative Teaching with Agentic Flows." Microsoft Research 2024.
4. Liu et al. "What Makes Good Data for Alignment? (DEITA)" ICLR 2024.
5. Wang et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." ACL 2023.
