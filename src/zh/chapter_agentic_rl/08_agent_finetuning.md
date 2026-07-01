# 10.8 专为 Agent 的微调：让模型真正会用工具

> 🔧 *"通用 SFT 教会模型说话；Agent SFT 教会模型按格式做事；Agentic-RL 则让模型在真实反馈中学会把任务做成。"*

第 10.2 节讲了通用的 SFT + LoRA 方法。但当你的目标是训练一个**能可靠使用工具、遵循 Agent 行为格式、减少幻觉调用**的模型时，数据构建和训练策略都需要专门设计。

本节聚焦：**如何微调出一个真正会用工具的 Agent 模型**。同时要记住一条边界：Agent SFT 是起点，不是终点。它解决“会不会按格式行动”，而 Agentic-RL 解决“行动以后能不能真的完成任务”。

---

## SFT 与 Agentic-RL 的分工：先学格式，再学成败

很多团队训练 Agent 时会把 SFT 当成全部方案：收集一批专家轨迹，让模型模仿，就期待模型能在真实环境中稳定完成任务。但 Agent 不是普通聊天机器人，它的输出会改变外部世界，因此训练目标必须分成两个阶段：

| 阶段 | 训练目标 | 学到的能力 | 主要局限 |
|------|----------|------------|----------|
| **Agent SFT** | 模仿高质量轨迹 | 工具格式、参数抽取、基础规划、回答风格 | 不知道动作执行后是否真的有效 |
| **Agentic-RL** | 根据环境反馈优化策略 | 工具选择、错误恢复、成本控制、长程任务完成 | 需要可执行环境和奖励设计 |

一句话总结：

> **SFT 让模型知道“专家通常怎么做”，Agentic-RL 让模型知道“这么做到底有没有用”。**

因此，本节的 Agent SFT 应被理解为 Agentic-RL 的准备阶段：先让模型具备基本工具语法和行为格式，再让它进入环境，通过成功、失败和代价信号继续学习。

---

## 通用 SFT 的局限：为什么不够用？

用通用对话数据训练出的模型，在 Agent 任务上往往出现以下问题：

> **❌ 幻觉工具调用**：调用根本不存在的工具（如输出 `search_google`，但实际工具列表中只有 `get_weather`、`web_search`、`calculator`）
>
> **❌ 格式不一致**：同样的工具，调用格式每次不同（JSON 格式、函数调用格式、自然语言格式混用）
>
> **❌ 多步推理断链**：第 3 步忘记了第 1 步的结果（步骤 1 查到股价 = 150 元，步骤 3 却说"我需要先查一下股价"）
>
> **❌ 滥用工具**：简单问题也要调用工具（用户问"3 + 5 等于多少"，模型却调用 `calculator(expr="3+5")`）

这些问题的根源是：**通用训练数据中工具调用场景稀少、格式混乱**。解决方法是用专门构建的 Agent 轨迹数据重新微调。

但即使用 Agent 轨迹做 SFT，也仍然存在一个根本限制：它只能学习“给定上下文下专家输出了什么”，不能学习“这个动作执行后环境发生了什么”。例如，训练样本可以告诉模型应该调用 `get_weather`，却无法让模型真正体验：API 超时、城市名歧义、权限不足、返回字段变更时应该如何调整策略。

这正是后续需要 Agentic-RL 的原因：**工具调用不是文本格式问题，而是环境交互问题**。

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

不同于通用对话，Agent 微调数据需要覆盖更多特殊场景。这里的关键不是把“成功答案”堆得越多越好，而是让模型看见三类能力：什么时候行动、怎么行动、行动失败后如何恢复。

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

这些指标能判断模型是否“会用工具”，但还不能完全判断它是否“能完成任务”。例如，工具选择准确率很高的模型，仍可能在 8 步任务中因为一步未检查中间结果而失败。因此，Agent SFT 的评估应分成两层：

| 评估层次 | 典型指标 | 含义 |
|---------|---------|------|
| **动作级评估** | 工具选择准确率、参数准确率、格式合法率 | 模型是否学会了单步工具调用 |
| **轨迹级评估** | 任务完成率、恢复成功率、平均成本、人工接管率 | 模型是否能把多步任务做成 |

如果动作级指标已经较高，但轨迹级指标仍然不稳定，就说明继续堆 SFT 数据的收益会下降，应该引入环境反馈和轨迹级奖励。

---

## 什么时候从 Agent SFT 进入 Agentic-RL？

Agent SFT 的目标不是把模型训练到“完美”，而是训练到足以进入环境探索。判断是否该进入 Agentic-RL，可以看四个信号：

| 信号 | 说明 | 下一步 |
|------|------|--------|
| **格式错误显著下降** | 工具调用 JSON 基本合法，参数字段基本稳定 | 可以开始让模型真实执行工具 |
| **工具选择达到可用水平** | 单步工具选择准确率达到业务阈值 | 引入结果奖励和工具成功率奖励 |
| **长任务仍然断链** | 多步任务中忘记上下文、重复调用、不会恢复 | 用轨迹级奖励训练规划与恢复 |
| **失败案例重复出现** | 同类 API 错误、权限错误、边界输入反复失败 | 将失败轨迹转为偏好对或 RL rollout |

进入 Agentic-RL 后，训练目标会从“预测专家下一句话”变成“最大化整条轨迹的长期收益”：

```text
SFT 目标：
给定上下文，预测专家下一步输出。

Agentic-RL 目标：
给定任务和环境，通过一系列动作最大化最终成功率、过程可靠性和成本收益。
```

这也是为什么普通 SFT 不足以训练强 Agent：SFT 看到的是静态数据，Agentic-RL 优化的是动态闭环。

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

## 小结

| 维度 | 通用 SFT | Agent SFT | Agentic-RL |
|------|---------|----------|------------|
| **数据格式** | 单轮对话 | 多轮轨迹（含工具调用+结果） | 真实或模拟环境中的 rollout |
| **关键能力** | 语言生成 | 工具选择、参数提取、错误恢复 | 长程规划、环境适应、成本控制 |
| **数据量** | 10K-1M | 1K-50K（质量优先） | 取决于环境采样成本和奖励密度 |
| **损失/目标** | 预测标准答案 | 模仿 assistant token | 最大化轨迹级奖励 |
| **评估指标** | BLEU/ROUGE | 工具调用准确率、任务完成率 | 成功率、恢复率、稳定性、成本 |
| **核心难点** | 语言多样性 | 格式一致性 + 工具泛化 | 奖励设计 + 探索效率 + 安全边界 |

> 🔗 **与第 11.3 节的关系**：本节解决的是“如何训练出第一个可用 Agent”。[11.3 Agentic 数据飞轮](../chapter_self_evolving/03_data_flywheel.md) 将介绍如何把真实运行中的成功、失败和环境反馈持续转化为训练数据，让 Agent 越用越强。

---

## 📝 本章练习

读完本章，先合上书用自己的话回答下面的问题，再展开参考答案对照。

**练习 1（概念）**：本章反复强调 Agentic-RL 采用 "SFT → RL" 两阶段范式。请解释：为什么"只做 SFT"和"跳过 SFT 直接做 RL"都不理想？这两个阶段各自不可替代的作用是什么？另外，本章提到"过度 SFT 反而会损害 RL 效果"，请说说这是为什么。

<details>
<summary>参考答案</summary>

可以用一句话先抓住分工：**SFT 教模型"专家通常怎么做"，RL 教模型"这么做到底有没有用"。**

**为什么只做 SFT 不够？**
SFT 本质是"模仿学习"（临帖练字）——给模型看大量专家示范，让它学会在某种输入下输出什么。它的能力上界被钉死在训练数据的质量上：数据里没有的推理策略，模型永远学不会。所以纯 SFT 无法涌现出像 DeepSeek-R1 那样的"自我反思""长链推理"等数据中未出现的新能力。

**为什么跳过 SFT 直接做 RL 不行？**
RL 是靠奖励信号自己摸索。如果从基座模型直接开始，模型连基本的输出格式都不稳定（标签不配对、工具调用语法混乱），采样出来的轨迹奖励几乎全是 0，梯度信号极弱，训练极不稳定、几乎无法收敛。SFT 的作用就是先把策略"初始化"到一个格式规范、行为合理的起点，让 RL 有一个像样的探索基础。

**为什么过度 SFT 会损害 RL？**
RL 的进步来自"探索"——模型需要对同一个问题生成多样化的回答，再从好坏对比中学习。如果 SFT 练得太狠，模型对每个输入都会非常"自信"地输出几乎相同的答案（输出分布过于尖锐），多样性消失。这样一来，在 GRPO 里同一问题采样的 G 个回答几乎一模一样，奖励的组内标准差趋近 0，标准化后优势全为零，梯度消失，RL 等于白做。所以本章给出的"SFT 毕业标准"是：Loss 收敛 + 格式正确率 ≥ 90% + **输出仍保有多样性**——做到"足够好"即可，而不是"做到最好"。

</details>

**练习 2（辨析）**：有同学说："GRPO 不过就是把 PPO 的 Critic 模型删掉了，所以它一定比 PPO 更弱。" 这句话对吗？请从"基准线（baseline）"的角度解释 GRPO 用组内均值替代 Critic 的原理，以及这种替代各自付出的代价。再补充回答：为什么 GRPO 训练时如果把采样 temperature 设得太低（比如 0.1），训练会"完全停滞"？

<details>
<summary>参考答案</summary>

**这句话不对。** "删掉 Critic"不等于"变弱"——GRPO 只是换了一种更省资源的方式来实现 Critic 原本的核心作用。

**Critic 到底是干什么的？**
在 PPO 里，Critic 的本质作用只有一个：提供一条"基准线"，把"绝对奖励"转换成"相对优势"，从而降低梯度方差。打个比方：你考了 85 分，这个分数本身没意义，要看全班平均——平均 60 分你就是发挥出色（优势为正），平均 90 分你就是失常（优势为负）。Critic 就是那个"预测全班平均分"的模型。

**GRPO 怎么替代它？**
GRPO 发现：既然只是要一条基准线，那干脆**对同一个问题采样 G 个回答，直接用这 G 个回答的奖励均值当基准线**：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

比组内平均好的回答优势为正（强化），差的为负（抑制）。这样就完全不需要训练和存储一个与主模型同等大小的 Critic，显存从约 3× 模型大小降到约 1.5×。

**各自的代价：**
- Critic 是"参数化函数逼近器"，理论上能泛化到没见过的状态，但需要额外训练、会有估计误差，而且误差会传播到策略更新里，增加不稳定性。
- 组内均值是"非参数统计量"，无需训练、没有误差传播，但**依赖采样质量**——需要为每个问题多采 G 个回答（增加采样成本），而且如果采样多样性不足，基准线就不准。

**为什么 temperature 太低会让训练停滞？**
GRPO 的全部学习信号都来自"组内回答之间的奖励差异"。temperature 控制采样的随机性/多样性：temperature 太低（如 0.1），模型对同一问题生成的 G 个回答几乎完全相同 → 它们的奖励也几乎相同 → 组内标准差 $\sigma_r \approx 0$ → 标准化后所有优势都≈0 → 梯度为零 → 参数不再更新，训练彻底停滞。这就是为什么本章建议把 temperature 设在 0.6–0.8：既保证回答有差异可供比较，又不至于太随机导致质量崩坏。

</details>

**练习 3（动手）**：假设你要训练一个"数学解题 Agent"，要求它先输出 `<think>...</think>` 写推理过程，再给出最终数字答案。请用 Python 写一个奖励函数 `math_reward(completion, ground_truth)`，至少综合三个维度：(1) 答案是否正确；(2) 格式是否规范（有非空的 think 块）；(3) 防止"在 think 里写乱码、直接凑出正确答案"的奖励黑客。最后说明：如果把这个奖励函数用在 GRPO 训练里，对同一道题采样 8 个回答后，组内标准化优势是怎么算出来的。

<details>
<summary>参考答案</summary>

核心思路：**不要只看最终答案对不对**，否则模型会学会"think 里写乱码凑答案"的捷径。要把"答案正确性 × 推理过程的真实性"绑在一起。

```python
import re

def math_reward(completion: str, ground_truth: str) -> float:
    """
    数学解题 Agent 的综合奖励，返回值 ∈ [0, 1]
    三个维度：答案正确性 + 格式规范 + 防奖励黑客
    """
    reward = 0.0

    # ── 维度 1：格式规范（think 块存在且非空）──────────────
    has_think = "<think>" in completion and "</think>" in completion
    think_content = ""
    if has_think:
        think_content = completion.split("<think>")[1].split("</think>")[0].strip()
        if len(think_content) >= 20:   # 有实质性推理，不是空壳
            reward += 0.3

    # ── 维度 2：答案正确性 ─────────────────────────────────
    answer_correct = False
    try:
        # 取 completion 中最后一个数字作为最终答案
        nums = re.findall(r'-?[\d,]+\.?\d*', completion)
        pred = float(nums[-1].replace(",", ""))
        true = float(ground_truth.replace(",", ""))
        rel_err = abs(pred - true) / (abs(true) + 1e-8)
        answer_correct = rel_err < 1e-2     # 允许 1% 相对误差
    except (ValueError, IndexError):
        answer_correct = False

    if answer_correct:
        reward += 0.5

    # ── 维度 3：防奖励黑客（think 必须是"正常语言"）─────────
    # 统计 think 中有效字符（中英文/数字/常见标点）的比例，
    # 比例太低说明可能是乱码凑答案 → 即使答案对也要打折
    if think_content:
        valid = len(re.findall(r'[\u4e00-\u9fff\w\s.,!?，。！？；：=+\-*/()]', think_content))
        coherence = valid / max(len(think_content), 1)
        if coherence >= 0.8:
            reward += 0.2          # 推理连贯，给满
        elif coherence < 0.5 and answer_correct:
            reward *= 0.5          # 答案对但推理是乱码 → 奖励减半，封死黑客捷径

    return max(0.0, min(1.0, reward))
```

**关键设计点：**
- 答案正确只给 0.5，留出空间让"格式 + 真实推理"也参与计分，避免模型只盯着答案。
- 第 3 维是防御核心：当 think 像乱码（有效字符比例 < 0.5）而答案又恰好对时，判定为"凑答案"，把总奖励直接砍半——让钻空子无利可图。

**用于 GRPO 时，组内标准化优势怎么算？**
对同一道题，用旧策略采样 8 个回答 $y_1,\dots,y_8$，分别算出奖励 $r_1,\dots,r_8$。然后做组内标准化：

```python
import numpy as np

def grpo_advantages(rewards, eps=1e-8):
    r = np.array(rewards, dtype=np.float64)
    mu, sigma = r.mean(), r.std()
    if sigma < eps:          # 8 个回答奖励都一样 → 无法区分好坏
        return [0.0] * len(r)
    return ((r - mu) / (sigma + eps)).tolist()

# 例：8 个回答的奖励
rewards = [0.93, 0.30, 1.00, 0.50, 0.30, 1.00, 0.45, 0.30]
print(grpo_advantages(rewards))
# 高于均值的回答 → 优势为正（强化其推理路径）
# 低于均值的回答 → 优势为负（抑制其推理路径）
```

即：先求这 8 个奖励的均值 $\mu_r$ 和标准差 $\sigma_r$，再用 $\hat{A}_i=(r_i-\mu_r)/(\sigma_r+\epsilon)$ 把"绝对奖励"变成"相对于本组平均的好坏"。这个优势会乘到该回答每个 token 的策略梯度上，从而"好回答的整条推理被强化，差回答的被抑制"。注意一个边界：如果 8 个奖励完全相同（$\sigma_r\approx0$），优势全为 0、不更新——这正好呼应练习 2 里 temperature 过低导致训练停滞的现象。

</details>

---

## 参考文献

1. Patil et al. "Gorilla: Large Language Model Connected with Massive APIs." NeurIPS 2023.
2. Qin et al. "ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs." ICLR 2024.
3. Mitra et al. "AgentInstruct: Toward Generative Teaching with Agentic Flows." Microsoft Research 2024.
4. Liu et al. "What Makes Good Data for Alignment? (DEITA)" ICLR 2024.
5. Wang et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." ACL 2023.