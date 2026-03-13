# Token、Temperature 与模型参数详解

模型参数是影响 LLM 输出质量、成本和稳定性的关键因素。理解这些参数，能让你更精准地控制 Agent 的行为。

## Token：模型的"基本单位"

Token 不等于字符，不等于单词，而是模型处理文本的**最小单位**。

```python
import tiktoken  # OpenAI 的 Token 计数库

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """计算文本的 Token 数量"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def visualize_tokens(text: str, model: str = "gpt-4o"):
    """可视化 Token 分割"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    print(f"文本：{text}")
    print(f"Token 数量：{len(tokens)}")
    print(f"Token 列表：{[encoding.decode([t]) for t in tokens]}")
    print()

# 英文分词示例
visualize_tokens("Hello, how are you today?")
# Token 列表：['Hello', ',', ' how', ' are', ' you', ' today', '?']
# Token 数量：7

# 中文分词示例（中文通常更多 Token）
visualize_tokens("你好，今天天气怎么样？")
# 中文每个字通常占 1-2 个 Token

# 代码的 Token 计数
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
visualize_tokens(code)
```

**Token 与计费的关系：**

> ⏰ *注：以下价格数据基于 2026 年 3 月各厂商官网公开定价，模型价格调整频繁，请以最新官方文档为准。*

```python
# Token 成本计算器（价格单位：美元/百万 Token，2026-03 数据）
PRICE_PER_1M_TOKENS = {
    "gpt-5": {"input": 2.0, "output": 8.0},           # 美元/百万 Token
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "claude-4-sonnet": {"input": 3.0, "output": 15.0},
    "deepseek-r2": {"input": 0.14, "output": 2.19},
}

def estimate_cost(
    input_text: str,
    expected_output_tokens: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """估算 API 调用成本"""
    input_tokens = count_tokens(input_text, model)
    
    price = PRICE_PER_1M_TOKENS.get(model, {"input": 1.0, "output": 2.0})
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (expected_output_tokens / 1_000_000) * price["output"]
    
    return {
        "input_tokens": input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "total_tokens": input_tokens + expected_output_tokens,
        "estimated_cost_usd": input_cost + output_cost,
        "estimated_cost_cny": (input_cost + output_cost) * 7.2  # 近似汇率
    }

# 估算成本
prompt = "请为我写一篇关于 Python 异步编程的500字文章"
cost = estimate_cost(prompt, 500, "gpt-4o-mini")
print(f"输入 Token：{cost['input_tokens']}")
print(f"预估总成本：¥{cost['estimated_cost_cny']:.4f}")
```

**Token 使用的关键规律：**
- 英文约 1 Token/词
- 中文约 1 Token/字
- 代码约 1-2 Token/行
- 数字和标点各占 1 Token

## Temperature：创造力旋钮

Temperature 控制输出的**随机性**，是最重要的参数之一：

```python
from openai import OpenAI
client = OpenAI()

def test_temperature(prompt: str, temperatures: list, runs: int = 3):
    """对比不同 Temperature 的输出效果"""
    
    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"Temperature = {temp}")
        print('='*50)
        
        for i in range(runs):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=50
            )
            print(f"  运行 {i+1}：{response.choices[0].message.content}")

# 测试创意写作（高 Temperature 更好）
test_temperature(
    "用一句话描述春天",
    temperatures=[0.0, 0.7, 1.5],
    runs=3
)
# Temperature=0.0：每次输出完全相同（确定性）
# Temperature=0.7：有一定变化，但合理
# Temperature=1.5：更多创意，但可能不连贯
```

**不同场景的 Temperature 推荐值：**

```python
TEMPERATURE_GUIDE = {
    "代码生成": 0.1,          # 要求精确，低随机性
    "数据提取/分类": 0.0,      # 完全确定性
    "问答/事实查询": 0.3,      # 稍微稳定
    "文案/摘要": 0.7,         # 平衡创意和准确
    "头脑风暴/创意": 1.0,      # 鼓励多样性
    "诗歌/创意写作": 1.2,      # 高创意
    "Agent 推理": 0.1,        # 推理需要稳定性
    "对话/闲聊": 0.8,         # 自然对话
}

def get_optimal_temperature(task_type: str) -> float:
    return TEMPERATURE_GUIDE.get(task_type, 0.7)
```

## Top-p：另一种控制随机性的方式

Top-p（也叫 Nucleus Sampling）从概率最高的词集合中采样，集合大小由 p 决定：

```python
# Top-p 与 Temperature 的配合使用
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一个关于机器人的短故事开头"}],
    temperature=0.8,   # 控制随机程度
    top_p=0.9,         # 只从概率总和 90% 的候选词中选择
    max_tokens=200
)
```

**Temperature vs Top-p 的区别：**

| 参数 | 机制 | 推荐做法 |
|------|------|---------|
| Temperature | 缩放概率分布 | 通常调这个 |
| Top-p | 截断低概率词 | 两个不要同时大幅调整 |

通常选择其中一个调整，另一个保持默认（temperature=1.0 或 top_p=1.0）。

## max_tokens：控制输出长度

```python
def chat_with_length_control(
    message: str,
    max_output_tokens: int = 500
) -> dict:
    """控制输出长度"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        max_tokens=max_output_tokens  # 最多生成 N 个 Token
    )
    
    usage = response.usage
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    
    return {
        "content": content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "finish_reason": finish_reason  # "stop"=正常结束, "length"=达到上限
    }

# 检查是否被截断
result = chat_with_length_control("写一篇500字的文章", max_output_tokens=100)
if result["finish_reason"] == "length":
    print("⚠️ 输出被 max_tokens 截断了！")
    print(f"已生成：{result['output_tokens']} tokens")
```

## Presence Penalty & Frequency Penalty：控制重复

```python
# 这两个参数帮助避免模型重复自己
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "列举10种不同的创业方向"}],
    
    # presence_penalty：惩罚已经出现过的词（降低重复）
    # 范围：-2.0 到 2.0，正值降低重复
    presence_penalty=0.5,
    
    # frequency_penalty：基于出现频率的惩罚（越用越不想用）
    # 范围：-2.0 到 2.0，正值降低高频词
    frequency_penalty=0.3,
)

# 适合用于：需要列举多样选项、生成不重复内容的场景
```

## stop：自定义停止条件

```python
# 让模型在特定字符串处停止生成
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "请按格式输出：\n名称：\n价格：\n描述："
    }],
    stop=["描述："],  # 在生成"描述："这个词之前停止
)

# 更实用的用法：结构化数据提取
def extract_until_marker(text: str, stop_marker: str) -> str:
    """提取直到某个标记符的内容"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        stop=[stop_marker]
    )
    return response.choices[0].message.content
```

## n：生成多个候选结果

```python
def generate_multiple_options(prompt: str, n: int = 3) -> list:
    """一次 API 调用生成多个候选"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        n=n,  # 生成 n 个不同的回复
        temperature=0.9  # 高 temperature 确保多样性
    )
    
    return [choice.message.content for choice in response.choices]

# 适合：标题生成、创意方案、A/B测试
titles = generate_multiple_options(
    "为一篇关于 Python 异步编程的技术博客生成3个吸引人的标题",
    n=3
)
for i, title in enumerate(titles, 1):
    print(f"候选 {i}：{title}")
```

## 完整参数参考与 Agent 实践建议

```python
def create_agent_call(
    messages: list,
    task_type: str = "general",
    **override_params
) -> dict:
    """
    Agent 调用的最佳实践封装
    根据任务类型自动选择最优参数
    """
    
    # 不同任务类型的参数预设
    task_presets = {
        "reasoning": {          # 推理、分析任务
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
        "code": {               # 代码生成
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        },
        "extraction": {         # 信息提取、分类
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 500,
        },
        "creative": {           # 创意写作
            "model": "gpt-4o",
            "temperature": 0.9,
            "max_tokens": 1000,
            "presence_penalty": 0.3,
        },
        "chat": {               # 普通对话
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "general": {            # 通用（默认）
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1500,
        }
    }
    
    params = task_presets.get(task_type, task_presets["general"])
    params.update(override_params)  # 允许覆盖参数
    params["messages"] = messages
    
    response = client.chat.completions.create(**params)
    
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        },
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason
    }

# 使用示例
result = create_agent_call(
    messages=[{"role": "user", "content": "帮我写一个二分查找算法"}],
    task_type="code"
)
print(f"模型：{result['model']}")
print(f"Token 消耗：{result['usage']['total']}")
print(f"代码：\n{result['content']}")
```

## 参数速查卡

![模型参数速查卡](../svg/chapter_llm_05_params_cheatsheet.svg)

---

## 本节小结

理解模型参数是 Agent 开发中的必备知识：

- **Token** 是计费和上下文限制的基本单位，中文约 1 字 1 Token
- **Temperature** 是最常调的参数：推理任务用 0.1，创意任务用 0.9+
- **max_tokens** 要留够余量，避免输出被截断
- **Penalty 参数** 在需要多样性输出时非常有用
- 不同任务类型应该使用不同的参数组合

掌握这些参数，你就能更精准地控制 Agent 的行为，同时优化成本和质量的平衡。

### 🤔 思考练习

1. 一个 Agent 需要同时完成"代码生成"和"创意写作"两种任务，你会如何为它们设置不同的 Temperature？
2. 如果你发现 Agent 的回复总是重复使用某些词汇，应该调整哪个参数？为什么？
3. 计算一下：如果你的 Agent 每天处理 1000 次请求，每次平均消耗 2000 Token，使用 GPT-4o-mini，月费用大约是多少？

---

*下一章：[第3章 开发环境搭建](../chapter_setup/README.md)*
