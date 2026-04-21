# 3.3 Few-shot / Zero-shot / Chain-of-Thought 提示策略

掌握了基础的 Prompt Engineering 之后，我们来学习几种经过研究验证的提示策略。这些策略在面对复杂任务时，能显著提升 LLM 的表现。

> 📄 **学术背景**：Few-shot 学习由 Brown 等人在 GPT-3 论文中系统化研究 [1]，证明了大模型仅通过几个示例就能适应新任务。Chain-of-Thought（思维链）由 Wei 等人提出 [2]，通过在 prompt 中加入"让我们一步步思考"就能大幅提升 LLM 的推理能力——在 GSM8K 数学推理基准上，CoT 将 PaLM 540B 的准确率从 17.9% 提升到 58.1%。

## Zero-shot：直接提问

**Zero-shot（零样本）** 是最简单的策略：直接告诉模型任务是什么，不提供任何示例。

```python
from openai import OpenAI

client = OpenAI()

def zero_shot_classify(text: str) -> str:
    """零样本情感分类"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "你是一个文本分类专家。"
            },
            {
                "role": "user",
                "content": f"""
对以下评论进行情感分类，只返回：正面、负面、中性 之一。

评论：{text}

分类结果："""
            }
        ]
    )
    return response.choices[0].message.content.strip()

# 测试
texts = [
    "这个产品太棒了，完全超出了我的预期！",
    "还可以吧，没什么特别的。",
    "质量很差，完全浪费钱，不推荐。"
]

for text in texts:
    result = zero_shot_classify(text)
    print(f"文本: {text[:20]}... → {result}")
```

**Zero-shot 适用场景：**
- 任务描述清晰、模型熟悉的常见任务
- 快速原型验证
- 对延迟和成本要求较高的场景

## Few-shot：示例引导学习

**Few-shot（少样本）** 通过提供几个示例，让模型理解期望的输入输出模式。

```python
def few_shot_classify(text: str) -> str:
    """少样本情感分类——通过示例引导"""
    
    # 精心挑选的示例（覆盖不同情况）
    examples = [
        ("这款手机拍照效果惊艳，续航也很棒！", "正面"),
        ("物流太慢了，等了两周才到，很失望。", "负面"),
        ("产品符合描述，包装完好。", "中性"),
        ("客服态度超好，问题解决得很及时，给好评！", "正面"),
        ("有点贵，但质量确实不错。", "中性"),
    ]
    
    # 构建 Few-shot Prompt
    few_shot_prompt = "对以下评论进行情感分类（正面/负面/中性）。\n\n"
    
    for example_text, label in examples:
        few_shot_prompt += f"评论：{example_text}\n情感：{label}\n\n"
    
    few_shot_prompt += f"评论：{text}\n情感："
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": few_shot_prompt}]
    )
    return response.choices[0].message.content.strip()

# Few-shot 在复杂分类任务上通常更稳定
result = few_shot_classify("不得不说，价格有些偏高，不过产品本身没什么大问题。")
print(f"分类结果：{result}")
```

**Few-shot 的关键技巧：**

```python
def create_few_shot_prompt(task_description: str, 
                            examples: list[tuple],
                            new_input: str) -> str:
    """
    通用 Few-shot Prompt 构建器
    
    Args:
        task_description: 任务描述
        examples: [(输入, 输出), ...] 示例列表
        new_input: 待处理的新输入
    """
    prompt = f"{task_description}\n\n"
    
    prompt += "## 示例\n\n"
    for i, (inp, out) in enumerate(examples, 1):
        prompt += f"示例 {i}：\n"
        prompt += f"输入：{inp}\n"
        prompt += f"输出：{out}\n\n"
    
    prompt += f"## 现在请处理\n"
    prompt += f"输入：{new_input}\n"
    prompt += "输出："
    
    return prompt

# 使用示例：代码注释生成
examples = [
    (
        "def add(a, b): return a + b",
        "# 将两个数相加并返回结果\ndef add(a, b): return a + b"
    ),
    (
        "def is_even(n): return n % 2 == 0",
        "# 判断数字是否为偶数，返回布尔值\ndef is_even(n): return n % 2 == 0"
    ),
]

new_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
prompt = create_few_shot_prompt(
    "为以下 Python 函数添加一行中文注释",
    examples,
    new_code
)
print(prompt)
```

**示例选择原则：**
1. **代表性**：覆盖任务的各种情况（边界case）
2. **多样性**：不要全是相似的例子
3. **质量**：示例本身必须正确
4. **顺序**：最后一个示例的风格影响最大

## Chain-of-Thought（CoT）：让模型"想出来"

**Chain-of-Thought（思维链）** 是一种革命性的提示策略：通过让模型展示推理过程，显著提升其在复杂问题上的准确率。

> 📄 **论文出处**：CoT 由 Google Brain 团队在论文 *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*（Wei et al., 2022）中首次提出。该论文发现，只要在 Few-shot 示例中加入推理步骤，模型在 GSM8K 数学推理数据集上的准确率就能从 17.7% 飙升到 58.1%——仅仅是改变了提示的格式，没有修改模型的任何参数。这个发现揭示了一个深刻的事实：**大型语言模型已经具备了推理能力，我们需要的只是用正确的方式"激发"它。**

![思维链推理示意图](../svg/chapter_llm_03_cot.svg)

```python
def solve_with_cot(problem: str) -> str:
    """使用思维链解决复杂问题"""
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": """解题时，请严格按照以下步骤：
1. 理解问题（用1-2句话复述问题）
2. 分析已知条件
3. 制定解题思路
4. 逐步推导
5. 得出结论并验证

每一步都要明确标注。"""
            },
            {
                "role": "user",
                "content": problem
            }
        ]
    )
    return response.choices[0].message.content

# 数学推理问题
problem = """
小明有若干个苹果。他先给了小红苹果总数的一半多1个，又给了小李苹果总数的四分之一，
此时小明还剩9个苹果。请问小明最初有多少个苹果？
"""

result = solve_with_cot(problem)
print(result)
```

**Zero-shot CoT 魔法咒语：**

> 📄 **论文出处**：*"Large Language Models are Zero-Shot Reasoners"*（Kojima et al., 2022）发现了一个令人惊讶的事实——只需在 Prompt 末尾加上 *"Let's think step by step"*（"让我们一步步思考"）这句话，就能在不提供任何推理示例的情况下触发 CoT 推理。这意味着模型的推理能力是"内建"的，只需要一个简单的触发词就能激活。

```python
def zero_shot_cot(question: str) -> str:
    """零样本思维链：加上魔法咒语即可触发推理"""
    
    # 第一步：触发推理
    response1 = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": f"{question}\n\n让我们一步步思考（Let's think step by step）："
            }
        ]
    )
    
    reasoning = response1.choices[0].message.content
    
    # 第二步：基于推理给出最终答案
    response2 = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": f"{question}\n\n让我们一步步思考："},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "基于以上推理，请给出最终简洁的答案："}
        ]
    )
    
    return {
        "reasoning": reasoning,
        "answer": response2.choices[0].message.content
    }

result = zero_shot_cot("如果一列火车以120km/h的速度行驶，需要多少分钟才能走完180公里？")
print("推理过程：", result["reasoning"])
print("\n最终答案：", result["answer"])
```

## 高级策略：Tree-of-Thought（ToT）

**Tree-of-Thought** 是 CoT 的升级版：让模型探索多条推理路径，选择最优解。

> 📄 **论文出处**：*"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"*（Yao et al., 2023）。与 CoT 的"一条路走到底"不同，ToT 让模型在每一步都生成多个候选想法（Thought），然后用评估函数判断哪些想法更有前景，最终像搜索树一样找到最优的推理路径。论文在 24 点游戏上的实验尤其惊艳——标准 CoT 仅解出 4% 的题目，ToT 达到了 74%。

```python
def tree_of_thought(problem: str, num_paths: int = 3) -> str:
    """
    思维树：生成多条推理路径，评估并选择最优解
    适用于复杂决策问题
    """
    
    # 步骤1：生成多条推理路径
    paths_prompt = f"""
问题：{problem}

请提供 {num_paths} 种不同的解决思路（每种思路用简短的标题开头，然后描述核心方法）：
"""
    
    paths_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": paths_prompt}]
    )
    
    paths = paths_response.choices[0].message.content
    
    # 步骤2：评估各路径
    eval_prompt = f"""
问题：{problem}

以下是几种解决思路：
{paths}

请评估每种思路的：
1. 可行性（1-10分）
2. 时间成本
3. 潜在风险

最终推荐哪种方案并说明原因。
"""
    
    eval_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": eval_prompt}]
    )
    
    return {
        "paths": paths,
        "evaluation": eval_response.choices[0].message.content
    }

# 示例：复杂决策问题
problem = """
我需要在3个月内学习 Python，用于数据分析工作。
我每天只有2小时学习时间，有一定的编程基础（学过简单的 HTML/CSS）。
应该如何规划学习路径？
"""

result = tree_of_thought(problem)
print("多路径探索：\n", result["paths"])
print("\n评估与推荐：\n", result["evaluation"])
```

## ReAct：推理与行动交织

**ReAct（Reasoning + Acting）** 是 Agent 开发中最重要的提示策略之一（第6章会深入讲解）。

> 📄 **论文出处**：*"ReAct: Synergizing Reasoning and Acting in Language Models"*（Yao et al., 2022）。ReAct 的核心洞察是：**纯推理（CoT）和纯行动（工具调用）都不够好，将两者交织在一起才能获得最佳效果。** 在 HotpotQA 多跳推理任务上，ReAct 比纯 CoT 提升了 6 个百分点；在 ALFWorld 交互式任务上，比纯行动模式提升了 34 个百分点。这篇论文直接奠定了现代 Agent 的基本架构。

```python
react_prompt = """
你是一个能够使用工具的 AI 助手。解决问题时，请严格按照以下格式：

思考（Thought）：分析当前情况，决定下一步
行动（Action）：选择并使用工具
观察（Observation）：记录工具返回的结果
... （重复直到问题解决）
答案（Answer）：最终答案

可用工具：
- search(query): 搜索互联网
- calculate(expression): 计算数学表达式
- get_weather(city): 获取天气信息

---
问题：北京今天的气温是多少摄氏度？折算成华氏度是多少？

思考：我需要先获取北京今天的气温，然后进行单位换算。
行动：get_weather("北京")
观察：北京今日气温：12°C
思考：已经获取了气温，现在需要将12°C换算成华氏度，公式是 F = C × 9/5 + 32
行动：calculate("12 * 9/5 + 32")
观察：53.6
答案：北京今天气温是 12°C，换算成华氏度是 53.6°F。
"""

# 这个模式将在第6章详细实现
```

## 提示策略选择指南

![\u63d0\u793a\u7b56\u7565\u9009\u62e9\u6307\u5357](../svg/chapter_llm_03_strategy_guide.svg)

## 实战：综合策略对比

```python
import time

def benchmark_strategies(question: str) -> dict:
    """对比不同策略在同一问题上的效果"""
    
    strategies = {
        "zero_shot": {
            "messages": [{"role": "user", "content": question}]
        },
        "cot": {
            "messages": [{"role": "user", "content": f"{question}\n\n让我们一步步思考："}]
        },
        "few_shot_cot": {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个逻辑推理专家，解题时先分析条件，再逐步推导。"
                },
                {
                    "role": "user",
                    "content": """示例：
问题：5个人分12个苹果，平均每人得几个？
分析：总量12，人数5，做除法
计算：12 ÷ 5 = 2.4
答案：平均每人得 2.4 个苹果

现在解答：""" + question
                }
            ]
        }
    }
    
    results = {}
    for name, config in strategies.items():
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            **config
        )
        elapsed = time.time() - start
        
        results[name] = {
            "answer": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "time": f"{elapsed:.2f}s"
        }
    
    return results

# 测试复杂推理问题
question = "一个班级有40名学生，其中60%喜欢数学，75%喜欢语文。至少有多少名学生同时喜欢数学和语文？"
results = benchmark_strategies(question)

for strategy, data in results.items():
    print(f"\n策略：{strategy}")
    print(f"Token 消耗：{data['tokens']}")
    print(f"耗时：{data['time']}")
    print(f"回答：{data['answer'][:200]}...")
```

---

## 小结

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| Zero-shot | 常见标准任务 | 简单、快速、省 Token | 复杂任务效果差 |
| Few-shot | 需要特定格式/风格 | 稳定可控 | 消耗更多 Token |
| CoT | 推理、计算、多步骤 | 准确率高 | 慢、Token 多 |
| ToT | 复杂决策问题 | 探索多方案 | 最慢、最贵 |
| ReAct | 需要工具调用的 Agent | 融合推理和行动 | 实现复杂 |

选择合适的策略是 Agent 开发的重要技能——不是"越复杂越好"，而是"恰到好处"。

### 📖 延伸阅读：核心论文

本节涉及的提示策略都有扎实的学术研究基础。以下是最重要的论文，按发表时间排序：

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|---------|
| *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | Wei et al. (Google Brain) | 2022 | 首次提出 CoT，证明在示例中加入推理步骤可大幅提升数学和逻辑推理能力 |
| *Large Language Models are Zero-Shot Reasoners* | Kojima et al. | 2022 | 发现 "Let's think step by step" 一句话就能激活 Zero-shot CoT |
| *Self-Consistency Improves Chain of Thought Reasoning* | Wang et al. (Google Brain) | 2023 | 提出自我一致性（Self-Consistency）：多次采样 CoT 路径，取多数投票结果，进一步提升推理准确率 |
| *ReAct: Synergizing Reasoning and Acting in Language Models* | Yao et al. (Princeton) | 2022 | 将推理和行动交织，奠定了现代 Agent 的 ReAct 架构 |
| *Tree of Thoughts: Deliberate Problem Solving with LLMs* | Yao et al. (Princeton) | 2023 | CoT 的升级版，多路径探索 + 回溯搜索，在复杂推理任务上大幅超越 CoT |

> 💡 **前沿进展**：2024-2025 年以来，推理模型成为 LLM 发展的核心方向。OpenAI 的 o1/o3/o4-mini 系列模型、Anthropic 的 Claude 4 Extended Thinking、DeepSeek-R2 等模型将 CoT 推理"内化"到了模型本身（而非依赖提示词），在数学、编程竞赛和科学推理中展现了惊人的能力。Google 的 Gemini 2.5 Pro 也引入了"Thinking Mode"。这表明 CoT 已从一种"提示技巧"演变为模型训练的核心范式——未来的 LLM 将越来越"会想"。对于 Agent 开发者而言，推理模型让 Agent 在复杂多步任务中的规划能力大幅提升。

> 📖 **更多论文解读**：ReAct 的深度解读请见 [6.6 论文解读：规划与推理前沿研究](../chapter_planning/06_paper_readings.md)，Self-Consistency 在幻觉缓解中的应用请见 [17.6 论文解读：安全与可靠性前沿研究](../chapter_security/06_paper_readings.md)。

---

## 参考文献

[1] BROWN T B, MANN B, RYDER N, et al. Language models are few-shot learners[C]//NeurIPS. 2020.

[2] WEI J, WANG X, SCHUURMANS D, et al. Chain-of-thought prompting elicits reasoning in large language models[C]//NeurIPS. 2022.

[3] KOJIMA T, GU S, REID M, et al. Large language models are zero-shot reasoners[C]//NeurIPS. 2022.

[4] WANG X, WEI J, SCHUURMANS D, et al. Self-consistency improves chain of thought reasoning in language models[C]//ICLR. 2023.

[5] YAO S, YU D, ZHAO J, et al. Tree of thoughts: Deliberate problem solving with large language models[C]//NeurIPS. 2023.

---

*下一节：[3.4 模型 API 调用入门（OpenAI / 开源模型）](./04_api_basics.md)*
