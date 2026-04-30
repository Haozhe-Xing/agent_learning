# 5.1 Agent 如何"思考"？

> 🎯 **本节学习目标**：建立 Agent 推理的心理模型，理解 OODA 循环在 AI 决策中的应用。

Agent 的"思考"本质上是**在上下文中组织信息、推导结论、制定计划的过程**。理解这个过程，是设计高效 Agent 的前提。

## 思考的本质：上下文中的推理

### 心理模型：为什么直接询问通常无效？
如果把 LLM 比作一个阅读速度极快的"阅读者"，直接提问就像是让它在还没看完题目的情况下给出结论。LLM 只有在输出每一个 Token 时，才在“思考”。因此，引导推理的关键是**强制要求模型在输出最终答案前，先把思维过程展示出来（CoT 策略）**。

### 实践练习：重构推理过程
对比以下两种代码，思考为什么右侧的结构化推理更稳定？

```python
# 练习：尝试在 structured_thinking 中加入一个“检查是否有逻辑漏洞”的环节
def structured_thinking(question: str) -> str:
    system_prompt = """请使用结构化分析：
    1. 【问题拆解】...
    2. 【推理步骤】...
    3. 【自我验证】... # 在这里尝试增加检查是否存在逻辑跳跃的检查逻辑
    """
    # ...
```

---

## 认知框架：OODA 循环

Agent 的决策可以用 OODA 循环来理解：

![OODA决策循环](../svg/chapter_planning_01_ooda_loop.svg)

```python
class OODAAgent:
    """基于 OODA 循环的 Agent 框架"""
    
    def __init__(self):
        self.context = {}  # 当前情境理解
    
    def observe(self, input_data: str) -> str:
        """观察：收集和整理当前环境信息"""
        prompt = f"""
分析以下输入，提取关键信息：
{input_data}

请识别：
1. 用户的明确需求
2. 隐含的期望
3. 可能的障碍
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )
        observation = response.choices[0].message.content
        self.context["observation"] = observation
        return observation
    
    def orient(self, observation: str) -> str:
        """定位：在已知知识框架中理解当前情况"""
        prompt = f"""
基于以下观察，进行情境评估：
{observation}

请分析：
1. 这个任务属于哪类问题？
2. 有哪些可用的方法和工具？
3. 主要的风险和挑战是什么？
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )
        orientation = response.choices[0].message.content
        self.context["orientation"] = orientation
        return orientation
    
    def decide(self, orientation: str) -> str:
        """决策：制定行动计划"""
        prompt = f"""
基于情境评估，制定具体行动计划：
{orientation}

请给出：
1. 推荐的行动方案（第一选择）
2. 备选方案
3. 执行步骤（按优先级排序）
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response.choices[0].message.content
        self.context["decision"] = decision
        return decision
    
    def act(self, plan: str, user_input: str) -> str:
        """行动：执行计划并生成最终响应"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": f"执行计划：\n{plan}\n\n用自然语言给用户一个清晰的回答。"
                },
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    
    def process(self, user_input: str) -> str:
        """完整的 OODA 循环"""
        obs = self.observe(user_input)
        orientation = self.orient(obs)
        decision = self.decide(orientation)
        result = self.act(decision, user_input)
        return result
```

## 元认知：Agent 的自我意识

高级 Agent 具备元认知能力——能够思考自己的思考过程：

```python
def metacognitive_reasoning(problem: str) -> dict:
    """元认知推理：Agent 能评估自己的置信度和局限性"""
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": """回答时，始终进行元认知评估：
1. 我对这个问题的知识有多可靠？（置信度 0-10）
2. 哪些方面我可能存在盲区？
3. 是否需要额外工具或信息？
4. 我的回答基于哪些假设？"""
            },
            {"role": "user", "content": problem}
        ]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "self_assessed_by_llm": True
    }

# 测试元认知
result = metacognitive_reasoning("量子计算机什么时候能超越传统计算机？")
print(result["answer"])
```

## 推理模式对比

![推理模式四大类型](../svg/chapter_planning_01_reasoning_modes.svg)

---

## 小结

Agent 的"思考"依赖于：
- 结构化的推理框架（CoT、OODA 等）
- 元认知能力（知道自己不知道什么）
- 不同的推理模式（演绎、归纳、溯因、类比）

---

*下一节：[5.2 ReAct：推理 + 行动框架](./02_react_framework.md)*
