# Prompt Engineering：与模型对话的艺术

如果说 LLM 是一台功能强大的机器，那么 Prompt Engineering（提示词工程）就是操作这台机器的技术。好的 Prompt 能让模型发挥出惊人的能力，糟糕的 Prompt 则可能让同一个模型产生令人沮丧的输出。

## 什么是 Prompt Engineering？

Prompt Engineering 是指**通过精心设计输入文本（Prompt），引导 LLM 产生期望输出的技术**。

这不仅仅是"会问问题"——它涉及对模型行为的深入理解，以及系统性地设计、测试、迭代提示词的方法论。

## 消息结构：System / User / Assistant

在调用 OpenAI 等 API 时，对话由三种角色的消息组成：

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",      # 系统指令：定义模型的角色和行为
            "content": "你是一个专业的 Python 编程助手，总是提供简洁、可运行的代码示例。"
        },
        {
            "role": "user",        # 用户输入
            "content": "如何用 Python 读取一个 CSV 文件？"
        },
        {
            "role": "assistant",   # 模型历史回复（多轮对话时使用）
            "content": "你可以使用 pandas 库..."
        },
        {
            "role": "user",
            "content": "能不用 pandas，只用标准库吗？"
        }
    ]
)

print(response.choices[0].message.content)
```

**三种角色的作用：**

| 角色 | 作用 | 优先级 |
|------|------|--------|
| `system` | 设定模型的角色、行为准则、输出格式 | 最高 |
| `user` | 用户的实际输入和请求 | 中等 |
| `assistant` | 模型的历史回复，用于多轮对话 | 参考 |

## System Prompt：塑造模型人格的利器

System Prompt 是 Prompt Engineering 中最重要的工具之一。它在整个对话过程中始终生效，就像给模型下达"工作守则"。

```python
# 示例：专业客服 Agent 的 System Prompt
system_prompt = """
你是"小智"，一名专业的技术支持工程师，服务于 TechCo 公司。

## 你的职责
- 帮助用户解决软件和硬件技术问题
- 提供清晰、步骤化的解决方案
- 遇到无法解决的问题，礼貌地引导用户联系高级支持

## 行为准则
1. 始终保持礼貌和专业
2. 每次回复先确认用户的问题，再提供解决方案
3. 给出步骤时，使用数字编号
4. 如果不确定，明确说明"这需要进一步排查"，不要猜测

## 回复格式
- 简洁明了，避免不必要的废话
- 代码用代码块包裹
- 关键信息用**加粗**标注

## 禁止行为
- 不讨论竞争对手产品
- 不提供可能导致数据丢失的操作建议
- 不超出技术支持范围
"""
```

**好的 System Prompt 应该包含：**

1. **角色定义**：明确"你是谁"
2. **职责边界**：能做什么、不能做什么
3. **行为准则**：如何思考和回答
4. **输出格式**：期望的回复格式
5. **特殊规则**：针对特定场景的约束

## 结构化输出：让模型按格式输出

Agent 开发中经常需要模型返回结构化数据（如 JSON），以便程序解析。

```python
import json
from openai import OpenAI

client = OpenAI()

def extract_task_info(user_input: str) -> dict:
    """从用户自然语言中提取任务信息"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},  # 强制 JSON 输出
        messages=[
            {
                "role": "system",
                "content": """你是一个任务解析助手。从用户输入中提取任务信息，
                返回以下 JSON 格式：
                {
                    "title": "任务标题",
                    "priority": "high/medium/low",
                    "deadline": "截止日期（YYYY-MM-DD 格式，无则为 null）",
                    "tags": ["标签1", "标签2"],
                    "description": "任务描述"
                }"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    
    return json.loads(response.choices[0].message.content)

# 测试
result = extract_task_info("明天下午三点之前把项目报告发给老板，很重要！")
print(result)
# 输出：
# {
#     "title": "提交项目报告",
#     "priority": "high",
#     "deadline": "2024-01-15",
#     "tags": ["报告", "项目"],
#     "description": "将项目报告发送给老板"
# }
```

## 角色扮演：激活模型的专业能力

通过让模型扮演特定角色，可以激活其在该领域的专业知识：

```python
# 让模型扮演不同专家来分析同一问题
def analyze_from_perspective(topic: str, role: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"你是一位资深的{role}，请从你的专业角度分析以下问题，"
                           f"提供专业、深入的见解。"
            },
            {"role": "user", "content": f"请分析：{topic}"}
        ]
    )
    return response.choices[0].message.content

topic = "AI Agent 技术在未来五年的发展趋势"

# 从不同视角分析
tech_view = analyze_from_perspective(topic, "AI 技术研究员")
biz_view = analyze_from_perspective(topic, "科技行业投资人")
ethics_view = analyze_from_perspective(topic, "AI 伦理学家")
```

## 约束与格式化：精确控制输出

```python
# 精确控制输出格式的 Prompt 技巧
def generate_product_description(product_info: dict) -> str:
    prompt = f"""
请为以下产品生成营销描述。

产品信息：
- 名称：{product_info['name']}
- 类别：{product_info['category']}
- 主要特性：{', '.join(product_info['features'])}
- 目标用户：{product_info['target_users']}

## 输出要求
1. 总字数：50-80字
2. 语气：专业但亲切
3. 必须包含一个具体的使用场景
4. 结尾加一句号召性用语
5. 不要使用"最好的"、"第一"等极端词汇

## 输出格式
直接输出描述文本，不需要任何解释或前言。
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

product = {
    "name": "SmartNote AI 笔记本",
    "category": "数字办公工具",
    "features": ["AI 总结", "语音转文字", "跨设备同步"],
    "target_users": "职场人士和学生"
}
print(generate_product_description(product))
```

## 迭代优化：Prompt 调试方法论

Prompt Engineering 不是一次性的工作，而是持续迭代的过程：

```python
class PromptTester:
    """Prompt 测试和对比工具"""
    
    def __init__(self, client):
        self.client = client
        self.results = []
    
    def test_prompt(self, 
                    system_prompt: str, 
                    test_cases: list, 
                    model: str = "gpt-4o-mini") -> dict:
        """测试一个 Prompt 在多个测试用例上的表现"""
        
        results = []
        for test_input in test_cases:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_input}
                ]
            )
            results.append({
                "input": test_input,
                "output": response.choices[0].message.content,
                "tokens": response.usage.total_tokens
            })
        
        return {
            "prompt": system_prompt,
            "results": results,
            "avg_tokens": sum(r["tokens"] for r in results) / len(results)
        }
    
    def compare_prompts(self, prompts: list, test_cases: list):
        """对比多个 Prompt 版本"""
        for i, prompt in enumerate(prompts):
            print(f"\n=== Prompt 版本 {i+1} ===")
            result = self.test_prompt(prompt, test_cases)
            for r in result["results"]:
                print(f"\n输入: {r['input']}")
                print(f"输出: {r['output']}")
                print(f"Token 消耗: {r['tokens']}")

# 使用示例
tester = PromptTester(client)

# 对比两种 System Prompt 的效果
prompts = [
    "你是一个助手，帮用户回答问题。",  # 版本 1：模糊
    """你是一个专业的 Python 编程教练。
    回答规则：
    1. 先解释概念（1-2句）
    2. 提供代码示例
    3. 说明常见错误
    每次回复控制在200字以内。"""  # 版本 2：清晰
]

test_cases = [
    "什么是列表推导式？",
    "如何处理文件读写异常？"
]

tester.compare_prompts(prompts, test_cases)
```

## Prompt 设计的黄金原则

经过大量实践总结，以下原则能显著提升 Prompt 质量：

| 原则 | 说明 | 反例 | 正例 |
|------|------|------|------|
| **明确性** | 清晰说明任务 | "写点东西" | "写一篇300字的产品介绍" |
| **上下文** | 提供足够背景 | "优化这段代码" | "优化这段 Python 代码，要求时间复杂度 O(n)" |
| **格式指定** | 明确输出格式 | （无要求） | "以 JSON 格式返回，包含 name 和 score 字段" |
| **角色定义** | 激活专业知识 | （无角色） | "你是一位有10年经验的 Python 工程师" |
| **示例引导** | 用例子展示期望 | （无示例） | "例如：输入A → 输出B" |
| **约束边界** | 明确不该做什么 | （无限制） | "不超过100字，不使用专业术语" |

---

## 小结

Prompt Engineering 是 Agent 开发的核心技能之一。好的 Prompt 能让同一个模型产生截然不同的输出质量。关键要点：

- **System Prompt** 是定义 Agent 行为的最重要工具
- **结构化输出**（JSON）让 Agent 的工具调用更可靠
- **迭代测试**是提升 Prompt 质量的正确方法
- **明确性**、**上下文**、**格式**是高质量 Prompt 的三要素

---

*下一节：[2.3 Few-shot / Zero-shot / Chain-of-Thought 提示策略](./03_prompting_strategies.md)*
