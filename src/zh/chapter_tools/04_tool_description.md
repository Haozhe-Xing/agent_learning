# 4.4 工具描述的编写技巧

> 🎯 **本节目标**：掌握让 LLM "听懂"工具的描述编写方法——这是 Agent 工程中 **投入产出比最高** 的技能之一。

---

## 为什么这一节单独拿出来讲？

因为**描述质量直接决定了 Agent 的智商**，而大多数开发者在这里花的时间远远不够。

一个真实的数据点：

```
同样的 GPT-4.1 模型 + 同样的工具代码：
- 模糊描述 → 工具调用准确率 ~60%
- 精心优化的描述 → 工具调用准确率 ~90%+

差距来自哪里？完全来自 description 字段的写法。
```

LLM 选择工具时看不到你的函数实现——它只能看到你写的 `description`。**这段文字就是 LLM 理解工具的唯一窗口。**

---

## 六要素检查清单

### 1️⃣ 一句话功能说明（最重要）

LLM 扫描工具列表时首先看的就是第一句话。

```python
# ❌ 太模糊 —— LLM 不知道什么时候该用
"description": "处理文本"

# ❌ 太技术化 —— LLM 不是程序员，不需要知道内部机制
"description": "基于 Transformer 的 NLP 文本处理管道"

# ✅ 动词 + 对象 + 核心特征
"description": "将任意语言的文本翻译成指定目标语言，保持原文格式和语气"
```

**公式：动词 + 操作对象 + 核心能力边界**

### 2️⃣ 适用场景（何时用）

```python
"description": """查询股票实时行情数据。

适合用于：
- 获取当前股价和涨跌幅
- 查询市值、市盈率等基本面指标
- 比较多只股票的关键数据"""
```

### 3️⃣ 不适用场景（何时不用的）

> 这是最容易被忽略、但效果最显著的要素。

```python
# 不加这行时，LLM 可能犯的错误：
"用户：怎么写一封商务邮件？" → LLM 错误调用了 send_email 工具

# 加上后：
"不适合用于：
- 邮件写作建议（直接回答即可）
- 邮件格式咨询（直接回答即可）
- 仅在用户明确要求'发送''寄出'等动作时才调用"
```

**原理**：明确告诉 LLM "什么时候不该用"，能大幅降低误调用率。这在工具数量增多时尤为关键。

### 4️⃣ 参数描述

不只是写类型，还要写**格式要求和示例值**：

```python
"symbol": {
    "type": "string",
    "description": """
    股票代码，格式要求：
    - 美股：AAPL, TSLA, GOOGL（纯大写字母）
    - A股：600036.SS（上海）, 000001.SZ（深圳）
    - 港股：0700.HK
    
    示例："AAPL", "000001.SZ", "0700.HK"
    """  # ← 这种描述远比单行 "股票代码" 有效得多
}
```

### 5️⃣ 返回值说明

告诉 LLM 工具返回什么格式的数据，方便它解读结果并组织回答：

```python
"description": """...
返回格式：
{
    "symbol": "AAPL",
    "price": 150.5,
    "change_percent": "+1.2%",
    "currency": "USD",
    "last_updated": "2024-01-15 15:30:00"
}
查询失败时返回：{"error": "错误原因"}"""
```

### 6️⃣ 限制与注意事项

```python
"description": """执行 SQL 数据库查询。

⚠️ 重要限制：
- 只允许 SELECT 查询，不支持写入操作
- 单次最多返回 1000 条记录
- 查询超时 30 秒

可用表：users, orders, products, inventory
如需了解表结构，请先使用 get_table_schema 工具"""
```

---

## 多工具场景：如何区分相似工具

当 Agent 有多个功能相近的工具时，描述中要**强调差异化特征**：

```python
search_tools = [
    {
        "name": "search_web",
        "description": """通用网络搜索。
        返回搜索结果摘要（标题+链接+简介），不包含完整页面内容。
        用于发现信息来源。"""
    },
    {
        "name": "browse_url",
        "description": """访问指定 URL 并返回完整页面内容。
        当你已经知道具体 URL 需要详细阅读时使用。
        ⚠️ 不用于搜索/发现新内容。"""
    },
    {
        "name": "search_academic",
        "description": """学术论文搜索（ArXiv / Google Scholar）。
        返回论文标题、作者、摘要、引用数。
        仅用于查找学术研究和论文，不适用于新闻或产品信息。"""
    }
]
```

注意每个描述中都包含了 **"不用于什么"** 来避免混淆。

---

## 描述模板：直接可用

下面是一个通用模板，覆盖了所有要素：

```python
def make_tool_schema(name, one_liner, when_to_use, when_not_to_use,
                     params, return_format, warnings=None):
    parts = [one_liner, ""]
    if when_to_use:
        parts += ["适合用于："] + [f"- {x}" for x in when_to_use] + [""]
    if when_not_to_use:
        parts += ["不适合用于："] + [f"- {x}" for x in when_not_to_use] + [""]
    parts += [f"返回：{return_format}", ""]
    if warnings:
        parts += ["⚠️ 注意："] + [f"- {x}" for x in warnings]

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "\n".join(parts),
            "parameters": {
                "type": "object",
                "properties": params,
                "required": [k for k, v in params.items() if not v.get("default")]
            }
        }
    }
```

---

## 如何测试描述质量？

不要猜——用数据说话：

```python
def test_tool_selection(tool_schemas, test_cases):
    """给 LLM 各种输入，看它是否正确选择/跳过了对应工具"""
    for case in test_cases:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # 用便宜的模型测试就够了
            messages=[{"role": "user", "content": case["input"]}],
            tools=[tool_schemas],
            tool_choice="auto"
        )
        msg = response.choices[0].message
        actual = msg.tool_calls[0].function.name if msg.tool_calls else "(无调用)"
        expected = case["expect_call"]
        print(f"{'✅' if actual == expected else '❌'} "
              f"输入: {case['input'][:40]} | 期望: {expected} | 实际: {actual}")
```

测试用例应该覆盖三种情况：
1. **正确调用**：明显需要该工具的输入
2. **正确跳过**：不需要该工具的输入
3. **边界模糊**：容易误判的边缘案例

---

## 小结

| 要素 | 作用 | 常见错误 |
|------|------|---------|
| 一句话功能 | LLM 的第一判断依据 | 写成技术术语而非自然语言 |
| 适用场景 | 引导正向使用 | 缺失或太笼统 |
| **不适用场景** | **防止误调用（效果最显著）** | 大多数人忘记写 |
| 参数描述 | 减少参数格式错误 | 只有类型没有示例 |
| 返回值格式 | 帮助 LLM 解读结果 | 完全缺失 |
| 注意事项 | 防止危险操作 | 安全相关工具常遗漏 |

> 💡 **终极建议**：每写完一个工具描述，把自己代入 LLM 的角色读一遍——如果你只能看到这些文字，能不能判断什么时候该用这个工具？如果不能，继续优化。

---

*下一节：[4.5 实战：搜索引擎 + 计算器 Agent](./05_practice_search_calc.md)*
