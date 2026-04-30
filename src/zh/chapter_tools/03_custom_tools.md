# 3.3 自定义工具的设计与实现

> 🎯 **本节目标**：学会如何设计高质量的工具——不是"怎么写代码"，而是"怎么思考设计"。

---

## 核心观点：工具设计质量 > 模型能力

很多 Agent 表现不佳的原因，不是模型不够聪明，而是**工具没设计好**：

> - 工具描述模糊 → LLM 不知道什么时候该用 → 要么不用，要么乱用
> - 参数没有验证 → LLM 传入非法值 → 工具崩溃
> - 错误信息不清晰 → LLM 无法理解发生了什么 → 不能自我纠错

一个设计精良的工具 + GPT-4.1-mini 的效果，往往优于设计糟糕的工具 + GPT-4.1。**工具就是 Agent 的“感官器官”——眼睛（搜索）、耳朵（监听）、手（操作）。感官越灵敏、信号越清晰，大脑的决策就越准确。**

---

## 设计原则一：单一职责

> 一个工具只做一件事，做好这一件事。

这不是软件工程教条——它对 Agent 有直接的量化影响。

```python
# ❌ 瑞士军刀式工具：模型会困惑"我到底该传什么参数？"
def search_analyze_summarize(
    query: str,
    analyze: bool = True,    # 要分析吗？
    summarize: bool = True,   # 要总结吗？
    format: str = "text",     # 什么格式？
    max_results: int = 10,    # 多少结果？
): ...
# 结果：LLM 经常漏填参数或传错值

# ✅ 职责清晰的独立工具：模型自由组合
def search_web(query: str) -> list: ...     # 只搜索
def analyze_data(data: list) -> dict: ...   # 只分析
def summarize_text(text: str) -> str: ...   # 只摘要
# 结果：LLM 可以根据任务灵活选择和组合
```

**为什么这很重要？** 因为 LLM 选择工具时是在做多分类问题。每增加一个可选参数或功能分支，决策空间就指数级膨胀。3 个单职责工具的组合空间是 $2^3 = 8$ 种用法；而 1 个多功能工具的参数组合可能达到几十种，模型更容易出错。

---

## 设计原则二：描述写给 LLM 看

工具的 `description` 不是给人看的文档——**它是给另一个 AI 读的"使用说明书"**。这意味着写法要完全不同于传统 API 文档。

### 好的 vs. 不好的描述对比

| 维度 | 不好的 | 好的 |
|------|--------|------|
| 功能说明 | "处理邮件" | "向指定邮箱发送邮件，支持 HTML 格式" |
| 使用时机 | （缺失） | "仅在用户明确要求发送时调用" |
| 排除时机 | （缺失） | "不适用于读取邮件或查询收件箱" |
| 参数格式 | `{"email": string}` | `"邮箱地址，如 boss@company.com"` |

### 描述的黄金公式

> **一句话功能说明**（动词 + 对象 + 核心特征）→ **适用场景**（何时用）→ **不适用场景**（何时不用）→ **返回值格式**（让 LLM 知道拿到结果后该怎么解读）

---

## 设计原则三：输入必须验证

LLM 生成的参数不总是合法的。它可能传入：

```python
# 这些都是真实会发生的情况
""                    # 空字符串
"user@example"        # 格式不完整的邮箱
"A" * 10000           # 超长文本导致内存问题
None                  # 缺失必需参数
{"nested": "object"}  # 期望字符串但传了对象
```

### 最小可行的输入验证模式

```python
from pydantic import BaseModel, field_validator

class EmailInput(BaseModel):
    to: str
    subject: str
    body: str

    @field_validator("to")
    @classmethod
    def check_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"无效的邮箱地址：{v}")
        return v

    @field_validator("subject")
    @classmethod
    def check_length(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("主题超过200字限制")
        return v

def send_email_safe(to: str, subject: str, body: str) -> str:
    try:
        # Pydantic 自动完成所有验证
        email_input = EmailInput(to=to, subject=subject, body=body)
        return f"✅ 邮件已发送至 {email_input.to}"
    except Exception as e:
        # 关键：返回人类可读的错误，不要抛异常！
        return f"❌ 发送失败：{e}"
```

**为什么用 Pydantic 而非手写 if-else？**
- 类型声明即验证规则，代码即文档
- 错误信息自动包含字段名和原因
- 与 OpenAI Structured Outputs 天然配合

---

## 设计原则四：错误信息是给 LLM 看的

这是最容易被忽略的原则：

```python
# ❌ 这种错误信息对人类友好，但对 LLM 几乎无用
raise ValueError("Invalid input")

# ✅ 这种错误信息让 LLM 能理解并采取行动
return json.dumps({
    "error": "邮箱地址格式不正确",
    "expected": "如 user@example.com",
    "received": "user@example",  # 告诉 LLM 它传了什么
    "suggestion": "请检查邮箱地址是否包含 @ 和域名后缀"
})
```

好的错误信息能让 LLM 自行纠错：
- 参数格式错 → LLM 修正格式重试
- API 超时 → LLM 换更简单的查询
- 权限不足 → LLM 告知用户需要授权

---

## 设计原则五：考虑缓存

Agent 在一次推理中可能多次调用同一个工具：

> 用户：“北京天气怎么样？适合户外活动吗？湿度大吗？” → LLM 可能在推理过程中分别查询天气 2-3 次 → 如果每次都调付费 API → 浪费錢和时间

解决方案：对相同参数的结果做短期缓存（TTL 通常设为几分钟到几小时）。

> 💡 缓存不需要自己实现。大多数 Agent 框架（LangChain、CrewAI）都有内置的工具缓存机制。了解这个概念就够，具体实现交给框架。

---

## 一个完整的工具设计清单

在把任何新工具接入 Agent 之前，用这张清单自检：

```
□ 单一职责：这个工具是否只做一件事？
□ 名称清晰：函数名能否一眼看出功能？（get_weather ✅ / process ✅）
□ 描述完整：是否包含 适用/不适用/返回格式？
□ 参数有类型注解 + 描述？
□ 输入验证：Pydantic 或等价的校验逻辑？
□ 错误处理：所有异常都被捕获并转为可读字符串？
□ 安全评估：最坏情况下这个工具能造成什么破坏？
□ 必要时加了缓存？
```

---

## 📝 动手练习

**练习**：为以下需求设计一个工具 Schema（只写 JSON 定义，不写函数体）：

> 需求：一个股票查询工具，可以查股票的当前价格、涨跌幅、市值。支持 A 股（带 .SS/.SZ 后缀）和美股（纯字母代码）。

<details>
<summary>参考方案</summary>

```python
{
    "type": "function",
    "function": {
        "name": "get_stock_info",
        "description": """查询股票的实时行情数据。
适合用于：
- 获取股票当前价格和涨跌幅
- 查询公司市值等基本面指标
不适合用于：
- K线图/历史数据（请使用 get_stock_history）
- 技术指标计算（请在获取数据后自行计算）

返回格式：{symbol, price, change_percent, market_cap, currency}""",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": """股票代码格式：
- 美股：AAPL, GOOGL, TSLA（纯大写字母）
- A股：600036.SS（上海）, 000001.SZ（深圳）
- 港股：0700.HK"""
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["price", "change_percent", "market_cap", "pe", "pb"]
                    },
                    "description": "要查询的指标，默认全部"
                }
            },
            "required": ["symbol"]
        }
    }
}
```
关键点：
1. description 中明确区分了适用和不适用场景
2. symbol 的描述包含了三种市场的代码格式示例
3. metrics 用 enum 限制了可选值范围，减少模型幻觉

</details>

---

## 小结

| 原则 | 核心要点 |
|------|---------|
| 单一职责 | 一个工具一件事，降低 LLM 决策复杂度 |
| 描述质量 | 写给 LLM 看，不是给人看——包含适用/不适用场景 |
| 输入验证 | 用 Pydantic 做类型安全 + 格式校验 |
| 错误信息 | 返回给 LLM 的结构化错误，而非抛异常 |
| 缓存策略 | 对重复调用的耗时/收费工具加缓存 |
| 安全自检 | 每个工具上线前评估最坏情况的影响 |

---

*下一节：[3.4 工具描述的编写技巧](./04_tool_description.md)*
