# 4.2 Function Calling 机制详解

Function Calling 是 OpenAI 于 2023 年 6 月推出的重要特性，让模型能够输出结构化的函数调用指令。2024 年 8 月，OpenAI 进一步推出了 **Structured Outputs**（结构化输出），通过 `strict: true` 参数确保模型生成的参数 100% 符合 JSON Schema，大幅提升了生产环境的可靠性。理解其完整机制，是构建可靠 Agent 的基础。

在传统的 LLM 使用中，模型只能输出纯文本。但现实世界的许多任务——查天气、发邮件、查数据库——都需要与外部系统交互。Function Calling 解决的正是这个问题：**它让 LLM 不再只是"说"，还能"做"。**

核心思想其实很朴素：我们告诉模型"你有哪些工具可以用"，模型分析用户的问题后，如果觉得需要使用工具，就输出一段结构化的 JSON，告诉我们"请帮我调用这个函数，参数是这些"。模型本身并不执行函数，执行由我们的代码完成，然后把结果再反馈给模型，让它组织最终的回答。

## Function Calling 的完整流程

![Function Calling 流程](../svg/chapter_tools_02_function_calling.svg)

> 🎬 **交互式动画**：观看用户、LLM 和工具引擎之间的消息传递全过程——包含多轮工具调用的完整 5 步通信协议，理解 Function Calling 如何让 LLM 从"只会说"变成"会做事"。
>
> <a href="../animations/function_calling.html" target="_blank" style="display:inline-block;padding:8px 16px;background:#9C27B0;color:white;border-radius:6px;text-decoration:none;font-weight:bold;">▶ 打开 Function Calling 交互动画</a>

整个流程分为 5 个步骤：

1. **定义工具**：用 JSON Schema 描述函数的名称、参数和用途
2. **发送请求**：将用户消息和工具列表一起发给 LLM
3. **模型决策**：LLM 分析后决定是直接回答还是调用工具
4. **执行工具**：我们的代码执行模型指定的函数，获取结果
5. **生成回答**：将工具结果返回给 LLM，由它生成最终回复

这个过程可能会循环多次——比如用户问"北京天气怎么样？如果下雨就帮我发邮件提醒"，模型会先调用天气工具，根据结果再决定是否调用邮件工具。

![Function Calling 完整流程](../svg/chapter_tools_02_fc_flow.svg)

> 上图展示了多轮工具调用的完整消息流：每一轮循环中，工具结果以 `role: "tool"` 的消息追加到历史，LLM 基于完整上下文决定是继续调用工具还是给出最终回答。

## 完整代码实现

下面是一个完整的 Function Calling 示例。我们定义两个工具（查天气和发邮件），然后实现一个执行循环，让 Agent 能够自动调用这些工具。

```python
import json
from openai import OpenAI
import requests

client = OpenAI()

# ========== 第一步：定义工具函数 ==========

def get_weather(city: str, unit: str = "celsius") -> dict:
    """获取城市天气（模拟实现）"""
    # 实际项目中调用真实天气 API
    weather_data = {
        "北京": {"temp": 15, "condition": "晴", "humidity": 40},
        "上海": {"temp": 18, "condition": "多云", "humidity": 65},
        "广州": {"temp": 25, "condition": "小雨", "humidity": 80},
    }
    
    data = weather_data.get(city, {"temp": 20, "condition": "未知", "humidity": 50})
    
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
        data["unit"] = "°F"
    else:
        data["unit"] = "°C"
    
    return {
        "city": city,
        "temperature": f"{data['temp']}{data['unit']}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%"
    }

def send_email(to: str, subject: str, body: str) -> dict:
    """发送邮件（模拟实现）"""
    # 实际项目中调用邮件 API
    print(f"[模拟] 发送邮件至 {to}")
    print(f"主题：{subject}")
    print(f"内容：{body[:50]}...")
    
    return {
        "status": "success",
        "message_id": "MSG-12345",
        "recipient": to
    }

# ========== 第二步：定义 OpenAI 工具格式 ==========
# 注意：description 字段至关重要——模型正是通过它来判断"何时该用这个工具"。
# 描述越精确（包括适用场景、参数含义、返回内容），模型的调用决策就越准确。
# strict: True 启用结构化输出，确保模型生成的参数 100% 符合 JSON Schema（推荐生产环境开启）

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前实时天气信息，包括温度、天气状况和湿度。仅用于查询天气，不用于预报未来天气。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，celsius=摄氏度，fahrenheit=华氏度"
                    }
                },
                "required": ["city", "unit"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "向指定邮箱发送邮件。仅在用户明确要求发送邮件时调用，不要在未经用户确认的情况下自动发送。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱地址"
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题"
                    },
                    "body": {
                        "type": "string",
                        "description": "邮件正文内容"
                    }
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False
            }
        }
    }
]

# 工具函数映射表
tool_functions = {
    "get_weather": get_weather,
    "send_email": send_email,
}

# ========== 第三步：实现 Agent 执行循环 ==========

def run_agent(user_message: str) -> str:
    """运行 Agent，处理工具调用循环"""
    
    messages = [{"role": "user", "content": user_message}]
    
    print(f"\n用户：{user_message}")
    
    while True:
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # 添加助手消息到历史
        messages.append(message)
        
        # 情况1：模型直接回答，不需要工具
        if finish_reason == "stop":
            print(f"\nAssistant：{message.content}")
            return message.content
        
        # 情况2：模型要求调用工具
        if finish_reason == "tool_calls":
            # 处理所有工具调用（可能同时调用多个）
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"\n[工具调用] {func_name}({func_args})")
                
                # 执行工具（加入错误处理：工具失败时将错误信息反馈给模型，让它自行决策）
                func = tool_functions.get(func_name)
                if func:
                    try:
                        result = func(**func_args)
                        result_str = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e), "status": "failed"})
                else:
                    result_str = json.dumps({"error": f"未知工具 {func_name}", "status": "failed"})
                
                print(f"[工具结果] {result_str}")
                
                # 将工具结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })
            
            # 继续循环，让 LLM 处理工具结果
            continue
        
        # 其他情况（超时等）
        break
    
    return "处理失败"

# ========== 测试 ==========

# 测试1：查询天气
run_agent("北京和上海今天天气怎么样？")

# 测试2：复合任务
run_agent("查询北京天气，如果气温低于10度，发邮件给 boss@company.com 提醒带伞")
```

让我们逐步理解上面代码的关键设计：

**工具定义的 JSON Schema**：注意 `tools` 列表中每个工具的格式——`name` 是函数名，`description` 告诉模型"什么时候该用这个工具"，`parameters` 描述参数类型和含义。这个 Schema 会随请求一起发送给 LLM，模型正是通过它来理解工具的能力边界。

**工具描述的质量至关重要**：`description` 字段是模型决策的核心依据。一个好的工具描述应该包含：工具的适用场景（什么时候用）、不适用场景（什么时候不用）、参数的具体含义和格式。描述模糊会导致模型误调用或漏调用——比如把 `send_email` 描述为"处理邮件相关任务"，模型可能在用户只是问"怎么写邮件"时也去调用它。

**`strict: true` 模式**：在工具定义中加入 `"strict": True`，可以启用结构化输出，确保模型生成的参数严格符合 JSON Schema。开启后需要同时设置 `"additionalProperties": False`，并将所有参数都放入 `required` 列表。这在生产环境中强烈推荐，可以避免参数缺失或类型错误导致的运行时崩溃。

**错误处理**：工具执行应该包裹在 `try/except` 中。当工具调用失败时，不要直接抛出异常中断流程，而是将错误信息以 JSON 格式返回给模型——模型会根据错误信息自行决策，比如换个参数重试、或者告知用户操作失败。

**执行循环（Agent Loop）**：`run_agent` 函数的核心是一个 `while True` 循环。每次循环中，我们把完整的消息历史（包括之前的工具调用结果）发给 LLM。模型返回后，我们检查 `finish_reason`：如果是 `"stop"`，说明模型已经给出最终回答；如果是 `"tool_calls"`，说明模型需要使用工具。执行完工具后，我们把结果以 `role: "tool"` 的消息追加到历史中，然后继续下一轮循环。

**工具函数映射表**：`tool_functions` 字典将函数名字符串映射到实际的 Python 函数。这个设计让工具的调用和注册可以解耦——你只需要在字典中添加新的映射，就能让 Agent 获得新的能力。

## tool_choice 参数控制

`tool_choice` 参数让你精确控制模型使用工具的策略。这在不同场景下非常有用：

- **auto**（默认）：模型自己判断是否需要工具，大多数情况下这就够了
- **none**：强制模型不使用任何工具，适合你只想要纯文本回答的场景
- **required**：强制模型必须使用工具，适合你确定这个请求一定需要工具的场景
- **指定工具**：强制使用某一个特定工具，适合测试或某些确定性很高的场景

```python
# tool_choice 控制模型使用工具的策略

# 1. "auto"（默认）：LLM 自己决定
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 2. "none"：禁止使用工具，直接回答
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    tool_choice="none"
)

# 3. "required"：强制必须使用工具
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    tool_choice="required"
)

# 4. 指定使用某个工具
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

## 并行工具调用

当用户提出一个包含多个独立子任务的请求时——比如"同时查询北京、上海、广州的天气"——让模型逐个调用工具效率很低。GPT-4 支持**并行工具调用**（Parallel Tool Calls），即在一次请求中返回多个工具调用指令，我们可以同时执行它们。

这个特性对性能至关重要。假设你有 3 个独立的 API 调用，串行执行需要 3 倍的等待时间，而并行执行只需要最慢那个的时间。下面的代码使用 `concurrent.futures` 实现了真正的并行执行：

> ⚠️ **注意**：并行调用适用于工具之间**相互独立**的场景。如果工具之间有顺序依赖（比如先查天气、再根据天气结果决定是否发邮件），应设置 `parallel_tool_calls=False`，让模型串行调用，避免后一个工具依赖前一个工具的结果时出现逻辑错误。

```python
def run_parallel_tools(user_message: str) -> str:
    """支持并行工具调用的 Agent"""
    
    messages = [{"role": "user", "content": user_message}]
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True  # 允许并行调用
    )
    
    message = response.choices[0].message
    
    if message.tool_calls:
        print(f"并行调用 {len(message.tool_calls)} 个工具：")
        
        # 并行执行所有工具（可以用 asyncio 实现真并行）
        import concurrent.futures
        
        def execute_tool(tool_call):
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func = tool_functions.get(func_name)
            result = func(**func_args) if func else "未知工具"
            return tool_call.id, json.dumps(result, ensure_ascii=False)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_tool, tc) for tc in message.tool_calls]
            results = [f.result() for f in futures]
        
        messages.append(message)
        
        for tool_call_id, result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result
            })
        
        # 获取最终回复
        final_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return message.content

# 测试并行调用
result = run_parallel_tools("同时查询北京、上海、广州三个城市的天气")
print(result)

# 禁用并行调用（工具之间有顺序依赖时使用）
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False  # 强制串行调用，适合有依赖关系的工具链
)
```

## 使用 Pydantic 自动生成工具 Schema

你可能已经注意到，手写 JSON Schema 是一件繁琐且容易出错的事——参数名拼错、类型写错都可能导致模型调用失败。一个更优雅的做法是利用 Pydantic 模型来自动生成 Schema。

Pydantic 的 `model_json_schema()` 方法可以将 Python 类型注解自动转换为标准的 JSON Schema，这样你只需要定义一个 Pydantic 模型来描述工具的输入参数，Schema 就自动生成了。这不仅减少了出错的可能，还让代码更具可维护性——修改参数时只需改 Pydantic 模型，Schema 会自动更新。

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
import inspect
import json

def create_tool_from_pydantic(
    func,
    input_model: type[BaseModel],
    description: str
) -> dict:
    """从 Pydantic 模型自动生成工具定义"""
    schema = input_model.model_json_schema()
    
    # 移除 Pydantic 特有的字段
    schema.pop("title", None)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": schema
        }
    }

# 示例
class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )

weather_tool = create_tool_from_pydantic(
    get_weather,
    WeatherInput,
    "获取指定城市的天气信息"
)

print(json.dumps(weather_tool, ensure_ascii=False, indent=2))
```

---

## Function Calling 与 MCP 的关系

学完 Function Calling，你可能会听到另一个概念：**MCP（Model Context Protocol）**，这是 Anthropic 于 2024 年底推出、并在 2025 年迅速成为行业标准的工具调用协议。它们的关系可以这样理解：

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 定位 | 单一平台的工具调用机制 | 跨平台标准化协议 |
| 平台依赖 | 绑定特定 LLM 平台（如 OpenAI API） | 平台无关，支持任意模型 |
| 工具复用 | 工具代码与模型强耦合 | 工具以独立服务形式存在，可复用 |
| 适用场景 | 快速集成、单模型项目 | 多模型、工具需要跨项目复用的场景 |
| 学习曲线 | 低，直接调用 API | 稍高，需要理解 Client/Server 架构 |

Function Calling 是理解 Agent 工具调用的**基础**，MCP 是在此之上的**标准化演进**。掌握了 Function Calling 的原理，理解 MCP 会非常自然——本质上 MCP 就是把工具定义和执行从代码中抽离出来，变成一个独立的、可复用的服务。

---

## 小结

Function Calling 的核心要点：

| 要点 | 说明 |
|------|------|
| 工具定义 | JSON Schema 格式，描述函数名、参数、用途 |
| 描述质量 | `description` 是模型决策的核心依据，需明确适用/不适用场景 |
| strict 模式 | `strict: true` + `additionalProperties: false`，确保参数 100% 符合 Schema |
| 执行循环 | LLM → 工具调用请求 → 执行 → 结果反馈 → LLM |
| 错误处理 | 工具失败时将错误 JSON 返回给模型，让模型自行决策重试或报错 |
| tool_choice | 控制工具使用策略（auto/none/required/指定） |
| 并行调用 | 独立工具可并行执行；有顺序依赖时用 `parallel_tool_calls=False` |
| MCP | Function Calling 的标准化演进，实现跨平台工具复用 |

---

*下一节：[4.3 自定义工具的设计与实现](./03_custom_tools.md)*
