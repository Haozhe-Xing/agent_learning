# ── 工具1：搜索（基于 DuckDuckGo 免费 API，无需 API Key）────────

# Instant Answer 返回摘要与相关主题；它不保证覆盖任意网页查询。

def search_web(query: str, num_results: int = 5) -> str:
    """调用 DuckDuckGo Instant Answer API，返回文本摘要。"""
    if (not isinstance(query, str) or not query.strip() or len(query) > 200
            or type(num_results) is not int or not 1 <= num_results <= 10):
        return "搜索参数错误：query须为1–200字符非空文本，num_results须为1–10的整数。"
    try:
        import requests
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        data = requests.get(url, params=params, timeout=10).json()

        results = []
        if data.get("AbstractText"):
            results.append(f"即时答案：{data['AbstractText']}\n来源：{data.get('AbstractURL', '未提供')}")
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text'][:200]}\n来源：{topic.get('FirstURL', '未提供')}")

        return "\n".join(results) if results \
            else f"未找到结果，建议换更具体的关键词。"
    except Exception as e:
        return f"搜索失败：{e}。请检查网络连接。"


# ── 工具2：显式解析允许的数学表达式 ────────────────────────
# safe_math.py 与配套脚本放在同一目录；实现见 examples/chapter03/
from safe_math import safe_math

def calculate(expression: str) -> str:
    """只允许白名单数学语法，并限制长度、节点数、数值和指数。"""
    try:
        return f"{expression} = {safe_math(expression)}"
    except (ValueError, SyntaxError, TypeError, ArithmeticError) as exc:
        return f"计算错误：{exc}"


# ── 工具3：单位换算 ──────────────────────────────────────────
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """在相同量纲内换算长度、重量、面积或温度。"""
    import math
    if type(value) not in (int, float) or not math.isfinite(value):
        return "数值必须是有限数字"
    groups = {
        "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
                   "inch": 0.0254, "foot": 0.3048, "mile": 1609.344, "yard": 0.9144},
        "mass": {"kg": 1.0, "g": 0.001, "mg": 0.000001,
                 "pound": 0.45359237, "ounce": 0.028349523125, "ton": 1000.0},
        "area": {"m2": 1.0, "km2": 1e6, "cm2": 0.0001,
                 "acre": 4046.8564224, "hectare": 10000.0},
    }
    temperatures = {"celsius", "fahrenheit", "kelvin"}
    if from_unit in temperatures and to_unit in temperatures:
        c = (value - 32) * 5 / 9 if from_unit == "fahrenheit" else value - 273.15 if from_unit == "kelvin" else value
        if c < -273.15:
            return "温度低于绝对零度"
        result = c * 9 / 5 + 32 if to_unit == "fahrenheit" else c + 273.15 if to_unit == "kelvin" else c
        return f"{value} {from_unit} = {result:.6g} {to_unit}"
    for units in groups.values():
        if from_unit in units and to_unit in units:
            result = value * units[from_unit] / units[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
    return f"不支持的单位或量纲不匹配：{from_unit} → {to_unit}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": """在互联网上搜索实时信息。

适合用于：
- 查询最新新闻、事件、价格等实时数据
- 获取人物、地点、事件的背景信息
- 查找技术文档和教程

不适合用于：
- 数学计算（使用 calculate 工具）
- 单位换算（使用 unit_converter 工具）
- 你已经知道答案的问题""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，建议简洁精准。"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认5，最大10",
                        "default": 5
                    }
                },
                "required": ["query"],
                "additionalProperties": False   # 关键：禁止额外字段
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": """精确计算数学表达式。

支持：基本运算（+,-,*,/,**）、数学函数（sqrt/sin/cos/log等）、常量（pi/e）

不适合：
- 需要查询的事实（如"光速是多少"）→ 用 search_web
- 单位换算 → 用 unit_converter
- 大段文本处理""",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，使用 Python 语法。乘方用 **，不用 ^"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": """单位换算，支持长度、重量、温度、面积等常见单位转换。

适合：m/km/mile、kg/pound、°C/°F 等
不适合：货币汇率、时区转换""",
            "parameters": {
                "type": "object",
                "properties": {
                    "value":    {"type": "number", "description": "要换算的数值"},
                    "from_unit": {"type": "string", "description": "原始单位，如 m, km, celsius"},
                    "to_unit":   {"type": "string", "description": "目标单位，如 mile, pound, fahrenheit"}
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False
            }
        }
    }
]

import json

TOOL_FUNCTIONS = {"search_web": search_web, "calculate": calculate,
                  "unit_converter": unit_converter}

class SearchCalcAgent:
    """搜索 + 计算 Agent"""

    SYSTEM_PROMPT = """你是一个能够搜索和计算的智能助手。
你有三个工具：search_web（搜索）、calculate（计算）、unit_converter（换算）。

使用策略：
- 需要实时数据 → 先 search_web
- 需要精确计算 → 用 calculate，不要手动算
- 复杂问题可组合多个工具
- 给出答案时说明信息来源
- 简洁准确，重点突出"""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(timeout=30.0, max_retries=1)
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        MAX_STEPS = 8   # 安全护栏：防无限循环

        for step in range(MAX_STEPS):
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=True
            )
            message = response.choices[0].message
            self.messages.append(message)

            # 模型直接给出最终答案 → 结束
            if response.choices[0].finish_reason == "stop":
                return message.content

            # 模型请求工具调用 → 执行并把结果回传
            if response.choices[0].finish_reason == "tool_calls":
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                        result = TOOL_FUNCTIONS[func_name](**func_args)
                    except (ValueError, TypeError, KeyError) as exc:
                        result = f"工具调用失败：{exc}"

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

        return "已达到最大步骤数限制，请简化问题重试。"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search and calculator teaching Agent")
    parser.add_argument("--demo", action="store_true", help="offline deterministic tool demo")
    args = parser.parse_args()
    if args.demo:
        print(calculate("384400 * 1000 / 299792458"))
        print(calculate("sqrt(144)"))
        print(unit_converter(1, "km", "m"))
        return
    agent = SearchCalcAgent()
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "exit":
            break
        if query == "reset":
            agent.messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
            continue
        try:
            print(agent.chat(query))
        except Exception as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
