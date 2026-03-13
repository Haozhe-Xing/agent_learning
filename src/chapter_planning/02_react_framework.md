# ReAct：推理 + 行动框架

ReAct（Reasoning + Acting）是 Agent 开发中最重要的框架之一，来自 2022 年普林斯顿大学和 Google Brain 的论文《ReAct: Synergizing Reasoning and Acting in Language Models》（Yao et al.）。它将 LLM 的推理能力与工具使用结合起来，创造出更可靠、更透明的 Agent 行为。

> 📄 **论文背景**：在 ReAct 之前，LLM 的推理（Chain-of-Thought）和行动（工具调用）是两个独立的研究方向。CoT 让模型"会想"但"不会做"——推理时无法获取外部信息；而工具调用让模型"会做"但"不会想"——盲目执行而不解释理由。ReAct 的核心洞察是：**推理为行动提供方向，行动为推理提供依据，两者交替进行才能解决复杂问题。**
>
> 论文在多个任务上的实验证明了这一点：在 HotpotQA（多跳问答）上，ReAct 比纯 CoT 提升了约 6 个百分点的准确率，因为 CoT 只能依赖模型已有知识，而 ReAct 可以通过搜索获取最新事实。在 ALFWorld（交互式文字游戏）上，ReAct 比纯行动（Act-only）提升了 34 个百分点，因为显式推理帮助 Agent 避免了盲目尝试。

## ReAct 的核心思想

ReAct 的关键创新是**将思考过程显式化**：Agent 在行动之前先"大声说出"自己的推理，然后再采取行动。

![ReAct 推理+行动循环](../svg/chapter_planning_02_react_loop.svg)

**与传统 Agent 的对比：**

| 特点 | 传统 Agent（黑盒） | ReAct Agent（透明） |
|------|:---:|:---:|
| 思考过程 | 隐藏 | 显式写入上下文 |
| 可解释性 | 差 | 好 |
| 自我纠错 | 难 | 容易 |
| 幻觉风险 | 高 | 低（工具验证） |

## 从零实现 ReAct Agent

```python
import json
import re
from openai import OpenAI
from typing import Callable

client = OpenAI()

class ReActAgent:
    """
    ReAct Agent 实现
    显式推理 + 工具调用交织执行
    """
    
    def __init__(self, tools: dict[str, Callable], tool_descriptions: str):
        """
        Args:
            tools: 工具名 → 工具函数 的映射
            tool_descriptions: 工具描述文本
        """
        self.tools = tools
        self.tool_descriptions = tool_descriptions
        self.scratchpad = []  # 记录推理过程
    
    def _build_system_prompt(self) -> str:
        return f"""你是一个能够使用工具解决问题的 AI 助手。

可用工具：
{self.tool_descriptions}

解题格式（严格遵守）：
思考：[分析当前情况，决定下一步]
行动：工具名[参数]
观察：[工具返回的结果，系统填充]
...（重复直到问题解决）
最终答案：[综合所有信息给出答案]

注意事项：
- 每次只选择一个工具
- 行动格式必须是：工具名[参数]
- 等待观察结果后再继续
- 如果不需要工具，直接写"最终答案："
"""
    
    def _parse_action(self, text: str) -> tuple[str, str] | None:
        """从文本中解析 Action"""
        # 匹配 "行动：工具名[参数]" 格式
        pattern = r'行动：(\w+)\[([^\]]*)\]'
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)
        return None
    
    def _is_final_answer(self, text: str) -> bool:
        """检查是否有最终答案"""
        return "最终答案：" in text
    
    def _extract_final_answer(self, text: str) -> str:
        """提取最终答案"""
        idx = text.find("最终答案：")
        if idx != -1:
            return text[idx + len("最终答案："):].strip()
        return text
    
    def run(self, question: str, max_steps: int = 8) -> str:
        """
        运行 ReAct 循环
        
        Args:
            question: 用户问题
            max_steps: 最大步骤数
        
        Returns:
            最终答案
        """
        self.scratchpad = []
        
        print(f"\n🔍 问题：{question}")
        print("=" * 60)
        
        # 构建初始消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"请回答以下问题：{question}"}
        ]
        
        for step in range(max_steps):
            # 调用 LLM
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stop=["观察："],  # 在"观察："前停止，等待工具执行
                max_tokens=500
            )
            
            output = response.choices[0].message.content
            print(f"\n{output}")
            
            self.scratchpad.append(output)
            
            # 检查是否有最终答案
            if self._is_final_answer(output):
                answer = self._extract_final_answer(output)
                print(f"\n✅ 最终答案：{answer}")
                return answer
            
            # 解析行动
            action = self._parse_action(output)
            if not action:
                # 没有找到行动，可能直接给出答案了
                if output.strip():
                    return output
                break
            
            tool_name, tool_input = action
            
            # 执行工具
            tool_func = self.tools.get(tool_name)
            if tool_func:
                try:
                    observation = tool_func(tool_input)
                except Exception as e:
                    observation = f"工具执行错误：{str(e)}"
            else:
                observation = f"未知工具：{tool_name}"
            
            # 打印观察结果
            obs_text = f"观察：{observation}"
            print(obs_text)
            self.scratchpad.append(obs_text)
            
            # 将完整的交互添加到消息历史
            messages.append({
                "role": "assistant",
                "content": output + "\n" + obs_text
            })
        
        return "已达到最大步骤数，无法给出确定答案"


# ============================
# 工具定义
# ============================

import math
import requests

def search(query: str) -> str:
    """模拟搜索（实际项目用真实搜索API）"""
    # 预设一些知识库
    knowledge = {
        "python创始人": "Python 由 Guido van Rossum 创建，1991年首次发布",
        "光速": "光在真空中的速度约为 299,792,458 米/秒（约30万公里/秒）",
        "地球周长": "地球赤道周长约为 40,075 公里",
        "水的沸点": "在标准大气压下，水的沸点是100摄氏度（212华氏度）",
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    
    return f"搜索'{query}'：找到相关信息，建议进行计算验证"

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        # ⚠️ 安全警告：eval() 存在安全风险，生产环境请使用 simpleeval 等安全替代方案
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

def get_current_date(_: str = "") -> str:
    """获取当前日期"""
    import datetime
    return datetime.datetime.now().strftime("%Y年%m月%d日")

# 工具描述（关键！）
tool_descriptions = """
- search[查询词]：搜索互联网获取信息
  示例：search[Python 创始人是谁]
  
- calculate[表达式]：计算数学表达式
  支持：+, -, *, /, **, sqrt, sin, cos, log, pi
  示例：calculate[sqrt(144) + pi * 2]
  
- get_current_date[]：获取今天的日期
  示例：get_current_date[]
"""

# ============================
# 测试
# ============================

agent = ReActAgent(
    tools={
        "search": search,
        "calculate": calculate,
        "get_current_date": get_current_date
    },
    tool_descriptions=tool_descriptions
)

# 测试1：需要搜索 + 计算
result = agent.run("光绕地球赤道一圈需要多少毫秒？")

# 测试2：需要多步推理
result = agent.run("Python 是哪年发布的？距离今年有多少年了？")
```

## ReAct 的优势与局限

**优势：**
- 推理过程透明可解释
- 支持自我纠错（观察到错误结果可以重新思考）
- 减少幻觉（每步都有工具验证）

**局限：**
- Token 消耗较大（每步都要写思考过程）
- 在简单任务上显得繁琐
- 可能陷入循环（需要 max_steps 保护）

---

## 小结

ReAct 框架的三大要素：
- **显式推理**：思考过程写在上下文中
- **行动-观察循环**：工具调用与推理交替进行
- **自然终止**：最终答案时结束循环

> 📖 **想深入了解 ReAct 及规划推理的学术前沿？** 请阅读 [6.6 论文解读：规划与推理前沿研究](./06_paper_readings.md)，涵盖 ReAct、MRKL、Plan-and-Solve、Reflexion、CRITIC 等核心论文的深度解读。
>
> 💡 **实践建议**：ReAct 是目前大多数 Agent 框架（LangChain、LlamaIndex）的默认架构。但它并非万能——对于不需要工具的纯推理任务，CoT 更高效；对于流程固定的工作流，直接编排（LangGraph）更可控。选择合适的架构，而非一味使用 ReAct。

---

*下一节：[6.3 任务分解：将复杂问题拆解为子任务](./03_task_decomposition.md)*
