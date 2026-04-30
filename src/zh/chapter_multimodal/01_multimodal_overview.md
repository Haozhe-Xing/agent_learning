# 22.1 多模态能力概述

> **本节目标**：了解多模态大模型的能力边界和典型应用场景。

---

## 什么是多模态？

![多模态 Agent 处理流程](../svg/chapter_multimodal_01_pipeline.svg)

"多模态"是指同时处理和理解多种信息类型（模态）的能力：

| 模态 | 输入 | 输出 |
|------|------|------|
| 文本 | 自然语言问题 | 文字回答 |
| 图像 | 图片、截图 | 图片描述、生成的图像 |
| 语音 | 语音指令 | 语音回复 |
| 视频 | 视频片段 | 视频描述、关键帧分析 |

GPT-4.1 是典型的多模态模型——它能同时理解文本和图像输入。

---

## 多模态 Agent 的应用场景

```python
MULTIMODAL_USE_CASES = {
    "图像分析助手": {
        "输入": "用户上传一张照片",
        "Agent 做": "识别内容、提取文字、分析场景",
        "示例": "拍照菜单自动翻译、拍照数学题自动解答"
    },
    "语音交互助手": {
        "输入": "用户说话",
        "Agent 做": "语音转文字 → 理解意图 → 执行 → 语音回复",
        "示例": "智能音箱、车载助手"
    },
    "文档处理助手": {
        "输入": "包含图文的 PDF/PPT",
        "Agent 做": "理解图表和文字，生成摘要和分析",
        "示例": "自动分析财报、总结研究论文"
    },
    "创意设计助手": {
        "输入": "文字描述 + 参考图片",
        "Agent 做": "生成符合需求的设计图",
        "示例": "Logo 设计、UI 设计稿生成"
    }
}
```

---

## 支持多模态的主流模型

| 模型 | 文本理解 | 图像理解 | 图像生成 | 语音 |
|------|---------|---------|---------|------|
| GPT-4.1 | ✅ | ✅ | ✅（DALL-E） | ✅ |
| Claude 4 | ✅ | ✅ | ❌ | ❌ |
| Gemini 2.5 Pro | ✅ | ✅ | ✅ | ✅ |
| 通义千问 | ✅ | ✅ | ✅ | ✅ |

> ⏰ *注：上表基于 2026 年 3 月各模型公开的能力信息，模型能力更新频繁，请以官方文档为准。*

---

## 在 Python 中使用多模态

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_image(image_path: str, question: str) -> str:
    """用 GPT-4o 分析图像"""
    
    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content


# 使用示例
result = analyze_image(
    "screenshot.png",
    "这张截图中有什么内容？请详细描述。"
)
print(result)
```

---

## 跨模态融合架构

![多模态 Agent 设计模式](../svg/chapter_multimodal_02_modality_fusion.svg)

多模态 Agent 的核心挑战是如何将不同模态的信息融合成统一的理解。主流的融合方式有三种：

```python
# 三种跨模态融合架构

FUSION_ARCHITECTURES = {
    "early_fusion（早期融合）": {
        "原理": "将所有模态的原始数据拼接后一起送入模型",
        "优点": "模型能学习到模态之间的低层特征关联",
        "缺点": "计算量大，对模型架构有要求",
        "代表": "GPT-4o（原生多模态输入）"
    },
    "late_fusion（晚期融合）": {
        "原理": "各模态独立处理后，在决策层合并结果",
        "优点": "模块化、灵活，可以独立优化各模态",
        "缺点": "可能丢失模态之间的交互信息",
        "代表": "传统 Pipeline（OCR + NLP 分开处理）"
    },
    "hybrid_fusion（混合融合）": {
        "原理": "在多个层级同时进行跨模态交互",
        "优点": "兼顾低层特征和高层语义",
        "缺点": "架构复杂，训练成本高",
        "代表": "Gemini（多层级跨模态注意力）"
    }
}
```

对于 Agent 开发者来说，**你通常不需要自己实现融合架构**——选择合适的多模态模型 API 即可。但理解这些架构有助于你：
- 判断哪些任务适合当前模型的能力
- 在模型理解出错时，分析可能的原因
- 设计更好的多模态 Prompt

---

## 多模态 Agent 的设计模式

实际开发中，多模态 Agent 通常采用以下设计模式：

### 模式一：模态路由器

根据用户输入的模态类型，路由到不同的处理管道：

```python
from openai import OpenAI

client = OpenAI()

class ModalityRouter:
    """根据输入模态路由到合适的处理流程"""
    
    def __init__(self):
        self.handlers = {
            "text": self._handle_text,
            "image": self._handle_image,
            "audio": self._handle_audio,
        }
    
    async def route(self, input_data: dict) -> str:
        modality = input_data.get("type", "text")
        handler = self.handlers.get(modality, self._handle_text)
        return await handler(input_data)
    
    async def _handle_text(self, data: dict) -> str:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # 纯文本用便宜模型
            messages=[{"role": "user", "content": data["content"]}]
        )
        return response.choices[0].message.content
    
    async def _handle_image(self, data: dict) -> str:
        response = client.chat.completions.create(
            model="gpt-4.1",  # 图像理解需要多模态模型
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": data.get("question", "描述这张图片")},
                    {"type": "image_url", "image_url": {"url": data["image_url"]}}
                ]
            }]
        )
        return response.choices[0].message.content
    
    async def _handle_audio(self, data: dict) -> str:
        # 先转录，再理解
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(data["audio_path"], "rb")
        )
        # 转录后作为文本处理
        return await self._handle_text({"content": transcript.text})
```

### 模式二：多模态增强链

在文本 Agent 的基础上，通过工具增加多模态能力：

```python
from langchain_core.tools import tool

@tool
def analyze_image_tool(image_url: str, question: str = "描述图片内容") -> str:
    """分析图片内容。传入图片URL和问题，返回分析结果。"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content

@tool
def generate_image_tool(prompt: str) -> str:
    """根据文字描述生成图片。返回生成的图片URL。"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return f"图片已生成：{response.data[0].url}"

@tool
def transcribe_audio_tool(audio_path: str) -> str:
    """将音频文件转为文字。支持 mp3、wav、m4a 等格式。"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    return transcript.text

# 将这些工具注册到文本 Agent 中，Agent 就获得了多模态能力
multimodal_tools = [analyze_image_tool, generate_image_tool, transcribe_audio_tool]
```

> 💡 **设计建议**：模式二（工具增强）是**最推荐**的入门方式——你可以在已有的文本 Agent 基础上，逐步添加多模态工具，而不用重写整个架构。

---

## 多模态 Agent vs 纯文本 Agent

多模态 Agent 与纯文本 Agent 的关键区别：

```python
# 纯文本 Agent：只能处理文字
text_agent_response = agent.run("分析这个数据")
# 需要用户手动把数据粘贴成文本

# 多模态 Agent：可以直接处理图片、文件
multimodal_response = agent.run(
    "分析这张财务报表截图中的数据",
    images=["financial_report.png"]
)
# Agent 自动识别表格内容 → 提取数据 → 进行分析
```

**多模态开发的三大挑战**：

| 挑战 | 说明 | 应对策略 |
|------|------|---------|
| 模态理解偏差 | LLM 可能误读图像内容 | 多轮确认 + 结构化提取 |
| Token 消耗高 | 图像编码占用大量 Token | 图像压缩 + 按需传输 |
| 延迟增加 | 多模态推理比纯文本慢 | 异步处理 + 流式输出 |

---

## 图像 Token 成本优化

多模态 API 中，图像消耗的 Token 数与图像尺寸直接相关。合理控制图像大小可以**大幅降低成本**：

```python
from PIL import Image
import io
import base64

def optimize_image_for_api(
    image_path: str,
    max_size: int = 1024,
    quality: int = 85
) -> str:
    """优化图像尺寸和质量，降低 API Token 消耗
    
    GPT-4o 的图像 Token 计算规则（2026-03）：
    - low detail: 固定 85 tokens
    - high detail: 基于 512x512 分块，每块 170 tokens + 85 base
    
    因此，将图像缩小到 1024px 以内可以显著降低成本。
    """
    img = Image.open(image_path)
    
    # 等比缩放
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # 转为 JPEG 并压缩
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_with_detail_control(
    image_path: str,
    question: str,
    detail: str = "auto"
) -> str:
    """控制图像分析精度（low/high/auto）
    
    low: 快速概览，适合简单分类、是否包含某物
    high: 精细分析，适合文字识别、细节提取
    auto: 让模型自己判断
    """
    image_b64 = optimize_image_for_api(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": detail  # 控制精度
                    }
                }
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 多模态 | 同时处理文本、图像、语音等多种信息 |
| 主流模型 | GPT-4.1、Claude 4、Gemini 2.5 Pro |
| 典型场景 | 图像分析、语音交互、文档处理、创意设计 |
| 核心流程 | 输入编码 → 跨模态融合 → 推理 → 多模态输出 |
| 关键挑战 | 模态理解偏差、Token 消耗、延迟控制 |

> **下一节预告**：我们将深入学习图像理解和生成能力。

---

[22.2 图像理解与生成](./02_image_understanding.md)
