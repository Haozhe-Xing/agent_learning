# 图像理解与生成

> **本节目标**：掌握用 GPT-4o 分析图像和用 DALL-E 生成图像的技术，深入理解视觉模型的能力边界与 Prompt 技巧。

![视觉模型能力边界与图像生成技术](../svg/chapter_multimodal_02_image.svg)

---

## 视觉模型的能力与边界

在动手编码之前，我们需要客观了解当前视觉语言模型（VLM）能做什么、不能做什么。理解能力边界有助于在 Agent 设计中设置合理预期，避免构建不切实际的功能。

### 能力评估维度

| 能力维度 | 表现水平 | 典型应用 | 注意事项 |
|---------|---------|---------|---------|
| **场景描述** | ⭐⭐⭐⭐⭐ | 图片内容概述、场景分类 | 最成熟的能力 |
| **文字识别（OCR）** | ⭐⭐⭐⭐ | 发票提取、截图转文字 | 手写体和模糊文字准确率下降 |
| **图表分析** | ⭐⭐⭐⭐ | 读取柱状图/折线图数据 | 密集数据点容易遗漏或误读 |
| **空间推理** | ⭐⭐⭐ | "左边的物体比右边大吗？" | 精确距离/角度估计不可靠 |
| **细粒度计数** | ⭐⭐ | "图中有几个人？" | 数量超过 10 时容易出错 |
| **专业领域识别** | ⭐⭐⭐ | 医学影像、电路图 | 准确率不足以替代专业人员 |
| **多图推理** | ⭐⭐⭐⭐ | 前后对比、找不同 | 超过 4 张图时注意力分散 |

> ⚠️ **关键认知**：视觉模型是"看得见但不一定看得准"的助手。在构建 Agent 时，对于精确计数、空间测量、专业诊断等场景，应将模型输出视为"参考意见"而非"确定事实"，必要时引入人类确认环节（参考第 12 章 Human-in-the-Loop 模式）。

### 多轮视觉对话

GPT-4o 支持在对话历史中保留图像上下文，实现多轮视觉对话。这在实际 Agent 中非常有用——用户可以先上传一张图，然后通过多轮追问逐步深入分析：

```python
class MultiTurnVisionChat:
    """多轮视觉对话管理器"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.messages: list[dict] = []
    
    def send_image(self, image_path: str, question: str) -> str:
        """发送图片并提问（首轮）"""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/png;base64,{data}"}}
            ]
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000
        )
        
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def follow_up(self, question: str) -> str:
        """追问（后续轮次，无需重新发送图片）"""
        self.messages.append({"role": "user", "content": question})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000
        )
        
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

# 使用示例：多轮分析一张架构图
chat = MultiTurnVisionChat()
print(chat.send_image("system_arch.png", "请描述这张系统架构图的整体结构"))
print(chat.follow_up("图中的数据流向是怎样的？"))
print(chat.follow_up("你觉得这个架构有什么潜在的瓶颈？"))
```

---

## 图像分析的 Prompt Engineering

与纯文本 Prompt 不同，视觉任务的 Prompt 需要特别注意**引导模型关注正确的区域和维度**：

### 通用原则

1. **明确任务类型**：告诉模型你需要描述、提取、比较还是判断
2. **指定输出格式**：要求 JSON、表格或结构化文本，避免模型自由发挥
3. **提供领域上下文**：如"这是一张电商产品图"比"描述这张图"效果好得多
4. **分步引导**：对于复杂分析，让模型先整体概述再细节拆解

### 常用 Prompt 模板

```python
# 结构化信息提取
EXTRACT_PROMPT = """请从这张图片中提取以下信息，以 JSON 格式返回：
{
  "document_type": "文档类型",
  "key_fields": {"字段名": "字段值", ...},
  "confidence": "high/medium/low"
}
如果某个字段无法识别，填写 null。"""

# 对比分析
COMPARE_PROMPT = """请对比这两张图片：
1. 列出所有不同之处（至少检查：颜色、文字、布局、元素数量）
2. 对每个差异评估其显著程度（高/中/低）
3. 以表格形式输出结果"""

# 图表数据提取
CHART_PROMPT = """这是一张数据图表。请：
1. 识别图表类型（柱状图/折线图/饼图/散点图等）
2. 提取所有可读的数据点，以表格形式呈现
3. 总结图表展示的主要趋势或结论
注意：如果某些数值不清晰，请标注"约"并给出估算值。"""
```

---

## 图像理解

### 封装图像分析工具

```python
from openai import OpenAI
import base64
import httpx

class VisionTool:
    """图像分析工具"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def analyze_local_image(
        self,
        image_path: str,
        prompt: str = "请描述这张图片的内容"
    ) -> str:
        """分析本地图片"""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # 自动检测图片格式
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp"
        }.get(ext, "image/png")
        
        return self._call_vision(
            prompt,
            f"data:{mime_type};base64,{image_data}"
        )
    
    def analyze_url_image(
        self,
        image_url: str,
        prompt: str = "请描述这张图片的内容"
    ) -> str:
        """分析网络图片"""
        return self._call_vision(prompt, image_url)
    
    def compare_images(
        self,
        image_paths: list[str],
        prompt: str = "请比较这些图片的异同"
    ) -> str:
        """比较多张图片"""
        content = [{"type": "text", "text": prompt}]
        
        for path in image_paths:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"}
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _call_vision(self, prompt: str, image_url: str) -> str:
        """调用视觉 API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content
```

---

## 图像生成

### 使用 DALL-E 生成图像

```python
class ImageGenerator:
    """图像生成工具（基于 DALL-E）"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1
    ) -> list[str]:
        """根据描述生成图像"""
        
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,        # "1024x1024", "1792x1024", "1024x1792"
            quality=quality,  # "standard" 或 "hd"
            n=n
        )
        
        return [img.url for img in response.data]
    
    def edit_image(
        self,
        image_path: str,
        prompt: str,
        mask_path: str = None
    ) -> str:
        """编辑已有图像"""
        
        with open(image_path, "rb") as f:
            image_file = f.read()
        
        kwargs = {
            "model": "dall-e-2",
            "image": image_file,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }
        
        if mask_path:
            with open(mask_path, "rb") as f:
                kwargs["mask"] = f.read()
        
        response = self.client.images.edit(**kwargs)
        return response.data[0].url
```

---

## 实用示例

```python
# 示例 1：OCR —— 从图片中提取文字
vision = VisionTool()

text = vision.analyze_local_image(
    "receipt.jpg",
    "请提取这张发票中的所有文字信息，以结构化的JSON格式返回"
)
print(text)

# 示例 2：图表分析
analysis = vision.analyze_local_image(
    "sales_chart.png",
    "请分析这张销售图表，指出关键趋势和数据要点"
)
print(analysis)

# 示例 3：代码截图转代码
code = vision.analyze_local_image(
    "code_screenshot.png",
    "请将截图中的代码转写为文本，保持格式"
)
print(code)
```

---

## 小结

| 功能 | API | 说明 |
|------|-----|------|
| 图像分析 | GPT-4o Vision | 理解图片内容、提取文字 |
| 图片比较 | GPT-4o Vision | 多图输入，分析异同 |
| 图像生成 | DALL-E 3 | 根据文字描述生成图片 |
| 图像编辑 | DALL-E 2 | 修改已有图片 |

---

[下一节：21.3 语音交互集成 →](./03_voice_interaction.md)
