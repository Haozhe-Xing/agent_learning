# 图像理解与生成

> **本节目标**：掌握用 GPT-4o 分析图像和用 DALL-E 生成图像的技术。

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

[下一节：18.3 语音交互集成 →](./03_voice_interaction.md)
