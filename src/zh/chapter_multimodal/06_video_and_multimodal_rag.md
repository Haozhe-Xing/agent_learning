# 22.6 视频理解与多模态 RAG

> **本节目标**：掌握视频理解 Agent 的实现方法，了解多模态 RAG 的架构设计和工程实践。

---

## 视频理解：从图片到时间维度

视频是"带时间轴的图片序列"，但视频理解远不是简单地逐帧分析——它需要理解**时间维度的因果关系**：谁先做了什么、动作如何演变、场景如何切换。

### 视频理解的三层能力

```python
VIDEO_UNDERSTANDING_LEVELS = {
    "第一层：帧级理解": {
        "能力": "识别单帧中的物体、文字、场景",
        "示例": "视频第15秒出现了红色的汽车",
        "技术": "截帧 + 图像理解模型",
        "难度": "⭐⭐",
    },
    "第二层：片段级理解": {
        "能力": "理解连续几秒内的动作和事件",
        "示例": "这个人从坐姿站了起来，走向门口",
        "技术": "多帧联合推理 / 视频专用模型",
        "难度": "⭐⭐⭐",
    },
    "第三层：视频级理解": {
        "能力": "理解整段视频的主题、叙事和因果",
        "示例": "这是一段烹饪教学视频，教的是红烧肉的做法",
        "技术": "长视频编码 + 层次化摘要",
        "难度": "⭐⭐⭐⭐",
    },
}
```

### 两种实现路径

**路径一：截帧 + 视觉模型（适用于所有多模态模型）**

```python
from openai import OpenAI
import base64
import cv2

client = OpenAI()


class VideoUnderstandingAgent:
    """视频理解 Agent（截帧方案）"""
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
    
    def extract_key_frames(
        self,
        video_path: str,
        interval_seconds: float = 5.0,
        max_frames: int = 20
    ) -> list[tuple[float, str]]:
        """从视频中按间隔提取关键帧
        
        Args:
            video_path: 视频文件路径
            interval_seconds: 采样间隔（秒）
            max_frames: 最大帧数（控制成本）
        
        Returns:
            [(时间戳, base64编码图像), ...]
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frames = []
        frame_interval = int(fps * interval_seconds)
        
        for i in range(0, total_frames, frame_interval):
            if len(frames) >= max_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 编码为 JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buffer).decode()
            timestamp = i / fps
            
            frames.append((timestamp, img_b64))
        
        cap.release()
        return frames
    
    def analyze_video(
        self,
        video_path: str,
        question: str,
        interval_seconds: float = 5.0,
        max_frames: int = 10
    ) -> str:
        """分析视频内容
        
        Args:
            video_path: 视频文件路径
            question: 要回答的问题
            interval_seconds: 截帧间隔
            max_frames: 最大帧数
        """
        # 1. 提取关键帧
        frames = self.extract_key_frames(
            video_path, interval_seconds, max_frames
        )
        
        # 2. 构建多帧 Prompt
        content = [
            {
                "type": "text",
                "text": f"""以下是一段视频的关键帧截图（按时间顺序排列），每张图标注了时间戳。
请根据这些截图回答问题。

问题：{question}

关键帧："""
            }
        ]
        
        for timestamp, img_b64 in frames:
            content.append({
                "type": "text",
                "text": f"\n[时间: {timestamp:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"  # 控制成本
                }
            })
        
        # 3. 调用多模态模型
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def generate_timeline(
        self,
        video_path: str,
        max_frames: int = 20
    ) -> list[dict]:
        """生成视频时间线摘要
        
        Returns:
            [{"timestamp": 0.0, "description": "..."}, ...]
        """
        frames = self.extract_key_frames(
            video_path, interval_seconds=3.0, max_frames=max_frames
        )
        
        content = [{
            "type": "text",
            "text": "请为以下视频关键帧生成时间线摘要。"
                    "对每帧用一句话描述正在发生的事，以 JSON 数组格式返回：\n"
                    '[{"time": "0.0s", "event": "..."}, ...]\n\n关键帧：'
        }]
        
        for timestamp, img_b64 in frames:
            content.append({
                "type": "text",
                "text": f"\n[{timestamp:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"
                }
            })
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result.get("timeline", [])


# 使用示例
agent = VideoUnderstandingAgent()

# 分析教学视频
summary = agent.analyze_video(
    "python_tutorial.mp4",
    "这个视频讲了什么？请总结主要知识点"
)
print(summary)

# 生成时间线
timeline = agent.generate_timeline("meeting_recording.mp4")
for event in timeline:
    print(f"[{event['time']}] {event['event']}")
```

**路径二：原生视频模型（Gemini 2.5 Pro）**

Gemini 2.5 Pro 原生支持最长 1 小时的视频输入，无需截帧：

```python
import google.generativeai as genai

def analyze_video_native(video_path: str, question: str) -> str:
    """使用 Gemini 2.5 Pro 原生视频理解
    
    优势：
    - 无需截帧，模型直接处理视频流
    - 理解时间维度的因果关系
    - 支持长视频（最长 1 小时）
    """
    # 上传视频文件
    video_file = genai.upload_file(path=video_path)
    
    # 等待文件处理完成
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    
    # 分析视频
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(
        [video_file, question],
        request_options={"timeout": 300}
    )
    
    return response.text


# 视频问答
answer = analyze_video_native(
    "product_demo.mp4",
    "视频中展示了产品的哪些核心功能？请按时间顺序列出"
)
```

---

## 多模态 RAG：检索图文混合内容

传统 RAG（第6章）只能检索文本。但在真实场景中，知识库往往包含图文混排的文档——技术手册中有架构图，论文中有实验图表，PPT 中有流程图。**多模态 RAG** 让 Agent 能够检索和理解这些混合内容。

### 架构设计

多模态 RAG 有三种主流架构：

```python
MULTIMODAL_RAG_ARCHITECTURES = {
    "架构一：文本优先（Text-first）": {
        "流程": "OCR/图像描述 → 纯文本 Embedding → 文本检索",
        "优点": "复用现有 RAG 基础设施，成本低",
        "缺点": "丢失视觉信息（布局、颜色、空间关系）",
        "适用": "以文字为主的文档（合同、发票）",
    },
    "架构二：多模态 Embedding（Multimodal Embedding）": {
        "流程": "图像+文本 → 统一向量空间 → 跨模态检索",
        "优点": "可以用文字检索图片、用图片检索文字",
        "缺点": "需要专门的跨模态 Embedding 模型",
        "适用": "图文混排文档（PPT、论文、手册）",
    },
    "架构三：原生多模态（Native Multimodal）": {
        "流程": "直接将图像送入多模态 LLM 进行理解",
        "优点": "信息零损失，理解最准确",
        "缺点": "成本高、速度慢",
        "适用": "图像理解质量要求极高的场景",
    },
}
```

### 实战：基于文本优先的多模态 RAG

最实用的方案——将文档中的图片通过视觉模型转为文字描述，然后走标准 RAG 流程：

```python
from openai import OpenAI
import base64

client = OpenAI()


class MultimodalDocumentProcessor:
    """多模态文档处理器"""
    
    def __init__(self):
        self.vision_client = OpenAI()
    
    def process_page(self, text: str, images: list[str]) -> str:
        """处理单页文档（含文字和图片）
        
        Args:
            text: 页面文字内容
            images: 页面图片路径列表
        """
        parts = [f"## 页面文本\n\n{text}"]
        
        for i, img_path in enumerate(images, 1):
            # 用视觉模型描述图片内容
            description = self._describe_image(img_path)
            parts.append(f"\n## 图表 {i}\n\n{description}")
        
        return "\n".join(parts)
    
    def _describe_image(self, image_path: str) -> str:
        """用视觉模型生成图片的文字描述"""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        response = self.vision_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """请详细描述这张图片的内容。如果是：
- 数据图表：提取所有可见的数据点和标签
- 流程图：描述所有步骤和连接关系
- 架构图：列出所有组件和它们的交互方式
- 截图：描述界面布局和关键元素

请用结构化的文字描述，便于后续检索。"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=1000
        )
        
        return response.choices[0].message.content


class MultimodalRAG:
    """多模态 RAG 系统"""
    
    def __init__(self):
        self.processor = MultimodalDocumentProcessor()
        self.documents = []  # 处理后的文本块
        self.embeddings = []  # 对应的向量
    
    def ingest_document(self, pages: list[dict]) -> None:
        """导入文档
        
        Args:
            pages: [{"text": "页面文字", "images": ["img1.png", ...]}, ...]
        """
        for page in pages:
            processed = self.processor.process_page(
                page["text"], page.get("images", [])
            )
            
            # 分块
            chunks = self._split_text(processed, chunk_size=500)
            
            # Embedding
            for chunk in chunks:
                emb = self._get_embedding(chunk)
                self.documents.append(chunk)
                self.embeddings.append(emb)
    
    def query(self, question: str, top_k: int = 5) -> str:
        """多模态 RAG 查询"""
        import numpy as np
        
        # 1. 查询向量化
        query_emb = self._get_embedding(question)
        
        # 2. 相似度检索
        similarities = [
            np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
            )
            for doc_emb in self.embeddings
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved = [self.documents[i] for i in top_indices]
        
        # 3. 生成回答
        context = "\n\n---\n\n".join(retrieved)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""基于以下检索到的内容回答问题。
                
检索内容：
{context}

问题：{question}

请基于检索内容回答，如果检索内容不足以回答问题，请说明。"""
            }],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _split_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """简单文本分块"""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def _get_embedding(self, text: str) -> list[float]:
        """获取文本向量"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


# 使用示例
rag = MultimodalRAG()

# 导入含图片的文档
rag.ingest_document([
    {
        "text": "系统架构概览：本系统采用微服务架构...",
        "images": ["architecture_diagram.png"]
    },
    {
        "text": "性能测试结果：在 1000 并发用户下...",
        "images": ["performance_chart.png"]
    }
])

# 查询（可以用自然语言检索图片中的内容）
answer = rag.query("系统的整体架构是怎样的？各服务如何交互？")
print(answer)
```

### 实战：多模态 Embedding 方案

使用跨模态 Embedding 模型（如 CLIP），实现"以文搜图"和"以图搜文"：

```python
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class CrossModalRetriever:
    """跨模态检索器（基于 CLIP）"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        self.text_items = []   # 文本条目
        self.image_items = []  # 图像条目
        self.text_embs = []    # 文本向量
        self.image_embs = []   # 图像向量
    
    def add_text(self, text: str, metadata: dict = None):
        """添加文本条目"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        self.text_items.append({"text": text, "meta": metadata})
        self.text_embs.append(emb[0].numpy())
    
    def add_image(self, image_path: str, metadata: dict = None):
        """添加图像条目"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        self.image_items.append({"path": image_path, "meta": metadata})
        self.image_embs.append(emb[0].numpy())
    
    def search_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """用文字搜索相关的文字和图片"""
        import numpy as np
        
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_emb = self.model.get_text_features(**inputs)
            query_emb = (query_emb / query_emb.norm(dim=-1, keepdim=True))[0].numpy()
        
        results = []
        
        # 搜索文本
        for i, text_emb in enumerate(self.text_embs):
            score = float(np.dot(query_emb, text_emb))
            results.append({
                "type": "text",
                "content": self.text_items[i]["text"],
                "score": score,
                "meta": self.text_items[i]["meta"]
            })
        
        # 搜索图像
        for i, image_emb in enumerate(self.image_embs):
            score = float(np.dot(query_emb, image_emb))
            results.append({
                "type": "image",
                "content": self.image_items[i]["path"],
                "score": score,
                "meta": self.image_items[i]["meta"]
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def search_by_image(self, query_image_path: str, top_k: int = 5) -> list[dict]:
        """用图片搜索相关的文字和图片"""
        import numpy as np
        
        image = Image.open(query_image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_emb = self.model.get_image_features(**inputs)
            query_emb = (query_emb / query_emb.norm(dim=-1, keepdim=True))[0].numpy()
        
        results = []
        
        for i, text_emb in enumerate(self.text_embs):
            score = float(np.dot(query_emb, text_emb))
            results.append({
                "type": "text",
                "content": self.text_items[i]["text"],
                "score": score,
            })
        
        for i, image_emb in enumerate(self.image_embs):
            score = float(np.dot(query_emb, image_emb))
            results.append({
                "type": "image",
                "content": self.image_items[i]["path"],
                "score": score,
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# 使用示例：以文搜图
retriever = CrossModalRetriever()
retriever.add_text("一张展示系统微服务架构的流程图")
retriever.add_image("architecture.png")

results = retriever.search_by_text("系统架构图")
for r in results:
    print(f"[{r['type']}] score={r['score']:.3f}: {r['content'][:50]}")
```

---

## 多模态 Agent 的完整设计模式

综合本章内容，一个生产级多模态 Agent 的架构如下：

```python
class ProductionMultimodalAgent:
    """生产级多模态 Agent"""
    
    def __init__(self):
        # 感知层：多模态输入处理
        self.vision = VisionTool()                    # 图像理解
        self.video = VideoUnderstandingAgent()        # 视频理解
        self.stt = SpeechToText()                     # 语音识别
        self.tts = TextToSpeech()                     # 语音合成
        
        # 知识层：多模态 RAG
        self.rag = MultimodalRAG()                    # 图文混合检索
        self.cross_modal = CrossModalRetriever()      # 跨模态检索
        
        # 行动层：多模态输出
        self.image_gen = ImageGenerator()             # 图像生成
        self.computer_use = SafeComputerUseAgent()    # 计算机操作
        
        # 编排层：统一入口
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
    
    async def process(self, user_input: dict) -> dict:
        """处理多模态输入并返回多模态输出
        
        Args:
            user_input: {
                "text": str | None,
                "image": str | None,    # 图片路径
                "video": str | None,    # 视频路径
                "audio": str | None,    # 音频路径
                "screenshot": str | None,  # 屏幕截图（Computer Use）
            }
        """
        # 1. 统一感知：将所有模态转为文本 + 结构化特征
        perception = await self._perceive(user_input)
        
        # 2. 知识检索：从多模态知识库中检索相关信息
        context = await self._retrieve(perception)
        
        # 3. 推理决策：综合感知和知识，决定输出模态和内容
        plan = await self._reason(perception, context)
        
        # 4. 多模态行动：执行计划，生成多模态输出
        output = await self._act(plan, user_input)
        
        return output
    
    async def _perceive(self, user_input: dict) -> dict:
        """统一感知层"""
        perception = {"text_parts": [], "visual_context": None}
        
        if user_input.get("audio"):
            text = self.stt.transcribe(user_input["audio"])
            perception["text_parts"].append(text)
        
        if user_input.get("text"):
            perception["text_parts"].append(user_input["text"])
        
        if user_input.get("image"):
            desc = self.vision.analyze_local_image(
                user_input["image"],
                "请描述这张图片的关键信息"
            )
            perception["text_parts"].append(f"[图片内容] {desc}")
            perception["visual_context"] = desc
        
        if user_input.get("video"):
            summary = self.video.analyze_video(
                user_input["video"],
                "总结视频的主要内容"
            )
            perception["text_parts"].append(f"[视频内容] {summary}")
        
        if user_input.get("screenshot"):
            # Computer Use 场景：理解屏幕状态
            screen_desc = self.vision.analyze_local_image(
                user_input["screenshot"],
                "描述当前屏幕上的主要界面元素和状态"
            )
            perception["text_parts"].append(f"[屏幕状态] {screen_desc}")
        
        return perception
    
    async def _retrieve(self, perception: dict) -> str:
        """多模态知识检索"""
        query = " ".join(perception["text_parts"])
        return self.rag.query(query, top_k=3)
    
    async def _reason(self, perception: dict, context: str) -> dict:
        """推理与规划"""
        query = " ".join(perception["text_parts"])
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": """你是一个多模态 Agent 的规划器。
根据用户输入和检索到的知识，决定：
1. 应该输出什么模态（text/image/audio/action）
2. 如果需要操作计算机，应该执行什么操作
3. 如果需要生成图片，应该用什么 prompt"""},
            {"role": "user", "content": f"用户输入：{query}\n\n检索知识：{context}"}
        ])
        
        return {"plan": response.content, "query": query}
    
    async def _act(self, plan: dict, user_input: dict) -> dict:
        """多模态行动"""
        result = {"text": "", "image": None, "audio": None, "action_taken": None}
        
        # 简化版：根据关键词判断输出类型
        plan_text = plan["plan"].lower()
        
        if "生成图片" in plan_text or "创建图像" in plan_text:
            urls = self.image_gen.generate(plan["query"])
            result["image"] = urls[0] if urls else None
            result["text"] = "已为您生成图片"
        
        elif "操作计算机" in plan_text or "点击" in plan_text:
            # Computer Use 场景
            result["text"] = "正在操作计算机..."
            result["action_taken"] = True
        
        else:
            # 普通文本回答
            result["text"] = plan["plan"]
        
        # 如果是语音输入，生成语音回复
        if user_input.get("audio"):
            audio_path = self.tts.speak(result["text"])
            result["audio"] = audio_path
        
        return result
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 视频理解 | 截帧方案（通用）或原生视频模型（Gemini） |
| 视频能力层次 | 帧级 → 片段级 → 视频级 |
| 多模态 RAG | 检索图文混合内容，三种架构可选 |
| 文本优先 RAG | 图片转描述 → 纯文本检索（最实用） |
| 跨模态检索 | CLIP 等模型实现"以文搜图""以图搜文" |
| 生产级架构 | 感知层 → 知识层 → 推理层 → 行动层 |

> 📄 **延伸阅读**：
> - Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021. (CLIP)
> - Google. "Gemini 2.5 Pro: Long Context & Video Understanding." Google AI Blog, 2025.
> - Chen et al. "LLaVA: Visual Instruction Tuning." NeurIPS, 2024.

---

> 🎓 **本章总结**：多模态 Agent 让 AI 突破了文字的边界。从图像理解、语音交互、视频分析到 Computer Use 操作计算机，多模态能力让 Agent 能够像人一样感知和操作真实世界。2025-2026 年，Computer Use Agent 和 GUI 自动化是最热门的方向——虽然距离人类水平的操作还有差距，但进步速度惊人。掌握多模态 Agent 的开发，是成为高级 Agent 工程师的必备技能。

---

[附录 A：常用 Prompt 模板大全](../appendix/prompt_templates.md)
