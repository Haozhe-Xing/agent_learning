# 23.6 Video Understanding and Multimodal RAG

> **Goal of this section**: Master how to implement video-understanding Agents, and understand the architecture design and engineering practice of multimodal RAG.

---

## Video Understanding: From Images to the Time Dimension

A video is "a sequence of images with a time axis", but video understanding is far more than analyzing frames one by one — it requires understanding **causal relationships along the time dimension**: who did what first, how an action evolved, how the scene changed.

### Three Levels of Video Understanding

```python
VIDEO_UNDERSTANDING_LEVELS = {
    "Level 1: Frame-level understanding": {
        "capability": "Identify objects, text, and scenes within a single frame",
        "example": "A red car appears at second 15 of the video",
        "technique": "Frame extraction + image understanding model",
        "difficulty": "⭐⭐",
    },
    "Level 2: Clip-level understanding": {
        "capability": "Understand actions and events across a few consecutive seconds",
        "example": "The person stands up from a seated position and walks toward the door",
        "technique": "Multi-frame joint reasoning / video-native model",
        "difficulty": "⭐⭐⭐",
    },
    "Level 3: Video-level understanding": {
        "capability": "Understand the topic, narrative, and causality of the whole video",
        "example": "This is a cooking tutorial teaching how to make braised pork",
        "technique": "Long-video encoding + hierarchical summarization",
        "difficulty": "⭐⭐⭐⭐",
    },
}
```

### Two Implementation Paths

**Path 1: Frame extraction + vision model (works with every multimodal model)**

```python
from openai import OpenAI
import base64
import cv2

client = OpenAI()


class VideoUnderstandingAgent:
    """Video understanding Agent (frame-extraction approach)"""
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
    
    def extract_key_frames(
        self,
        video_path: str,
        interval_seconds: float = 5.0,
        max_frames: int = 20
    ) -> list[tuple[float, str]]:
        """Extract key frames from a video at a fixed interval
        
        Args:
            video_path: path to the video file
            interval_seconds: sampling interval (seconds)
            max_frames: maximum number of frames (to control cost)
        
        Returns:
            [(timestamp, base64-encoded image), ...]
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
            
            # Encode as JPEG
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
        """Analyze the content of a video
        
        Args:
            video_path: path to the video file
            question: the question to answer
            interval_seconds: frame-extraction interval
            max_frames: maximum number of frames
        """
        # 1. Extract key frames
        frames = self.extract_key_frames(
            video_path, interval_seconds, max_frames
        )
        
        # 2. Build a multi-frame prompt
        content = [
            {
                "type": "text",
                "text": f"""Below are key-frame screenshots from a video (in chronological order); each image is labeled with its timestamp.
Answer the question based on these screenshots.

Question: {question}

Key frames:"""
            }
        ]
        
        for timestamp, img_b64 in frames:
            content.append({
                "type": "text",
                "text": f"\n[time: {timestamp:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"  # control cost
                }
            })
        
        # 3. Call the multimodal model
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
        """Generate a timeline summary of the video
        
        Returns:
            [{"timestamp": 0.0, "description": "..."}, ...]
        """
        frames = self.extract_key_frames(
            video_path, interval_seconds=3.0, max_frames=max_frames
        )
        
        content = [{
            "type": "text",
            "text": "Generate a timeline summary for the following video key frames. "
                    "Describe what is happening in each frame in one sentence, returned as a JSON array:\n"
                    '[{"time": "0.0s", "event": "..."}, ...]\n\nKey frames:'
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


# Usage example
agent = VideoUnderstandingAgent()

# Analyze a tutorial video
summary = agent.analyze_video(
    "python_tutorial.mp4",
    "What is this video about? Summarize the main knowledge points"
)
print(summary)

# Generate a timeline
timeline = agent.generate_timeline("meeting_recording.mp4")
for event in timeline:
    print(f"[{event['time']}] {event['event']}")
```

**Path 2: Native video models (Gemini 2.5 Pro)**

Gemini 2.5 Pro natively supports video input up to one hour long, with no frame extraction required:

```python
import google.generativeai as genai

def analyze_video_native(video_path: str, question: str) -> str:
    """Native video understanding with Gemini 2.5 Pro
    
    Advantages:
    - No frame extraction; the model processes the video stream directly
    - Understands causal relationships along the time dimension
    - Supports long videos (up to 1 hour)
    """
    # Upload the video file
    video_file = genai.upload_file(path=video_path)
    
    # Wait for file processing to finish
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    
    # Analyze the video
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(
        [video_file, question],
        request_options={"timeout": 300}
    )
    
    return response.text


# Video question answering
answer = analyze_video_native(
    "product_demo.mp4",
    "Which core product features are shown in the video? List them in chronological order"
)
```

---

## Multimodal RAG: Retrieving Mixed Text-and-Image Content

Traditional RAG (Chapter 6) can only retrieve text. But in real scenarios, a knowledge base usually contains documents that mix text and images — technical manuals with architecture diagrams, papers with experiment charts, slide decks with flowcharts. **Multimodal RAG** lets an Agent retrieve and understand this mixed content.

### Architecture Design

There are three mainstream architectures for multimodal RAG:

```python
MULTIMODAL_RAG_ARCHITECTURES = {
    "Architecture 1: Text-first": {
        "pipeline": "OCR / image captioning → pure-text embedding → text retrieval",
        "pros": "Reuses existing RAG infrastructure, low cost",
        "cons": "Loses visual information (layout, color, spatial relationships)",
        "best for": "Text-dominant documents (contracts, invoices)",
    },
    "Architecture 2: Multimodal Embedding": {
        "pipeline": "Images + text → unified vector space → cross-modal retrieval",
        "pros": "Search images with text and search text with images",
        "cons": "Requires a dedicated cross-modal embedding model",
        "best for": "Mixed text-and-image documents (slides, papers, manuals)",
    },
    "Architecture 3: Native Multimodal": {
        "pipeline": "Feed images directly into a multimodal LLM for understanding",
        "pros": "Zero information loss, the most accurate understanding",
        "cons": "High cost, slow",
        "best for": "Scenarios with extremely high image-understanding quality requirements",
    },
}
```

### Hands-On: Text-First Multimodal RAG

The most practical approach — turn the images in a document into text descriptions with a vision model, then run the standard RAG pipeline:

```python
from openai import OpenAI
import base64

client = OpenAI()


class MultimodalDocumentProcessor:
    """Multimodal document processor"""
    
    def __init__(self):
        self.vision_client = OpenAI()
    
    def process_page(self, text: str, images: list[str]) -> str:
        """Process a single document page (text plus images)
        
        Args:
            text: the page's text content
            images: list of image paths on the page
        """
        parts = [f"## Page text\n\n{text}"]
        
        for i, img_path in enumerate(images, 1):
            # Describe the image content with a vision model
            description = self._describe_image(img_path)
            parts.append(f"\n## Figure {i}\n\n{description}")
        
        return "\n".join(parts)
    
    def _describe_image(self, image_path: str) -> str:
        """Generate a text description of an image with a vision model"""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        response = self.vision_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Describe the content of this image in detail. If it is:
- A data chart: extract every visible data point and label
- A flowchart: describe all steps and their connections
- An architecture diagram: list all components and how they interact
- A screenshot: describe the interface layout and key elements

Please write a structured text description so that it is easy to retrieve later."""
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
    """Multimodal RAG system"""
    
    def __init__(self):
        self.processor = MultimodalDocumentProcessor()
        self.documents = []  # processed text chunks
        self.embeddings = []  # the corresponding vectors
    
    def ingest_document(self, pages: list[dict]) -> None:
        """Ingest a document
        
        Args:
            pages: [{"text": "page text", "images": ["img1.png", ...]}, ...]
        """
        for page in pages:
            processed = self.processor.process_page(
                page["text"], page.get("images", [])
            )
            
            # Chunking
            chunks = self._split_text(processed, chunk_size=500)
            
            # Embedding
            for chunk in chunks:
                emb = self._get_embedding(chunk)
                self.documents.append(chunk)
                self.embeddings.append(emb)
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Multimodal RAG query"""
        import numpy as np
        
        # 1. Vectorize the query
        query_emb = self._get_embedding(question)
        
        # 2. Similarity retrieval
        similarities = [
            np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
            )
            for doc_emb in self.embeddings
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved = [self.documents[i] for i in top_indices]
        
        # 3. Generate the answer
        context = "\n\n---\n\n".join(retrieved)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Answer the question based on the retrieved content below.
                
Retrieved content:
{context}

Question: {question}

Answer based on the retrieved content; if it is insufficient to answer the question, say so."""
            }],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _split_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Simple text chunking"""
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
        """Get the embedding vector of a text"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


# Usage example
rag = MultimodalRAG()

# Ingest a document containing images
rag.ingest_document([
    {
        "text": "System architecture overview: this system uses a microservice architecture...",
        "images": ["architecture_diagram.png"]
    },
    {
        "text": "Performance test results: under 1000 concurrent users...",
        "images": ["performance_chart.png"]
    }
])

# Query (you can retrieve content inside images with natural language)
answer = rag.query("What is the overall architecture of the system? How do the services interact?")
print(answer)
```

### Hands-On: Multimodal Embedding Approach

Use a cross-modal embedding model (such as CLIP) to implement "search images with text" and "search text with images":

```python
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class CrossModalRetriever:
    """Cross-modal retriever (based on CLIP)"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        self.text_items = []   # text entries
        self.image_items = []  # image entries
        self.text_embs = []    # text vectors
        self.image_embs = []   # image vectors
    
    def add_text(self, text: str, metadata: dict = None):
        """Add a text entry"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        self.text_items.append({"text": text, "meta": metadata})
        self.text_embs.append(emb[0].numpy())
    
    def add_image(self, image_path: str, metadata: dict = None):
        """Add an image entry"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        self.image_items.append({"path": image_path, "meta": metadata})
        self.image_embs.append(emb[0].numpy())
    
    def search_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for related text and images using text"""
        import numpy as np
        
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_emb = self.model.get_text_features(**inputs)
            query_emb = (query_emb / query_emb.norm(dim=-1, keepdim=True))[0].numpy()
        
        results = []
        
        # Search text
        for i, text_emb in enumerate(self.text_embs):
            score = float(np.dot(query_emb, text_emb))
            results.append({
                "type": "text",
                "content": self.text_items[i]["text"],
                "score": score,
                "meta": self.text_items[i]["meta"]
            })
        
        # Search images
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
        """Search for related text and images using an image"""
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


# Usage example: search images with text
retriever = CrossModalRetriever()
retriever.add_text("A flowchart showing the microservice architecture of the system")
retriever.add_image("architecture.png")

results = retriever.search_by_text("system architecture diagram")
for r in results:
    print(f"[{r['type']}] score={r['score']:.3f}: {r['content'][:50]}")
```

---

## The Complete Design Pattern for a Multimodal Agent

Pulling this chapter together, the architecture of a production-grade multimodal Agent looks like this:

```python
class ProductionMultimodalAgent:
    """Production-grade multimodal Agent"""
    
    def __init__(self):
        # Perception layer: multimodal input processing
        self.vision = VisionTool()                    # image understanding
        self.video = VideoUnderstandingAgent()        # video understanding
        self.stt = SpeechToText()                     # speech recognition
        self.tts = TextToSpeech()                     # speech synthesis
        
        # Knowledge layer: multimodal RAG
        self.rag = MultimodalRAG()                    # mixed text-and-image retrieval
        self.cross_modal = CrossModalRetriever()      # cross-modal retrieval
        
        # Action layer: multimodal output
        self.image_gen = ImageGenerator()             # image generation
        self.computer_use = SafeComputerUseAgent()    # computer operation
        
        # Orchestration layer: unified entry point
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
    
    async def process(self, user_input: dict) -> dict:
        """Process multimodal input and return multimodal output
        
        Args:
            user_input: {
                "text": str | None,
                "image": str | None,    # image path
                "video": str | None,    # video path
                "audio": str | None,    # audio path
                "screenshot": str | None,  # screenshot (Computer Use)
            }
        """
        # 1. Unified perception: convert every modality into text + structured features
        perception = await self._perceive(user_input)
        
        # 2. Knowledge retrieval: retrieve relevant information from the multimodal knowledge base
        context = await self._retrieve(perception)
        
        # 3. Reasoning and decision: combine perception and knowledge, decide the output modality and content
        plan = await self._reason(perception, context)
        
        # 4. Multimodal action: execute the plan and produce multimodal output
        output = await self._act(plan, user_input)
        
        return output
    
    async def _perceive(self, user_input: dict) -> dict:
        """Unified perception layer"""
        perception = {"text_parts": [], "visual_context": None}
        
        if user_input.get("audio"):
            text = self.stt.transcribe(user_input["audio"])
            perception["text_parts"].append(text)
        
        if user_input.get("text"):
            perception["text_parts"].append(user_input["text"])
        
        if user_input.get("image"):
            desc = self.vision.analyze_local_image(
                user_input["image"],
                "Describe the key information in this image"
            )
            perception["text_parts"].append(f"[image content] {desc}")
            perception["visual_context"] = desc
        
        if user_input.get("video"):
            summary = self.video.analyze_video(
                user_input["video"],
                "Summarize the main content of the video"
            )
            perception["text_parts"].append(f"[video content] {summary}")
        
        if user_input.get("screenshot"):
            # Computer Use scenario: understand the screen state
            screen_desc = self.vision.analyze_local_image(
                user_input["screenshot"],
                "Describe the main interface elements and state on the current screen"
            )
            perception["text_parts"].append(f"[screen state] {screen_desc}")
        
        return perception
    
    async def _retrieve(self, perception: dict) -> str:
        """Multimodal knowledge retrieval"""
        query = " ".join(perception["text_parts"])
        return self.rag.query(query, top_k=3)
    
    async def _reason(self, perception: dict, context: str) -> dict:
        """Reasoning and planning"""
        query = " ".join(perception["text_parts"])
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": """You are the planner of a multimodal Agent.
Based on the user input and the retrieved knowledge, decide:
1. Which modality to output (text/image/audio/action)
2. If the computer must be operated, which operation to perform
3. If an image must be generated, which prompt to use"""},
            {"role": "user", "content": f"User input: {query}\n\nRetrieved knowledge: {context}"}
        ])
        
        return {"plan": response.content, "query": query}
    
    async def _act(self, plan: dict, user_input: dict) -> dict:
        """Multimodal action"""
        result = {"text": "", "image": None, "audio": None, "action_taken": None}
        
        # Simplified version: decide the output type from keywords
        plan_text = plan["plan"].lower()
        
        if "generate image" in plan_text or "create image" in plan_text:
            urls = self.image_gen.generate(plan["query"])
            result["image"] = urls[0] if urls else None
            result["text"] = "Here is the image I generated for you"
        
        elif "operate computer" in plan_text or "click" in plan_text:
            # Computer Use scenario
            result["text"] = "Operating the computer..."
            result["action_taken"] = True
        
        else:
            # Ordinary text answer
            result["text"] = plan["plan"]
        
        # If the input was speech, generate a spoken reply
        if user_input.get("audio"):
            audio_path = self.tts.speak(result["text"])
            result["audio"] = audio_path
        
        return result
```

---

## Summary

| Concept | Description |
|------|------|
| Video understanding | Frame-extraction approach (universal) or native video model (Gemini) |
| Levels of video capability | Frame level → clip level → video level |
| Multimodal RAG | Retrieving mixed text-and-image content, with three architectures to choose from |
| Text-first RAG | Images turned into descriptions → pure-text retrieval (the most practical) |
| Cross-modal retrieval | Models such as CLIP enable "search images with text" and "search text with images" |
| Production-grade architecture | Perception layer → knowledge layer → reasoning layer → action layer |

> 📄 **Further reading**:
> - Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021. (CLIP)
> - Google. "Gemini 2.5 Pro: Long Context & Video Understanding." Google AI Blog, 2025.
> - Chen et al. "LLaVA: Visual Instruction Tuning." NeurIPS, 2024.

---

> 🎓 **Chapter summary**: Multimodal Agents let AI break past the boundary of text. From image understanding, voice interaction, and video analysis to Computer Use operating a computer, multimodal capabilities let an Agent perceive and act on the real world much like a human does. In 2025–2026, Computer Use Agents and GUI automation are the hottest directions — although they are still short of human-level operation, the pace of progress is astonishing. Mastering multimodal Agent development is an essential skill for becoming a senior Agent engineer.

---

[Appendix A: A Complete Collection of Common Prompt Templates](../appendix/prompt_templates.md)
