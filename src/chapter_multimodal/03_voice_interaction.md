# 语音交互集成

> **本节目标**：为 Agent 集成语音识别（STT）和语音合成（TTS）能力。

---

## 语音识别（Speech-to-Text）

使用 OpenAI 的 Whisper 模型将语音转为文字：

```python
from openai import OpenAI
from pathlib import Path

class SpeechToText:
    """语音转文字工具"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "zh"
    ) -> str:
        """将音频文件转为文字"""
        
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )
        
        return transcript
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: str = "zh"
    ) -> dict:
        """转录并返回时间戳"""
        
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        return {
            "text": transcript.text,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end
                }
                for seg in (transcript.segments or [])
            ]
        }


# 使用示例
stt = SpeechToText()
text = stt.transcribe("user_voice.mp3")
print(f"用户说：{text}")
```

---

## 语音合成（Text-to-Speech）

将 Agent 的文字回复转为语音：

```python
class TextToSpeech:
    """文字转语音工具"""
    
    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(self, voice: str = "nova"):
        self.client = OpenAI()
        self.voice = voice
    
    def speak(
        self,
        text: str,
        output_path: str = "response.mp3",
        speed: float = 1.0
    ) -> str:
        """将文字转为语音文件"""
        
        response = self.client.audio.speech.create(
            model="tts-1",       # 或 "tts-1-hd" 高清版
            voice=self.voice,
            input=text,
            speed=speed          # 0.25 - 4.0
        )
        
        response.stream_to_file(output_path)
        return output_path
    
    def stream_speak(self, text: str):
        """流式语音合成（边生成边播放）"""
        
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text
        )
        
        # 返回字节流，可以边接收边播放
        return response.content


# 使用示例
tts = TextToSpeech(voice="nova")
tts.speak("你好！我是你的AI助手，有什么可以帮你的吗？")
```

---

## 语音对话循环

```python
import asyncio

class VoiceConversation:
    """语音对话系统"""
    
    def __init__(self, agent_func):
        self.stt = SpeechToText()
        self.tts = TextToSpeech(voice="nova")
        self.agent = agent_func  # Agent 处理函数
    
    async def process_voice(self, audio_path: str) -> str:
        """处理一轮语音对话"""
        
        # 1. 语音 → 文字
        print("🎤 识别语音中...")
        user_text = self.stt.transcribe(audio_path)
        print(f"📝 识别结果: {user_text}")
        
        # 2. Agent 处理
        print("🤔 Agent 思考中...")
        response_text = await self.agent(user_text)
        print(f"💬 Agent 回复: {response_text}")
        
        # 3. 文字 → 语音
        print("🔊 生成语音...")
        audio_output = self.tts.speak(response_text)
        print(f"✅ 语音已保存: {audio_output}")
        
        return audio_output
```

---

## 小结

| 功能 | 模型 | 说明 |
|------|------|------|
| 语音识别 | Whisper | 支持 50+ 语言，准确率高 |
| 语音合成 | TTS-1 | 6 种音色可选 |
| 语音对话 | 组合使用 | STT → Agent → TTS 完整链路 |

---

[下一节：18.4 实战：多模态个人助理 →](./04_practice_multimodal_assistant.md)
