"""LLM provider abstraction.

The book needs code that runs *without* an API key for tests and demos, but
also runs against a real model in production. `FakeProvider` is deterministic
and offline; `OpenAIProvider` is lazy-imported so the package stays usable
without the `openai` SDK installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool" | "system"
    content: str


@dataclass
class Completion:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    # each item: {"name": str, "arguments": dict}


def _normalize_math(text: str) -> str:
    """Map Chinese operators to symbols so the offline provider can find an
    arithmetic expression."""
    text = (
        text.replace("加上", "+")
        .replace("加", "+")
        .replace("减去", "-")
        .replace("减", "-")
        .replace("乘以", "*")
        .replace("乘", "*")
        .replace("除以", "/")
        .replace("除", "/")
    )
    return text


class FakeProvider:
    """Deterministic, offline provider for tests and local demos.

    Behaviour is intentionally simple but *real*:
      - If the user message contains an arithmetic request, emit a
        `calculator` tool call.
      - If the last message is a tool result, return a final answer that
        quotes the result.
      - Otherwise return a neutral answer.
    """

    def complete(self, messages, tools=None, max_tokens=512, temperature=0.0) -> Completion:
        if messages and messages[-1].role == "tool":
            return Completion(content=f"计算结果是 {messages[-1].content.strip()}。")

        text = ""
        for m in reversed(messages):
            if m.role == "user":
                text = m.content
                break

        norm = _normalize_math(text)
        match = re.search(r"(\d+\s*[+\-*/]\s*\d+)", norm)
        if match:
            expr = match.group(1).replace(" ", "")
            return Completion(
                content="",
                tool_calls=[{"name": "calculator", "arguments": {"expression": expr}}],
            )
        return Completion(content="这是离线 FakeProvider 的回复（未配置真实模型）。")


class OpenAIProvider:
    """Real provider backed by any OpenAI-compatible chat completion API."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI  # lazy import

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages, tools=None, max_tokens=512, temperature=0.0) -> Completion:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            payload["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {"name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}
                )
        return Completion(content=msg.content or "", tool_calls=tool_calls)
