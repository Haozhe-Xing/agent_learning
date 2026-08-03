"""FastAPI service that wraps the reference agent.

Key honesty fix vs. the old book draft: `/chat` returns the **real** agent
output (or an error), never a hard-coded "这是 Agent 的回复" string.
`/stream` streams the real answer token-by-token over SSE.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent
from ..provider import FakeProvider, OpenAIProvider
from ..security.guardrails import GuardrailError, InjectionGuard

app = FastAPI(title="reference-agent", version="0.1.0")
_guard = InjectionGuard()


def _build_agent() -> Agent:
    # Offline by default so the book's examples run without an API key.
    # Set AGENT_REAL=1 (and OPENAI_API_KEY) to use a real model in production.
    if os.getenv("AGENT_REAL") == "1":
        # OpenAIProvider is imported lazily so the app runs without the SDK.
        return Agent(
            OpenAIProvider(
                model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        )
    return Agent(FakeProvider())


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        _guard.check_input(req.message)
    except GuardrailError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    try:
        answer = _build_agent().run(req.message)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {"answer": answer}


@app.post("/stream")
def stream(req: ChatRequest):
    try:
        _guard.check_input(req.message)
    except GuardrailError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    answer = _build_agent().run(req.message)

    def event_gen():
        for token in answer.split(" "):
            yield f"data: {json.dumps({'token': token + ' '}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
