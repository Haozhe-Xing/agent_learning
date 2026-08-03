from __future__ import annotations

from .provider import Message
from .tools import CalculatorTool, SearchTool, Tool

DEFAULT_TOOLS: list[Tool] = [CalculatorTool(), SearchTool()]


class Agent:
    """A minimal ReAct-style agent loop with real tool execution.

    The loop:
      1. Send conversation + tool specs to the provider.
      2. If the provider emits tool calls, execute them and append the
         results as `tool` messages.
      3. Repeat until the provider returns text with no tool calls, or the
         iteration budget is exhausted.
    """

    def __init__(self, provider, tools: list[Tool] | None = None, max_iterations: int = 5):
        self.provider = provider
        self.tools = {t.name: t for t in (tools or DEFAULT_TOOLS)}
        self.max_iterations = max_iterations

    def _tool_specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def run(self, user_input: str) -> str:
        messages = [Message("user", user_input)]
        for _ in range(self.max_iterations):
            completion = self.provider.complete(messages, tools=self._tool_specs())
            if not completion.tool_calls:
                return completion.content
            for tc in completion.tool_calls:
                tool = self.tools.get(tc["name"])
                if tool is None:
                    messages.append(Message("tool", f"unknown tool: {tc['name']}"))
                    continue
                try:
                    result = tool.run(**tc["arguments"])
                except Exception as exc:  # noqa: BLE001 - surfaced to the model
                    result = f"error: {exc}"
                messages.append(Message("tool", str(result)))
        # Budget exhausted: ask the model for a final summary.
        return self.provider.complete(messages, tools=self._tool_specs()).content
