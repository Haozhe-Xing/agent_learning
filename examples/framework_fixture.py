"""Deterministic tool messages for framework integration, never an online model.

Both introductory framework examples reuse the same calculator and scripted
model so their execution mechanisms can be compared with fixed inputs.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from examples.chapter03.safe_math import safe_math


@tool
def calculate(expression: str) -> dict:
    """Evaluate a bounded arithmetic expression, such as sqrt(144) + 2 * 3.

    Only the arithmetic and function allowlist in chapter03.safe_math is allowed.
    This tool does not run Python statements, access files, or fetch facts.
    """
    try:
        return {'status': 'ok', 'expression': expression, 'value': safe_math(expression)}
    except (ValueError, SyntaxError, TypeError, ArithmeticError) as exc:
        return {'status': 'invalid_expression', 'expression': expression, 'error': str(exc)}


class FixedToolChatModel(BaseChatModel):
    """Request calculate with the user's raw expression, then echo its result.

    No natural-language interpretation is performed. repeat_calls is an explicit
    fault fixture that never voluntarily stops, used to exercise execution limits.
    """
    repeat_calls: bool = False

    @property
    def _llm_type(self) -> str:
        return 'book-fixed-calculator-model'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage) and not self.repeat_calls:
            message = AIMessage(content=str(messages[-1].content))
        else:
            expression = next(str(item.content) for item in reversed(messages)
                              if isinstance(item, HumanMessage))
            number = sum(isinstance(item, AIMessage) for item in messages) + 1
            message = AIMessage(content='', tool_calls=[{
                'name': 'calculate', 'args': {'expression': expression},
                'id': f'fixture-call-{number}',
            }])
        return ChatResult(generations=[ChatGeneration(message=message)])
