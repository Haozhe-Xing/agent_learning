"""Runtime guardrails with FAIL-CLOSED defaults.

The previous book draft shipped a "LLM reviewer" that returned "safe" when
parsing failed — i.e. fail-OPEN. That is the wrong default for security.
Here every check raises on uncertainty instead of letting input through.
"""

from __future__ import annotations

import re


# Patterns are intentionally conservative. When a pattern matches we block;
# we do NOT try to be clever about borderline cases.
INJECTION_PATTERNS = [
    r"忽略.{0,20}(指令|规则|提示|系统|prompt|要求)",
    r"ignore (the )?(previous|above|prior|system|all) (instructions|prompt|rules)",
    r"disregard (the )?(previous|above|prior|system)",
    r"你(现在)?(是|变成|扮演).{0,12}(dAN|开发者模式|developer ?mode|越狱)",
    r"system prompt",
    r"reveal your (system )?prompt",
    r"打印.{0,20}提示词",
    r"把(你|系统).{0,8}(提示|prompt)告诉我",
]


class GuardrailError(Exception):
    pass


class InjectionGuard:
    def __init__(self, patterns: list[str] | None = None, fail_closed: bool = True):
        self.patterns = patterns or INJECTION_PATTERNS
        self.fail_closed = fail_closed

    def check_input(self, text: str) -> str:
        """Return the text if safe; raise GuardrailError if blocked.

        fail_closed=True means uncertain input is blocked, never passed.
        """
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise GuardrailError(f"blocked input matching /{pattern}/")
        return text

    def check_tool_result(self, text: str) -> str:
        """Tool outputs can carry injected instructions (data exfiltration)."""
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise GuardrailError("tool result contained injection-like content")
        return text
