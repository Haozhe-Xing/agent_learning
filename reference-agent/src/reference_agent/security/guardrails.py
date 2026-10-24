"""Bounded regex screening, not a general prompt-injection safety decision.

A match blocks the request. A non-match only passes this lexical screen;
service authorization and downstream resource permissions remain separate.
Malformed/oversized input is rejected. There is no semantic reviewer here.
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
        # Keep the old keyword for compatible callers, without suggesting that
        # unknown attacks can be detected. Disabling validation is unsupported.
        if fail_closed is not True:
            raise ValueError("fail_closed must be True; this is a lexical screen")
        selected = INJECTION_PATTERNS if patterns is None else patterns
        self.patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in selected)

    def check_input(self, text: str) -> str:
        """Return a bounded string with no pattern match, or raise.

        Passing this screen is not evidence of semantic safety or authority.
        Explicitly empty patterns disable lexical matching only, not bounds.
        """
        if type(text) is not str:
            raise GuardrailError("input must be text")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise GuardrailError("invalid input encoding") from exc
        if size > 16384:
            raise GuardrailError("input exceeds lexical screen limit")
        if any(pattern.search(text) for pattern in self.patterns):
            raise GuardrailError("input matched a lexical screening pattern")
        return text

    def check_tool_result(self, text: str) -> str:
        """Apply the same bounded lexical screen to untrusted tool text."""
        return self.check_input(text)
