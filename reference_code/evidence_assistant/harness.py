"""Host-enforced tool registry, call budget, and independent completion check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class ToolBudgetError(RuntimeError):
    code = 'tool_budget_exhausted'


@dataclass
class Harness:
    tools: dict[str, Callable]
    allowed: frozenset[str]
    max_calls: int = 8
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.max_calls < 1:
            raise ValueError('max_calls must be positive')

    def invoke(self, tool: str, arguments: dict):
        if len(self.trace) >= self.max_calls:
            raise ToolBudgetError('tool call budget exhausted')
        event = {'tool': tool, 'arguments': arguments, 'status': 'pending'}
        self.trace.append(event)
        if tool not in self.allowed or tool not in self.tools:
            event['status'] = 'denied'
            raise PermissionError(f'tool not authorized: {tool}')
        try:
            result = self.tools[tool](**arguments)
            event.update(status='completed', result=result)
            return result
        except Exception:
            event['status'] = 'failed'
            raise

    def finish(self, artifact, verify: Callable) -> dict:
        try:
            accepted = bool(verify(artifact))
        except Exception:
            accepted = False
        return {'status': 'verified' if accepted else 'verification_failed',
                'artifact': artifact, 'trace': self.trace[:]}
