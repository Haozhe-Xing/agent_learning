"""Bounded plan/execute/replan loop with replaceable planning and execution."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: dict = field(default_factory=dict)


class PlanAndExecuteAgent:
    def __init__(self, planner: Callable, tools: dict[str, Callable],
                 replanner: Callable | None = None, max_steps: int = 8, max_replans: int = 2):
        if max_steps < 1 or max_replans < 0:
            raise ValueError('invalid execution budgets')
        self.planner, self.tools, self.replanner = planner, tools, replanner
        self.max_steps, self.max_replans = max_steps, max_replans

    def run(self, task: str) -> dict:
        pending = deque(self.planner(task))
        trace, replans = [], 0
        if not pending:
            return {'status': 'failed', 'trace': [], 'replans': 0, 'error': 'empty plan'}
        while pending and len(trace) < self.max_steps:
            step = pending.popleft()
            try:
                if step.tool not in self.tools:
                    raise PermissionError(f'unregistered tool: {step.tool}')
                result = self.tools[step.tool](**step.arguments)
                trace.append({'tool': step.tool, 'arguments': step.arguments, 'result': result, 'ok': True})
            except (ValueError, TypeError, ArithmeticError, PermissionError, OSError, RuntimeError) as exc:
                code = getattr(exc, 'code', 'tool_failed')
                trace.append({'tool': step.tool, 'arguments': step.arguments, 'error': str(exc),
                              'error_code': code, 'ok': False})
                if code == 'tool_budget_exhausted':
                    return {'status': 'budget_exhausted', 'trace': trace, 'replans': replans}
                if self.replanner is None or replans >= self.max_replans:
                    return {'status': 'failed', 'trace': trace, 'replans': replans}
                replans += 1
                # Replace the queue; reassigning a list inside `for enumerate(plan)`
                # would leave the original iterator running the stale plan.
                pending = deque(self.replanner(task, trace[:], list(pending)))
                if not pending:
                    return {'status': 'failed', 'trace': trace, 'replans': replans, 'error': 'empty replacement plan'}
        return {'status': 'budget_exhausted' if pending else 'completed',
                'trace': trace, 'replans': replans}
