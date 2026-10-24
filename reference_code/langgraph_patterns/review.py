"""A bounded revision loop; exhaustion and adapter failures never approve."""
from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class AdapterError(Exception):
    """An expected reviewer/repair adapter failure, distinct from a code bug."""


class ReviewState(TypedDict):
    draft: dict[str, str]
    findings: list[str]
    attempts: int
    max_attempts: int
    status: str
    accepted: bool
    reason: str


def required_fields(draft: dict[str, str]) -> list[str]:
    """A small independent contract, not a general document-quality judge."""
    return [key for key in ('version', 'owner', 'rollback')
            if not isinstance(draft.get(key), str) or not draft[key].strip()]


def fixture_repair(draft: dict[str, str], findings: list[str]) -> dict[str, str]:
    """Fixed offline proposal. It only knows how to supply a rollback field."""
    repaired = dict(draft)
    if 'rollback' in findings:
        repaired['rollback'] = 'restore the previous release snapshot'
    return repaired


def route_review(state: ReviewState) -> Literal['accept', 'repair', 'exhausted', 'fail']:
    if state['status'] == 'failed':
        return 'fail'
    if not state['findings']:
        return 'accept'
    if state['attempts'] >= state['max_attempts']:
        return 'exhausted'
    return 'repair'


def accept_review(state: ReviewState) -> dict:
    return {'status': 'accepted', 'accepted': True, 'reason': 'all required fields passed'}


def exhaust_review(state: ReviewState) -> dict:
    return {'status': 'budget_exhausted', 'accepted': False,
            'reason': 'review attempt budget exhausted with unresolved findings'}


def retain_failure(state: ReviewState) -> dict:
    return {'status': 'failed', 'accepted': False}


def build_review_graph(*, reviewer: Callable = required_fields,
                       repairer: Callable = fixture_repair, checkpointer=None):
    def analyze(state: ReviewState) -> dict:
        attempt = state['attempts'] + 1
        try:
            findings = reviewer(dict(state['draft']))
        except AdapterError as exc:
            return {'attempts': attempt, 'status': 'failed', 'reason': str(exc)}
        if not isinstance(findings, list) or any(not isinstance(item, str) or not item for item in findings):
            return {'attempts': attempt, 'status': 'failed', 'reason': 'invalid reviewer response'}
        return {'findings': findings, 'attempts': attempt, 'status': 'running'}

    def repair(state: ReviewState) -> dict:
        try:
            draft = repairer(dict(state['draft']), list(state['findings']))
        except AdapterError as exc:
            return {'status': 'failed', 'reason': str(exc)}
        if not isinstance(draft, dict):
            return {'status': 'failed', 'reason': 'invalid repair response'}
        return {'draft': draft}

    def route_repair(state: ReviewState) -> Literal['analyze', 'fail']:
        return 'fail' if state['status'] == 'failed' else 'analyze'

    builder = StateGraph(ReviewState)
    for name, node in (('analyze', analyze), ('repair', repair), ('accept', accept_review),
                       ('exhausted', exhaust_review), ('fail', retain_failure)):
        builder.add_node(name, node)
    builder.add_edge(START, 'analyze')
    builder.add_conditional_edges('analyze', route_review,
                                  {name: name for name in ('accept', 'repair', 'exhausted', 'fail')})
    builder.add_conditional_edges('repair', route_repair, {'analyze': 'analyze', 'fail': 'fail'})
    for name in ('accept', 'exhausted', 'fail'):
        builder.add_edge(name, END)
    return builder.compile(checkpointer=checkpointer)


def review_input(draft: dict[str, str], max_attempts: int = 3) -> ReviewState:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError('max_attempts must be a positive integer')
    return {'draft': dict(draft), 'findings': [], 'attempts': 0, 'max_attempts': max_attempts,
            'status': 'running', 'accepted': False, 'reason': ''}


def run_review(draft: dict[str, str], *, max_attempts: int = 3, reviewer: Callable = required_fields,
               repairer: Callable = fixture_repair, recursion_limit: int | None = None) -> ReviewState:
    initial = review_input(draft, max_attempts)
    # A graph super-step limit is a backstop, separate from the business budget.
    limit = recursion_limit if recursion_limit is not None else 2 * max_attempts + 3
    return build_review_graph(reviewer=reviewer, repairer=repairer).invoke(
        initial, {'recursion_limit': limit})
