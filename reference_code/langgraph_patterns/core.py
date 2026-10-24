"""Parallel release checks: partial updates, reducers and an explicit join."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class Check(TypedDict):
    name: str
    passed: bool


class AuditState(TypedDict):
    draft: dict[str, str]
    checks: Annotated[list[Check], operator.add]
    accepted: bool


def check_metadata(state: AuditState) -> dict:
    draft = state['draft']
    passed = all(isinstance(draft.get(key), str) and draft[key].strip()
                 for key in ('version', 'owner'))
    return {'checks': [{'name': 'metadata', 'passed': bool(passed)}]}


def check_rollback(state: AuditState) -> dict:
    rollback = state['draft'].get('rollback')
    return {'checks': [{'name': 'rollback', 'passed': bool(
        isinstance(rollback, str) and rollback.strip())}]}


def summarize(state: AuditState) -> dict:
    # Names and outcomes matter; parallel list order is not a business rule.
    checks = {check['name']: check['passed'] for check in state['checks']}
    return {'accepted': set(checks) == {'metadata', 'rollback'} and all(checks.values())}


def build_audit_graph(*, checkpointer=None):
    builder = StateGraph(AuditState)
    builder.add_node('metadata', check_metadata)
    builder.add_node('rollback', check_rollback)
    builder.add_node('summarize', summarize)
    builder.add_edge(START, 'metadata')
    builder.add_edge(START, 'rollback')
    builder.add_edge(['metadata', 'rollback'], 'summarize')
    builder.add_edge('summarize', END)
    return builder.compile(checkpointer=checkpointer)


def audit_input(draft: dict[str, str]) -> AuditState:
    return {'draft': dict(draft), 'checks': [], 'accepted': False}
