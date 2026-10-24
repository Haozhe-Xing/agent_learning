"""Dynamic approval and a replay-safe local receipt, with no external actions.

The supplied policy is a local exercise, not identity authentication. The
SQLite receipt ledger is durable application data, not a graph checkpointer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class ApprovalState(TypedDict):
    request_id: str
    notice: str
    operation_digest: str
    decision: dict
    status: str
    reason: str
    receipt: dict


def operation_payload(state: ApprovalState) -> str:
    return json.dumps({'request_id': state['request_id'], 'notice': state['notice']},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def operation_digest(state: ApprovalState) -> str:
    return hashlib.sha256(operation_payload(state).encode('utf-8')).hexdigest()


class ReceiptLedger:
    """One local receipt per request ID; a changed payload is a conflict."""
    def __init__(self, path: str | Path = ':memory:'):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute('CREATE TABLE IF NOT EXISTS receipts '
                                '(request_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.connection.commit()

    def record(self, state: ApprovalState) -> dict:
        payload = operation_payload(state)
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO receipts VALUES (?, ?)',
                                    (state['request_id'], payload))
            saved = self.connection.execute('SELECT payload FROM receipts WHERE request_id = ?',
                                            (state['request_id'],)).fetchone()[0]
            if saved != payload:
                raise ValueError('request ID already belongs to a different operation')
        return {'request_id': state['request_id'], 'operation_digest': operation_digest(state)}

    def count(self) -> int:
        return self.connection.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def local_demo_policy(decision: dict, state: ApprovalState) -> bool:
    """A deterministic policy fixture; never use a claimed name as real auth."""
    return decision.get('reviewer') == 'reviewer-1'


def build_approval_graph(ledger: ReceiptLedger, *, checkpointer,
                         authorize: Callable[[dict, ApprovalState], bool],
                         on_approval_entry: Callable | None = None,
                         after_record: Callable | None = None):
    def prepare(state: ApprovalState) -> dict:
        if any(not isinstance(state.get(key), str) or not state[key].strip()
               for key in ('request_id', 'notice')):
            return {'status': 'failed', 'reason': 'request_id and notice must be nonempty'}
        return {'operation_digest': operation_digest(state), 'status': 'waiting', 'reason': ''}

    def approve(state: ApprovalState) -> dict:
        # This whole node starts again on resume. No business write goes here.
        if on_approval_entry is not None:
            on_approval_entry()
        decision = interrupt({'request_id': state['request_id'], 'notice': state['notice'],
                              'operation_digest': state['operation_digest']})
        if not isinstance(decision, dict) or type(decision.get('approved')) is not bool:
            return {'status': 'failed', 'reason': 'invalid decision payload'}
        if not decision['approved']:
            return {'status': 'denied', 'reason': 'reviewer declined the operation'}
        if decision.get('operation_digest') != state['operation_digest']:
            return {'status': 'failed', 'reason': 'decision does not match the pending operation'}
        if authorize(decision, state) is not True:
            return {'status': 'denied', 'reason': 'host authorization policy denied the decision'}
        return {'decision': decision, 'status': 'approved'}

    def record(state: ApprovalState) -> dict:
        if (state['decision'].get('approved') is not True
                or operation_digest(state) != state['operation_digest']
                or authorize(state['decision'], state) is not True):
            return {'status': 'failed', 'reason': 'operation or authorization changed before execution'}
        try:
            receipt = ledger.record(state)
        except ValueError as exc:
            return {'status': 'failed', 'reason': str(exc)}
        # Test hook: a crash after the committed effect but before checkpointing.
        if after_record is not None:
            after_record()
        return {'receipt': receipt, 'status': 'recorded', 'reason': 'local receipt recorded'}

    def after_prepare(state: ApprovalState) -> str:
        return 'approve' if state['status'] == 'waiting' else END

    def after_approve(state: ApprovalState) -> str:
        return 'record' if state['status'] == 'approved' else END

    builder = StateGraph(ApprovalState)
    builder.add_node('prepare', prepare)
    builder.add_node('approve', approve)
    builder.add_node('record', record)
    builder.add_edge(START, 'prepare')
    builder.add_conditional_edges('prepare', after_prepare, {'approve': 'approve', END: END})
    builder.add_conditional_edges('approve', after_approve, {'record': 'record', END: END})
    builder.add_edge('record', END)
    return builder.compile(checkpointer=checkpointer)


def approval_input(request_id: str = 'release-1', notice: str = 'Release 1.0 is ready') -> ApprovalState:
    return {'request_id': request_id, 'notice': notice, 'operation_digest': '', 'decision': {},
            'status': 'new', 'reason': '', 'receipt': {}}
