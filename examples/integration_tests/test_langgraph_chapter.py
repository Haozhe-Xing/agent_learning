"""Real offline LangGraph contracts for chapters 11.2, 11.4 and 11.5."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command

from reference_code.langgraph_patterns.approval import (
    ReceiptLedger, approval_input, build_approval_graph, local_demo_policy)
from reference_code.langgraph_patterns.core import audit_input, build_audit_graph
from reference_code.langgraph_patterns.review import (
    AdapterError, review_input, run_review)


class OfflineGraphTests(unittest.TestCase):
    def setUp(self):
        tracing = patch.dict('os.environ', {'LANGSMITH_TRACING': 'false', 'LANGCHAIN_TRACING_V2': 'false'})
        tracing.start()
        self.addCleanup(tracing.stop)
        self.connect = patch.object(socket.socket, 'connect', side_effect=AssertionError('network disabled'))
        self.create_connection = patch.object(socket, 'create_connection', side_effect=AssertionError('network disabled'))
        self.connect.start()
        self.create_connection.start()
        self.addCleanup(self.connect.stop)
        self.addCleanup(self.create_connection.stop)


class StateAndReducerTests(OfflineGraphTests):
    def test_parallel_checks_join_and_keep_both_updates(self):
        draft = {'version': '1.0', 'owner': 'team', 'rollback': 'restore snapshot'}
        original = copy.deepcopy(draft)
        result = build_audit_graph().invoke(audit_input(draft))
        self.assertTrue(result['accepted'])
        self.assertEqual({item['name'] for item in result['checks']}, {'metadata', 'rollback'})
        self.assertEqual(len(result['checks']), 2)
        self.assertEqual(draft, original)

    def test_failed_parallel_check_cannot_approve(self):
        result = build_audit_graph().invoke(audit_input({'version': '1.0', 'owner': 'team'}))
        self.assertFalse(result['accepted'])
        self.assertEqual(next(item['passed'] for item in result['checks'] if item['name'] == 'rollback'), False)

    def test_parallel_writers_without_reducer_raise_instead_of_last_writer_winning(self):
        from typing import TypedDict
        class State(TypedDict):
            value: str
        def left(_state):
            return {'value': 'left'}
        def right(_state):
            return {'value': 'right'}
        builder = StateGraph(State)
        builder.add_node('left', left)
        builder.add_node('right', right)
        builder.add_edge(START, 'left')
        builder.add_edge(START, 'right')
        builder.add_edge('left', END)
        builder.add_edge('right', END)
        with self.assertRaises(InvalidUpdateError):
            builder.compile().invoke({'value': ''})

    def test_add_messages_replaces_same_id_but_preserves_new_ids(self):
        messages = add_messages([HumanMessage(content='draft', id='1')],
                                [HumanMessage(content='corrected', id='1'),
                                 HumanMessage(content='next', id='2')])
        self.assertEqual([(item.id, item.content) for item in messages], [('1', 'corrected'), ('2', 'next')])

    def test_checkpoint_needs_thread_and_new_saver_does_not_restore_it(self):
        saver = InMemorySaver()
        self.assertIs(MemorySaver, InMemorySaver)
        app = build_audit_graph(checkpointer=saver)
        with self.assertRaises(ValueError):
            app.invoke(audit_input({}))
        config = {'configurable': {'thread_id': 'audit-1'}}
        app.invoke(audit_input({'version': '1.0'}), config)
        self.assertEqual(len(build_audit_graph(checkpointer=saver).get_state(config).values['checks']), 2)
        self.assertEqual(build_audit_graph(checkpointer=InMemorySaver()).get_state(config).values, {})


class BoundedReviewTests(OfflineGraphTests):
    def test_fixed_repair_is_rechecked_before_acceptance(self):
        result = run_review({'version': '1.0', 'owner': 'team'})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['findings'], [])

    def test_pass_on_last_allowed_review_is_accepted(self):
        result = run_review({'version': '1.0', 'owner': 'team', 'rollback': 'restore'}, max_attempts=1)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['attempts'], 1)

    def test_unresolved_findings_exhaust_budget_without_approval(self):
        repairs = []
        def no_repair(draft, findings):
            repairs.append(list(findings))
            return draft
        result = run_review({'version': '1.0', 'owner': 'team'}, max_attempts=2, repairer=no_repair)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['findings'], ['rollback'])
        self.assertEqual(len(repairs), 1)

    def test_malformed_or_expected_failed_reviewer_does_not_become_empty_findings(self):
        def malformed(_draft):
            return None
        def unavailable(_draft):
            raise AdapterError('review unavailable')
        for reviewer in (malformed, unavailable):
            result = run_review({}, reviewer=reviewer)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['accepted'])
            self.assertTrue(result['reason'])

    def test_repair_failure_is_retained(self):
        def unavailable(_draft, _findings):
            raise AdapterError('repair unavailable')
        result = run_review({}, repairer=unavailable)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'repair unavailable')

    def test_unexpected_programming_error_propagates(self):
        def broken(_draft):
            raise RuntimeError('bug in reviewer')
        with self.assertRaisesRegex(RuntimeError, 'bug in reviewer'):
            run_review({}, reviewer=broken)

    def test_graph_superstep_limit_is_an_exception_not_a_business_approval(self):
        with self.assertRaises(GraphRecursionError):
            run_review({}, recursion_limit=1)

    def test_nonpositive_attempt_budget_is_rejected(self):
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                review_input({}, value)


class ApprovalTests(OfflineGraphTests):
    def setUp(self):
        super().setUp()
        self.ledger = ReceiptLedger()
        self.addCleanup(self.ledger.close)
        self.saver = InMemorySaver()
        self.app = build_approval_graph(self.ledger, checkpointer=self.saver, authorize=local_demo_policy)
        self.config = {'configurable': {'thread_id': 'approval-1'}, 'recursion_limit': 8}

    def pause(self, app=None, config=None):
        paused = (app or self.app).invoke(approval_input(), config or self.config)
        self.assertEqual(paused['status'], 'waiting')
        self.assertEqual(self.ledger.count(), 0)
        return paused['__interrupt__'][0].value

    def decision(self, pending, **updates):
        return {'approved': True, 'reviewer': 'reviewer-1',
                'operation_digest': pending['operation_digest'], **updates}

    def test_interrupt_surfaces_operation_and_resumes_with_a_decision(self):
        pending = self.pause()
        self.assertEqual(self.app.get_state(self.config).next, ('approve',))
        result = self.app.invoke(Command(resume=self.decision(pending)), self.config)
        self.assertEqual(result['status'], 'recorded')
        self.assertEqual(self.ledger.count(), 1)
        self.assertEqual(self.app.get_state(self.config).next, ())

    def test_denial_bad_payload_unknown_reviewer_and_stale_digest_never_write(self):
        for index, updates, expected in (
                (0, {'approved': False}, 'denied'),
                (1, {'approved': 'yes'}, 'failed'),
                (2, {'reviewer': 'unknown'}, 'denied'),
                (3, {'operation_digest': 'stale'}, 'failed')):
            config = {'configurable': {'thread_id': f'case-{index}'}}
            pending = self.pause(config=config)
            result = self.app.invoke(Command(resume=self.decision(pending, **updates)), config)
            self.assertEqual(result['status'], expected)
            self.assertEqual(self.ledger.count(), 0)

    def test_resume_restarts_entire_approval_node(self):
        entries = []
        def observe_entry():
            entries.append('entered')
        app = build_approval_graph(self.ledger, checkpointer=self.saver, authorize=local_demo_policy,
                                   on_approval_entry=observe_entry)
        pending = self.pause(app=app)
        self.assertEqual(entries, ['entered'])
        app.invoke(Command(resume=self.decision(pending)), self.config)
        self.assertEqual(entries, ['entered', 'entered'])
        self.assertEqual(self.ledger.count(), 1)

    def test_nonboolean_authorization_never_passes_initial_approval(self):
        for index, response in enumerate((1, 'yes', {'allowed': True})):
            def authorize(_decision, _state, response=response):
                return response
            app = build_approval_graph(self.ledger, checkpointer=InMemorySaver(), authorize=authorize)
            config = {'configurable': {'thread_id': f'nonboolean-approval-{index}'}}
            pending = self.pause(app=app, config=config)
            result = app.invoke(Command(resume=self.decision(pending)), config)
            self.assertEqual(result['status'], 'denied')
            self.assertEqual(self.ledger.count(), 0)

    def test_execution_recheck_rejects_truthy_nonboolean_authorization(self):
        for index, response in enumerate((1, 'yes', {'allowed': True})):
            checks = []
            def authorize(_decision, _state, response=response):
                checks.append('checked')
                return True if len(checks) == 1 else response
            app = build_approval_graph(self.ledger, checkpointer=InMemorySaver(), authorize=authorize)
            config = {'configurable': {'thread_id': f'nonboolean-execution-{index}'}}
            pending = self.pause(app=app, config=config)
            result = app.invoke(Command(resume=self.decision(pending)), config)
            self.assertEqual(len(checks), 2)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(self.ledger.count(), 0)

    def test_recompiled_graph_with_same_saver_resumes_but_new_thread_has_no_checkpoint(self):
        pending = self.pause()
        other = {'configurable': {'thread_id': 'other'}}
        self.assertEqual(self.app.get_state(other).values, {})
        wrong_thread = self.app.invoke(Command(resume=self.decision(pending)), other)
        self.assertEqual(wrong_thread['status'], 'failed')
        self.assertEqual(self.ledger.count(), 0)
        rebuilt = build_approval_graph(self.ledger, checkpointer=self.saver, authorize=local_demo_policy)
        result = rebuilt.invoke(Command(resume=self.decision(pending)), self.config)
        self.assertEqual(result['status'], 'recorded')

    def test_editing_operation_after_approval_request_invalidates_execution(self):
        pending = self.pause()
        self.app.update_state(self.config, {'notice': 'a different operation'})
        result = self.app.invoke(Command(resume=self.decision(pending)), self.config)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ledger.count(), 0)

    def test_committed_effect_before_failed_checkpoint_is_not_duplicated_on_retry(self):
        calls = []
        def fail_once_after_commit():
            calls.append('record')
            if len(calls) == 1:
                raise RuntimeError('simulated failure after receipt commit')
        app = build_approval_graph(self.ledger, checkpointer=self.saver, authorize=local_demo_policy,
                                   after_record=fail_once_after_commit)
        pending = self.pause(app=app)
        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            app.invoke(Command(resume=self.decision(pending)), self.config)
        self.assertEqual(self.ledger.count(), 1)
        result = app.invoke(None, self.config)
        self.assertEqual(result['status'], 'recorded')
        self.assertEqual(self.ledger.count(), 1)
        self.assertEqual(len(calls), 2)

    def test_disk_receipt_survives_reopening_but_is_not_a_graph_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipts.sqlite'
            with ReceiptLedger(path) as ledger:
                ledger.record(approval_input())
            with ReceiptLedger(path) as ledger:
                ledger.record(approval_input())
                self.assertEqual(ledger.count(), 1)
                with self.assertRaisesRegex(ValueError, 'different operation'):
                    ledger.record(approval_input(notice='changed'))
                self.assertEqual(ledger.count(), 1)
        fresh = build_approval_graph(self.ledger, checkpointer=InMemorySaver(), authorize=local_demo_policy)
        self.assertEqual(fresh.get_state(self.config).values, {})


class ChapterEntryTests(OfflineGraphTests):
    def test_documented_module_entry_runs_from_repository_root(self):
        result = subprocess.run([sys.executable, '-m', 'reference_code.langgraph_patterns', 'core'],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['accepted'])

    def test_cli_success_and_failure_exit_codes(self):
        from reference_code.langgraph_patterns.__main__ import main
        for arguments, expected, status in (
                (['core'], 0, None),
                (['review'], 0, 'accepted'),
                (['review', '--no-repair', '--max-attempts', '2'], 1, 'budget_exhausted'),
                (['approval'], 1, 'denied'),
                (['approval', '--decision', 'approve'], 0, 'recorded')):
            with self.subTest(arguments=arguments), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(arguments), expected)
            result = json.loads(output.getvalue())
            if status:
                self.assertEqual(result['status'], status)

    def test_bilingual_python_blocks_run_offline(self):
        files = ('02_core_concepts.md', '04_conditional_routing.md', '05_human_in_the_loop.md')
        for language in ('zh', 'en'):
            for name in files:
                path = ROOT / 'src' / language / 'chapter_langgraph' / name
                for index, code in enumerate(re.findall(r'^```python\n(.*?)^```', path.read_text(), re.M | re.S)):
                    with self.subTest(language=language, page=name, block=index), \
                            contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(code, str(path), 'exec', dont_inherit=True),
                             {'__name__': '__chapter_example__'})


if __name__ == '__main__':
    unittest.main()
