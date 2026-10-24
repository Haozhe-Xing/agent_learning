"""Executable planning contracts and bilingual teaching examples, offline."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import json
import re
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference_code.planning_patterns import (
    FACTS, GOALS, Node, OmitRollbackOnce, check_brief, default_plan, run_plan,
    run_react, validate_plan, write_complete)


def omit_rollback(facts, feedback):
    result = deepcopy(facts)
    result.pop('rollback')
    return result


class OfflineCase(unittest.TestCase):
    def setUp(self):
        for target in ('socket.socket.connect', 'socket.create_connection'):
            guard = patch(target, side_effect=AssertionError('network disabled'))
            guard.start()
            self.addCleanup(guard.stop)


class PlanTests(OfflineCase):
    def test_reverse_list_is_scheduled_by_dependencies(self):
        self.assertEqual([n.id for n in validate_plan(list(reversed(default_plan())))], ['read', 'brief'])
        result = run_plan(nodes=list(reversed(default_plan())))
        self.assertTrue(result['accepted'])
        self.assertEqual([event['id'] for event in result['trace']], ['read', 'brief'])

    def test_invalid_dependency_and_goal_plans_never_execute(self):
        plans = [[], [Node('a', 'load'), Node('a', 'load')],
                 [Node('a', 'compose', ('missing',), GOALS)],
                 [Node('a', 'load', ('b',)), Node('b', 'compose', ('a',), GOALS)],
                 [Node('a', 'delete_files', (), GOALS)],
                 [Node('a', 'load', (), GOALS)],
                 [Node('a', 'load'), Node('b', 'compose', (), GOALS)],
                 [Node('a', 'load'), Node('b', 'compose', ('a',), ('budget',))]]
        for plan in plans:
            with self.subTest(plan=plan):
                result = run_plan(nodes=plan)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['accepted'])
                self.assertEqual(result['trace'], [])

    def test_complete_brief_has_independently_expected_values_and_sources(self):
        result = run_plan(max_steps=2)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['candidate']['budget'], {'value': '30 units', 'source': 'budget-v1#release'})
        self.assertEqual(result['candidate']['rollback']['source'], 'rollback-v1#release')
        self.assertEqual(result['budgets']['steps_used'], 2)

    def test_compose_cannot_consume_an_unrelated_branch_snapshot(self):
        for dependencies, accepted in [(('A',), False), (('A', 'B'), True)]:
            reads = []
            def reader():
                branch = 'A' if not reads else 'B'
                reads.append(branch)
                facts = deepcopy(FACTS)
                for goal in GOALS:
                    facts[goal]['source'] = f'read-{branch}#{goal}'
                return facts
            nodes = [Node('A', 'load'), Node('B', 'load'),
                     Node('brief_A', 'compose', dependencies, GOALS)]
            with self.subTest(dependencies=dependencies):
                result = run_plan(nodes=nodes, reader=reader, max_steps=3, max_replans=0)
                self.assertEqual(result['accepted'], accepted)
                self.assertEqual(reads, ['A', 'B'])
                if accepted:
                    self.assertEqual(result['candidate']['budget']['source'], 'read-B#budget')
                else:
                    self.assertEqual(result['status'], 'failed')
                    self.assertIsNone(result['candidate'])
                    self.assertIn('declared load ancestor', result['trace'][-1]['error'])

    def test_declared_coverage_does_not_approve_a_missing_output_point(self):
        result = run_plan(writer=omit_rollback, max_replans=0)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['verification']['missing_points'], ['rollback'])

    def test_wrong_value_or_source_is_not_supported(self):
        for field, value in [('value', '300 units'), ('source', 'invented')]:
            candidate = deepcopy(FACTS)
            candidate['budget'][field] = value
            verdict = check_brief(candidate, FACTS)
            self.assertEqual(verdict['status'], 'failed')
            self.assertIn('budget', verdict['unsupported_points'])

    def test_feedback_revision_rechecks_original_contract(self):
        result = run_plan(writer=OmitRollbackOnce())
        self.assertTrue(result['accepted'])
        self.assertEqual(result['replans'], 1)
        self.assertEqual([event['status'] for event in result['trace']],
                         ['succeeded', 'failed', 'succeeded', 'succeeded'])

    def test_critic_self_approval_cannot_approve_stubborn_output(self):
        def approving_critic(snapshot):
            return {'passed': True, 'score': 10, 'suggestion': 'declare success'}
        result = run_plan(writer=omit_rollback, reflector=approving_critic)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['budgets']['steps_used'], 4)

    def test_replan_replaces_old_queue_and_invalidates_artifacts(self):
        reads = []
        def reader():
            reads.append('read')
            facts = deepcopy(FACTS)
            for goal in GOALS:
                facts[goal]['source'] = f'read-{len(reads)}#{goal}'
            return facts
        nodes = default_plan() + [Node('obsolete', 'load', ('brief',))]
        result = run_plan(nodes=nodes, reader=reader, writer=OmitRollbackOnce())
        self.assertTrue(result['accepted'])
        self.assertEqual(len(reads), 2)
        self.assertNotIn('obsolete', [event['id'] for event in result['trace']])
        self.assertEqual(result['candidate']['budget']['source'], 'read-2#budget')

    def test_unbounded_plan_iterator_is_rejected_without_consumption(self):
        consumed = []
        def infinite():
            while True:
                consumed.append(True)
                yield Node('x', 'load')
        result = run_plan(nodes=infinite())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(consumed, [])

    def test_invalid_replacement_does_not_resume_old_plan(self):
        result = run_plan(writer=omit_rollback, replanner=lambda snapshot: [])
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['trace']), 2)

    def test_permission_denial_does_not_try_an_alternative(self):
        replans = []
        def denied_reader():
            raise PermissionError('source not allowed')
        def replacement(snapshot):
            replans.append(True)
            return default_plan()
        result = run_plan(reader=denied_reader, replanner=replacement)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['trace']), 1)
        self.assertEqual(replans, [])

    def test_source_adapter_failure_is_not_empty_success(self):
        def unavailable():
            raise OSError('source unavailable')
        result = run_plan(reader=unavailable)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['verification']['status'], 'not_started')
        self.assertEqual(result['trace'][0]['error'], 'source unavailable')

    def test_step_budget_counts_failed_attempts_and_replans_together(self):
        result = run_plan(writer=OmitRollbackOnce(), max_steps=3)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['budgets']['steps_used'], 3)
        self.assertEqual(result['verification']['status'], 'not_started')

    def test_noninteger_or_unbounded_budgets_fail_before_tools(self):
        for options in ({'max_steps': 0}, {'max_steps': True}, {'max_steps': 1.5},
                        {'max_replans': -1}, {'max_replans': True}, {'max_replans': 100}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_plan(**options)

    def test_writer_cannot_mutate_original_observations_to_fit_false_output(self):
        def mutate(facts, feedback):
            facts['budget']['value'] = '300 units'
            return facts
        result = run_plan(writer=mutate, max_replans=0)
        self.assertFalse(result['accepted'])
        self.assertIn('budget', result['verification']['unsupported_points'])
        self.assertEqual(FACTS['budget']['value'], '30 units')

    def test_updated_source_does_not_rewrite_frozen_requirements(self):
        def updated():
            facts = deepcopy(FACTS)
            facts['budget'] = {'value': '40 units', 'source': 'budget-v2#release'}
            return facts
        result = run_plan(reader=updated, max_replans=0)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['verification']['unsupported_points'], [])
        self.assertEqual(result['verification']['wrong_requirements'], ['budget'])

    def test_rereading_after_a_verified_brief_invalidates_old_acceptance(self):
        nodes = default_plan() + [Node('reread', 'load', ('brief',))]
        result = run_plan(nodes=nodes)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['verification']['status'], 'not_started')
        self.assertIsNone(result['candidate'])

    def test_critic_error_is_not_replaced_with_a_success(self):
        def broken(snapshot):
            raise RuntimeError('critic unavailable')
        result = run_plan(writer=omit_rollback, reflector=broken)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['accepted'])
        self.assertIn('critic unavailable', result['reason'])


class ReactiveTests(OfflineCase):
    def test_action_observation_finish_has_separate_decision_and_call_counts(self):
        result = run_react()
        self.assertTrue(result['accepted'])
        self.assertEqual(result['budgets']['decisions_used'], 3)
        self.assertEqual(result['budgets']['steps_used'], 2)

    def test_malformed_or_unregistered_decisions_cannot_become_answers(self):
        for proposal in ('all done', {'action': 'delete'}, {'finish': 1},
                         {'action': 'load', 'finish': True}):
            with self.subTest(proposal=proposal):
                result = run_react(chooser=lambda state: proposal)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['accepted'])
                self.assertEqual(result['trace'], [])

    def test_finish_before_checking_is_incomplete(self):
        result = run_react(chooser=lambda state: {'finish': True})
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['accepted'])

    def test_tool_and_decision_budgets_are_separate(self):
        chooser = lambda state: {'action': 'load'}
        limited = run_react(chooser=chooser, max_calls=1)
        self.assertEqual(limited['budgets']['steps_used'], 1)
        self.assertEqual(limited['status'], 'budget_exhausted')
        decisions = run_react(chooser=chooser, max_decisions=1)
        self.assertEqual(decisions['budgets']['decisions_used'], 1)
        self.assertEqual(decisions['status'], 'budget_exhausted')

    def test_revision_consumes_actual_observation_feedback(self):
        result = run_react(writer=OmitRollbackOnce())
        self.assertTrue(result['accepted'])
        self.assertEqual(result['budgets']['steps_used'], 3)
        self.assertEqual(result['budgets']['decisions_used'], 4)

    def test_keyboard_interrupt_is_not_turned_into_a_result(self):
        def cancelled(snapshot):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_react(chooser=cancelled)


class ChapterTests(OfflineCase):
    def test_cli_success_failure_and_repair(self):
        from reference_code.planning_patterns.__main__ import main
        for mode, accepted in [('plan', True), ('react', True), ('revise', True), ('stubborn', False)]:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main([mode])
            self.assertEqual(code, 0 if accepted else 1)
            self.assertEqual(json.loads(output.getvalue())['accepted'], accepted)

    def test_real_module_entry(self):
        result = subprocess.run([sys.executable, '-m', 'reference_code.planning_patterns', 'revise'],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['replans'], 1)

    def test_bilingual_blocks_are_equivalent_and_executable(self):
        pages = ('01_how_agents_think', '02_react_framework', '03_task_decomposition',
                 '04_reflection', '07_plan_and_execute')
        for page in pages:
            blocks = {}
            for lang in ('zh', 'en'):
                path = ROOT / 'src' / lang / 'chapter_planning' / (page + '.md')
                blocks[lang] = re.findall(r'^```python\n(.*?)^```', path.read_text(), re.M | re.S)
                self.assertTrue(blocks[lang])
                for index, block in enumerate(blocks[lang]):
                    with self.subTest(page=page, lang=lang, block=index), contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(block, str(path), 'exec', dont_inherit=True), {})
            self.assertEqual(blocks['zh'], blocks['en'])


if __name__ == '__main__':
    unittest.main()
