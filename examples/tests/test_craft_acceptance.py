"""Check the published CRAFT interface without executing generated source.

The runner mocks below exercise host-side acceptance only. They do not implement
or prove any process/container isolation or resource limit.
"""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from markdown_source import split_source

TOOL = {'function_name': 'add', 'code': 'def add(a, b): return a + b',
        'test_cases': [{'input': [1, 2], 'expected': 999}]}
CASES = [{'input': [1, 2], 'expected': 3}, {'input': [-1, 1], 'expected': 0}]


def chapter_class(language):
    path = ROOT / 'src' / language / 'chapter_skill/03_skill_learning.md'
    for label, line, code in split_source(path.read_text())[1]:
        if label == 'python' and 'class CRAFTSystem:' in code:
            namespace = {}
            exec(compile(code, f'{path}:{line}', 'exec'), namespace)
            return namespace['CRAFTSystem'], ast.parse(code)
    raise AssertionError(f'{path}: missing CRAFT fragment')


class CraftAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = {language: chapter_class(language) for language in ('zh', 'en')}

    def test_no_runner_rejects_without_executing_candidate_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'candidate-side-effect'
            candidate = {**TOOL, 'code': f'open({str(marker)!r}, "w").write("escaped")'}
            for language, (craft, _) in self.classes.items():
                with self.subTest(language=language):
                    self.assertFalse(craft(None, acceptance_cases=CASES)._verify_tool(candidate))
                    self.assertFalse(marker.exists())

    def test_model_self_tests_cannot_replace_independent_cases(self):
        for language, (craft, _) in self.classes.items():
            runner = Mock()
            runner.run_candidate.return_value = [999]
            with self.subTest(language=language):
                self.assertFalse(craft(None, runner=runner)._verify_tool(TOOL))
                runner.run_candidate.assert_not_called()

    def test_results_are_compared_with_host_expectations(self):
        for language, (craft, _) in self.classes.items():
            runner = Mock()
            runner.run_candidate.return_value = [999, 999]
            instance = craft(None, runner, deepcopy(CASES))
            with self.subTest(language=language):
                self.assertFalse(instance._verify_tool(TOOL))
                candidate = {**TOOL, 'test_cases': [{'input': [1, 2], 'expected': 3}]}
                self.assertFalse(instance._verify_tool(candidate))
                runner.run_candidate.return_value = [3, 0]
                self.assertTrue(instance._verify_tool(TOOL))

    def test_runner_gets_inputs_and_policy_but_not_expectations(self):
        for language, (craft, _) in self.classes.items():
            runner = Mock()
            runner.run_candidate.return_value = [3, 0]
            with self.subTest(language=language):
                self.assertTrue(craft(None, runner, deepcopy(CASES))._verify_tool(TOOL))
                arguments = runner.run_candidate.call_args.kwargs
                self.assertEqual(set(arguments), {'code', 'function_name', 'inputs', 'limits'})
                self.assertEqual(arguments['inputs'], [[1, 2], [-1, 1]])
                self.assertEqual(arguments['limits'], {'timeout_seconds': 2, 'cpu_seconds': 1,
                                                       'memory_mb': 128, 'output_bytes': 65536,
                                                       'network': False})

    def test_malformed_candidate_is_rejected_before_runner(self):
        invalid = [None, [], {}, {**TOOL, 'code': ''}, {**TOOL, 'code': 'x' * 4097},
                   {**TOOL, 'function_name': 'a.b'}, {**TOOL, 'function_name': 'for'},
                   {**TOOL, 'function_name': None}]
        for language, (craft, _) in self.classes.items():
            for candidate in invalid:
                with self.subTest(language=language, candidate=candidate):
                    runner = Mock()
                    self.assertFalse(craft(None, runner, CASES)._verify_tool(candidate))
                    runner.run_candidate.assert_not_called()

    def test_empty_or_malformed_cases_are_rejected_before_runner(self):
        invalid = [(), None, {'input': [1], 'expected': 1}, [{'input': [1]}],
                   [{'input': '1', 'expected': 1}], [{'input': [1], 'expected': float('nan')}],
                   [CASES[0]] * 33]
        for language, (craft, _) in self.classes.items():
            for cases in invalid:
                with self.subTest(language=language, cases=cases):
                    runner = Mock()
                    self.assertFalse(craft(None, runner, cases)._verify_tool(TOOL))
                    runner.run_candidate.assert_not_called()

    def test_success_flags_non_json_and_wrong_result_counts_do_not_pass(self):
        for language, (craft, _) in self.classes.items():
            for output in [True, {'status': 'passed'}, None, [], [3], [3, 0, 1],
                           [3, float('inf')], [3, object()], [3, False]]:
                with self.subTest(language=language, output=output):
                    runner = Mock()
                    runner.run_candidate.return_value = output
                    self.assertFalse(craft(None, runner, CASES)._verify_tool(TOOL))

    def test_runner_failure_is_rejection(self):
        for language, (craft, _) in self.classes.items():
            for error in [TimeoutError('runner timed out'), RuntimeError('runner unavailable')]:
                with self.subTest(language=language, error=type(error).__name__):
                    runner = Mock()
                    runner.run_candidate.side_effect = error
                    self.assertFalse(craft(None, runner, CASES)._verify_tool(TOOL))

    def test_rejected_candidate_never_reaches_storage_or_tool_use(self):
        for language, (craft, _) in self.classes.items():
            with self.subTest(language=language):
                instance = craft(None, acceptance_cases=CASES)
                instance._retrieve_tools = Mock(return_value=[])
                instance._create_tool = Mock(return_value=TOOL)
                instance._store_tool = Mock()
                instance._solve_with_tools = Mock()
                instance._direct_solve = Mock(return_value='No candidate approved')
                self.assertEqual(instance.solve('Add two numbers'), 'No candidate approved')
                instance._store_tool.assert_not_called()
                instance._solve_with_tools.assert_not_called()

    def test_fragment_contains_no_host_eval_or_exec(self):
        for language, (_, tree) in self.classes.items():
            calls = [node.func.id for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id in {'eval', 'exec', 'compile'}]
            with self.subTest(language=language):
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
