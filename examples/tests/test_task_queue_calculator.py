"""Execute the calculator boundaries printed in both task-queue chapters.

Only helper imports, tool functions and dispatch boundaries are loaded. Broker,
model, Celery and Temporal behavior are outside these offline tests.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from examples.chapter03.safe_math import safe_math
from markdown_source import split_source


def chapter_boundaries(language):
    path = ROOT / 'src' / language / 'chapter_22_deployment/08_task_queue_cost.md'
    functions = {}
    for label, line, code in split_source(path.read_text())[1]:
        if label != 'python':
            continue
        tree = ast.parse(code)
        helpers = [node for node in tree.body if (
            isinstance(node, ast.ImportFrom) and node.module == 'examples.chapter03.safe_math'
            or isinstance(node, ast.Import) and all(alias.name in {'json', 'logging'} for alias in node.names)
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'logger'
                                                   for target in node.targets))]
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in {'_tool_calculate', 'execute_tool', 'execute_tool_call',
                                 '_tool_search', '_tool_query_database'}:
                continue
            # Framework decorators are irrelevant to parsing/evaluating a tool
            # argument. Keep the actual function body and actual helper import.
            node.decorator_list = []
            namespace = {}
            exec(compile(ast.Module(body=[*helpers, node], type_ignores=[]),
                         f'{path}:{line}', 'exec'), namespace)
            functions[node.name] = namespace[node.name]
    if set(functions) != {'_tool_calculate', 'execute_tool', 'execute_tool_call',
                          '_tool_search', '_tool_query_database'}:
        raise AssertionError(f'{path}: missing tested calculator boundary')
    for function in functions.values():
        function.__globals__.update(functions)
    return functions


class TaskQueueCalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = {language: chapter_boundaries(language) for language in ('zh', 'en')}

    def boundaries(self):
        for language, functions in self.functions.items():
            yield language, 'celery', functions['_tool_calculate']
            activity = functions['execute_tool_call']

            def temporal(expression, activity=activity):
                return asyncio.run(activity('calculate', {'expression': expression}))['result']

            yield language, 'temporal', temporal

    def dispatchers(self):
        for language, functions in self.functions.items():
            execute = functions['execute_tool']

            def celery(name, arguments, execute=execute):
                task = SimpleNamespace(retry=Mock(side_effect=AssertionError('invalid arguments were retried')))
                return execute(task, name, arguments)['result']

            yield language, 'celery', celery
            activity = functions['execute_tool_call']

            def temporal(name, arguments, activity=activity):
                return asyncio.run(activity(name, arguments))['result']

            yield language, 'temporal', temporal

    def test_normal_arithmetic_reaches_each_real_boundary(self):
        for language, framework, calculate in self.boundaries():
            with self.subTest(language=language, framework=framework):
                self.assertEqual(calculate('sqrt(144) + 2 * 3'), 18)
                self.assertAlmostEqual(calculate('384400 * 1000 / 299792458'), 1.2822203819416758)
                self.assertEqual(calculate('round(2.56, 1)'), 2.6)

    def test_builtins_object_access_and_file_effects_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'escaped.txt'
            expressions = [
                '__import__("math").sqrt(144)',
                '().__class__.__bases__',
                'sqrt.__call__(4)',
                f'open({str(marker)!r}, "w").write("escaped")',
            ]
            for language, framework, calculate in self.boundaries():
                for expression in expressions:
                    with self.subTest(language=language, framework=framework, expression=expression):
                        with self.assertRaises(ValueError):
                            calculate(expression)
                        self.assertFalse(marker.exists())

    def test_non_arithmetic_inputs_do_not_become_results(self):
        for language, framework, calculate in self.boundaries():
            for expression in [True, None, {'expression': '2+2'}, 'True', '"text"',
                               '[1] * 100', 'sum(x for x in [1, 2])']:
                with self.subTest(language=language, framework=framework, expression=expression):
                    with self.assertRaises(ValueError):
                        calculate(expression)

    def test_length_complexity_exponent_and_numeric_limits(self):
        for language, framework, calculate in self.boundaries():
            for expression in ['9' * 257, '1+' * 40 + '1', '9**101', '1e101',
                               'exp(1000)', 'round(1, 101)']:
                with self.subTest(language=language, framework=framework, expression=expression):
                    with self.assertRaises((ValueError, OverflowError)):
                        calculate(expression)

    def test_invalid_math_is_an_error_not_a_successful_tool_result(self):
        for language, framework, calculate in self.boundaries():
            for expression in ['1/0', 'sqrt(-1)', '1+']:
                with self.subTest(language=language, framework=framework, expression=expression):
                    with self.assertRaises((ValueError, ArithmeticError, SyntaxError)):
                        calculate(expression)

    def test_both_chapters_import_the_shared_interpreter(self):
        for language, functions in self.functions.items():
            for name, function in functions.items():
                with self.subTest(language=language, function=name):
                    self.assertIs(function.__globals__['safe_math'], safe_math)

    def test_dispatch_accepts_json_objects_and_backend_dicts(self):
        for language, framework, dispatch in self.dispatchers():
            for arguments in [{'expression': 'sqrt(144) + 6'}, '{"expression": "sqrt(144) + 6"}']:
                with self.subTest(language=language, framework=framework, arguments=arguments):
                    self.assertEqual(dispatch('calculate', arguments), 18)

    def test_malformed_json_and_non_objects_are_permanent_errors(self):
        for language, framework, dispatch in self.dispatchers():
            for arguments in ['{', '', '[]', 'null', 'true', '1', '"2+2"', [], None, 42, True]:
                with self.subTest(language=language, framework=framework, arguments=arguments):
                    with self.assertRaises(ValueError) as caught:
                        dispatch('calculate', arguments)
                    # Temporal's retry policy compares concrete error names.
                    self.assertIs(type(caught.exception), ValueError)

    def test_non_finite_json_is_rejected_before_any_tool_dispatch(self):
        for language, framework, dispatch in self.dispatchers():
            for arguments in ['{"query": NaN}', '{"query": Infinity}', '{"query": 1e999}',
                              {'query': float('nan')}, {'query': object()}]:
                with self.subTest(language=language, framework=framework, arguments=arguments):
                    with self.assertRaises(ValueError):
                        dispatch('search', arguments)

    def test_unknown_tools_and_invalid_argument_shapes_do_not_retry(self):
        for language, framework, dispatch in self.dispatchers():
            for name in ['', None, [], 'not_registered']:
                with self.subTest(language=language, framework=framework, name=name):
                    with self.assertRaises(ValueError):
                        dispatch(name, {'expression': '2+2'})
            for arguments in [{}, {'expr': '2+2'}, {'expression': '__import__("math").sqrt(144)'},
                              {'expression': '1/0'}, {'expression': '1+'}]:
                with self.subTest(language=language, framework=framework, arguments=arguments):
                    with self.assertRaises((ValueError, TypeError, SyntaxError, ArithmeticError)):
                        dispatch('calculate', arguments)

    def test_no_other_python_block_reintroduces_eval_or_exec(self):
        for language in ('zh', 'en'):
            path = ROOT / 'src' / language / 'chapter_22_deployment/08_task_queue_cost.md'
            for label, line, code in split_source(path.read_text())[1]:
                if label != 'python':
                    continue
                calls = [node.func.id for node in ast.walk(ast.parse(code))
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id in {'eval', 'exec'}]
                with self.subTest(language=language, line=line):
                    self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
