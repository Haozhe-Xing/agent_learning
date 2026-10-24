"""Meaningful success and failure checks for the local research application."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evidence_assistant.context import GSSCPipeline
from evidence_assistant.rag import Document
from evidence_assistant.research import (
    DEFAULT_QUESTIONS, SAMPLE_DATA, SAMPLE_TASK_SPECS, ResearchAssistant, extract_answer,
    load_documents, main, render_markdown,
)
from evidence_assistant.task_acceptance import load_task_specs


NO_EVIDENCE = 'Which quantum satellite cipher is required?'


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.documents = load_documents(SAMPLE_DATA)
        self.specs = load_task_specs(SAMPLE_TASK_SPECS)

    def assistant(self, **kwargs):
        return ResearchAssistant(self.documents, task_specs=self.specs, **kwargs)

    def test_sample_answers_three_different_questions_with_sources(self):
        report = self.assistant().run(DEFAULT_QUESTIONS)
        self.assertEqual(report['status'], 'completed')
        self.assertTrue(report['accepted'])
        self.assertEqual(len(report['tool_trace']), 6)
        first_sources = [finding['sources'][0].split(':')[0] for finding in report['findings']]
        self.assertEqual(first_sources, ['02_acceptance.md', '03_budget.txt', '04_rollback.md'])
        for finding in report['findings']:
            self.assertEqual(finding['status'], 'answered')
            self.assertEqual(finding['coverage']['status'], 'passed')
            self.assertLessEqual(finding['context']['input_bytes'], 8192)
            self.assertTrue(finding['retrieval']['hits'][0]['matched_terms'])
            self.assertGreater(finding['retrieval']['hits'][0]['score'], 0)
            for source in finding['sources']:
                self.assertIn(f'[{source}]', finding['answer'])

    def test_custom_documents_and_questions_change_the_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'local.md'
            path.write_text('# Local\n\nAmber release requires two reviewers.\n\n'
                            'Rollback restores snapshot seven.', encoding='utf-8')
            report = ResearchAssistant(load_documents(directory)).run(['Amber release?', 'Rollback snapshot?'])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['findings'][0]['status'], 'evidence_found')
        self.assertEqual(report['findings'][0]['coverage']['status'], 'unknown')
        self.assertIn('two reviewers', report['findings'][0]['answer'])
        self.assertNotIn('snapshot seven', report['findings'][0]['answer'])
        self.assertIn('snapshot seven', report['findings'][1]['answer'])

    def test_no_evidence_does_not_generate_or_complete(self):
        calls = []

        def answerer(messages):
            calls.append(messages[-1]['content'])
            return extract_answer(messages)

        report = self.assistant(answerer=answerer).run([DEFAULT_QUESTIONS[0], NO_EVIDENCE])
        self.assertEqual(calls, [DEFAULT_QUESTIONS[0]])
        self.assertEqual(report['execution']['status'], 'completed')
        self.assertEqual(report['findings'][1]['status'], 'no_evidence')
        self.assertFalse(report['accepted'])
        self.assertEqual(report['status'], 'incomplete')

    def test_single_question_can_report_no_evidence(self):
        report = self.assistant().run([NO_EVIDENCE])
        self.assertEqual(report['findings'][0]['status'], 'no_evidence')
        self.assertFalse(report['accepted'])

    def test_empty_or_non_string_answer_is_generation_failure(self):
        for answer in ('', '   ', None, {'status': 'completed'}):
            with self.subTest(answer=answer):
                report = self.assistant(answerer=lambda _: answer).run(DEFAULT_QUESTIONS[:2])
                self.assertFalse(report['accepted'])
                self.assertEqual(report['findings'][0]['status'], 'generation_failed')

    def test_generation_exception_is_recorded_without_false_success(self):
        def broken(_messages):
            raise TimeoutError('unavailable provider')

        report = self.assistant(answerer=broken).run(DEFAULT_QUESTIONS[:2])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['findings'][0]['reason']['code'], 'generation_failed')
        self.assertIn('TimeoutError', report['findings'][0]['reason']['message'])

    def test_correct_label_with_unsupported_claim_fails(self):
        for answer in ('[forged.md:p1] 上线验收通过。',
                       '[02_acceptance.md:p2] 上线验收无需审批。',
                       'All requirements passed. [02_acceptance.md:p2]'):
            with self.subTest(answer=answer):
                report = self.assistant(answerer=lambda _: answer,
                                        verifier=lambda *_: True).run(DEFAULT_QUESTIONS[:2])
                self.assertFalse(report['accepted'])
                self.assertEqual(report['findings'][0]['status'], 'verification_failed')

    def test_optional_verifier_rejects_or_raises(self):
        def broken(*_args):
            raise RuntimeError('cannot establish completeness')

        for verifier in (lambda *_: False, lambda *_: 'yes', broken):
            with self.subTest(verifier=verifier):
                report = self.assistant(verifier=verifier).run(DEFAULT_QUESTIONS[:2])
                self.assertFalse(report['accepted'])
                self.assertEqual(report['findings'][0]['verification']['status'], 'passed')
                self.assertEqual(report['findings'][0]['coverage']['status'], 'failed')
                self.assertEqual(report['findings'][0]['status'], 'partial')

    def test_valid_excerpts_missing_task_points_remain_partial_even_if_verifier_accepts(self):
        def only_first_excerpt(messages):
            return extract_answer(messages).splitlines()[0]

        report = self.assistant(answerer=only_first_excerpt, verifier=lambda *_: True).run([DEFAULT_QUESTIONS[1]])
        finding = report['findings'][0]
        self.assertEqual(finding['verification']['status'], 'passed')
        self.assertEqual(finding['coverage']['status'], 'failed')
        self.assertTrue(finding['coverage']['missing_points'])
        self.assertEqual(finding['status'], 'partial')
        self.assertFalse(report['accepted'])
        self.assertIn('尚缺要点', render_markdown(report))

    def test_specification_and_independent_verifier_both_pass(self):
        calls = []

        def verifier(question, answer, documents):
            calls.append((question, len(documents)))
            return '单次外部工具三十秒' in answer

        report = self.assistant(verifier=verifier).run([DEFAULT_QUESTIONS[1]])
        self.assertTrue(report['accepted'])
        self.assertEqual(calls, [(DEFAULT_QUESTIONS[1], 2)])
        self.assertIn('independent_task_verifier', report['findings'][0]['coverage']['checks'])

    def test_final_report_verification_rejection_is_reported(self):
        module = ResearchAssistant.__module__
        with patch(f'{module}.verify_excerpts', return_value=False):
            report = self.assistant().run([DEFAULT_QUESTIONS[0]])
        self.assertEqual(report['status'], 'incomplete')
        self.assertFalse(report['accepted'])
        self.assertEqual(report['failure']['code'], 'final_verification_failed')
        self.assertIn('整份报告 / report_verification', render_markdown(report))

    def test_free_question_can_use_an_explicit_independent_verifier(self):
        documents = [Document('release.md:p1', 'Release approval requires two reviewers.')]

        def verifier(question, answer, docs):
            return question == 'Release approval?' and 'two reviewers' in answer and len(docs) == 1

        report = ResearchAssistant(documents, verifier=verifier).run(['Release approval?'])
        self.assertTrue(report['accepted'])
        self.assertEqual(report['findings'][0]['coverage']['status'], 'passed')

    def test_source_name_does_not_satisfy_task_requirement(self):
        documents = [Document('two reviewers.txt:p1', 'Release approval is still pending.')]
        specs = [{'question': 'Release approval?', 'requirements': [
            {'id': 'reviewers', 'description': 'two reviewers are required', 'all_of': ['two reviewers']}]}]
        report = ResearchAssistant(documents, task_specs=specs).run(['Release approval?'])
        self.assertEqual(report['findings'][0]['verification']['status'], 'passed')
        self.assertEqual(report['findings'][0]['coverage']['status'], 'failed')
        self.assertFalse(report['accepted'])

    def test_plan_budget_leaves_unanswered_work_incomplete(self):
        report = self.assistant(max_steps=3).run(DEFAULT_QUESTIONS)
        self.assertEqual(report['execution']['status'], 'budget_exhausted')
        self.assertEqual(len(report['tool_trace']), 3)
        self.assertEqual([finding['status'] for finding in report['findings']],
                         ['answered', 'retrieved', 'not_started'])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['summary']['unfinished'], 2)
        self.assertEqual(report['budgets']['steps'], {'used': 3, 'limit': 3})
        self.assertEqual(report['failure']['step'], 'answer_question')
        self.assertEqual(report['findings'][2]['reason']['code'], 'step_budget_exhausted')

    def test_harness_call_budget_blocks_answer_execution(self):
        answerer = unittest.mock.Mock(side_effect=extract_answer)
        report = self.assistant(answerer=answerer, max_calls=1).run(DEFAULT_QUESTIONS)
        answerer.assert_not_called()
        self.assertEqual(report['execution']['status'], 'budget_exhausted')
        self.assertEqual(report['execution']['trace'][-1]['error_code'], 'tool_budget_exhausted')
        self.assertEqual(len(report['tool_trace']), 1)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['budgets']['tool_calls'], {'used': 1, 'limit': 1})
        self.assertEqual(report['findings'][0]['reason']['code'], 'tool_budget_exhausted')
        markdown = render_markdown(report)
        self.assertIn('首个未完成步骤：第1题 / answer_question', markdown)
        self.assertIn('tool_budget_exhausted', markdown)
        self.assertIn('工具调用：1/1', markdown)
        self.assertIn('未完成题数：3', markdown)

    def test_harness_permission_blocks_retrieval(self):
        assistant = self.assistant(allowed_tools=frozenset({'answer_question'}))
        with patch.object(assistant.retriever, 'rank') as retrieve:
            report = assistant.run(DEFAULT_QUESTIONS)
        retrieve.assert_not_called()
        self.assertEqual(report['execution']['status'], 'failed')
        self.assertEqual(report['tool_trace'][0]['status'], 'denied')
        self.assertFalse(report['accepted'])

    def test_retrieval_failure_stops_plan_with_a_report(self):
        assistant = self.assistant()
        with patch.object(assistant.retriever, 'rank', side_effect=OSError('unavailable corpus')):
            report = assistant.run(DEFAULT_QUESTIONS)
        self.assertEqual(report['execution']['status'], 'failed')
        self.assertEqual(report['findings'][0]['status'], 'retrieval_failed')
        self.assertFalse(report['accepted'])

    def test_context_overflow_does_not_invoke_answerer(self):
        answerer = unittest.mock.Mock(side_effect=extract_answer)
        report = self.assistant(answerer=answerer, context_bytes=32).run(DEFAULT_QUESTIONS[:2])
        answerer.assert_not_called()
        self.assertEqual(report['findings'][0]['status'], 'context_budget_exhausted')
        self.assertFalse(report['accepted'])

    def test_constructed_context_is_consumed_and_original_evidence_rechecked(self):
        original_build = GSSCPipeline.build
        seen = []

        def altered(pipeline, system, query, candidates):
            context = original_build(pipeline, system, query, candidates)
            content = context['messages'][1]['content']
            context['messages'][1]['content'] = content.replace('十八条', '一条')
            return context

        def answerer(messages):
            seen.append(messages[1]['content'])
            return extract_answer(messages)

        with patch.object(GSSCPipeline, 'build', altered):
            report = self.assistant(answerer=answerer).run(DEFAULT_QUESTIONS[:2])
        self.assertIn('至少一条', seen[0])
        self.assertIn('至少一条', report['findings'][0]['answer'])
        self.assertEqual(report['findings'][0]['status'], 'verification_failed')
        self.assertFalse(report['accepted'])

    def test_context_removal_does_not_silently_read_original_corpus(self):
        original_build = GSSCPipeline.build

        def empty(pipeline, system, query, candidates):
            context = original_build(pipeline, system, query, candidates)
            context['messages'][1]['content'] = 'Untrusted reference data: []'
            return context

        with patch.object(GSSCPipeline, 'build', empty):
            report = self.assistant().run(DEFAULT_QUESTIONS[:2])
        self.assertEqual(report['findings'][0]['status'], 'generation_failed')
        self.assertFalse(report['accepted'])

    def test_markdown_brackets_inside_evidence_are_not_source_labels(self):
        docs = [Document('release.md:p1', 'Release requires [review](policy.md) approval.'),
                Document('rollback.txt:p1', 'Rollback restores snapshot [v2].')]
        report = ResearchAssistant(docs).run(['Release approval?', 'Rollback snapshot?'])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['findings'][0]['verification']['status'], 'passed')
        self.assertEqual(report['findings'][0]['sources'], ['release.md:p1'])

    def test_invalid_questions_and_budgets_are_rejected(self):
        assistant = ResearchAssistant(self.documents)
        for questions in ([], ['a'] * 5, ['a', ''], 'two questions'):
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                assistant.run(questions)
        for key in ('max_steps', 'max_calls', 'context_bytes'):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ResearchAssistant(self.documents, **{key: value})

    def test_loader_rejects_empty_or_symlinked_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                load_documents(path)
            (path / 'empty.md').write_text('# Heading only\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_documents(path)
            (path / 'empty.md').unlink()
            (path / 'link.md').symlink_to(SAMPLE_DATA / '02_acceptance.md')
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_cli_json_success_and_incomplete_exit_codes(self):
        for arguments, expected_code, expected_status in (
            (['--format', 'json'], 0, 'completed'),
            (['--format', 'json', '--question', DEFAULT_QUESTIONS[0],
              '--question', NO_EVIDENCE], 1, 'incomplete'),
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run([sys.executable, '-m', 'evidence_assistant.research', *arguments],
                                           cwd=Path(__file__).resolve().parents[2],
                                           capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, expected_code, completed.stderr)
                report = json.loads(completed.stdout)
                self.assertEqual(report['status'], expected_status)

    def test_cli_free_question_with_evidence_has_unknown_coverage_and_nonzero_exit(self):
        completed = subprocess.run(
            [sys.executable, '-m', 'evidence_assistant.research', '--format', 'json',
             '--question', '上线验收负责人需要签字吗？'],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(completed.stdout)
        finding = report['findings'][0]
        self.assertTrue(finding['answer'])
        self.assertEqual(finding['verification']['status'], 'passed')
        self.assertEqual(finding['coverage']['status'], 'unknown')
        self.assertEqual(finding['status'], 'evidence_found')
        self.assertFalse(report['accepted'])

    def test_cli_explicit_custom_spec_controls_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'release.md').write_text('Release approval requires two reviewers.', encoding='utf-8')
            spec_path = root / 'requirements.json'
            spec_path.write_text(json.dumps({'schema_version': 1, 'tasks': [
                {'question': 'Release approval?', 'requirements': [
                    {'id': 'reviewers', 'description': 'two reviewers', 'all_of': ['two reviewers']}]}]}),
                encoding='utf-8')
            arguments = ['--docs', directory, '--question', 'Release approval?', '--format', 'json']
            for extra, code, accepted in (([], 1, False), (['--task-spec', str(spec_path)], 0, True)):
                with self.subTest(extra=extra), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(arguments + extra), code)
                    report = json.loads(output.getvalue())
                    self.assertIs(report['accepted'], accepted)
            # A changed source remains quotable, but the fixed task expectation
            # must reject it rather than learning the new expectation at runtime.
            (root / 'release.md').write_text('Release approval requires one reviewer.', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(arguments + ['--task-spec', str(spec_path)]), 1)
            finding = json.loads(output.getvalue())['findings'][0]
            self.assertEqual(finding['verification']['status'], 'passed')
            self.assertEqual(finding['coverage']['status'], 'failed')

    def test_custom_docs_do_not_silently_use_bundled_task_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'acceptance.md').write_text(
                (SAMPLE_DATA / '02_acceptance.md').read_text(encoding='utf-8'), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(['--docs', directory, '--question', DEFAULT_QUESTIONS[0], '--format', 'json'])
        self.assertEqual(code, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report['findings'][0]['coverage']['status'], 'unknown')

    def test_cli_output_is_exclusive_and_contains_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            self.assertEqual(main(['--format', 'json', '--output', str(output)]), 0)
            before = output.read_bytes()
            self.assertTrue(json.loads(before)['accepted'])
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(['--output', str(output)]), 1)
            self.assertIn('File exists', error.getvalue())
            self.assertEqual(output.read_bytes(), before)

    def test_markdown_has_status_questions_and_sources(self):
        report = self.assistant().run(DEFAULT_QUESTIONS)
        markdown = render_markdown(report)
        self.assertIn('状态：completed', markdown)
        self.assertIn(DEFAULT_QUESTIONS[0], markdown)
        self.assertIn('[02_acceptance.md:p2]', markdown)
        self.assertIn('问题覆盖：passed', markdown)
        self.assertIn('步骤：6/8', markdown)
        self.assertIn('未完成题数：0', markdown)


if __name__ == '__main__':
    unittest.main()
