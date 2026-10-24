"""Offline acceptance checks: retrieval hits do not establish task completion."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evidence_assistant.research_agent import ResearchAgent
from evidence_assistant.agentic_rag import AgenticRAG
from evidence_assistant.rag import Document, DocumentQAAgent, answer_accepted


POLICY = Document('policy', 'Returns are accepted within seven days.')


class AnswerAcceptanceTests(unittest.TestCase):
    def test_offline_baseline_passes_with_explicit_validation_scope(self):
        result = DocumentQAAgent([POLICY]).ask('returns')
        self.assertTrue(answer_accepted(result))
        self.assertEqual(result['sources'], ['policy'])
        self.assertEqual(result['retrieved_sources'], ['policy'])
        self.assertEqual(result['retrieval']['status'], 'succeeded')
        self.assertEqual(result['generation']['status'], 'succeeded')
        self.assertEqual(result['verification']['semantic_support'], 'not_checked')

    def test_empty_retrieval_does_not_generate(self):
        answerer = Mock()
        result = DocumentQAAgent([POLICY], answerer).ask('astronomy')
        self.assertEqual(result['status'], 'no_evidence')
        self.assertEqual(result['retrieval']['status'], 'empty')
        self.assertEqual(result['generation']['status'], 'not_started')
        self.assertEqual(result['verification']['status'], 'not_run')
        answerer.assert_not_called()

    def test_blank_and_non_string_outputs_fail_before_verification(self):
        for output in ['', ' \n\t ', None, 0, ['answer'],
                       {'status': 'failed', 'answer': 'Seven days [policy]'}]:
            with self.subTest(output=output):
                verifier = Mock(return_value=True)
                result = DocumentQAAgent([POLICY], lambda _: output, verifier).ask('returns')
                self.assertEqual(result['status'], 'generation_failed')
                self.assertEqual(result['answer'], '')
                self.assertEqual(result['sources'], [])
                self.assertEqual(result['generation']['status'], 'failed')
                self.assertEqual(result['verification']['status'], 'not_run')
                self.assertFalse(answer_accepted(result))
                verifier.assert_not_called()

    def test_provider_errors_are_identifiable_and_only_transient_errors_retry(self):
        for error, retryable in [(TimeoutError('slow'), True), (ConnectionError('offline'), True),
                                 (RuntimeError('broken'), False)]:
            with self.subTest(error=type(error).__name__):
                result = DocumentQAAgent([POLICY], Mock(side_effect=error)).ask('returns')
                self.assertEqual(result['status'], 'generation_failed')
                self.assertEqual(result['generation']['error'], type(error).__name__)
                self.assertEqual(result['retryable'], retryable)

    def test_retrieval_failure_is_distinct_from_generation(self):
        qa = DocumentQAAgent([POLICY], Mock())
        qa.retrieve = Mock(side_effect=ValueError('bad index'))
        result = qa.ask('returns')
        self.assertEqual(result['status'], 'retrieval_failed')
        self.assertEqual(result['retrieval']['error'], 'ValueError')
        self.assertEqual(result['generation']['status'], 'not_started')
        self.assertFalse(result['retryable'])
        qa.answerer.assert_not_called()

    def test_missing_unknown_and_unretrieved_citations_fail(self):
        documents = [POLICY, Document('warranty', 'Warranty repairs last one year.')]
        for answer, error in [('Seven days.', 'missing_citations'),
                              ('Seven days [invented]', 'unknown_citations'),
                              ('Seven days [policy] [warranty]', 'unknown_citations')]:
            with self.subTest(answer=answer):
                result = DocumentQAAgent(documents, lambda _: answer).ask('returns')
                self.assertEqual(result['status'], 'verification_failed')
                self.assertEqual(result['verification']['error'], error)
                self.assertEqual(result['sources'], [])
                self.assertEqual(result['retrieved_sources'], ['policy'])
                self.assertFalse(answer_accepted(result))

    def test_sources_follow_answer_citations_instead_of_retrieval_list(self):
        documents = [POLICY, Document('other', 'Returns require a receipt.')]
        result = DocumentQAAgent(documents, lambda _: 'Seven days [policy]').ask('returns')
        self.assertEqual(result['retrieved_sources'], ['policy', 'other'])
        self.assertEqual(result['sources'], ['policy'])
        self.assertTrue(answer_accepted(result))

    def test_independent_verifier_can_reject_cited_but_incorrect_answer(self):
        # This expected answer is independent of the generator, not inferred from a citation.
        def verifier(question, answer, documents):
            self.assertEqual(question, 'returns')
            self.assertEqual(documents, (POLICY,))
            return answer == 'Seven days [policy]'
        qa = DocumentQAAgent([POLICY], lambda _: 'Ninety days [policy]', verifier)
        result = qa.ask('returns')
        self.assertEqual(result['status'], 'verification_failed')
        self.assertEqual(result['verification']['error'], 'task_requirements_not_met')
        qa.answerer = lambda _: 'Seven days [policy]'
        result = qa.ask('returns')
        self.assertTrue(answer_accepted(result))
        self.assertIn('task_requirements', result['verification']['checks'])

    def test_verifier_error_and_non_boolean_success_do_not_pass(self):
        for verifier in [Mock(side_effect=RuntimeError('bad rubric')), lambda *_: 1,
                         lambda *_: {'status': 'passed'}]:
            with self.subTest(verifier=verifier):
                result = DocumentQAAgent([POLICY], lambda _: 'Seven days [policy]', verifier).ask('returns')
                self.assertEqual(result['status'], 'verification_failed')
                self.assertEqual(result['verification']['status'], 'failed')

    def test_keyboard_interrupt_is_never_converted_to_a_result(self):
        with self.assertRaises(KeyboardInterrupt):
            DocumentQAAgent([POLICY], Mock(side_effect=KeyboardInterrupt)).ask('returns')
        with self.assertRaises(KeyboardInterrupt):
            DocumentQAAgent([POLICY], verifier=Mock(side_effect=KeyboardInterrupt)).ask('returns')

    def test_research_requires_each_answer_to_pass(self):
        for output in ['', 'Seven days.', 'Seven days [unknown]', {'status': 'failed'}]:
            with self.subTest(output=output):
                result = ResearchAgent([POLICY], lambda _: output).run(['returns'])
                self.assertEqual(result['status'], 'incomplete')
                # Executing all steps is distinct from meeting their acceptance criteria.
                self.assertEqual(result['execution']['status'], 'completed')
        self.assertEqual(ResearchAgent([POLICY], verifier=lambda *_: False).run(['returns'])['status'], 'incomplete')
        self.assertEqual(ResearchAgent([POLICY]).run(['returns'])['status'], 'completed')
        self.assertEqual(ResearchAgent([POLICY]).run([])['status'], 'incomplete')

    def test_acceptance_rechecks_attribution_and_explicit_failure(self):
        valid = DocumentQAAgent([POLICY]).ask('returns')
        for change in [{'status': 'generation_failed'}, {'answer': ''}, {'sources': ['unknown']},
                       {'retrieved_sources': []}, {'verification': {'status': 'failed'}}]:
            with self.subTest(change=change):
                self.assertFalse(answer_accepted({**valid, **change}))

    def test_agentic_rag_only_retries_appropriate_failures_within_budget(self):
        answerer = Mock(side_effect=[TimeoutError('slow'), 'Seven days [policy]'])
        result = AgenticRAG(DocumentQAAgent([POLICY], answerer), 2).run(['returns', 'returns'])
        self.assertTrue(answer_accepted(result))
        self.assertEqual(len(result['trace']), 2)
        answerer = Mock(side_effect=TimeoutError('slow'))
        result = AgenticRAG(DocumentQAAgent([POLICY], answerer), 2).run(iter(['returns'] * 5))
        self.assertEqual(result['status'], 'generation_failed')
        self.assertEqual(len(result['trace']), 2)
        self.assertEqual(answerer.call_count, 2)

    def test_agentic_rag_stops_on_verification_or_permanent_failure(self):
        for answerer in [Mock(return_value=''), Mock(return_value='Seven days [unknown]'),
                         Mock(side_effect=RuntimeError('broken'))]:
            with self.subTest(answerer=answerer):
                result = AgenticRAG(DocumentQAAgent([POLICY], answerer), 2).run(['returns', 'returns'])
                self.assertFalse(answer_accepted(result))
                self.assertNotEqual(result['status'], 'no_evidence')
                self.assertEqual(len(result['trace']), 1)
                self.assertEqual(answerer.call_count, 1)

    def test_agentic_empty_queries_do_not_succeed(self):
        result = AgenticRAG(DocumentQAAgent([POLICY])).run([])
        self.assertEqual(result['status'], 'no_evidence')
        self.assertEqual(result['trace'], [])


if __name__ == '__main__':
    unittest.main()
