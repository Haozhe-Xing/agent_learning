"""Contrastive retrieval and independent task-coverage acceptance tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evidence_assistant.rag import Document, DocumentQAAgent, terms
from evidence_assistant.task_acceptance import check_question_coverage, load_task_specs


PROJECT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT / 'sample_task_specs.json'
CASES_PATH = Path(__file__).with_name('fixtures') / 'research_cases.json'


class ExplainedRetrievalTests(unittest.TestCase):
    def test_rank_explains_scores_and_preserves_retrieve_interface(self):
        docs = [Document('weak', '验收记录。'),
                Document('one', '上线验收需要清单。'),
                Document('two', '上线验收需要审批。')]
        qa = DocumentQAAgent(docs)
        hits = qa.rank('上线验收需要满足哪些条件？')
        self.assertEqual([hit['document'].source for hit in hits], ['one', 'two'])
        self.assertEqual(qa.retrieve('上线验收需要满足哪些条件？'), [docs[1], docs[2]])
        self.assertEqual(hits[0]['matched_terms'], ['上线', '线验', '验收'])
        self.assertEqual(hits[0]['score'], 3)
        self.assertEqual(qa.rank('上线验收', limit=1), hits[:1])

    def test_generic_phrases_do_not_match_or_create_boundary_bigrams(self):
        self.assertEqual(terms('需要满足哪些条件如何要求具体情况'), set())
        self.assertNotIn('收需', terms('验收需要满足条件'))
        qa = DocumentQAAgent([Document('release', '上线验收需要满足全部条件。')])
        self.assertEqual(qa.rank('火星移民需要满足哪些条件？'), [])
        self.assertEqual(qa.ask('需要满足哪些条件？')['status'], 'no_evidence')

    def test_short_keywords_and_original_english_baselines_remain_supported(self):
        docs = [Document('moon', 'Moon distance is 100 km.'),
                Document('distance', 'Distance: 100000 m.'),
                Document('policy', 'Returns are accepted within seven days.'),
                Document('zh', '退货期限为七天')]
        qa = DocumentQAAgent(docs)
        self.assertEqual(qa.ask('returns')['sources'], ['policy'])
        self.assertEqual([hit['document'].source for hit in qa.rank('Moon distance')], ['moon', 'distance'])
        self.assertEqual(qa.ask('退货期限')['sources'], ['zh'])
        self.assertEqual(qa.ask('退货')['sources'], ['zh'])

    def test_rank_limit_and_empty_queries(self):
        qa = DocumentQAAgent([Document('policy', 'Returns require a receipt.')])
        self.assertEqual(qa.rank('returns', limit=0), [])
        self.assertEqual(qa.rank(''), [])
        for query, limit in ((None, 3), ('returns', -1), ('returns', True)):
            with self.subTest(query=query, limit=limit), self.assertRaises(ValueError):
                qa.rank(query, limit)


class TaskCoverageTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_task_specs(SPEC_PATH)

    def test_independent_contrastive_cases(self):
        for case in json.loads(CASES_PATH.read_text(encoding='utf-8'))['cases']:
            with self.subTest(case=case['id']):
                documents = [Document(**document) for document in case['documents']]
                qa = DocumentQAAgent(documents)
                self.assertEqual([document.source for document in qa.retrieve(case['question'])],
                                 case['expected_sources'])
                generated = qa.ask(case['question'])
                answer = case.get('answer', generated['answer'])
                coverage = check_question_coverage(case['question'], answer, specs=self.specs,
                                                   documents=documents)
                self.assertEqual(coverage['status'], case['expected_coverage'])
                self.assertEqual([detail['id'] for detail in coverage['details']
                                  if detail['missing_fragments']], case['missing_ids'])
                if coverage['status'] in {'failed', 'unknown'} and case['expected_sources']:
                    # A valid cited excerpt still cannot establish question coverage.
                    self.assertEqual(generated['status'], 'answered')

    def test_no_spec_never_infers_coverage_from_quotes(self):
        answer = '[policy] Returns are accepted within seven days.'
        result = check_question_coverage('returns', answer)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'no_task_spec_or_verifier')

    def test_verifier_must_pass_and_cannot_bypass_missing_requirements(self):
        question = self.specs[0]['question']
        answer = '[release] 上线验收需要资料负责人确认发布清单。'
        result = check_question_coverage(question, answer, specs=self.specs, verifier=lambda *_: True)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('20条固定问题中至少18条返回正确来源', result['missing_points'])
        for value in (False, 'yes', 1, None):
            with self.subTest(value=value):
                result = check_question_coverage('custom', 'an excerpt', verifier=lambda *_: value)
                self.assertEqual(result['status'], 'failed')

    def test_independent_verifier_can_accept_a_custom_task(self):
        documents = [Document('policy', 'Returns are accepted within seven days.')]
        verifier = Mock(return_value=True)
        result = check_question_coverage('returns', '[policy] seven days',
                                         verifier=verifier, documents=documents)
        self.assertEqual(result['status'], 'passed')
        verifier.assert_called_once_with('returns', '[policy] seven days', tuple(documents))

    def test_verifier_error_fails_and_interrupt_propagates(self):
        result = check_question_coverage('custom', 'answer', verifier=Mock(side_effect=RuntimeError))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_type'], 'RuntimeError')
        with self.assertRaises(KeyboardInterrupt):
            check_question_coverage('custom', 'answer', verifier=Mock(side_effect=KeyboardInterrupt))

    def test_spec_and_verifier_are_both_required_when_configured(self):
        spec = [{'question': 'Release?', 'requirements': [
            {'id': 'approval', 'description': 'approval recorded', 'all_of': ['approval recorded']}]}]
        self.assertEqual(check_question_coverage('release?', 'Approval\nrecorded', specs=spec)['status'], 'passed')
        self.assertEqual(check_question_coverage('Release?', 'approval recorded', specs=spec,
                                                verifier=lambda *_: False)['status'], 'failed')
        self.assertEqual(check_question_coverage('Release?', 'missing', specs=spec,
                                                verifier=lambda *_: True)['status'], 'failed')

    def test_source_labels_cannot_supply_required_answer_points(self):
        spec = [{'question': 'Release?', 'requirements': [
            {'id': 'approval', 'description': 'approval recorded', 'all_of': ['approval recorded']}]}]
        for answer in ('[approval recorded] Release delayed.', '  [approval recorded] Release delayed.'):
            result = check_question_coverage('Release?', answer, specs=spec)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['missing_points'], ['approval recorded'])
        spec[0]['requirements'][0]['all_of'] = ['[review](policy.md)']
        self.assertEqual(check_question_coverage(
            'Release?', '[release] Release requires [review](policy.md).', specs=spec)['status'], 'passed')

    def test_malformed_or_vacuous_specs_are_rejected(self):
        valid = {'schema_version': 1, 'tasks': copy.deepcopy(self.specs)}
        invalid = [[], {'schema_version': True, 'tasks': self.specs},
                   {'schema_version': 1, 'tasks': []}, {**valid, 'ignored': 'extra'}]
        for change in ('empty_requirements', 'blank_fragment', 'duplicate_question', 'duplicate_id'):
            payload = copy.deepcopy(valid)
            if change == 'empty_requirements':
                payload['tasks'][0]['requirements'] = []
            elif change == 'blank_fragment':
                payload['tasks'][0]['requirements'][0]['all_of'] = [' ']
            elif change == 'duplicate_question':
                payload['tasks'].append(copy.deepcopy(payload['tasks'][0]))
            else:
                payload['tasks'][0]['requirements'].append(copy.deepcopy(payload['tasks'][0]['requirements'][0]))
            invalid.append(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.json'
            for payload in invalid:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_task_specs(path)

    def test_empty_answers_do_not_pass_even_with_a_permissive_verifier(self):
        for answer in ('', ' ', None):
            with self.subTest(answer=answer):
                result = check_question_coverage('custom', answer, verifier=lambda *_: True)
                self.assertEqual(result['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
