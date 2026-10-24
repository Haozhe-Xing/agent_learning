"""Independent expected cases and fault injection for the shared project."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evidence_assistant import demo
from evidence_assistant.acceptance import verify_light_travel
from evidence_assistant.contracts import calculate_travel_time
from evidence_assistant.rag import Document


CASES = json.loads((Path(__file__).parent / 'fixtures/light_travel_cases.json').read_text())


class ProjectContractTests(unittest.TestCase):
    def setUp(self):
        self.documents = [Document('short', 'Moon distance is 100 km.')]

    def test_independent_fixture_table_across_all_stages(self):
        for case in CASES:
            documents = [Document(**doc) for doc in case['documents']]
            for stage in demo.STAGES:
                with self.subTest(case=case['id'], stage=stage):
                    result = demo.run_stage(stage, documents)
                    self.assertEqual(result['accepted'], case['accepted'])
                    if case['accepted']:
                        self.assertEqual(result['status'], 'completed')
                        self.assertTrue(math.isclose(result['seconds'], case['seconds'], rel_tol=1e-12))
                    else:
                        self.assertEqual(result['status'], 'failed')
                        self.assertEqual(result['error']['code'], case['error'])
                        self.assertNotIn('seconds', result)

    def test_replacing_default_fact_changes_every_stage(self):
        with patch.object(demo, 'FACT', self.documents[0]):
            for stage in demo.STAGES:
                result = demo.run_stage(stage)
                self.assertTrue(result['accepted'])
                self.assertAlmostEqual(result['seconds'], 0.00033356409519815205, places=16)

    def test_memory_preference_controls_output(self):
        result = demo.run_stage('memory', self.documents, output_unit='milliseconds')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['memory']['output_unit'], 'milliseconds')
        self.assertAlmostEqual(result['output']['value'], 0.33356409519815205)

    def test_memory_is_read_instead_of_merely_displayed(self):
        with patch.object(demo.MemoryStore, 'records', return_value={'output_unit': 'milliseconds'}):
            result = demo.run_stage('memory', self.documents)
        self.assertEqual(result['output']['unit'], 'milliseconds')
        self.assertFalse(result['accepted'])  # The independent task requested seconds.

    def test_plan_step_budget_is_not_success(self):
        result = demo.run_stage('planning', self.documents, max_steps=1)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        self.assertNotIn('seconds', result)

    def test_harness_budget_is_not_success(self):
        result = demo.run_stage('harness', self.documents, max_calls=1)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['error']['code'], 'tool_budget_exhausted')
        self.assertEqual(result['verification']['status'], 'verification_failed')

    def test_context_budget_blocks_calculation(self):
        with patch.object(demo, 'calculate_travel_time') as calculate:
            result = demo.run_stage('context', self.documents, context_max_bytes=50)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['accepted'])
        calculate.assert_not_called()

    def test_unaccepted_retrieval_blocks_calculation(self):
        with patch.object(demo.DocumentQAAgent, 'ask', return_value={'sources': ['short']}), \
                patch.object(demo, 'calculate_travel_time') as calculate:
            result = demo.run_stage('rag', self.documents)
        self.assertFalse(result['accepted'])
        calculate.assert_not_called()

    def test_constructed_context_is_actually_consumed(self):
        payload = 'Untrusted reference data: ' + json.dumps([
            {'source': 'short', 'content': 'Moon distance is 200 km.'}])
        with patch.object(demo.GSSCPipeline, 'build', return_value={
                'messages': [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': payload}]}):
            result = demo.run_stage('context', self.documents)
        self.assertAlmostEqual(result['seconds'], 2 * 0.00033356409519815205, places=16)
        self.assertFalse(result['accepted'])  # Changed context cannot impersonate original evidence.

    def test_independent_check_rejects_corrupted_calculator(self):
        def broken(quantity, unit):
            calculation = calculate_travel_time(quantity, unit)
            return replace(calculation, seconds=42, output_value=42)
        with patch.object(demo, 'calculate_travel_time', side_effect=broken):
            result = demo.run_stage('harness', self.documents)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['verification']['status'], 'verification_failed')

    def test_independent_check_rejects_parser_unit_bug(self):
        original = demo.extract_distance
        def broken(records):
            return replace(original(records), unit='m')
        with patch.object(demo, 'extract_distance', side_effect=broken):
            result = demo.run_stage('harness', self.documents)
        self.assertFalse(result['accepted'])

    def test_tampered_quantity_and_provenance_are_rejected(self):
        valid = demo.run_stage('harness', self.documents)
        changes = [('quantity', 'value', '200'), ('quantity', 'unit', 'm'),
                   ('quantity', 'source', 'other'), ('quantity', 'quote', 'invented text'),
                   ('quantity', 'evidence_sha256', '0' * 64),
                   ('calculation', 'seconds', 42), ('calculation', 'speed_metres_per_second', 1),
                   ('output', 'value', 42), ('output', 'unit', 'milliseconds')]
        for field, key, value in changes:
            with self.subTest(field=field, key=key):
                bad = deepcopy(valid)
                bad[field][key] = value
                self.assertFalse(verify_light_travel(bad, self.documents, 'seconds')['accepted'])
        for key in ('source', 'text', 'sha256'):
            bad = deepcopy(valid)
            bad['provenance'][0][key] = 'tampered'
            self.assertFalse(verify_light_travel(bad, self.documents, 'seconds')['accepted'])

    def test_final_answer_value_unit_and_source_are_verified(self):
        valid = demo.run_stage('tools', self.documents)
        for answer in ('', '42 seconds [short]', '0.00033356409519815205 milliseconds [short]',
                       '0.00033356409519815205 seconds [unknown]',
                       '0.00033356409519815205 seconds [short] [unknown]'):
            bad = dict(valid, answer=answer)
            self.assertFalse(verify_light_travel(bad, self.documents, 'seconds')['accepted'])

    def test_non_finite_and_boolean_numbers_are_rejected(self):
        valid = demo.run_stage('tools', self.documents)
        for value in (float('nan'), float('inf'), True, None, '0.00033356409519815205'):
            self.assertFalse(verify_light_travel(dict(valid, seconds=value), self.documents, 'seconds')['accepted'])

    def test_acceptance_does_not_trust_success_flag(self):
        self.assertFalse(verify_light_travel({'accepted': True, 'status': 'completed'},
                                           self.documents, 'seconds')['accepted'])

    def test_tiny_values_cannot_be_zeroed_by_absolute_tolerance(self):
        documents = [Document('tiny', 'Distance: 0.000000001 m.')]
        valid = demo.run_stage('tools', documents)
        self.assertTrue(valid['accepted'])
        valid.update(seconds=0, answer='0 seconds [tiny]')
        valid['calculation'].update(seconds=0, output_value=0)
        valid['output']['value'] = 0
        self.assertFalse(verify_light_travel(valid, documents, 'seconds')['accepted'])

    def test_unsupported_precision_and_underflow_are_explicit_failures(self):
        for text in ('Distance: 1.' + '0' * 60 + '1 km.',
                     'Distance: 0.' + '0' * 330 + '1 m.', 'Distance: １００ km.',
                     'Distance: 0.0000000000001 m.'):
            for stage in demo.STAGES:
                result = demo.run_stage(stage, [Document('boundary', text)])
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['error']['code'], 'invalid_quantity')

    def test_cli_failure_exit_code_and_alternate_input(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'distance.txt'
            for text, code in [('Moon distance is 100 km.', 0), ('Moon distance is unknown.', 1)]:
                path.write_text(text)
                run = subprocess.run([sys.executable, '-m', 'evidence_assistant.demo', '--file', str(path)],
                                     cwd=root, capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, code, run.stderr)
                self.assertEqual(json.loads(run.stdout)[0]['accepted'], code == 0)

    def test_invalid_source_and_units_have_structured_failures(self):
        for docs, unit in [([Document('bad/source', 'Moon distance is 100 km.')], 'seconds'),
                           (self.documents, 'minutes'), ([Document('blank', '')], 'seconds')]:
            result = demo.run_stage('tools', docs, output_unit=unit)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['accepted'])
            self.assertIn('code', result['error'])


if __name__ == '__main__':
    unittest.main()
