"""Protect module identity, existing patches and standalone project execution."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROJECT = ROOT / 'reference_code/evidence_assistant'


class ReferenceCodeMigrationTests(unittest.TestCase):
    def test_legacy_modules_are_the_canonical_module_objects(self):
        for name in ('acceptance', 'context', 'contracts', 'demo', 'harness', 'memory', 'planning', 'rag'):
            with self.subTest(module=name):
                canonical = importlib.import_module('reference_code.evidence_assistant.' + name)
                legacy = importlib.import_module('examples.learning_path.' + name)
                self.assertIs(legacy, canonical)
                self.assertEqual(Path(canonical.__file__).resolve().parent, PROJECT)

    def test_patch_through_legacy_demo_changes_canonical_execution(self):
        from reference_code.evidence_assistant import demo
        from reference_code.evidence_assistant.rag import Document
        from reference_code.evidence_assistant.contracts import calculate_travel_time

        with patch('examples.learning_path.demo.FACT', Document('changed', 'Distance: 100 km.')):
            result = demo.run_stage('harness')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['distance_km'], 100)

        def corrupted(quantity, output_unit):
            return replace(calculate_travel_time(quantity, output_unit), seconds=42, output_value=42)

        with patch('examples.learning_path.demo.calculate_travel_time', side_effect=corrupted) as calculate:
            result = demo.run_stage('harness')
        calculate.assert_called_once()
        self.assertFalse(result['accepted'])

    def test_patch_through_legacy_memory_reaches_canonical_dependency(self):
        from reference_code.evidence_assistant.demo import run_stage
        with patch('examples.learning_path.memory.MemoryStore.records',
                   return_value={'output_unit': 'milliseconds'}) as records:
            result = run_stage('memory')
        records.assert_called_once()
        self.assertEqual(result['output']['unit'], 'milliseconds')
        self.assertFalse(result['accepted'])  # The task still requested seconds.

    def test_chapter_entry_points_reuse_project_classes(self):
        from examples.chapter05.research_agent import ResearchAgent as chapter_research
        from examples.chapter06.agentic_rag import AgenticRAG as chapter_rag
        from reference_code.evidence_assistant.research_agent import ResearchAgent
        from reference_code.evidence_assistant.agentic_rag import AgenticRAG
        self.assertIs(chapter_research, ResearchAgent)
        self.assertIs(chapter_rag, AgenticRAG)

    def test_legacy_cli_keeps_the_original_six_stage_behavior(self):
        result = subprocess.run([sys.executable, '-m', 'examples.learning_path', '--all'],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = json.loads(result.stdout)
        self.assertEqual([item['stage'] for item in outputs],
                         ['tools', 'memory', 'planning', 'rag', 'context', 'harness'])
        self.assertTrue(all(item['accepted'] for item in outputs))

    def test_project_can_be_copied_and_run_without_the_repository(self):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'evidence_assistant'
            shutil.copytree(PROJECT, copied, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            unrelated = Path(temporary) / 'unrelated'
            unrelated.mkdir()
            # No examples/, reference-agent/, tools/, or repository package is
            # present in this temporary checkout.
            result = subprocess.run([sys.executable, str(copied / 'run_tests.py')],
                                    cwd=unrelated, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertRegex(result.stderr, r'Ran \d+ tests')
            commands = [([sys.executable, str(copied / 'run.py'), '--help'], unrelated),
                        ([sys.executable, '-m', 'evidence_assistant', '--help'], temporary),
                        ([sys.executable, '-m', 'evidence_assistant.demo', '--all'], temporary)]
            for command, directory in commands:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=directory, env=env,
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
            for command, directory in [
                ([sys.executable, str(copied / 'run.py'), '--format', 'json'], unrelated),
                ([sys.executable, '-m', 'evidence_assistant', '--format', 'json'], temporary),
            ]:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=directory, env=env,
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report['status'], 'completed')
                    self.assertIs(report['accepted'], True)
                    self.assertEqual(len(report['findings']), 3)
                    self.assertTrue(all(finding['status'] == 'answered' and finding['sources']
                                        for finding in report['findings']))


if __name__ == '__main__':
    unittest.main()
