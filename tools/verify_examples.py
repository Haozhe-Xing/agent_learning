"""Run offline companion entry points and emit a reviewable JSON record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [
    ['-m', 'reference_code.tool_calling'],
    ['-m', 'reference_code.planning_patterns'],
    ['-m', 'reference_code.security_boundary'],
    ['examples/chapter03/search_calc_agent.py', '--demo'],
    ['examples/chapter04/long_term_memory.py'],
    ['examples/chapter04/layered_memory_agent.py'],
    ['examples/chapter04/scratchpad_agent.py'],
    ['examples/chapter05/plan_and_execute.py'],
    ['examples/chapter05/research_agent.py'],
    ['examples/chapter06/doc_qa_agent.py'],
    ['examples/chapter06/agentic_rag.py'],
    ['examples/chapter07/gssc_pipeline.py'],
    ['examples/chapter08/harness_demo.py'],
    ['-m', 'reference_code.evidence_assistant.demo', '--all'],
    ['-m', 'reference_code.evidence_assistant'],
]
FRAMEWORK_ENTRIES = [
    ['-m', 'reference_code.memory_rag'],
    ['-m', 'reference_code.vector_retrieval'],
    ['-m', 'reference_code.langchain_patterns'],
    ['-m', 'reference_code.langgraph_patterns'],
    ['-m', 'examples.chapter12.first_agent'],
    ['-m', 'examples.chapter13.first_graph_agent'],
    ['examples/chapter04/personal_assistant.py', '--help'],
    ['examples/chapter12/customer_service.py'],
    ['examples/chapter12/customer_service.py', '--query', 'check ORD-2'],
    ['examples/chapter13/content_workflow.py'],
    ['examples/chapter13/content_workflow.py', '--always-low'],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    results = []
    entries = ENTRIES + (FRAMEWORK_ENTRIES if args.integration else [])
    for entry in entries:
        try:
            proc = subprocess.run([sys.executable, *entry], cwd=ROOT, capture_output=True, text=True, timeout=60)
            item = {'command': ['python', *entry], 'exit_code': proc.returncode,
                    'stdout': proc.stdout, 'stderr': proc.stderr}
        except subprocess.TimeoutExpired:
            item = {'command': ['python', *entry], 'exit_code': -1, 'error': 'timeout'}
        results.append(item)
        print(('PASS' if item['exit_code'] == 0 else 'FAIL') + ' ' + ' '.join(entry))
    report = {'python': sys.version.split()[0], 'mode': 'offline-fixtures', 'results': results}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if any(item['exit_code'] for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
