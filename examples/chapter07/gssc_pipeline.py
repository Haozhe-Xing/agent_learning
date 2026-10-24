"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.context import ContextItem, GSSCPipeline

def main():
    import json
    items = [ContextItem('Essential evidence', 'fixture', required=True),
             ContextItem('Older conversation. ' * 100, 'history', priority=9)]
    print(json.dumps(GSSCPipeline(max_bytes=600).build('Use evidence only.', 'Answer the task.', items), indent=2))

if __name__ == '__main__':
    main()
