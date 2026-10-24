"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.rag import Document, DocumentQAAgent
from reference_code.evidence_assistant.agentic_rag import AgenticRAG

def main():
    import json
    qa = DocumentQAAgent([Document('fixture', 'Returns are allowed within seven days.')])
    print(json.dumps(AgenticRAG(qa).run(['unknown warranty', 'returns']), indent=2))

if __name__ == '__main__':
    main()
