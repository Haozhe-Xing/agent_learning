"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.rag import Document, DocumentQAAgent, OpenAIAnswerer

DOCUMENTS = [Document('handbook', 'The fixture return window is seven days.'),
             Document('shipping', 'The fixture shipping time is two days.')]

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description='Offline lexical evidence baseline; optional real generation')
    parser.add_argument('--question', default='What is the return window?')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--model', default='gpt-4.1-mini')
    args = parser.parse_args()
    answerer = OpenAIAnswerer(args.model) if args.real else None
    print(json.dumps(DocumentQAAgent(DOCUMENTS, answerer).ask(args.question), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
