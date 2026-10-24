"""Evidence-grounded research on authorized local documents; offline by default."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reference_code.evidence_assistant.rag import Document, OpenAIAnswerer
from reference_code.evidence_assistant.research_agent import ResearchAgent


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, help='authorized UTF-8 source document')
    parser.add_argument('--question', default='What is the return window?')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--model', default='gpt-4.1-mini')
    args = parser.parse_args()
    doc = Document(args.file.name, args.file.read_text()) if args.file else Document('fixture', 'The fixture return window is seven days.')
    agent = ResearchAgent([doc], OpenAIAnswerer(args.model) if args.real else None)
    print(json.dumps(agent.run([args.question]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
