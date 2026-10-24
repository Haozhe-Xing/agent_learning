"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.memory import LayeredMemoryAgent, MemoryStore

def main():
    store = MemoryStore(':memory:')
    agent = LayeredMemoryAgent(store, 'demo', working_limit=2)
    for note in ['collect evidence', 'calculate', 'verify answer']:
        agent.messages(note)
    print(list(agent.working))
    assert len(agent.working) == 2
    store.close()

if __name__ == '__main__':
    main()
