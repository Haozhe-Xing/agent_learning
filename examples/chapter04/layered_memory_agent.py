"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.memory import LayeredMemoryAgent, MemoryStore

def main():
    store = MemoryStore(':memory:')
    agent = LayeredMemoryAgent(store, 'demo', working_limit=2)
    agent.update_core_memory('name', 'Ada', consent=True)
    agent.archive_content('project', 'customer churn prediction', consent=True)
    for query in ['hello', 'work', 'churn']:
        messages = agent.messages(query)
    import json
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print('Working items:', len(agent.working))
    agent.forget()
    store.close()

if __name__ == '__main__':
    main()
