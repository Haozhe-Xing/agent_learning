"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.memory import MemoryStore

def main():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'memory.sqlite'
        store = MemoryStore(path)
        store.put('demo', 'preference', 'concise answers', consent=True)
        store.close()
        store = MemoryStore(path)
        print(store.records('demo'))
        store.delete('demo')
        assert store.records('demo') == {}
        store.close()

if __name__ == '__main__':
    main()
