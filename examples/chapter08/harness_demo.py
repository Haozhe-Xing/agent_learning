"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.harness import Harness

def main():
    import json
    harness = Harness({'calculator': lambda a, b: a * b}, frozenset({'calculator'}), max_calls=2)
    value = harness.invoke('calculator', {'a': 12, 'b': 8})
    try:
        harness.invoke('delete_file', {'path': 'anything'})
    except PermissionError:
        pass
    print(json.dumps(harness.finish(value, lambda result: result == 96), indent=2))

if __name__ == '__main__':
    main()
