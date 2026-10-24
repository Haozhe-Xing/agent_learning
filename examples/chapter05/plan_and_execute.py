"""Runnable companion entry; run from any directory using Python 3.11+."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference_code.evidence_assistant.planning import PlanAndExecuteAgent, Step

def main():
    agent = PlanAndExecuteAgent(
        lambda _: [Step('unavailable')],
        {'calculate': lambda a, b: a / b},
        replanner=lambda task, trace, remaining: [Step('calculate', {'a': 384400000, 'b': 299792458})])
    import json
    print(json.dumps(agent.run('Compute fixture light travel time'), indent=2))

if __name__ == '__main__':
    main()
