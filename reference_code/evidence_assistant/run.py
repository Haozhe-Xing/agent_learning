"""Run this reference project from any working directory."""
from pathlib import Path
import sys

if __package__:
    from .research import main
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evidence_assistant.research import main


if __name__ == '__main__':
    raise SystemExit(main())
