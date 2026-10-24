"""Run the project's standard-library tests, including after copying its folder."""
from pathlib import Path
import sys
import unittest


def main() -> int:
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project.parent))
    suite = unittest.defaultTestLoader.discover(str(project / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
