"""python -m reference_code.security_boundary; no keys or network."""
import json

from .regression import run_regression


def main():
    result = run_regression()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
