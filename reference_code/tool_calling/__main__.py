"""Run the fixed local search/calculation exercise; no online mode."""
import argparse
import json
from .core import run_agent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-calls', type=int, default=3)
    parser.add_argument('--max-rounds', type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = run_agent(max_calls=args.max_calls, max_rounds=args.max_rounds)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
