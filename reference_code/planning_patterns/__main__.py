"""Run a local release-brief task; no model, network, arbitrary code or file writes."""
import argparse
from copy import deepcopy
import json
from .workflow import OmitRollbackOnce, run_plan, run_react


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['plan', 'react', 'revise', 'stubborn'], default='plan')
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--max-replans', type=int, default=1)
    args = parser.parse_args(argv)
    def stubborn(facts, feedback):
        result = deepcopy(facts)
        result.pop('rollback')
        return result
    try:
        if args.mode == 'react':
            result = run_react(max_calls=args.max_steps)
        else:
            options = {'max_steps': args.max_steps, 'max_replans': args.max_replans}
            if args.mode == 'revise':
                options['writer'] = OmitRollbackOnce()
            elif args.mode == 'stubborn':
                options['writer'] = stubborn
            result = run_plan(**options)
    except ValueError as exc:
        result = {'status': 'failed', 'accepted': False, 'reason': str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
