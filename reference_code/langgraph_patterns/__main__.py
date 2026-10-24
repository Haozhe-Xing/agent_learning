"""Run local graph exercises without credentials, models, or network access."""
from __future__ import annotations

import argparse
import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .approval import ReceiptLedger, approval_input, build_approval_graph, local_demo_policy
from .core import audit_input, build_audit_graph
from .review import run_review


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exercise', choices=['core', 'review', 'approval'], nargs='?', default='core')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--recursion-limit', type=int)
    parser.add_argument('--no-repair', action='store_true', help='demonstrate unresolved review findings')
    parser.add_argument('--decision', choices=['approve', 'deny'], default='deny',
                        help='a local decision fixture; no real authentication or external action')
    args = parser.parse_args(argv)
    draft = {'version': '1.0', 'owner': 'release-team'}
    try:
        if args.exercise == 'core':
            result = build_audit_graph().invoke(audit_input({
                **draft, 'rollback': 'restore the previous release snapshot'}))
            success = result['accepted']
        elif args.exercise == 'review':
            kwargs = {'max_attempts': args.max_attempts, 'recursion_limit': args.recursion_limit}
            if args.no_repair:
                def keep_draft(draft, _findings):
                    return draft
                kwargs['repairer'] = keep_draft
            result = run_review(draft, **kwargs)
            success = result['accepted']
        else:
            with ReceiptLedger() as ledger:
                app = build_approval_graph(ledger, checkpointer=InMemorySaver(), authorize=local_demo_policy)
                config = {'configurable': {'thread_id': 'local-approval'}, 'recursion_limit': 8}
                paused = app.invoke(approval_input(), config)
                pending = paused['__interrupt__'][0].value
                result = app.invoke(Command(resume={
                    'approved': args.decision == 'approve', 'reviewer': 'reviewer-1',
                    'operation_digest': pending['operation_digest']}), config)
                result['receipt_count'] = ledger.count()
                success = result['status'] == 'recorded'
    except (ValueError, GraphRecursionError) as exc:
        result, success = {'status': 'failed', 'reason': str(exc)}, False
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
