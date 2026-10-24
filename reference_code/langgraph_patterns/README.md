# LangGraph lifecycle exercises

Run from the repository root with Python 3.11+ and the versions in
`examples/requirements.txt` (`langgraph==1.2.16`). No model, key, or network
is needed.

```bash
python -m reference_code.langgraph_patterns core
python -m reference_code.langgraph_patterns review
python -m reference_code.langgraph_patterns review --no-repair --max-attempts 2
python -m reference_code.langgraph_patterns approval --decision approve
python -m unittest examples.integration_tests.test_langgraph_chapter -v
```

- `core.py`: two parallel required-field checks, an append reducer, and an
  explicit join. Acceptance only covers those named checks.
- `review.py`: independent required-field review, a fixed repair proposal,
  attempt limits, failure states, and a separate graph super-step backstop.
- `approval.py`: dynamic `interrupt`/`Command(resume=...)`, operation binding,
  a local policy fixture, and an idempotent SQLite receipt.

An unresolved review, denied approval, or failed run exits 1. The approval CLI
defaults to denial. It keeps checkpoints and receipts in memory; it sends
nothing outside the process. The library's `ReceiptLedger(path)` can persist
receipts, but these business records are not LangGraph checkpoints.

`local_demo_policy` checks a fixed demonstration name, not authenticated
identity. A real host must supply authentication, thread/operation permissions,
approval expiry, and any remote tool's idempotency protocol. Durable graph
checkpointers, cross-process resume, approval UI, and model quality are outside
this exercise's implemented scope.

The authorization callback must return the boolean `True` explicitly at both
decision review and the execution-time recheck. Truthy values such as `1`,
`"yes"`, or a nonempty dictionary do not authorize a write.
