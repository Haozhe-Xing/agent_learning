# Chapter 19: Security and Reliability

**Reading levels**: 19.1–19.5 are core; 19.7–19.8 are runtime/evaluation labs; 19.6 is selected research. Revisit these boundaries from the first tool call onward.

## This chapter's iteration

- **Previous limitation**: No pattern match, a valid citation label or a successful tool exit does not establish authority or task completion. An ordinary child process is not a secure sandbox.
- **This chapter adds**: Separate content from authority and support from coverage. The independent `reference_code/security_boundary/` lab enforces closed proposals, host identity/resource scope, content-bound approval, current-policy checks, local receipts, budgets, field projection and explicit consent.
- **How to verify**: Start with 11 frozen local cases checking legitimate and unauthorized effects. Then run `examples/integration_tests/test_security_chapter.py` for rejection, errors, retries and executable prose. Separate lexical tests live in `reference-agent/tests/test_security.py`; they do not establish general semantic safety.
- **Next step**: Carry the permission matrix, actual effects, unknown/errors and outstanding deployment requirements into [Chapter 20](../chapter_22_deployment/README.md).

## Keep a moderate mainline

The [local evidence research assistant](../learning_project.md) remains the core project. This chapter reuses existing QA support/coverage, then adds an optional standard-library authorization lab. It reads fictional in-memory documents and writes fictional local notices. It has no real sending, arbitrary code execution, login UI, disk persistence or OS sandbox.

```bash
python -m reference_code.security_boundary
python -m unittest examples.integration_tests.test_security_chapter -v
```

Run from the repository root with Python 3.11+, without keys. The directory can be copied independently; see `reference_code/security_boundary/README.md`. Local tests, real-model red teaming, concurrency, remote idempotency and deployment isolation have separate evidence requirements.

## Chapter structure

| Section | Checkable outcome |
|---|---|
| [19.1 Prompt injection](./01_prompt_injection.md) | Separate false-positive/negative signals from authorization |
| [19.2 Hallucination and factuality](./02_hallucination.md) | Separate labels, excerpt support, task coverage and freshness |
| [19.3 Permissions and sandboxes](./03_permission_sandbox.md) | Current policy, ownership, bound approval and an isolation contract |
| [19.4 Data protection](./04_data_protection.md) | Field projection, explicit consent, actor scope and deletion boundaries |
| [19.5 Controllability and alignment](./05_alignment.md) | Specific authorization, receipt retries, acceptance and stopping |
| [19.6 Selected papers](./06_paper_readings.md) | The same eight versions, settings, methods, limits and exercise |
| [19.7 Runtime guardrails](./07_guardrails_runtime.md) | Actual order, budgets, failure states and audit |
| [19.8 Red teaming](./08_red_teaming.md) | Frozen benign/attack cases, independent effects and complete denominators |

Afterward, explain who grants permission and where it is checked before the actual operation; what producer and verifier each see; how a failed response avoids duplicate effects; and which risks remain untested. More complex execution services and real-model experiments are extensions, not prerequisites for the core assignment.

*Next: [19.1 Prompt Injection](./01_prompt_injection.md)*
