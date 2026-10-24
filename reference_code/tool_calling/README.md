# Local function-calling exercise

Run from the repository root with Python 3.11+:

```bash
python -m reference_code.tool_calling
python -m reference_code.tool_calling --max-calls 1
python -m unittest examples.integration_tests.test_tool_calling_chapter -v
```

The first command completes the frozen Moon fixture. The one-call budget returns
`incomplete`, `accepted: false`, exit 1. Search examines injected in-memory
material only. No model client, web request, file tool, or shell is created.
The provider is a three-turn script consuming actual previous tool results.

`core.py` provides strict bounded JSON argument decoding, explicit host
permissions, call-ID correspondence, separate model/tool budgets, structured
errors and a frozen independent rubric. Tool errors stop the task; admitted
batches receive one result per ID, with later calls skipped after a failure.
Budget-rejected batches are not admitted into the conversation.

The implementation reuses the repository's single `safe_math` implementation
in `examples/chapter03/safe_math.py` and lexical retrieval from
`reference_code/evidence_assistant/rag.py`. This directory is a separate companion
entry point, not a package promised to work when copied without those modules.
Core execution/tests require only the standard library. SDK integration tests
use OpenAI 2.54.0 and an injected HTTP mock; there is no online CLI mode.

The verifier accepts only the frozen question, evidence, calculation and output
format. It does not establish general factual or semantic correctness. Byte,
round and call limits do not implement token billing, wall-clock cancellation,
process isolation, retries, persistent state or production authentication.
