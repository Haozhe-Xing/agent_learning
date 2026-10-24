# Offline LangChain patterns

Companion code for sections 10.1, 10.2, 10.4, 10.6–10.8. Tested with
Python 3.12, langchain 1.4.6, langchain-core 1.6.12,
langchain-openai 1.7.2 and langsmith 0.14.11. Use the repository's framework
integration environment; no dependencies are installed by this directory.

From the repository root:

```bash
python -m reference_code.langchain_patterns
python -m reference_code.langchain_patterns --trace-file /tmp/lc-pattern-trace.json
python -m unittest examples.integration_tests.test_langchain_chapter -v
```

The optional trace file must not already exist. It records lifecycle IDs,
allowlisted caller metadata and error types, without prompt/output text.
Treat those metadata identifiers as data too; this is not a privacy classifier.

`workflows.py` has real Runnable composition, a fixed chat provider, complete
JSON plus Pydantic validation, selected-exception retries, scoped cache inputs,
and explicit partial-stream errors. `local_trace.py` records local callbacks.
The CLI disables LangSmith tracing even if an environment variable enabled it.
Direct library callers choose their own tracing context, as the book examples
show. No LangSmith client or online model is invoked by these exercises.

The fixed provider returns scripted messages; it does not understand arbitrary
questions. The schema's seven-day policy and source ID are frozen fixture
requirements. The intake classifier is an English substring rule. None provides
general semantic verification, authorization, an HTTP service, persistent
execution, or a process sandbox. Sections 10.3/10.5 supply separate Agent examples.
