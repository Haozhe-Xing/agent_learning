# Memory and RAG mechanisms

Small offline exercises for chapters 4.1–4.4 and 6.1–6.2. Python 3.11+;
Chroma 1.5.9 comes from the repository's pinned `examples/requirements.txt`.
No model calls, tokenizer downloads, embedding downloads, or network loaders.
Run from the repository root:

```bash
python -m reference_code.memory_rag history
python -m reference_code.memory_rag working
python -m reference_code.memory_rag chunks
python -m reference_code.memory_rag vectors
python -m unittest discover -s examples/integration_tests -p test_memory_rag_chapters.py -v
```

- `BoundedHistory`: complete text-only turns; fixed instructions/current request
  cannot be dropped. The hard budget counts serialized UTF-8 bytes, not tokens.
  Summaries stay untrusted data. No summarization model or tool-call protocol.
- `TaskState`: owner/task checks and bounded observable results. Independent
  predicates distinguish verified/rejected/observed data; no private reasoning.
- `load_text` / `split_document`: bounded local UTF-8 `.txt` / `.md`; deterministic
  character windows with exact source version and offsets. They do not parse
  Markdown sections or guarantee semantic completeness. Root checks assume a
  trusted filesystem rather than a concurrent hostile process.
- `VectorMemory`: real persistent Chroma; explicit Boolean consent, owner-scoped
  reads/deletes, explicit vectors, cosine metric, finite/dimension validation.
  The manual fixture vectors establish database behavior only. They are neither
  a semantic encoder nor a semantic-retrieval benchmark.

The calling host must authenticate identities and authorize access; selecting an
owner string is not authentication. Logical deletion removes current queryable
records, not independently retained backups, exported traces, or secure disk
residue. These mechanisms do not implement a production memory service.
