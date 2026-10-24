# 4.7 Practice: MemGPT/Letta Memory Architecture

This executable baseline illustrates Core, Working, and Archive memory. It is an independent teaching implementation rather than a Letta SDK substitute or production security system. The module is `reference_code/evidence_assistant/memory.py`.

| Layer | Implementation | Retention |
|---|---|---|
| Core | User-scoped SQLite records | Explicit consent; survives reopening |
| Working | Bounded deque | Per-instance recent state; oldest items expire |
| Archive | Separate SQLite records | Keyword search on demand |

Writes use explicit `update_core_memory` and `archive_content` commands. Automatic preference inference is an additional model feature; verify the storage contract before adding it.

```bash
python examples/chapter04/layered_memory_agent.py
python examples/chapter04/long_term_memory.py
python examples/chapter04/scratchpad_agent.py
```

```python
from reference_code.evidence_assistant.memory import LayeredMemoryAgent, MemoryStore
store = MemoryStore(":memory:")
agent = LayeredMemoryAgent(store, "demo", working_limit=2)
agent.update_core_memory("name", "Ada", consent=True)
agent.archive_content("project", "churn prediction", consent=True)
assert "churn prediction" in agent.messages("churn")[1]["content"]
agent.forget()
assert agent.search_archive("churn") == {}
store.close()
```

Use a filesystem path instead of `:memory:` to check persistence. Unit tests exercise reopening, user isolation, deletion, and bounded working state. `examples/integration_tests/test_frameworks_and_storage.py` separately uses actual ChromaDB with deterministic embeddings.

Core memory is still unverified data. All layers enter reference messages rather than system rules. User IDs partition records but do not authenticate a requester. Writes without explicit consent fail. Forgetting removes the current user's stored core and archive records and current-instance working state; it does not delete another user's records. Define backup and log deletion separately in production.

After specifying required operations, use the [official Letta documentation](https://docs.letta.com/) for its current versioned API. Recheck authorization, retrieval, and deletion after an upgrade.
