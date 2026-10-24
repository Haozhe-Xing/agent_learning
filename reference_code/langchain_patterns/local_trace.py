"""A local callback recorder which does not collect prompt/output text."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler


class LocalTrace(BaseCallbackHandler):
    """Record chain/model lifecycle identifiers and allowlisted metadata.

    Metadata values are caller-selected identifiers, not an automatic privacy
    classifier. This recorder does not provide token/cost or wall-clock metrics.
    A capacity bound drops excess diagnostics without changing task acceptance.
    """
    def __init__(self, max_events: int = 1000):
        if type(max_events) is not int or max_events < 1:
            raise ValueError('max_events must be a positive integer')
        self.max_events = max_events
        self.events: list[dict] = []
        self.dropped = 0
        self._lock = Lock()

    def _record(self, event, kind, run_id, parent_run_id, metadata=None, error=None):
        item = {'event': event, 'kind': kind, 'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None}
        if metadata:
            item['metadata'] = {
                key: metadata[key] for key in ('case_id', 'app_version')
                if isinstance(metadata.get(key), str) and len(metadata[key]) <= 64
            }
        if error is not None:
            item['error_type'] = type(error).__name__
        with self._lock:
            if len(self.events) >= self.max_events:
                self.dropped += 1
            else:
                self.events.append(item)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       metadata=None, **kwargs):
        self._record('start', 'chain', run_id, parent_run_id, metadata)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._record('end', 'chain', run_id, parent_run_id)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record('error', 'chain', run_id, parent_run_id, error=error)

    def on_chat_model_start(self, serialized, messages, *, run_id,
                            parent_run_id=None, metadata=None, **kwargs):
        self._record('start', 'chat_model', run_id, parent_run_id, metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._record('end', 'chat_model', run_id, parent_run_id)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record('error', 'chat_model', run_id, parent_run_id, error=error)

    def write_new(self, path: str | Path):
        """Write a new local JSON file; never overwrite an existing artifact."""
        with self._lock:
            payload = {'events': self.events[:], 'dropped': self.dropped}
        with Path(path).open('x', encoding='utf-8') as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write('\n')
