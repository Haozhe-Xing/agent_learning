"""Bounded dialogue, observable task state, and source-preserving text chunks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('non-empty text required')
    return value


def _positive(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError('positive integer required')
    return value


def json_bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8'))


class BudgetExceeded(ValueError):
    pass


class BoundedHistory:
    """Bound serialized UTF-8 input bytes; this is not a model token counter.

    Only completed user/assistant turns enter history. Current input and fixed
    instructions are mandatory. A supplied summary remains untrusted user data.
    """
    def __init__(self, system: str, max_input_bytes: int = 1200, max_turns: int = 4):
        self.system = _text(system)
        self.max_input_bytes = _positive(max_input_bytes)
        self.max_turns = _positive(max_turns)
        self.turns: list[tuple[str, str]] = []

    def append_turn(self, user: str, assistant: str) -> None:
        self.turns.append((_text(user), _text(assistant)))
        self.turns = self.turns[-self.max_turns:]

    def build(self, query: str, *, summary: str = '') -> dict:
        query = _text(query)
        if not isinstance(summary, str):
            raise ValueError('summary must be text')
        def render(turns, reference=''):
            messages = [{'role': 'system', 'content': self.system}]
            if reference:
                messages.append({'role': 'user', 'content': 'Untrusted history summary: '
                                 + json.dumps(reference, ensure_ascii=False)})
            for user, assistant in turns:
                messages.extend([{'role': 'user', 'content': user},
                                 {'role': 'assistant', 'content': assistant}])
            return messages + [{'role': 'user', 'content': query}]
        if json_bytes(render([])) > self.max_input_bytes:
            raise BudgetExceeded('fixed instructions and current query exceed byte budget')
        selected = []
        # Preserve the newest contiguous sequence of complete turns.
        for turn in reversed(self.turns):
            proposal = [turn] + selected
            if json_bytes(render(proposal)) > self.max_input_bytes:
                break
            selected = proposal
        use_summary = bool(summary) and json_bytes(render(selected, summary)) <= self.max_input_bytes
        messages = render(selected, summary if use_summary else '')
        return {'messages': messages, 'input_bytes': json_bytes(messages),
                'dropped_turns': len(self.turns) - len(selected),
                'summary_omitted': bool(summary) and not use_summary}


class TaskState:
    """Task-scoped observable results, not private reasoning or system policy.

    The host supplies the authenticated owner/task and independent verifier.
    A caller who can select any owner is not authenticated by this data structure.
    """
    def __init__(self, owner: str, task_id: str, max_entries: int = 8):
        self.owner, self.task_id = _text(owner), _text(task_id)
        self.max_entries = _positive(max_entries)
        self._entries = {}

    def _scope(self, owner, task_id):
        if (owner, task_id) != (self.owner, self.task_id):
            raise PermissionError('working state belongs to another owner or task')

    def record(self, owner: str, task_id: str, key: str, value, *, source: str,
               verifier: Callable | None = None) -> None:
        self._scope(owner, task_id)
        _text(key); _text(source)
        if key not in self._entries and len(self._entries) >= self.max_entries:
            raise BudgetExceeded('working-state capacity reached; no silent eviction')
        if json_bytes(value) > 2048:
            raise BudgetExceeded('working-state value exceeds byte limit')
        # Round-trip to detach mutable values supplied by a caller.
        detached = json.loads(json.dumps(value, allow_nan=False))
        if detached != value:
            raise ValueError('use JSON string keys and list arrays without implicit conversion')
        value = detached
        if verifier is None:
            status = 'observed'
        else:
            try:
                # A verifier may inspect or mutate its input, but cannot replace
                # the bounded JSON value retained as the original observation.
                verification_input = json.loads(json.dumps(value))
                status = 'verified' if verifier(verification_input) is True else 'rejected'
            except Exception:
                status = 'rejected'
        self._entries[key] = {'value': value, 'source': source, 'status': status}

    def snapshot(self, owner: str, task_id: str) -> dict:
        self._scope(owner, task_id)
        return json.loads(json.dumps(self._entries))

    def clear(self, owner: str, task_id: str) -> None:
        self._scope(owner, task_id)
        self._entries.clear()


@dataclass(frozen=True)
class Document:
    source: str
    version: str
    text: str


def load_text(root: str | Path, relative: str, *, max_bytes: int = 100_000) -> Document:
    """Load one local UTF-8 .txt/.md file inside a trusted root; no URL fetches.

    Root checks suit a local exercise. They are not an OS sandbox against a
    hostile process concurrently replacing files or changing the filesystem.
    """
    _positive(max_bytes); _text(relative)
    root = Path(root).resolve(strict=True)
    requested = Path(relative)
    if requested.is_absolute():
        raise PermissionError('use a relative path within the document root')
    path = (root / requested).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError('document outside root or not a regular file')
    if path.suffix.lower() not in {'.txt', '.md'}:
        raise ValueError('this loader supports local .txt and .md only')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise BudgetExceeded('document exceeds byte limit')
    return Document(path.relative_to(root).as_posix(), hashlib.sha256(raw).hexdigest(),
                    raw.decode('utf-8'))


def split_document(document: Document, max_chars: int = 200, overlap: int = 30) -> list[dict]:
    """Exact character windows with source offsets and no synthetic text.

    Offsets are Python Unicode code-point indices, not byte/token/page offsets.
    No paragraph, Markdown, table, or semantic integrity guarantee is made.
    """
    _positive(max_chars)
    if not isinstance(document, Document) or not isinstance(document.text, str):
        raise ValueError('a loaded text Document is required')
    _text(document.source)
    if document.version != hashlib.sha256(document.text.encode('utf-8')).hexdigest():
        raise ValueError('document text does not match its source version digest')
    if type(overlap) is not int or not 0 <= overlap < max_chars:
        raise ValueError('overlap must satisfy 0 <= overlap < max_chars')
    chunks, start = [], 0
    while start < len(document.text):
        end = min(start + max_chars, len(document.text))
        identity = json.dumps([document.source, document.version, max_chars, overlap, start, end])
        chunks.append({'id': hashlib.sha256(identity.encode()).hexdigest(),
                       'source': document.source, 'version': document.version,
                       'start_char': start, 'end_char': end,
                       'text': document.text[start:end]})
        if end == len(document.text):
            break
        start = end - overlap
    return chunks
