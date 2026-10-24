"""Explicit field projection and consent-scoped ephemeral notes.

No general PII detection, encryption, authentication or durable deletion claim.
"""
from __future__ import annotations

from .runtime import Rejected, _identifier, _text


def project_record(kind: str, data: dict) -> dict:
    fields = {'order_status': ('order_id', 'status'), 'notice': ('title', 'summary')}
    if type(kind) is not str or kind not in fields or type(data) is not dict:
        raise Rejected('unknown_projection')
    if not all(key in data for key in fields[kind]):
        raise Rejected('missing_field')
    # Allowlist selected fields; unknown tasks never fall back to all fields.
    return {key: _text(data[key], 600) for key in fields[kind]}


class ScopedNotes:
    """Trusted host passes actor, never an actor copied from a model argument.

    One object is one local process store. There are no disk writes, remote
    backups, access tokens or externally authenticated sessions in this lab.
    """
    def __init__(self):
        self._notes = {}

    def put(self, actor: str, note_id: str, text: str, *, consent: bool):
        _identifier(actor); _identifier(note_id); _text(text, 600)
        if consent is not True:
            raise Rejected('consent_required')
        key = (actor, note_id)
        if key not in self._notes and len(self._notes) >= 100:
            raise Rejected('note_capacity')
        self._notes[key] = text

    def get(self, actor: str, note_id: str):
        _identifier(actor); _identifier(note_id)
        return self._notes.get((actor, note_id))

    def delete(self, actor: str, note_id: str) -> bool:
        _identifier(actor); _identifier(note_id)
        return self._notes.pop((actor, note_id), None) is not None
