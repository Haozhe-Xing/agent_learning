"""Local host authorization lab, not an OS sandbox or a model detector.

Only this Python process is trusted. Its backend and approval registry must
never be exposed as tools. One Runtime has one host-selected principal/policy.
The fixture is synchronous, single-process and deliberately has no I/O tools.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
import uuid


class Rejected(ValueError):
    """Fixed reason code; never includes the rejected content."""


def _text(value, maximum: int) -> str:
    if type(value) is not str or not value.strip():
        raise Rejected('invalid_text')
    try:
        size = len(value.encode('utf-8'))
    except UnicodeEncodeError as exc:
        raise Rejected('invalid_text') from exc
    if size > maximum or any(ord(c) < 32 for c in value):
        raise Rejected('invalid_text')
    return value


def _identifier(value) -> str:
    if type(value) is not str or not re.fullmatch(r'[a-zA-Z0-9_-]{1,48}', value):
        raise Rejected('invalid_identifier')
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejected('duplicate_key')
        result[key] = value
    return result


def _constant(value):
    raise Rejected('nonfinite_json')


@dataclass(frozen=True)
class Proposal:
    request_id: str
    tool: str
    # An immutable sorted tuple; callers cannot alter an approved proposal.
    arguments: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        data = [self.request_id, self.tool, dict(self.arguments)]
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def parse_proposal(raw: str) -> Proposal:
    """Strict closed schema. Neither principal nor approval is model input."""
    if type(raw) is not str:
        raise Rejected('invalid_json')
    try:
        size = len(raw.encode('utf-8'))
    except UnicodeEncodeError as exc:
        raise Rejected('invalid_json') from exc
    if size > 4096:
        raise Rejected('invalid_json')
    try:
        data = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise Rejected('invalid_json') from exc
    if type(data) is not dict or set(data) != {'request_id', 'tool', 'arguments'}:
        raise Rejected('invalid_schema')
    request_id = _identifier(data['request_id'])
    tool = data['tool']
    if type(tool) is not str or tool not in {'read_document', 'publish_notice'}:
        raise Rejected('unknown_tool')
    args = data['arguments']
    fields = {'document_id'} if tool == 'read_document' else {'destination', 'text'}
    if type(args) is not dict or set(args) != fields:
        raise Rejected('invalid_arguments')
    if tool == 'read_document':
        checked = {'document_id': _identifier(args['document_id'])}
    else:
        checked = {'destination': _identifier(args['destination']), 'text': _text(args['text'], 600)}
    return Proposal(request_id, tool, tuple(sorted(checked.items())))


def proposal_json(request_id: str, tool: str, **arguments) -> str:
    """Fixture builder; execute() still validates its entire result."""
    return json.dumps(dict(request_id=request_id, tool=tool, arguments=arguments), ensure_ascii=False)


@dataclass(frozen=True)
class HostPolicy:
    principal: str
    tools: frozenset[str]
    documents: frozenset[str] = frozenset()
    destinations: frozenset[str] = frozenset()

    def __post_init__(self):
        _identifier(self.principal)
        # Snapshot even when the trusted caller supplied a mutable set.
        for field in ('tools', 'documents', 'destinations'):
            values = getattr(self, field)
            if not isinstance(values, (set, frozenset)):
                raise ValueError('policy fields must be sets')
            object.__setattr__(self, field, frozenset(_identifier(v) for v in values))
        if not self.tools <= {'read_document', 'publish_notice'}:
            raise ValueError('unknown policy tool')


class ApprovalRegistry:
    """Trusted host-only registry; no model-visible approval tool.

    A real host must authenticate the reviewer and show exact action content.
    This fixture represents that decision with the literal boolean True.
    """
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._records = {}

    def issue(self, principal: str, raw: str, *, approved: bool, ttl: float = 60) -> str:
        _identifier(principal)
        proposal = parse_proposal(raw)
        if approved is not True or proposal.tool != 'publish_notice':
            raise Rejected('approval_not_granted')
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or not 0 < ttl <= 300:
            raise Rejected('invalid_ttl')
        if len(self._records) >= 100:
            raise Rejected('approval_capacity')
        token = uuid.uuid4().hex
        self._records[token] = (principal, proposal.digest, self._clock() + ttl)
        return token

    def valid(self, token, principal: str, proposal: Proposal) -> bool:
        if type(token) is not str:
            return False
        record = self._records.get(token)
        return (record is not None and record[:2] == (principal, proposal.digest)
                and self._clock() < record[2])


class LocalBackend:
    """Fictional records/receipts. No file access, network, shell or real send."""
    def __init__(self):
        self._documents = {
            'alice_policy': ('alice', 'Refund requests require a receipt.', 'CANARY_ALICE_PRIVATE'),
            'bob_policy': ('bob', 'Internal Bob-only draft.', 'CANARY_BOB_PRIVATE'),
        }
        self._receipts = {}
        self._notices = []
        self.read_calls = 0
        self.publish_calls = 0

    def owned(self, principal: str, document_id: str) -> bool:
        record = self._documents.get(document_id)
        return record is not None and record[0] == principal

    def read(self, principal: str, document_id: str) -> dict:
        # Defense in depth at the actual data operation, not only the caller.
        if not self.owned(principal, document_id):
            raise Rejected('document_denied')
        self.read_calls += 1
        return {'document_id': document_id, 'excerpt': self._documents[document_id][1]}

    @property
    def notices(self) -> list[dict]:
        return [dict(item) for item in self._notices]

    def previous(self, principal: str, proposal: Proposal):
        record = self._receipts.get((principal, proposal.request_id))
        if record and record[0] != proposal.digest:
            raise Rejected('id_conflict')
        return dict(record[1]) if record else None

    def publish(self, principal: str, proposal: Proposal) -> dict:
        previous = self.previous(principal, proposal)
        if previous is not None:
            return previous
        # Synchronous in-memory effect + receipt; not a distributed transaction.
        receipt = {'receipt_id': 'local-' + str(len(self._receipts) + 1), 'destination': dict(proposal.arguments)['destination']}
        self._notices.append({'principal': principal, **dict(proposal.arguments)})
        self._receipts[(principal, proposal.request_id)] = (proposal.digest, receipt)
        self.publish_calls += 1
        return dict(receipt)


class Runtime:
    def __init__(self, policy: HostPolicy, backend: LocalBackend, approvals: ApprovalRegistry,
                 *, max_attempts: int = 8):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
            raise ValueError('max_attempts must be an integer in 1..100')
        self._policy = policy
        self.backend = backend
        self.approvals = approvals
        self.max_attempts = max_attempts
        self.attempts = 0
        self._audit = []
        self.run_id = uuid.uuid4().hex

    @property
    def audit(self) -> list[dict]:
        return [dict(item) for item in self._audit]

    def replace_policy(self, policy: HostPolicy):
        """Trusted host revocation path; cannot switch the authenticated actor."""
        if not isinstance(policy, HostPolicy) or policy.principal != self._policy.principal:
            raise ValueError('principal cannot change within a run')
        self._policy = policy

    def _authorize(self, proposal: Proposal):
        policy = self._policy
        if proposal.tool not in policy.tools:
            raise Rejected('tool_denied')
        args = dict(proposal.arguments)
        if proposal.tool == 'read_document':
            if args['document_id'] not in policy.documents or not self.backend.owned(policy.principal, args['document_id']):
                raise Rejected('document_denied')
        elif args['destination'] not in policy.destinations:
            raise Rejected('destination_denied')

    def execute(self, raw: str, *, approval_token=None) -> dict:
        if self.attempts >= self.max_attempts:
            # No new audit entry: its storage is bounded by the attempt budget.
            return {'status': 'rejected', 'code': 'budget_exhausted'}
        self.attempts += 1  # Invalid JSON/denial/errors also consume attempts.
        tool = 'unparsed'
        try:
            proposal = parse_proposal(raw)
            tool = proposal.tool
            self._authorize(proposal)
            if tool == 'read_document':
                value = self.backend.read(self._policy.principal, dict(proposal.arguments)['document_id'])
            else:
                value = self.backend.previous(self._policy.principal, proposal)
                if value is None:
                    if not self.approvals.valid(approval_token, self._policy.principal, proposal):
                        raise Rejected('approval_required')
                    # Recheck immediately before the operation. This lab is synchronous.
                    self._authorize(proposal)
                    value = self.backend.publish(self._policy.principal, proposal)
            result = {'status': 'succeeded', 'value': value}
        except Rejected as exc:
            result = {'status': 'rejected', 'code': str(exc)}
        except Exception:
            # No raw exception, arguments, document content or proof in output/audit.
            result = {'status': 'failed', 'code': 'tool_error'}
        self._audit.append({'run_id': self.run_id, 'attempt': self.attempts,
                            'tool': tool, 'status': result['status'], 'code': result.get('code', 'ok')})
        return result


def injection_signal(text: str) -> str:
    """A narrow lexical signal; no_match NEVER means safe."""
    _text(text, 4096)
    return 'matched' if re.search(r'ignore previous instructions', text, re.I) else 'no_match'
