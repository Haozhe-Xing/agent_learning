"""Executable contracts for controlled decimal-distance fixtures, not arbitrary prose."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
import re
from typing import Sequence

from .rag import Document


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    text: str
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DistanceQuantity:
    value: str
    unit: str
    source: str
    quote: str
    evidence_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Calculation:
    distance_metres: float
    speed_metres_per_second: int
    seconds: float
    output_value: float
    output_unit: str

    def to_dict(self) -> dict:
        return asdict(self)


DISTANCE = re.compile(
    r'(?:Moon(?: mean)? distance(?: in this teaching fixture)? is |Distance: )'
    r'([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s+([A-Za-z]+)\.?', re.IGNORECASE)


def evidence_records(documents: Sequence[Document]) -> list[EvidenceRecord]:
    if not documents:
        raise ContractError('missing_evidence', 'at least one distance document is required')
    if len(documents) > 3:
        raise ContractError('invalid_input', 'this fixture accepts at most three documents')
    records, seen = [], set()
    for doc in documents:
        if not isinstance(doc, Document) or not isinstance(doc.source, str) or not re.fullmatch(
                r'[A-Za-z0-9_-]{1,64}', doc.source):
            raise ContractError('invalid_input', 'source must be a short, explicit fixture identity')
        if doc.source in seen:
            raise ContractError('invalid_input', 'duplicate source identity')
        if not isinstance(doc.text, str) or not doc.text.strip() or len(doc.text.encode('utf-8')) > 4000:
            raise ContractError('invalid_input', 'evidence text must contain 1–4000 UTF-8 bytes')
        seen.add(doc.source)
        records.append(EvidenceRecord(doc.source, doc.text, sha256(doc.text.encode('utf-8')).hexdigest()))
    return records


def extract_distance(records: Sequence[EvidenceRecord]) -> DistanceQuantity:
    if not records:
        raise ContractError('missing_evidence', 'no evidence reached the extraction tool')
    quantities, metres = [], []
    for record in records:
        match = DISTANCE.fullmatch(record.text.strip())
        if match is None:
            raise ContractError('invalid_quantity', 'expected one supported decimal distance statement')
        value, unit = match.groups()
        unit = unit.lower()
        if sum(char.isdigit() for char in value) > 24:
            raise ContractError('invalid_quantity', 'fixture numbers support at most 24 decimal digits')
        if unit not in {'m', 'km'}:
            raise ContractError('unsupported_unit', f'unsupported distance unit: {unit}')
        amount = Decimal(value)
        normal = amount * (1000 if unit == 'km' else 1)
        if not Decimal('1e-12') <= normal <= Decimal('1e18'):
            raise ContractError('invalid_quantity', 'distance must be between 1e-12 and 1e18 metres')
        quantities.append(DistanceQuantity(value, unit, record.source, record.text, record.sha256))
        metres.append(normal)
    if any(value != metres[0] for value in metres[1:]):
        raise ContractError('conflicting_evidence', 'distance documents disagree after unit conversion')
    return quantities[0]


def calculate_travel_time(quantity: DistanceQuantity, output_unit: str) -> Calculation:
    if output_unit not in {'seconds', 'milliseconds'}:
        raise ContractError('unsupported_output_unit', 'choose seconds or milliseconds')
    if quantity.unit not in {'m', 'km'}:
        raise ContractError('unsupported_unit', 'calculation requires an m or km quantity')
    if (not isinstance(quantity.value, str) or not re.fullmatch(
            r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)', quantity.value)
            or sum(char.isdigit() for char in quantity.value) > 24):
        raise ContractError('invalid_quantity', 'calculation requires a supported decimal number')
    metres = Decimal(quantity.value) * (1000 if quantity.unit == 'km' else 1)
    if not metres.is_finite() or not Decimal('1e-12') <= metres <= Decimal('1e18'):
        raise ContractError('invalid_quantity', 'distance is outside the calculation contract')
    seconds = float(metres / Decimal(299792458))
    return Calculation(float(metres), 299792458, seconds,
                       seconds * (1000 if output_unit == 'milliseconds' else 1), output_unit)
