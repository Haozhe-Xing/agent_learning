"""Independent checks against caller evidence, without the producer's arithmetic.

Source hashes check trace consistency, not a real-world document's authenticity.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import math
import re
from typing import Sequence

from .rag import Document


def _trusted_distance(text: str) -> tuple[Decimal, str, Decimal]:
    # A separate whole-document parser for the declared fixture grammar.
    body = text.strip().lower()
    if body.endswith('.'):
        body = body[:-1]
    prefixes = ('moon mean distance in this teaching fixture is ',
                'moon distance is ', 'distance: ', 'moon mean distance is ',
                'moon distance in this teaching fixture is ')
    prefix = next((item for item in prefixes if body.startswith(item)), None)
    if prefix is None:
        raise ValueError('unsupported evidence statement')
    parts = body[len(prefix):].split()
    if len(parts) != 2 or not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)', parts[0]):
        raise ValueError('invalid distance')
    if len(re.sub('[^0-9]', '', parts[0])) > 24:
        raise ValueError('distance precision exceeds the fixture contract')
    value, unit = Decimal(parts[0]), parts[1]
    if unit == 'km':
        metres = value * Decimal('1000')
    elif unit == 'm':
        metres = value
    else:
        raise ValueError('unsupported unit')
    if not value.is_finite() or not Decimal('1e-12') <= metres <= Decimal('1e18'):
        raise ValueError('distance out of range')
    return value, unit, metres


def _close(actual, expected: Decimal) -> bool:
    if type(actual) not in (int, float) or not math.isfinite(actual):
        return False
    difference = abs(Decimal(str(actual)) - expected)
    return difference <= abs(expected) * Decimal('1e-12')


def verify_light_travel(artifact: dict, documents: Sequence[Document], output_unit: str) -> dict:
    """Return a decision with reasons; never accept the producer's own 'success'."""
    try:
        with localcontext() as context:
            context.prec = 50
            if not documents or len(documents) > 3 or output_unit not in {'seconds', 'milliseconds'}:
                raise ValueError('invalid task inputs')
            trusted, parsed = {}, {}
            for doc in documents:
                if (not isinstance(doc.source, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', doc.source)
                        or doc.source in trusted or not isinstance(doc.text, str)
                        or not 0 < len(doc.text.encode('utf-8')) <= 4000):
                    raise ValueError('invalid source identity or text')
                trusted[doc.source] = {'source': doc.source, 'text': doc.text,
                                      'sha256': sha256(doc.text.encode('utf-8')).hexdigest()}
                parsed[doc.source] = _trusted_distance(doc.text)
            expected_metres = next(iter(parsed.values()))[2]
            if any(values[2] != expected_metres for values in parsed.values()):
                raise ValueError('conflicting evidence')
            records = artifact.get('provenance', [])
            if (not isinstance(records, list) or len(records) != len(trusted)
                    or any(not isinstance(record, dict) for record in records)
                    or len({record.get('source') for record in records}) != len(trusted)
                    or {record['source']: record for record in records} != trusted):
                raise ValueError('provenance does not match task evidence')
            quantity = artifact['quantity']
            source = quantity['source']
            if source not in trusted:
                raise ValueError('unknown quantity source')
            expected_value, expected_unit, _ = parsed[source]
            if (not isinstance(quantity['value'], str) or Decimal(quantity['value']) != expected_value
                    or quantity['unit'] != expected_unit or quantity['quote'] != trusted[source]['text']
                    or quantity['evidence_sha256'] != trusted[source]['sha256']):
                raise ValueError('quantity or citation does not match original evidence')
            expected_seconds = expected_metres / Decimal('299792458')
            expected_output = expected_seconds * (1000 if output_unit == 'milliseconds' else 1)
            calc, output = artifact['calculation'], artifact['output']
            if (not _close(calc['distance_metres'], expected_metres)
                    or type(calc['speed_metres_per_second']) is not int
                    or calc['speed_metres_per_second'] != 299792458
                    or not _close(calc['seconds'], expected_seconds)
                    or not _close(calc['output_value'], expected_output)
                    or calc['output_unit'] != output_unit
                    or not _close(artifact['seconds'], expected_seconds)
                    or not _close(artifact['distance_km'], expected_metres / 1000)
                    or output['unit'] != output_unit or not _close(output['value'], expected_output)):
                raise ValueError('calculation or output does not match task inputs')
            answer = artifact['answer']
            match = re.fullmatch(
                r'([+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?) (seconds|milliseconds) \[([A-Za-z0-9_-]+)\]',
                answer) if isinstance(answer, str) else None
            if (match is None or match[2] != output_unit or match[3] != source
                    or not _close(float(match[1]), expected_output)):
                raise ValueError('final answer must match the verified value, unit and quantity source')
    except (KeyError, TypeError, ValueError, AttributeError, InvalidOperation, OverflowError) as exc:
        return {'status': 'failed', 'accepted': False, 'reason': str(exc)}
    return {'status': 'passed', 'accepted': True,
            'checks': ['source_identity', 'source_digest', 'quantity', 'unit', 'calculation', 'output', 'citation']}
