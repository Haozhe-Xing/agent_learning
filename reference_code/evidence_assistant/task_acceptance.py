"""Independent, explicit task coverage contracts for the teaching application.

This module never imports the answer generator or retrieves its own evidence.
Literal requirements are appropriate for a frozen excerpt task, not general
natural-language entailment. Unknown questions stay unknown without an explicit
specification or a caller-supplied independent verifier.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Callable, Sequence


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize(value: str) -> str:
    # Whitespace/case normalization permits formatting changes, not synonyms.
    return ''.join(value.split()).casefold()


def _validate(specs) -> list[dict]:
    if not isinstance(specs, list) or not specs or len(specs) > 64:
        raise ValueError('tasks must be a non-empty list of at most 64 specifications')
    questions = set()
    for spec in specs:
        if (not isinstance(spec, dict) or set(spec) != {'question', 'requirements'}
                or not _text(spec['question'])):
            raise ValueError('each task needs question and requirements only')
        question = _normalize(spec['question'])
        if question in questions:
            raise ValueError('task questions must be unique')
        questions.add(question)
        requirements = spec['requirements']
        if not isinstance(requirements, list) or not 1 <= len(requirements) <= 32:
            raise ValueError('each task needs 1–32 requirements')
        identifiers = set()
        for requirement in requirements:
            if (not isinstance(requirement, dict)
                    or set(requirement) != {'id', 'description', 'all_of'}
                    or not _text(requirement['id']) or not _text(requirement['description'])
                    or not isinstance(requirement['all_of'], list)
                    or not 1 <= len(requirement['all_of']) <= 16
                    or any(not _text(fragment) for fragment in requirement['all_of'])):
                raise ValueError('requirements need id, description, and 1–16 non-empty all_of fragments')
            if requirement['id'] in identifiers:
                raise ValueError('requirement ids must be unique within a task')
            identifiers.add(requirement['id'])
    return specs


def load_task_specs(path: str | Path) -> list[dict]:
    """Load the explicit schema_version=1 JSON contract; reject malformed specs."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'tasks'}
            or type(payload['schema_version']) is not int or payload['schema_version'] != 1):
        raise ValueError('task specification needs schema_version=1 and tasks')
    return _validate(payload['tasks'])


def check_question_coverage(question: str, answer: str, *, specs: list[dict] | None = None,
                            verifier: Callable | None = None, documents: Sequence = ()) -> dict:
    """Check frozen requirements and/or a separate task verifier.

    Source authenticity and quote validity are checked elsewhere. A configured
    specification and verifier must both pass. Without a matching specification,
    an explicit verifier returning exactly True can establish task acceptance.
    Exceptions fail closed, while KeyboardInterrupt remains interruptible.
    """
    result = {'status': 'unknown', 'matched_points': [], 'missing_points': [],
              'reason': 'no_task_spec_or_verifier', 'checks': [], 'details': []}
    if not _text(question) or not _text(answer):
        return {**result, 'status': 'failed', 'reason': 'empty_question_or_answer'}
    validated = _validate(specs) if specs is not None else []
    specification = next((spec for spec in validated
                          if _normalize(spec['question']) == _normalize(question)), None)
    if specification is not None:
        result['checks'].append('literal_task_requirements')
        # Attribution is metadata, not an answer point. Strip only the explicit
        # line-prefix label; brackets inside an excerpt remain actual content.
        body = re.sub(r'(?m)^[ \t]*\[[^\[\]\r\n]+\][ \t]+', '', answer)
        normalized_answer = _normalize(body)
        for requirement in specification['requirements']:
            missing = [fragment for fragment in requirement['all_of']
                       if _normalize(fragment) not in normalized_answer]
            field = 'missing_points' if missing else 'matched_points'
            result[field].append(requirement['description'])
            result['details'].append({'id': requirement['id'],
                                      'description': requirement['description'],
                                      'missing_fragments': missing})
        result.update(status='failed' if result['missing_points'] else 'passed',
                      reason='missing_required_points' if result['missing_points'] else 'requirements_met')
    if verifier is not None:
        result['checks'].append('independent_task_verifier')
        try:
            accepted = verifier(question, answer, tuple(documents)) is True
        except Exception as exc:
            return {**result, 'status': 'failed', 'reason': 'verifier_error',
                    'error_type': type(exc).__name__}
        if not accepted:
            result.update(status='failed', reason='task_verifier_rejected')
        elif result['status'] == 'unknown':
            result.update(status='passed', reason='task_verifier_passed')
    return result
