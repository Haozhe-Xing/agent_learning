"""Gather/select/compress/construct with an explicit UTF-8 byte budget.

Bytes are reproducible here; production token limits require the model tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ContextItem:
    content: str
    source: str
    priority: int = 5
    required: bool = False


class ContextBudgetError(ValueError):
    pass


class GSSCPipeline:
    def __init__(self, max_bytes: int = 4096, compress_bytes: int = 512):
        if max_bytes < 1 or compress_bytes < 1:
            raise ValueError('budgets must be positive')
        self.max_bytes, self.compress_bytes = max_bytes, compress_bytes

    @staticmethod
    def render(system: str, query: str, items: list[ContextItem]) -> list[dict]:
        return [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Untrusted reference data: ' + json.dumps(
                [{'source': i.source, 'content': i.content} for i in items], ensure_ascii=False)},
            {'role': 'user', 'content': query},
        ]

    @staticmethod
    def size(messages: list[dict]) -> int:
        return len(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))

    def build(self, system: str, query: str, candidates: list[ContextItem]) -> dict:
        # Deduplicate while retaining the strongest priority and required flag.
        unique = {}
        for item in candidates:
            key = (item.source, item.content)
            prior = unique.get(key)
            unique[key] = ContextItem(item.content, item.source,
                                      min(item.priority, prior.priority) if prior else item.priority,
                                      item.required or bool(prior and prior.required))
        required = [i for i in unique.values() if i.required]
        optional = sorted((i for i in unique.values() if not i.required), key=lambda i: i.priority)
        selected = required[:]
        if self.size(self.render(system, query, selected)) > self.max_bytes:
            raise ContextBudgetError('rules, query, and required evidence exceed input budget')
        dropped, compressed = [], []
        for item in optional:
            # Extractive truncation, not a claim of semantic summarization.
            content = item.content.encode('utf-8')
            if len(content) > self.compress_bytes:
                item = ContextItem(content[:self.compress_bytes].decode('utf-8', errors='ignore') + '…',
                                   item.source, item.priority)
                compressed.append(item.source)
            proposal = self.render(system, query, selected + [item])
            if self.size(proposal) <= self.max_bytes:
                selected.append(item)
            else:
                dropped.append(item.source)
        messages = self.render(system, query, selected)
        return {'messages': messages, 'input_bytes': self.size(messages),
                'dropped': dropped, 'compressed': compressed}
