"""Rank proposals, preserving source identity; no answer acceptance here."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log
import re

from .store import Record, VectorStore, positive, text


@dataclass(frozen=True)
class Ranked:
    record: Record
    score: float


def terms(value: str) -> list[str]:
    """Teaching tokenizer: ASCII identifiers plus adjacent CJK bigrams.

    This is not general multilingual segmentation, stemming, or semantic search.
    """
    output = re.findall(r'[a-z0-9_]+', value.casefold())
    for block in re.findall(r'[\u4e00-\u9fff]+', value):
        output.extend([block] if len(block) == 1 else
                      [block[i:i+2] for i in range(len(block) - 1)])
    return output


def bm25_search(query: str, records: list[Record], *, limit: int = 10) -> list[Ranked]:
    text(query); positive(limit)
    if len({r.id for r in records}) != len(records):
        raise ValueError('lexical corpus ids must be unique')
    counters = [Counter(terms(r.text)) for r in records]
    average = sum(sum(c.values()) for c in counters) / len(records) if records else 0
    if not average:
        return []
    scores = [0.0] * len(records)
    for term in set(terms(query)):
        frequency = sum(term in c for c in counters)
        idf = log(1 + (len(records) - frequency + 0.5) / (frequency + 0.5))
        for index, counter in enumerate(counters):
            tf = counter[term]
            denominator = tf + 1.2 * (1 - 0.75 + 0.75 * sum(counter.values()) / average)
            scores[index] += idf * tf * 2.2 / denominator
    return sorted((Ranked(r, score) for r, score in zip(records, scores) if score > 0),
                  key=lambda hit: (-hit.score, hit.record.id))[:limit]


def fuse_ranks(rankings: list[list[Record]], *, limit: int = 5,
               rank_constant: int = 60) -> list[Ranked]:
    positive(limit); positive(rank_constant, 10000)
    records, totals = {}, Counter()
    for ranking in rankings:
        seen = set()
        for rank, record in enumerate(ranking, start=1):
            if record.id in seen:
                raise ValueError('duplicate id in one ranking would create extra votes')
            seen.add(record.id)
            if record.id in records and records[record.id] != record:
                raise ValueError('same id refers to conflicting source records')
            records[record.id] = record
            totals[record.id] += 1 / (rank_constant + rank)
    return [Ranked(records[key], score) for key, score in
            sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def hybrid_search(store: VectorStore, query: str, *, limit: int = 5,
                  candidates: int = 10, source: str | None = None) -> list[Ranked]:
    positive(limit); positive(candidates)
    if limit > candidates:
        raise ValueError('candidate budget must cover final limit')
    corpus = store.records(source=source)
    vector = store.search(query, limit=candidates, source=source)
    lexical = bm25_search(query, corpus, limit=candidates)
    return fuse_ranks([[h.record for h in vector], [h.record for h in lexical]], limit=limit)


def rerank(query: str, candidates: list[Ranked], scorer, *, limit: int = 3,
           min_score: float = 0.5) -> list[Ranked]:
    """Validate complete candidate score proposals, including tiny candidate sets.

    Scores remain uncalibrated proposals, not verified relevance or truth.
    A provider error propagates; no silent fallback labels candidates relevant.
    """
    text(query); positive(limit)
    if type(min_score) not in (int, float) or not isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError('minimum proposed score must be in [0, 1]')
    if not candidates:
        return []
    records = {hit.record.id: hit.record for hit in candidates}
    if len(records) != len(candidates):
        raise ValueError('duplicate candidate ids')
    proposals = scorer(query, [{'id': r.id, 'text': r.text, 'source': r.source,
                                'version': r.version} for r in records.values()])
    if not isinstance(proposals, list) or len(proposals) != len(records):
        raise ValueError('one score proposal per candidate required')
    scored, seen = [], set()
    for item in proposals:
        if not isinstance(item, dict) or set(item) != {'id', 'score'}:
            raise ValueError('each proposal must contain id and score only')
        key, score = item['id'], item['score']
        if not isinstance(key, str) or key not in records or key in seen:
            raise ValueError('unknown or duplicate candidate id')
        seen.add(key)
        if type(score) not in (int, float) or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError('proposed score must be finite and in [0, 1]')
        if score >= min_score:
            scored.append(Ranked(records[key], float(score)))
    return sorted(scored, key=lambda h: (-h.score, h.record.id))[:limit]


def query_variants(original: str, proposals: list[str], *, max_variants: int = 2) -> list[str]:
    text(original); positive(max_variants, 10)
    if not isinstance(proposals, list) or len(proposals) > max_variants:
        raise ValueError('query proposal budget exceeded')
    result, seen = [], set()
    for query in [original, *proposals]:
        text(query)
        if len(query) > 200:
            raise ValueError('query exceeds character budget')
        cleaned = query.strip()
        if cleaned.casefold() not in seen:
            seen.add(cleaned.casefold()); result.append(cleaned)
    return result
