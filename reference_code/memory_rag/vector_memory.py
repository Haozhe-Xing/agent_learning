"""Real local Chroma mechanics with caller-supplied vectors; no embedding service."""
from __future__ import annotations

import math
from pathlib import Path
import re


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('expected a bounded ASCII identity')
    return value


class VectorMemory:
    """Owner-filtered teaching store, not a multi-tenant authentication service.

    The host must supply an authenticated owner on every call. Fixed vectors in
    the demo/tests exercise storage and cosine ranking, not semantic quality.
    """
    def __init__(self, path: str | Path, *, dimension: int = 3):
        if type(dimension) is not int or not 1 <= dimension <= 4096:
            raise ValueError('invalid embedding dimension')
        import chromadb
        from chromadb.config import Settings
        self.dimension = dimension
        self.client = chromadb.PersistentClient(str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            'chapter_memory_fixture', embedding_function=None,
            metadata={'hnsw:space': 'cosine', 'dimension': dimension, 'embedding_version': 'manual-fixture-v1'})
        meta = self.collection.metadata or {}
        if (meta.get('hnsw:space'), meta.get('dimension'), meta.get('embedding_version')) != (
                'cosine', dimension, 'manual-fixture-v1'):
            raise ValueError('existing collection has a different vector contract')

    def _vector(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != self.dimension:
            raise ValueError('vector dimension mismatch')
        if any(type(x) not in (int, float) for x in values):
            raise ValueError('vectors must contain finite numbers')
        try:
            numbers = [float(x) for x in values]
        except (ValueError, OverflowError) as exc:
            raise ValueError('vector values exceed finite numeric range') from exc
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('vectors must contain finite numbers')
        norm = math.hypot(*numbers)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('vector must have a finite nonzero norm')
        return [x / norm for x in numbers]

    def put(self, owner: str, record_id: str, text: str, vector, *, source: str,
            consent: bool = False) -> None:
        owner, record_id = _id(owner), _id(record_id)
        if consent is not True:
            raise PermissionError('explicit Boolean consent required for persistent writes')
        if (not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 8192
                or not isinstance(source, str) or not source.strip() or len(source) > 240):
            raise ValueError('bounded text and provenance required')
        vector = self._vector(vector)
        self.collection.upsert(ids=[f'{owner}:{record_id}'], documents=[text], embeddings=[vector],
                               metadatas=[{'owner': owner, 'record_id': record_id, 'source': source}])

    def search(self, owner: str, vector, *, limit: int = 5, source: str | None = None,
               min_cosine: float = -1.0) -> list[dict]:
        owner, vector = _id(owner), self._vector(vector)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be in [1,100]')
        if type(min_cosine) not in (int, float) or not -1 <= min_cosine <= 1:
            raise ValueError('cosine threshold must be in [-1,1]')
        where = {'owner': owner}
        if source is not None:
            if not isinstance(source, str) or not source.strip():
                raise ValueError('source filter must be non-empty text')
            where = {'$and': [where, {'source': source}]}
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(query_embeddings=[vector], n_results=min(limit, count),
                                       where=where, include=['documents', 'metadatas', 'distances'])
        rows = zip(result['ids'][0], result['documents'][0], result['metadatas'][0], result['distances'][0])
        return [{'id': i, 'text': text, 'source': meta['source'], 'cosine': 1 - distance}
                for i, text, meta, distance in rows if 1 - distance >= min_cosine]

    def delete(self, owner: str, record_id: str | None = None) -> None:
        owner = _id(owner)
        args = {'where': {'owner': owner}}
        if record_id is not None:
            args['ids'] = [f'{owner}:{_id(record_id)}']
        self.collection.delete(**args)
