"""Small single-writer collection with explicit embedding provenance.

Metadata filters are data selection, not authentication or tenant isolation.
No model downloads or provider calls are built into this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Callable


def positive(value: int, maximum: int = 100) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'integer in [1, {maximum}] required')
    return value


def text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('non-empty text required')
    return value


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    source: str
    version: str

    def __post_init__(self):
        for value in (self.id, self.text, self.source, self.version):
            text(value)


@dataclass(frozen=True)
class Hit:
    record: Record
    distance: float

    @property
    def cosine(self) -> float:
        # Only meaningful because this store explicitly configures cosine space.
        return 1.0 - self.distance


class VectorStore:
    """At most 2,000 chunks; validated batches are at most 100 records.

    `embed(texts)` must return finite, nonzero vectors in input order, all from
    the declared embedding revision/preprocessing. Same dimension is not enough.
    Caller-supplied embedding_id is provenance, not cryptographic attestation.
    """
    def __init__(self, path: str | Path, embed: Callable, *, embedding_id: str,
                 dimension: int, collection_name: str = 'book_retrieval'):
        import chromadb
        from chromadb.config import Settings
        self.embed = embed
        self.dimension = positive(dimension, 65536)
        self.embedding_id = text(embedding_id)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False))
        expected = {'embedding_id': self.embedding_id, 'dimension': self.dimension,
                    'schema': 'book_retrieval_v1', 'normalization': 'stable_l2_v1'}
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata=expected,
            configuration={'hnsw': {'space': 'cosine'}})
        if (self.collection.metadata != expected or
                self.collection.configuration.get('hnsw', {}).get('space') != 'cosine'):
            raise ValueError('collection provenance/configuration differs; build a new collection')

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embed(texts)
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
            raise ValueError('embedding batch size differs from input')
        checked = []
        for vector in vectors:
            if not isinstance(vector, (list, tuple)) or len(vector) != self.dimension:
                raise ValueError('embedding dimension differs')
            if any(type(v) not in (int, float) for v in vector):
                raise ValueError('embedding values must be finite real numbers')
            try:
                converted = [float(v) for v in vector]
            except (OverflowError, ValueError) as error:
                raise ValueError('embedding values must fit finite float64') from error
            if not all(isfinite(v) for v in converted):
                raise ValueError('embedding values must be finite real numbers')
            scale = max(abs(v) for v in converted)
            if scale == 0:
                raise ValueError('zero vector has no cosine direction')
            # Chroma stores float32. Stable normalization prevents finite Python
            # values from becoming all-zero or infinite on that conversion.
            scaled = [v / scale for v in converted]
            norm = sqrt(sum(v * v for v in scaled))
            checked.append([v / norm for v in scaled])
        return checked

    def add(self, records: list[Record]) -> None:
        if not records:
            return
        positive(len(records))
        if any(not isinstance(record, Record) for record in records):
            raise ValueError('Record values required')
        ids = [record.id for record in records]
        if len(set(ids)) != len(ids) or self.collection.get(ids=ids, include=[])['ids']:
            raise ValueError('duplicate id; replace a source explicitly rather than silently ignoring it')
        if self.collection.count() + len(records) > 2000:
            raise ValueError('teaching collection capacity exceeded')
        vectors = self._embed([record.text for record in records])
        # Validate the whole batch before the write. Storage failures still need
        # explicit recovery; this wrapper does not promise a multi-call transaction.
        self.collection.add(ids=ids, documents=[r.text for r in records],
                            embeddings=vectors,
                            metadatas=[{'source': r.source, 'version': r.version} for r in records])

    def records(self, *, source: str | None = None) -> list[Record]:
        kwargs = {'where': {'source': text(source)}} if source is not None else {}
        result = self.collection.get(limit=2001, include=['documents', 'metadatas'], **kwargs)
        if len(result['ids']) > 2000:
            raise ValueError('teaching collection capacity exceeded')
        return sorted((Record(i, d, m['source'], m['version']) for i, d, m in
                       zip(result['ids'], result['documents'], result['metadatas'])),
                      key=lambda r: r.id)

    def search(self, query: str, *, limit: int = 5, source: str | None = None,
               min_cosine: float = -1.0) -> list[Hit]:
        text(query); positive(limit)
        if type(min_cosine) not in (int, float) or not isfinite(min_cosine) or not -1 <= min_cosine <= 1:
            raise ValueError('cosine threshold must be finite and in [-1, 1]')
        count = self.collection.count()
        if not count:
            return []  # In particular, do not embed or query with n_results=0.
        if count > 2000:
            raise ValueError('teaching collection capacity exceeded')
        kwargs = {'where': {'source': text(source)}} if source is not None else {}
        result = self.collection.query(
            query_embeddings=self._embed([query]), n_results=min(limit, count),
            include=['documents', 'metadatas', 'distances'], **kwargs)
        # Chroma always returns ids. It is not a valid include item in this version.
        hits = [Hit(Record(i, d, m['source'], m['version']), float(distance))
                for i, d, m, distance in zip(result['ids'][0], result['documents'][0],
                                            result['metadatas'][0], result['distances'][0])]
        return sorted((h for h in hits if h.cosine >= min_cosine),
                      key=lambda h: (h.distance, h.record.id))

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={'source': text(source)})
