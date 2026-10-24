"""Deliberately imperfect fixed vectors; not a semantic embedding model."""
from .store import Record, VectorStore

RECORDS = [
    Record('http404-v1:0', 'Error 404: check the requested resource path.', 'http404.md', 'v1'),
    Record('http500-v1:0', 'Error 500: inspect the server exception log.', 'http500.md', 'v1'),
    Record('budget-v1:0', 'The report budget is 30 units.', 'budget.md', 'v1'),
]


def fixed_embeddings(texts):
    vectors = {RECORDS[0].text: [0.8, 0.6, 0.0], RECORDS[1].text: [1.0, 0.0, 0.0],
               RECORDS[2].text: [0.0, 1.0, 0.0], '404': [1.0, 0.0, 0.0],
               'birthday': [0.0, 0.0, 1.0]}
    try:
        return [vectors[value][:] for value in texts]
    except KeyError as error:
        raise ValueError('fixture has no vector for this text; supply an embedding adapter') from error


def fixture_scorer(query, candidates):
    if query != '404':
        raise ValueError('fixture scorer covers query 404 only')
    return [{'id': r['id'], 'score': 0.9 if '404' in r['text'] else 0.1} for r in candidates]


def build_fixture(path):
    store = VectorStore(path, fixed_embeddings, embedding_id='book-fixture-v1:exact-text', dimension=3)
    store.add(RECORDS)
    return store
