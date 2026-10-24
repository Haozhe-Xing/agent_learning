"""Run with python -m reference_code.vector_retrieval, or python __main__.py."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'vector_retrieval'

import json
from tempfile import TemporaryDirectory
from .fixture import build_fixture, fixture_scorer
from .retrieval import hybrid_search, rerank


def main():
    with TemporaryDirectory() as directory:
        store = build_fixture(directory)
        vector = store.search('404', limit=3)
        hybrid = hybrid_search(store, '404', limit=3)
        proposed = rerank('404', hybrid, fixture_scorer)
        print(json.dumps({'vector_ids': [h.record.id for h in vector],
                          'hybrid_ids': [h.record.id for h in hybrid],
                          'reranker_proposed_ids': [h.record.id for h in proposed],
                          'answer_accepted': False,
                          'scope': 'fixed vectors and proposed ranking only'}, indent=2))


if __name__ == '__main__':
    main()
