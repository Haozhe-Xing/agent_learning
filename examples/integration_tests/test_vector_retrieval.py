"""Real Chroma 1.5.9, exact fixtures, and adversarial ranking proposals."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest

from reference_code.vector_retrieval import Record, VectorStore, Ranked, bm25_search, fuse_ranks, hybrid_search, rerank, query_variants
from reference_code.vector_retrieval.fixture import RECORDS, build_fixture, fixed_embeddings, fixture_scorer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('network use outside this offline fixture')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def test_real_cosine_distance_and_source_filter(tmp_path):
    store = build_fixture(tmp_path)
    hits = store.search('404', limit=3)
    assert [h.record.id for h in hits] == ['http500-v1:0', 'http404-v1:0', 'budget-v1:0']
    assert [h.distance for h in hits] == pytest.approx([0.0, 0.2, 1.0], abs=1e-6)
    assert store.search('404', source='http404.md')[0].record == RECORDS[0]
    assert store.search('404', source='missing.md') == []
    assert store.search('birthday', min_cosine=0.8) == []


def test_empty_store_does_not_embed_and_invalid_query_budget_rejects(tmp_path):
    def fail(_):
        raise AssertionError('empty store must not request embeddings')
    store = VectorStore(tmp_path, fail, embedding_id='empty-v1', dimension=3)
    assert store.search('404') == []
    assert store.records() == []
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):store.search('404', limit=limit)
    for threshold in (float('nan'), float('inf'), 2, True):
        with pytest.raises(ValueError):store.search('404', min_cosine=threshold)


def test_real_cosine_normalizes_lengths_and_keeps_negative_similarity(tmp_path):
    vectors={'positive':[5.0,0.0],'negative':[-3.0,0.0],'query':[1.0,0.0]}
    store=VectorStore(tmp_path,lambda values:[vectors[v]for v in values],embedding_id='directions-v1',dimension=2)
    store.add([Record('pos','positive','pos.txt','v1'),Record('neg','negative','neg.txt','v1')])
    hits=store.search('query')
    assert [h.record.id for h in hits]==['pos','neg']
    assert [h.cosine for h in hits]==pytest.approx([1.0,-1.0],abs=1e-6)


@pytest.mark.parametrize('magnitude',[1e-100,1e300])
def test_float32_storage_does_not_destroy_extreme_finite_directions(tmp_path,magnitude):
    vectors={'positive':[magnitude,0.0],'negative':[-magnitude,0.0],'query':[magnitude,0.0]}
    store=VectorStore(tmp_path,lambda values:[vectors[v]for v in values],embedding_id='extreme-v1',dimension=2)
    store.add([Record('pos','positive','pos.txt','v1'),Record('neg','negative','neg.txt','v1')])
    assert [h.cosine for h in store.search('query')]==pytest.approx([1.0,-1.0],abs=1e-6)


def test_reopen_and_source_deletion_in_an_independent_process(tmp_path):
    build_fixture(tmp_path)
    script = '''
import sys
from reference_code.vector_retrieval import VectorStore
from reference_code.vector_retrieval.fixture import fixed_embeddings
store = VectorStore(sys.argv[1], fixed_embeddings, embedding_id='book-fixture-v1:exact-text', dimension=3)
assert len(store.records()) == 3
store.delete_source('http404.md')
assert [r.id for r in store.records()] == ['budget-v1:0', 'http500-v1:0']
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    reopened = VectorStore(tmp_path, fixed_embeddings, embedding_id='book-fixture-v1:exact-text', dimension=3)
    assert reopened.records(source='http404.md') == []
    assert len(reopened.records()) == 2


def test_incompatible_embedding_revision_dimension_or_distance_rejects(tmp_path):
    store = build_fixture(tmp_path)
    for embedding_id,dimension in [('different-v1',3),('book-fixture-v1:exact-text',2)]:
        with pytest.raises(ValueError):VectorStore(tmp_path, fixed_embeddings, embedding_id=embedding_id, dimension=dimension)
    store.collection.modify(configuration={'hnsw': {'ef_search': 120}})
    assert len(store.records()) == 3
    # A same-named unrelated collection with default l2 is not accepted.
    other = tmp_path/'other'
    import chromadb
    chromadb.PersistentClient(path=str(other)).create_collection('book_retrieval', embedding_function=None)
    with pytest.raises(ValueError):VectorStore(other, fixed_embeddings, embedding_id='book-fixture-v1:exact-text', dimension=3)


@pytest.mark.parametrize('vectors', [[], [[1, 0]], [[0, 0, 0]], [[float('nan'), 0, 0]], [[True, 0, 0]], [[1, '0', 0]], [[10**1000,0,0]]])
def test_invalid_embedding_batch_never_writes(tmp_path, vectors):
    store = VectorStore(tmp_path, lambda _: vectors, embedding_id='bad-v1', dimension=3)
    with pytest.raises(ValueError):store.add([RECORDS[0]])
    assert store.records() == []


def test_duplicate_ids_fail_before_embedding_or_partial_write(tmp_path):
    store = build_fixture(tmp_path)
    store.embed = lambda _: (_ for _ in ()).throw(AssertionError('must reject before embedding'))
    with pytest.raises(ValueError):store.add([RECORDS[0], RECORDS[0]])
    with pytest.raises(ValueError):store.add([replace(RECORDS[0], text='changed')])
    assert store.records(source='http404.md') == [RECORDS[0]]


def test_bm25_and_rank_fusion_share_stable_ids(tmp_path):
    store = build_fixture(tmp_path)
    assert [h.record.id for h in bm25_search('404', store.records())] == ['http404-v1:0']
    hybrid = hybrid_search(store, '404', limit=3)
    assert [h.record.id for h in hybrid] == ['http404-v1:0', 'http500-v1:0', 'budget-v1:0']
    assert hybrid[0].score == pytest.approx(1/62 + 1/61)
    assert len({h.record.id for h in hybrid}) == 3
    assert bm25_search('birthday', store.records()) == []
    assert bm25_search('404', []) == []


def test_invalid_hybrid_budget_never_calls_embedder(tmp_path):
    store=build_fixture(tmp_path)
    store.embed=lambda _:(_ for _ in ()).throw(AssertionError('invalid budget reached embedder'))
    for limit,candidates in [(0,10),(-1,10),(True,10),(2,1),(1,0)]:
        with pytest.raises(ValueError):hybrid_search(store,'404',limit=limit,candidates=candidates)


def test_fusion_preserves_provenance_and_rejects_extra_votes():
    a,b=RECORDS[:2]
    with pytest.raises(ValueError):fuse_ranks([[a,a]])
    with pytest.raises(ValueError):fuse_ranks([[a],[replace(a,version='v2')]])
    # Equal text in distinct sources is not a duplicate record.
    same_text=replace(a,id='independent',source='other.md')
    assert {h.record.id for h in fuse_ranks([[a,same_text]])} == {a.id,'independent'}


def test_tiny_candidate_set_is_scored_and_can_be_entirely_rejected():
    seen=[]
    def irrelevant(query, candidates):
        seen.append((query,candidates))
        return [{'id':r['id'],'score':0.1}for r in candidates]
    assert rerank('404',[Ranked(RECORDS[1],1)],irrelevant,limit=3) == []
    assert len(seen)==1
    assert rerank('404',[],lambda *_: (_ for _ in ()).throw(AssertionError())) == []


@pytest.mark.parametrize('proposals', [[], [{'id':'unknown','score':0.8}],
    [{'id':RECORDS[0].id,'score':True}], [{'id':RECORDS[0].id,'score':float('nan')}],
    [{'id':RECORDS[0].id,'score':2}], [{'id':RECORDS[0].id,'score':0.9,'verified':True}],
    {'id':RECORDS[0].id,'score':0.9}])
def test_malformed_reranker_proposals_do_not_become_relevance(proposals):
    with pytest.raises(ValueError):rerank('404',[Ranked(RECORDS[0],1)],lambda *_:proposals)


def test_missing_duplicate_ids_and_provider_errors_do_not_fallback_to_success():
    candidates=[Ranked(r,1)for r in RECORDS[:2]]
    with pytest.raises(ValueError):rerank('404',candidates,lambda *_:[{'id':RECORDS[0].id,'score':0.8}]*2)
    with pytest.raises(TimeoutError):rerank('404',candidates,lambda *_:(_ for _ in ()).throw(TimeoutError('fixture')))


def test_query_variants_keep_original_and_obey_budget():
    assert query_variants('404',[' 404 ','missing path']) == ['404','missing path']
    for proposals in (['a','b','c'],[''],[1],'text'):
        with pytest.raises(ValueError):query_variants('404',proposals)


@pytest.mark.parametrize('language',['zh','en'])
@pytest.mark.parametrize('page',['03_embeddings_vectordb.md','04_retrieval_strategies.md'])
def test_actual_published_python_examples_execute(language,page):
    source=(ROOT/'src'/language/'chapter_rag'/page).read_text()
    blocks=re.findall(r'^```python\s*\n(.*?)^```',source,flags=re.M|re.S)
    assert blocks, 'page must retain an executable practice'
    namespace={'__name__':'__book_example__'}
    for block in blocks:exec(compile(block,str(ROOT/'src'/language/'chapter_rag'/page),'exec'),namespace)
