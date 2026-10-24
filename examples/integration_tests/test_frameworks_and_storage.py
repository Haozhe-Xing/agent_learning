"""Real dependencies, local fixtures only: no live model or embedding downloads."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
from types import SimpleNamespace

import pytest

from examples.chapter12.customer_service import build_agent
from examples.chapter13.content_workflow import build_workflow


def test_langchain_tool_call_and_owner_isolation():
    agent = build_agent('alice')
    for query, expected in [('check ORD-1', 'shipped'), ('check ORD-2', 'access denied'),
                            ('complaint ORD-1', 'Write permission denied')]:
        result = agent.invoke({'messages': [{'role': 'user', 'content': query}]}, config={'recursion_limit': 8})
        assert expected in result['messages'][-1].content
        assert any(message.type == 'tool' for message in result['messages'])


def test_langchain_complaint_idempotency():
    agent = build_agent('alice', allow_writes=True)
    outputs = [agent.invoke({'messages': [{'role': 'user', 'content': 'complaint ORD-1'}]},
                           config={'recursion_limit': 8})['messages'][-1].content for _ in range(2)]
    assert outputs == ['TICKET-1', 'TICKET-1']


@pytest.mark.parametrize('scores,revisions,accepted', [([9], 0, True), ([4, 9], 1, True), ([4, 5, 6], 2, False)])
def test_real_langgraph_budget(scores, revisions, accepted):
    values = iter(scores)
    graph = build_workflow(lambda _: next(values), lambda text: text + ' revised')
    result = graph.invoke({'topic': 'fixture'}, config={'recursion_limit': 12})
    assert result['revision_count'] == revisions
    assert result['accepted'] is accepted


def test_langgraph_rejects_invalid_review():
    graph = build_workflow(lambda _: 99, lambda text: text)
    with pytest.raises(ValueError):
        graph.invoke({'topic': 'fixture'})


def test_actual_chroma_restart_isolation_and_deletion(tmp_path):
    from examples.chapter04.personal_assistant import PersonalAssistant

    def embed(input, **kwargs):
        # Explicit deterministic vectors avoid model downloads. The database is real.
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0])])
    responses = iter(['hello', json.dumps({'memories': [{'content': 'likes tea', 'importance': 5}]})])
    def complete(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))])
    client = SimpleNamespace(embeddings=SimpleNamespace(create=embed),
                             chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    path = tmp_path / 'chroma'
    alice = PersonalAssistant('alice', remember=True, client=client, persist_dir=path)
    alice.chat('I like tea')
    reopened = PersonalAssistant('alice', client=client, persist_dir=path)
    assert reopened.recall('tea')[0]['content'] == 'likes tea'
    bob = PersonalAssistant('bob', client=client, persist_dir=path)
    assert bob.recall('tea') == []
    records = reopened.memory.get(where={'user_id': 'alice'}, include=[])
    reopened.memory.delete(ids=records['ids'])
    assert PersonalAssistant('alice', client=client, persist_dir=path).recall('tea') == []


def test_chroma_persists_across_processes(tmp_path):
    import subprocess, sys
    create = """
import chromadb, sys
client = chromadb.PersistentClient(path=sys.argv[1])
collection = client.get_or_create_collection('restart_test', embedding_function=None)
collection.add(ids=['one'], documents=['fixture record'], embeddings=[[1.0, 0.0]], metadatas=[{'user_id':'alice'}])
"""
    read = """
import chromadb, sys
collection = chromadb.PersistentClient(path=sys.argv[1]).get_collection('restart_test', embedding_function=None)
assert collection.get(where={'user_id':'alice'})['documents'] == ['fixture record']
collection.delete(where={'user_id':'alice'})
assert collection.count() == 0
"""
    for script in [create, read]:
        result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'restart')],
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
