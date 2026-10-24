"""Exercise the installed LangGraph APIs, without remote model calls."""
from __future__ import annotations

import json
from pathlib import Path
import re
import socket
import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
import pytest

from examples.chapter13.first_graph_agent import DEFAULT_EXPRESSION, build_graph
from examples.framework_fixture import FixedToolChatModel


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('offline graph tests must not open a network connection')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def test_graph_routes_through_real_tool_node_then_stops():
    graph = build_graph()
    events = list(graph.stream({'messages': [HumanMessage(content=DEFAULT_EXPRESSION)]},
                               config={'recursion_limit': 8}, stream_mode='updates'))
    assert [next(iter(event)) for event in events] == ['agent', 'tools', 'agent']
    call = events[0]['agent']['messages'][0].tool_calls[0]
    assert call['name'] == 'calculate'
    tool_message = events[1]['tools']['messages'][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == call['id']
    assert json.loads(tool_message.content)['value'] == 18
    assert not events[-1]['agent']['messages'][0].tool_calls


def test_graph_without_tool_request_ends_after_agent_node():
    class DirectModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content='No calculation requested in this fixture.')

    graph = build_graph(DirectModel())
    events = list(graph.stream({'messages': [HumanMessage(content='fixture')]},
                               config={'recursion_limit': 8}, stream_mode='updates'))
    assert [next(iter(event)) for event in events] == ['agent']


def test_graph_recursion_limit_stops_repeated_tool_requests():
    graph = build_graph(FixedToolChatModel(repeat_calls=True))
    with pytest.raises(GraphRecursionError):
        graph.invoke({'messages': [HumanMessage(content='2 + 3')]}, config={'recursion_limit': 4})


def test_graph_rejects_empty_input_instead_of_inventing_a_question():
    with pytest.raises(ValueError, match='input message'):
        build_graph().invoke({'messages': []}, config={'recursion_limit': 8})


def test_graph_does_not_automatically_persist_thread_state():
    with pytest.raises(ValueError, match='checkpointer'):
        build_graph().get_state({'configurable': {'thread_id': 'not-persisted'}})


@pytest.mark.parametrize('language', ['zh', 'en'])
def test_documented_minimal_graph_is_executable(language, capsys):
    root = Path(__file__).resolve().parents[2]
    path = root / 'src' / language / 'chapter_langgraph/03_first_graph_agent.md'
    blocks = re.findall(r'```python\n(.*?)\n```', path.read_text(encoding='utf-8'), flags=re.DOTALL)
    assert len(blocks) == 1
    namespace = {}
    exec(compile(blocks[0], str(path), 'exec'), namespace)
    events = namespace['events']
    assert [next(iter(event)) for event in events] == ['agent', 'tools', 'agent']
    assert json.loads(events[1]['tools']['messages'][0].content)['value'] == 18
    assert 'agent' in capsys.readouterr().out


@pytest.mark.parametrize('arguments,code,status', [
    ([], 0, 'completed'),
    (['--stream'], 0, 'completed'),
    (['--expression', "__import__('os').getcwd()"], 1, 'tool_failed'),
    (['--recursion-limit', '1'], 1, 'budget_exhausted'),
])
def test_graph_cli_reports_success_and_failures(arguments, code, status):
    result = subprocess.run([sys.executable, '-m', 'examples.chapter13.first_graph_agent', *arguments],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == code, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records[-1]['status'] == status
    if status == 'completed':
        assert records[-1]['mode'] == 'offline_fixture'
