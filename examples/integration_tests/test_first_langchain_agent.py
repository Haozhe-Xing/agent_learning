"""Actual LangChain loop and middleware, with all outbound sockets blocked."""
from __future__ import annotations

import json
import socket

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples import framework_fixture
from examples.chapter12.first_agent import build_agent, main, run
from examples.framework_fixture import FixedToolChatModel, calculate


@pytest.fixture(autouse=True)
def no_online_requests(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')

    def blocked(*args, **kwargs):
        raise AssertionError('offline integration must not open a network connection')

    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def test_real_framework_runs_model_tool_model():
    result = build_agent().invoke({'messages': [{'role': 'user', 'content': 'sqrt(144) + 2 * 3'}]},
                                  config={'recursion_limit': 16})
    messages = result['messages']
    assert [message.type for message in messages] == ['human', 'ai', 'tool', 'ai']
    assert messages[1].tool_calls[0]['args'] == {'expression': 'sqrt(144) + 2 * 3'}
    assert json.loads(messages[2].content)['value'] == 18
    assert messages[2].tool_call_id == messages[1].tool_calls[0]['id']
    assert messages[-1].content == messages[2].content


@pytest.mark.parametrize('expression', ["__import__('os').getcwd()", '2 ** 1000', '1 / 0', '1e101'])
def test_invalid_arithmetic_is_an_explicit_failure(expression):
    report = run(expression)
    assert report['status'] == 'incomplete'
    assert report['tool_results'][0]['status'] == 'invalid_expression'
    assert 'value' not in report['tool_results'][0]


def test_model_call_budget_stops_before_second_model_call():
    report = run('12 * 8', max_model_calls=1)
    assert report['status'] == 'budget_exhausted'
    assert report['error'] == 'ModelCallLimitExceededError'


def test_repeated_tool_calls_stop_before_second_execution(monkeypatch):
    executed = []
    original = framework_fixture.safe_math

    def observed(expression):
        executed.append(expression)
        return original(expression)

    monkeypatch.setattr(framework_fixture, 'safe_math', observed)
    report = run('12 * 8', FixedToolChatModel(repeat_calls=True),
                 max_model_calls=4, max_tool_calls=1)
    assert report['status'] == 'budget_exhausted'
    assert report['error'] == 'ToolCallLimitExceededError'
    assert executed == ['12 * 8']


def test_cli_and_tool_schema(capsys):
    assert calculate.args_schema.model_json_schema()['properties']['expression']['type'] == 'string'
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)['tool_results'][0]['value'] == 18
    assert main(['--expression', 'sqrt(-1)']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'incomplete'


def test_invalid_call_limits_do_not_start_an_agent():
    for limit in (0, -1, True):
        with pytest.raises(ValueError):
            build_agent(max_tool_calls=limit)


def test_unregistered_model_tool_request_is_not_a_success():
    class WrongToolModel(FixedToolChatModel):
        def _generate(self, messages, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                return super()._generate(messages, **kwargs)
            message = AIMessage(content='', tool_calls=[{
                'name': 'missing_tool', 'args': {}, 'id': 'bad-call'}])
            return ChatResult(generations=[ChatGeneration(message=message)])

    report = run('12 * 8', WrongToolModel())
    assert report['status'] == 'incomplete'
    assert report['tool_results'][0]['status'] == 'tool_error'
