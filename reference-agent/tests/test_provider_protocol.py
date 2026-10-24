import json

import httpx
import pytest
from openai import OpenAI

from reference_agent.agent import Agent
from reference_agent.provider import Completion, OpenAIProvider
from reference_agent.tools.calculator import safe_eval


def test_sdk_round_trip_preserves_tool_call_ids():
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-123', 'type': 'function', 'function': {'name': 'calculator', 'arguments': '{"expression":"12*8"}'}}]}
            reason = 'tool_calls'
        else:
            assistant, tool = body['messages'][-2:]
            assert assistant['role'] == 'assistant'
            assert assistant['tool_calls'][0]['id'] == 'call-123'
            assert tool == {'role': 'tool', 'content': '96', 'tool_call_id': 'call-123'}
            message, reason = {'role': 'assistant', 'content': '96'}, 'stop'
        return httpx.Response(200, json={'id': 'chat-test', 'object': 'chat.completion', 'created': 0,
                             'model': 'fixture', 'choices': [{'index': 0, 'message': message, 'finish_reason': reason}]})
    provider = OpenAIProvider(api_key='test-not-a-live-key')
    provider.client = OpenAI(api_key='test-not-a-live-key', base_url='https://fixture.invalid/v1',
                             http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    assert Agent(provider).run('12*8') == '96'
    assert len(requests) == 2


def test_exhaustion_does_not_make_an_extra_model_call():
    class Loop:
        count = 0
        def complete(self, messages, tools=None):
            self.count += 1
            return Completion(tool_calls=[{'name': 'calculator', 'arguments': {'expression': '1+1'}}])
    provider = Loop()
    assert 'incomplete' in Agent(provider, max_iterations=2).run('loop')
    assert provider.count == 2


@pytest.mark.parametrize('expression', ['True', '9**9999999', '().__class__', '[1]*100', '9'*257])
def test_calculator_rejects_code_and_resource_abuse(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)
