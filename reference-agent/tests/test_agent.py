from reference_agent.agent import Agent
from reference_agent.provider import FakeProvider


def test_calculator_tool_use():
    agent = Agent(FakeProvider())
    out = agent.run("请帮我计算 12*8")
    assert "96" in out


def test_chinese_operator():
    agent = Agent(FakeProvider())
    out = agent.run("23 加 19 的结果是多少")
    assert "42" in out


def test_no_api_key_fallback_answer():
    agent = Agent(FakeProvider())
    out = agent.run("今天天气怎么样")
    assert out  # non-empty, deterministic offline answer


def test_openai_round_trip_preserves_parallel_tool_call_ids():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from reference_agent.provider import OpenAIProvider

    def tool_call(call_id, arguments):
        function = SimpleNamespace(name="calculator", arguments=arguments)
        return SimpleNamespace(id=call_id, function=function)

    def reply(content=None, tool_calls=None):
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    provider = OpenAIProvider(api_key="test-key")
    provider.client = Mock()
    create_completion = provider.client.chat.completions.create
    create_completion.side_effect = [
        reply(tool_calls=[tool_call("call_first", '{"expression":"12*8"}'),
                          tool_call("call_second", '{"expression":"23+19"}')]),
        reply(content="96 and 42"),
    ]

    assert Agent(provider).run("Calculate 12*8 and 23+19.") == "96 and 42"
    messages = create_completion.call_args.kwargs["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"] == [
        {"id": "call_first", "type": "function",
         "function": {"name": "calculator", "arguments": '{"expression":"12*8"}'}},
        {"id": "call_second", "type": "function",
         "function": {"name": "calculator", "arguments": '{"expression":"23+19"}'}},
    ]
    assert messages[2:] == [
        {"role": "tool", "content": "96", "tool_call_id": "call_first"},
        {"role": "tool", "content": "42", "tool_call_id": "call_second"},
    ]
