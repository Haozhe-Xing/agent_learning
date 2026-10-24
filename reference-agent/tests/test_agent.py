from unittest.mock import Mock

from reference_agent.agent import Agent, DEFAULT_TOOLS
from reference_agent.provider import Completion, FakeProvider


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


def test_explicit_empty_tools_cannot_execute_default_tools(monkeypatch):
    runs = []
    for tool in DEFAULT_TOOLS:
        run = Mock(wraps=tool.run)
        monkeypatch.setattr(tool, "run", run)
        runs.append(run)

    class UnexpectedToolProvider:
        offered_tools = []

        def complete(self, messages, tools=None):
            self.offered_tools.append(tools)
            if messages[-1].role == "user":
                return Completion(tool_calls=[
                    {"name": "calculator", "arguments": {"expression": "12*8"}},
                    {"name": "search", "arguments": {"query": "agent"}},
                ])
            return Completion(content="; ".join(message.content for message in messages if message.role == "tool"))

    provider = UnexpectedToolProvider()
    answer = Agent(provider, tools=[]).run("Do not use any tools")
    assert provider.offered_tools == [[], []]
    assert answer == "unknown tool: calculator; unknown tool: search"
    for run in runs:
        run.assert_not_called()


def test_none_tools_still_uses_defaults():
    assert "96" in Agent(FakeProvider(), tools=None).run("12*8")
