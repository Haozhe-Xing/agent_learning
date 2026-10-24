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
