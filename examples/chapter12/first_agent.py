"""LangChain 1.4.6 create_agent loop: an offline fixture by default."""
from __future__ import annotations

import argparse
import json

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langgraph.errors import GraphRecursionError

from examples.framework_fixture import FixedToolChatModel, calculate


def build_agent(model=None, *, max_model_calls: int = 3, max_tool_calls: int = 2):
    """Bound model/tool calls per invocation; these are not elapsed-time limits."""
    for value in (max_model_calls, max_tool_calls):
        if type(value) is not int or value < 1:
            raise ValueError('call limits must be positive integers')
    return create_agent(
        model=FixedToolChatModel() if model is None else model,
        tools=[calculate],
        system_prompt='Use calculate for arithmetic. Report rejected input as a failure.',
        middleware=[
            ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior='error'),
            ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior='error'),
        ],
    )


def run(expression: str, model=None, *, max_model_calls: int = 3, max_tool_calls: int = 2) -> dict:
    """Expose execution and actual tool results, without claiming task acceptance."""
    mode = 'offline-fixture' if model is None else 'injected-model'
    agent = build_agent(model, max_model_calls=max_model_calls, max_tool_calls=max_tool_calls)
    try:
        state = agent.invoke({'messages': [{'role': 'user', 'content': expression}]},
                             config={'recursion_limit': 16})
    except (ModelCallLimitExceededError, ToolCallLimitExceededError, GraphRecursionError) as exc:
        return {'mode': mode, 'status': 'budget_exhausted', 'error': type(exc).__name__}
    except Exception as exc:
        return {'mode': mode, 'status': 'failed', 'error': type(exc).__name__}
    messages = state['messages']
    results = []
    for message in messages:
        if message.type != 'tool':
            continue
        try:
            result = json.loads(message.content)
        except (ValueError, TypeError):
            result = {'status': 'tool_error', 'error': str(message.content)}
        if not isinstance(result, dict) or message.status == 'error':
            result = {'status': 'tool_error', 'error': str(message.content)}
        results.append(result)
    last = messages[-1]
    complete = (bool(results) and all(result.get('status') == 'ok' for result in results)
                and last.type == 'ai' and not last.tool_calls and bool(last.content))
    return {'mode': mode, 'status': 'completed' if complete else 'incomplete',
            'answer': messages[-1].content, 'tool_results': results,
            'model_calls': sum(message.type == 'ai' for message in messages),
            'tool_calls': len(results),
            'verification_scope': 'Tool execution only; independent task acceptance is separate.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression', default='sqrt(144) + 2 * 3')
    parser.add_argument('--max-model-calls', type=int, default=3)
    parser.add_argument('--max-tool-calls', type=int, default=2)
    parser.add_argument('--real', action='store_true', help='explicitly call the configured online model')
    parser.add_argument('--model', default='gpt-4.1-mini')
    args = parser.parse_args(argv)
    model = None
    if args.real:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=args.model, timeout=30, max_retries=1)
    try:
        report = run(args.expression, model, max_model_calls=args.max_model_calls,
                     max_tool_calls=args.max_tool_calls)
    except ValueError as exc:
        parser.error(str(exc))
    if args.real:
        report['mode'] = 'real-model'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
