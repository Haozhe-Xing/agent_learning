"""An explicit LangGraph tool loop; the default model is an offline fixture.

Run from the repository root with ``python -m examples.chapter13.first_graph_agent``.
The fixture treats the input as an arithmetic expression, not natural language.
"""
from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from examples.framework_fixture import FixedToolChatModel, calculate


DEFAULT_EXPRESSION = 'sqrt(144) + 2 * 3'


def build_graph(model=None):
    """Build a message-state graph, without persistence or human interrupts."""
    model = FixedToolChatModel() if model is None else model
    bound_model = model.bind_tools([calculate])

    def agent_node(state: MessagesState) -> dict:
        messages = state.get('messages', [])
        if not messages:
            raise ValueError('provide at least one input message')
        response = bound_model.invoke([
            SystemMessage(content='Use calculate for arithmetic. Preserve tool failures in the reply.'),
            *messages,
        ])
        return {'messages': [response]}

    graph = StateGraph(MessagesState)
    graph.add_node('agent', agent_node)
    graph.add_node('tools', ToolNode([calculate]))
    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', tools_condition)
    graph.add_edge('tools', 'agent')
    return graph.compile()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression', default=DEFAULT_EXPRESSION)
    parser.add_argument('--recursion-limit', type=int, default=8)
    parser.add_argument('--stream', action='store_true', help='print per-node message updates')
    parser.add_argument('--real', action='store_true', help='explicitly use a remote model instead of the fixture')
    parser.add_argument('--model', default='gpt-4.1-mini')
    args = parser.parse_args(argv)
    if args.recursion_limit < 1:
        parser.error('--recursion-limit must be positive')
    model = None
    if args.real:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=args.model, timeout=30, max_retries=1)
    graph = build_graph(model)
    config = {'recursion_limit': args.recursion_limit}
    state = {'messages': [HumanMessage(content=args.expression)]}
    try:
        if args.stream:
            messages = []
            for event in graph.stream(state, config=config, stream_mode='updates'):
                for node, update in event.items():
                    messages.extend(update.get('messages', []))
                    print(json.dumps({'node': node, 'messages': [message.model_dump(mode='json')
                                                               for message in update.get('messages', [])]}))
        else:
            messages = graph.invoke(state, config=config)['messages']
    except GraphRecursionError:
        print(json.dumps({'status': 'budget_exhausted', 'reason': 'graph_recursion_limit',
                          'recursion_limit': args.recursion_limit}))
        return 1
    failures = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.status == 'error':
                failures.append(str(message.content))
            elif message.name == 'calculate':
                payload = json.loads(str(message.content))
                if payload.get('status') != 'ok':
                    failures.append(payload)
    print(json.dumps({'status': 'tool_failed' if failures else 'completed',
                      'mode': 'real' if args.real else 'offline_fixture',
                      'answer': messages[-1].content, 'tool_failures': failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
