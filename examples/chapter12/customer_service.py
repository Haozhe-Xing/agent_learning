"""Runnable LangChain v1 customer service Agent with a deterministic test model.

All orders are fixtures. Authorization is enforced in the tools, not the prompt.
"""
from __future__ import annotations

import re
from typing import Any


def build_agent(owner: str = 'alice', allow_writes: bool = False, model=None):
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    orders = {'ORD-1': {'owner': 'alice', 'status': 'shipped'},
              'ORD-2': {'owner': 'bob', 'status': 'pending'}}
    tickets = {}

    @tool
    def check_order(order_id: str) -> str:
        """Read the authenticated user's order status by order ID."""
        record = orders.get(order_id)
        if record is None or record['owner'] != owner:
            return 'Order unavailable or access denied.'
        return record['status']

    @tool
    def submit_complaint(order_id: str, issue: str) -> str:
        """Create an idempotent test complaint for an owned order with write permission."""
        if not allow_writes:
            return 'Write permission denied.'
        if order_id not in orders or orders[order_id]['owner'] != owner:
            return 'Order unavailable or access denied.'
        if not issue.strip():
            return 'Issue must not be empty.'
        key = (owner, order_id, issue.strip())
        tickets.setdefault(key, f'TICKET-{len(tickets)+1}')
        return tickets[key]

    @tool
    def search_faq(query: str) -> str:
        """Read a fixture FAQ; unsupported questions return an explicit limitation."""
        return 'Fixture return window: seven days.' if 'return' in query.casefold() else 'No matching FAQ.'

    class FixtureModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return 'book-fixture-tool-model'

        def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                reply = AIMessage(content=str(messages[-1].content))
            else:
                query = str(messages[-1].content)
                order = re.search(r'ORD-\d+', query)
                if 'complaint' in query.casefold() and order:
                    name, args = 'submit_complaint', {'order_id': order[0], 'issue': 'fixture complaint'}
                elif order:
                    name, args = 'check_order', {'order_id': order[0]}
                else:
                    name, args = 'search_faq', {'query': query}
                reply = AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': 'fixture-call'}])
            return ChatResult(generations=[ChatGeneration(message=reply)])

    agent = create_agent(model=model or FixtureModel(),
                         tools=[check_order, submit_complaint, search_faq],
                         system_prompt='Use tools for order facts. Never claim a denied write succeeded.')
    return agent


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='check ORD-1')
    parser.add_argument('--user', choices=['alice', 'bob'], default='alice')
    parser.add_argument('--allow-writes', action='store_true')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--model', default='gpt-4.1-mini')
    args = parser.parse_args()
    model = None
    if args.real:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=args.model, timeout=30, max_retries=1)
    agent = build_agent(args.user, args.allow_writes, model)
    result = agent.invoke({'messages': [{'role': 'user', 'content': args.query}]},
                          config={'recursion_limit': 8})
    print(result['messages'][-1].content)


if __name__ == '__main__':
    main()
