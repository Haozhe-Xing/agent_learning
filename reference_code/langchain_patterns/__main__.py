"""Run every small pattern with fixed providers and external tracing disabled."""
from __future__ import annotations

import argparse
import json

from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from . import (FIXTURE_ANSWER, LocalTrace, build_answer_chain, build_intake_flow,
               build_retry_example, build_scoped_chain, build_structured_chain,
               stream_events)


def run_examples() -> tuple[dict, LocalTrace]:
    trace = LocalTrace()
    result = {}
    with tracing_context(enabled=False):
        result['answer'] = build_answer_chain().invoke(
            {'fact': FIXTURE_ANSWER, 'question': 'Return policy?'},
            config={'callbacks': [trace], 'metadata': {'case_id': 'return-1'}},
        )
        result['flow'] = build_intake_flow().invoke({'question': 'Refund policy?'})
        result['structured'] = build_structured_chain().invoke({
            'fact': FIXTURE_ANSWER, 'question': 'Return policy?',
        }).model_dump()
        retry, attempts = build_retry_example()
        result['retry'] = {'result': retry.invoke('fixture-ok'), **attempts}

        def interrupted(_):
            yield 'partial'
            raise TimeoutError('fixed midstream failure')

        fallback_calls = []
        fallback = RunnableLambda(lambda _: fallback_calls.append(True) or 'backup')
        chain = RunnableLambda(interrupted).with_fallbacks(
            [fallback], exceptions_to_handle=(TimeoutError,))
        result['stream'] = list(stream_events(chain, 'fixture'))
        result['fallback_calls_after_partial'] = len(fallback_calls)
        model = FakeListChatModel(responses=['first', 'second'], cache=InMemoryCache())
        cached = build_scoped_chain(model)
        request = {'scope': 'team-a', 'revision': 'v1', 'question': 'fixture'}
        result['cache'] = [cached.invoke(request), cached.invoke(request),
                           cached.invoke({**request, 'revision': 'v2'})]
    assert result['answer'] == FIXTURE_ANSWER
    assert result['retry']['attempts'] == 2
    assert [event['event'] for event in result['stream']] == ['delta', 'error']
    assert result['fallback_calls_after_partial'] == 0
    assert result['cache'] == ['first', 'first', 'second']
    result['local_trace_events'] = len(trace.events)
    result['provider'] = 'fixed-fixtures; no model inference or remote tracing'
    return result, trace


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-file', help='optional new local JSON trace file')
    args = parser.parse_args(argv)
    result, trace = run_examples()
    if args.trace_file:
        trace.write_new(args.trace_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
