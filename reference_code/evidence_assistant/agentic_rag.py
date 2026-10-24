"""Retry only recoverable retrieval failures within a fixed query budget."""
from __future__ import annotations

from itertools import islice
from .rag import answer_accepted


class AgenticRAG:
    def __init__(self, qa, max_queries=2):
        if max_queries < 1:
            raise ValueError('max_queries must be positive')
        self.qa, self.max_queries = qa, max_queries

    def run(self, queries):
        trace = []
        result = {'status': 'no_evidence', 'answer': 'No supporting evidence found.', 'sources': [],
                  'retrieved_sources': [], 'retrieval': {'status': 'empty'},
                  'generation': {'status': 'not_started'}, 'verification': {'status': 'not_run'},
                  'retryable': False}
        for query in islice(queries, self.max_queries):
            result = self.qa.ask(query)
            trace.append({'query': query, 'status': result['status']})
            if answer_accepted(result):
                return {**result, 'trace': trace}
            # Reformulate an empty search or retry a transient provider failure only.
            if not result.get('retryable', False):
                break
        return {**result, 'trace': trace}
