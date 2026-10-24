"""Bounded multi-question research with per-answer acceptance checks."""
from __future__ import annotations

from .rag import DocumentQAAgent, answer_accepted
from .planning import PlanAndExecuteAgent, Step


class ResearchAgent:
    def __init__(self, documents, answerer=None, verifier=None):
        self.qa = DocumentQAAgent(documents, answerer, verifier)

    def run(self, questions):
        findings = []
        def collect(question):
            result = self.qa.ask(question)
            findings.append({'question': question, **result})
            return result['status']
        plan = PlanAndExecuteAgent(lambda _: [Step('research', {'question': q}) for q in questions],
                                   {'research': collect}, max_steps=8)
        execution = plan.run('Research supplied questions')
        status = ('completed' if execution['status'] == 'completed' and findings and
                  all(answer_accepted(f) for f in findings) else 'incomplete')
        return {'status': status, 'findings': findings, 'execution': execution}
