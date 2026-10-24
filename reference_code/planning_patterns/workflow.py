"""Bounded, offline release-brief planning with independent source/goal checks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from reference_code.evidence_assistant.planning import PlanAndExecuteAgent, Step

GOALS = ('approval', 'budget', 'rollback')
# Frozen task requirements are separate from the proposal writer and source data.
REQUIREMENTS = {'approval': 'release-owner approval recorded', 'budget': '30 units',
                'rollback': 'restore release-1.0 snapshot'}
FACTS = {
    'approval': {'value': 'release-owner approval recorded', 'source': 'acceptance-v1#approval'},
    'budget': {'value': '30 units', 'source': 'budget-v1#release'},
    'rollback': {'value': 'restore release-1.0 snapshot', 'source': 'rollback-v1#release'},
}


def budget(value: int, *, allow_zero=False, maximum=30) -> int:
    if type(value) is not int or not (0 if allow_zero else 1) <= value <= maximum:
        raise ValueError('budget must be a bounded integer')
    return value


@dataclass(frozen=True)
class Node:
    id: str
    operation: str
    depends_on: tuple[str, ...] = ()
    covers: tuple[str, ...] = ()


def default_plan() -> list[Node]:
    return [Node('read', 'load'), Node('brief', 'compose', ('read',), GOALS)]


def validate_plan(nodes: list[Node]) -> list[Node]:
    """Validate a bounded DAG, input prerequisites and declared goal coverage."""
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 12:
        raise ValueError('plan must contain 1..12 nodes, not an unbounded iterator')
    if any(not isinstance(n, Node) or not isinstance(n.id, str) or not n.id.strip()
           or n.operation not in {'load', 'compose'} or not isinstance(n.depends_on, tuple)
           or not isinstance(n.covers, tuple) for n in nodes):
        raise ValueError('invalid node or unregistered operation')
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('duplicate node ids')
    for node in nodes:
        if any(not isinstance(key, str) or key not in by_id for key in node.depends_on):
            raise ValueError('unknown dependency')
        allowed = set(GOALS) if node.operation == 'compose' else set()
        if any(not isinstance(key, str) or key not in allowed for key in node.covers):
            raise ValueError('operation cannot claim these goals')
    ordered, done, ancestors = [], set(), {}
    while len(ordered) < len(nodes):
        ready = [n for n in nodes if n.id not in done and set(n.depends_on) <= done]
        if not ready:
            raise ValueError('cyclic dependencies')
        for node in ready:
            ancestors[node.id] = set(node.depends_on)
            for dependency in node.depends_on:
                ancestors[node.id].update(ancestors[dependency])
            if node.operation == 'compose' and not any(
                    by_id[key].operation == 'load' for key in ancestors[node.id]):
                raise ValueError('compose needs a load ancestor, not just an earlier list position')
            ordered.append(node)
            done.add(node.id)
    if set().union(*(set(n.covers) for n in nodes)) != set(GOALS):
        raise ValueError('plan omits required goals')
    return ordered


def check_brief(candidate, facts: dict) -> dict:
    """Check exact fixture claims against both original observations and goals."""
    missing, unsupported, wrong = [], [], []
    if not isinstance(candidate, dict) or set(candidate) - set(GOALS):
        return {'status': 'failed', 'missing_points': list(GOALS),
                'unsupported_points': [], 'wrong_requirements': [], 'reason': 'invalid brief shape'}
    for goal in GOALS:
        if goal not in candidate:
            missing.append(goal)
            continue
        claim = candidate[goal]
        if not isinstance(claim, dict) or set(claim) != {'value', 'source'} or claim != facts.get(goal):
            unsupported.append(goal)
        if not isinstance(claim, dict) or claim.get('value') != REQUIREMENTS[goal]:
            wrong.append(goal)
    passed = not (missing or unsupported or wrong)
    return {'status': 'passed' if passed else 'failed', 'missing_points': missing,
            'unsupported_points': unsupported, 'wrong_requirements': wrong,
            'reason': 'fixture checks passed' if passed else 'brief does not meet the frozen contract'}


def read_fixture() -> dict:
    return deepcopy(FACTS)


def write_complete(facts: dict, feedback: dict) -> dict:
    return deepcopy(facts)


class OmitRollbackOnce:
    """Fixed faulty proposal followed by a feedback-driven revision."""
    def __call__(self, facts, feedback):
        result = deepcopy(facts)
        if 'rollback' not in feedback.get('missing_points', []):
            result.pop('rollback')
        return result


def reflect_missing(snapshot: dict) -> dict:
    return {'missing_points': snapshot['verification']['missing_points'],
            'suggestion': 'revise using the original observations; do not change the requirements'}


def replace_plan(snapshot: dict) -> list[Node]:
    return default_plan()


class NeedsRevision(ValueError):
    code = 'requirements_not_met'


class Session:
    def __init__(self, reader=read_fixture, writer=write_complete):
        self.reader, self.writer = reader, writer
        self.trace, self.feedback = [], {}
        self.reset_artifacts()

    def reset_artifacts(self):
        self.facts, self.candidate = None, None
        self.loaded_by = None
        self.verification = {'status': 'not_started', 'missing_points': list(GOALS)}
        self.completed = set()

    def snapshot(self):
        return deepcopy({'loaded': self.facts is not None, 'candidate': self.candidate,
                         'verification': self.verification, 'trace': self.trace})

    def execute(self, operation: str, node_id: str, *, load_ancestors=None):
        event = {'id': node_id, 'operation': operation, 'status': 'started'}
        self.trace.append(event)
        try:
            if operation == 'load':
                facts = self.reader()
                if not isinstance(facts, dict) or set(facts) != set(GOALS):
                    raise ValueError('source adapter did not return all fixture observations')
                if (any(not isinstance(row, dict) or set(row) != {'value', 'source'}
                        or any(not isinstance(item, str) or not item.strip() for item in row.values())
                        for row in facts.values())
                        or len(json.dumps(facts, allow_nan=False).encode()) > 8192):
                    raise ValueError('invalid or oversized source observations')
                self.facts = deepcopy(facts)
                self.loaded_by = node_id
                # A new read supersedes the inputs supporting any old brief.
                self.candidate = None
                self.verification = {'status': 'not_started', 'missing_points': list(GOALS)}
                result = {'observations': deepcopy(self.facts)}
            elif operation == 'compose':
                if self.facts is None:
                    raise ValueError('required observations are unavailable')
                if load_ancestors is not None and self.loaded_by not in load_ancestors:
                    raise ValueError('current source snapshot is not from a declared load ancestor')
                candidate = self.writer(deepcopy(self.facts), deepcopy(self.feedback))
                if len(json.dumps(candidate, allow_nan=False).encode()) > 8192:
                    raise ValueError('candidate exceeds byte budget')
                self.candidate = deepcopy(candidate)
                self.verification = check_brief(self.candidate, self.facts)
                if self.verification['status'] != 'passed':
                    raise NeedsRevision(self.verification['reason'])
                result = deepcopy(self.verification)
            else:
                raise PermissionError('unregistered operation')
            self.completed.add(node_id)
            event['status'] = 'succeeded'
            event['result'] = deepcopy(result)
            return result
        except (ValueError, TypeError, OSError, RuntimeError, PermissionError) as exc:
            event.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise

    def report(self, status, *, accepted=False, steps_limit, replans=0, replans_limit=0, reason=''):
        return {'status': status, 'accepted': accepted, 'reason': reason,
                'candidate': deepcopy(self.candidate), 'verification': deepcopy(self.verification),
                'trace': deepcopy(self.trace), 'replans': replans,
                'budgets': {'steps_used': len(self.trace), 'steps_limit': steps_limit,
                            'replans_used': replans, 'replans_limit': replans_limit}}


def run_plan(*, nodes=None, reader=read_fixture, writer=write_complete,
             replanner=replace_plan, reflector=reflect_missing, max_steps=6, max_replans=1) -> dict:
    """Use the shared bounded queue; add DAG contracts and final acceptance."""
    budget(max_steps); budget(max_replans, allow_zero=True, maximum=3)
    session, replans, denied = Session(reader, writer), 0, False
    try:
        active = validate_plan(default_plan() if nodes is None else nodes)
        def queue(ordered):
            return [Step('execute', {'node_id': n.id}) for n in ordered]
        def initial(_task):
            return queue(active)
        def execute(node_id):
            nonlocal denied
            node = next(n for n in active if n.id == node_id)
            if not set(node.depends_on) <= session.completed:
                raise ValueError('dependency did not complete successfully')
            # This exercise retains one snapshot, not an artifact store/join.
            ancestors = set(node.depends_on)
            for predecessor in reversed(active):
                if predecessor.id in ancestors:
                    ancestors.update(predecessor.depends_on)
            try:
                return session.execute(node.operation, node_id, load_ancestors=ancestors)
            except PermissionError:
                denied = True
                raise
        def replan(_task, trace, remaining):
            nonlocal active, replans
            if denied or trace[-1].get('error_code') != 'requirements_not_met':
                return []  # Never seek an alternative after permission/adapter failure.
            replans += 1
            snapshot = session.snapshot()
            snapshot['remaining'] = [s.arguments['node_id'] for s in remaining]
            feedback = reflector(deepcopy(snapshot))
            if not isinstance(feedback, dict) or len(json.dumps(feedback, allow_nan=False).encode()) > 4096:
                raise ValueError('invalid or oversized reflection proposal')
            replacement = validate_plan(replanner(deepcopy(snapshot)))
            active = replacement
            session.reset_artifacts()  # Conservative invalidation; no stale result reuse.
            session.feedback = deepcopy(feedback)
            return queue(active)
        executor = PlanAndExecuteAgent(initial, {'execute': execute},
                                       replan if replanner is not None else None,
                                       max_steps=max_steps, max_replans=max_replans)
        execution = executor.run('Prepare the local release brief')
        passed = execution['status'] == 'completed' and session.verification['status'] == 'passed'
        if passed:
            status, reason = 'completed', 'all required fixture claims independently checked'
        elif execution['status'] == 'budget_exhausted':
            status, reason = 'budget_exhausted', 'step budget exhausted'
        elif session.verification['status'] == 'failed' and replanner is not None and replans >= max_replans:
            status, reason = 'budget_exhausted', 'replan budget exhausted; requirements remain unmet'
        else:
            status, reason = 'failed', 'execution or independent verification failed'
        result = session.report(status, accepted=passed, steps_limit=max_steps, replans=replans,
                                replans_limit=max_replans, reason=reason)
        result['execution_status'] = execution['status']
        return result
    except (ValueError, TypeError, OSError, RuntimeError, PermissionError) as exc:
        return session.report('failed', steps_limit=max_steps, replans=replans,
                              replans_limit=max_replans, reason=f'{type(exc).__name__}: {exc}')


def choose_fixture(snapshot):
    if not snapshot['loaded']:
        return {'action': 'load'}
    if snapshot['verification']['status'] == 'passed':
        return {'finish': True}
    return {'action': 'compose'}


def run_react(*, chooser=choose_fixture, writer=write_complete, max_decisions=5, max_calls=3) -> dict:
    """A structured action/observation exercise, not a reproduction of ReAct."""
    budget(max_decisions); budget(max_calls)
    session = Session(writer=writer)
    for decision_count in range(1, max_decisions + 1):
        try:
            decision = chooser(session.snapshot())
            if isinstance(decision, dict) and set(decision) == {'finish'} and decision['finish'] is True:
                passed = session.verification['status'] == 'passed'
                result = session.report('completed' if passed else 'incomplete', accepted=passed,
                                        steps_limit=max_calls, reason='independent checks decide completion')
                break
            if (not isinstance(decision, dict) or set(decision) != {'action'}
                    or decision['action'] not in {'load', 'compose'}):
                raise ValueError('expected one registered action or explicit finish')
            if len(session.trace) >= max_calls:
                result = session.report('budget_exhausted', steps_limit=max_calls, reason='tool call budget exhausted')
                break
            try:
                session.execute(decision['action'], f'decision-{decision_count}')
            except NeedsRevision:
                session.feedback = reflect_missing(session.snapshot())
        except (ValueError, TypeError, OSError, RuntimeError, PermissionError) as exc:
            result = session.report('failed', steps_limit=max_calls, reason=f'{type(exc).__name__}: {exc}')
            break
    else:
        result = session.report('budget_exhausted', steps_limit=max_calls, reason='decision budget exhausted')
    result['budgets'].update(decisions_used=decision_count, decisions_limit=max_decisions)
    return result
