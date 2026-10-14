"""Three independent planning audits and evidence-accounted consolidation."""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json
import time

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.planner import DecompositionBudget
from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import (
    AUDIT_DIMENSIONS, PLAN_AUDIT_TOOLS, accepted_research_nodes, audit_report_plan,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_rendering import render_planning_input

PLAN = {"nodes": [{"id": "derive", "title": "Establish result", "task": "Investigate and check the result",
                    "depends_on": []}], "rationale": "One complete investigation"}
CONTEXT = {"task": "Determine whether the proposed explanation holds.",
           "recovery_capsule": json.dumps({"previous_epoch": 0, "material": [
               {"id": "old", "evidence": "FULL_RECOVERY with ```fences```", "unknown_field": "KEEP_ME"}
           ], "additional_context": "PRESERVE_THIS_TOO"})}
BUDGET = DecompositionBudget.for_node(current_depth=0, max_depth=1, max_children=4)
PASS = {"verdict": "PASS", "issues": []}
TOOL_DIMENSIONS = {tool["function"]["name"]: d for d, tool in PLAN_AUDIT_TOOLS.items()}


def finding(text="A candidate explanation was made mandatory"):
    return {"node_ids": ["derive"], "issue": text, "repair": "Test the explanation without requiring it to hold"}


def review(*issues):
    return {"verdict": "REVISE" if issues else "PASS", "issues": list(issues)}


def response(name, payload):
    return LLMResponse(tool_calls=[{"function": {"name": name, "arguments": json.dumps(payload)}}])


class Audits:
    def __init__(self, *, findings=None, synthesis=None, delivery=(), barrier=False):
        self.findings = findings or {}
        self.synthesis = synthesis or {"plan_review": PASS, "dispositions": []}
        self.delivery = set(delivery)
        self.calls = []
        self.started = set()
        self.barrier = asyncio.Event() if barrier else None

    async def chat(self, messages, tools=None):
        name = tools[0]["function"]["name"]
        dimension = TOOL_DIMENSIONS[name]
        self.calls.append((dimension, deepcopy(messages)))
        if dimension in AUDIT_DIMENSIONS:
            self.started.add(dimension)
            if self.barrier:
                if len(self.started) == 3:
                    self.barrier.set()
                await self.barrier.wait()
        if dimension == "synthesis":
            payload = self.synthesis
        else:
            payload = {"plan_review": self.findings.get(dimension, PASS)}
            if dimension == "delivery":
                payload.update(has_delivery=bool(self.delivery), nodes=[
                    {"id": n["id"], "kind": "report_delivery" if n["id"] in self.delivery else "research",
                     "reason": "Classified actual responsibility"}
                    for n in self.plan["nodes"]])
        return response(name, payload)


async def audit(client, *, plan=None, diagnostics=None):
    client.plan = plan or deepcopy(PLAN)
    return await audit_report_plan(
        plan=client.plan, planner_input=deepcopy(CONTEXT), budget=BUDGET, llm=client,
        deadline=time.monotonic() + 2, diagnostics=diagnostics, epoch_id=1,
    )


@pytest.mark.asyncio
async def test_all_clean_runs_concurrently_with_identical_complete_inputs_and_skips_synthesis(tmp_path):
    client = Audits(barrier=True)
    result = await asyncio.wait_for(audit(client, diagnostics=DiagnosticRecorder(tmp_path)), timeout=3)
    assert [dimension for dimension, _ in client.calls] == list(AUDIT_DIMENSIONS)
    inputs = [messages[1]["content"] for _, messages in client.calls]
    assert len(set(inputs)) == 1
    assert all(text in inputs[0] for text in (CONTEXT["task"], "FULL_RECOVERY with ```fences```",
                                             "KEEP_ME", "PRESERVE_THIS_TOO", PLAN["rationale"]))
    assert len({messages[0]["content"] for _, messages in client.calls}) == 3
    assert result["plan_review"] == PASS and result["synthesis_skipped"]
    assert {p.stem for p in (tmp_path / 'report_delivery/epoch-1/audit-round-1').glob('*.json')} == set(AUDIT_DIMENSIONS)
    assert (tmp_path / 'report_delivery/epoch-1/plan-audit-1.json').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension", AUDIT_DIMENSIONS)
async def test_finding_in_any_dimension_triggers_synthesis_and_preserves_sources(dimension):
    issue = finding()
    client = Audits(findings={dimension: review(issue)}, synthesis={
        "plan_review": review(issue), "dispositions": [{
            "source_id": f"{dimension}:1", "issue_numbers": [1], "reason": "Candidate prescribes the conclusion",
        }],
    })
    result = await audit(client)
    assert result["plan_review"] == review(issue) and not result["synthesis_skipped"]
    assert Counter(d for d, _ in client.calls) == Counter([*AUDIT_DIMENSIONS, "synthesis"])
    original_input = client.calls[0][1][1]["content"]
    synthesis_input = client.calls[-1][1][1]["content"]
    assert synthesis_input.startswith(original_input + '\n\n## Independent audit findings')
    assert f'Source issue {dimension}:1' in synthesis_input
    assert issue['issue'] in synthesis_input and issue['repair'] in synthesis_input
    rendered = render_planning_input(planner_input=CONTEXT,
                                     constraints={"max_research_nodes": 4, "runtime_owns_final_delivery": True},
                                     candidate_plan=PLAN, audit_feedback=result)
    assert f'Source {dimension}:1' in rendered
    assert 'Candidate prescribes the conclusion' in rendered


@pytest.mark.asyncio
async def test_terminal_delivery_detection_is_synthesized_but_does_not_force_revision():
    plan = deepcopy(PLAN)
    plan['nodes'].append({'id': 'draft', 'title': 'Report', 'task': 'Write final report', 'depends_on': ['derive']})
    client = Audits(delivery=['draft'])
    result = await audit(client, plan=plan)
    assert result['has_delivery'] and result['plan_review'] == PASS
    assert client.calls[-1][0] == 'synthesis'
    assert [n.id for n in accepted_research_nodes(plan, result)] == ['derive']


@pytest.mark.asyncio
async def test_runtime_guards_cannot_be_waived_by_synthesis_pass():
    plan = deepcopy(PLAN)
    plan['nodes'][0]['depends_on'] = ['draft']
    plan['nodes'].append({'id': 'draft', 'title': 'Report', 'task': 'Write final report', 'depends_on': []})
    result = await audit(Audits(delivery=['draft']), plan=plan)
    assert result['model_plan_review'] == PASS
    assert result['plan_review']['verdict'] == 'REVISE'
    assert any('successors' in s for s in result['runtime_findings'])
    with pytest.raises(ValueError, match='requiring revision'):
        accepted_research_nodes(plan, result)


@pytest.mark.asyncio
async def test_synthesis_deduplicates_and_explains_rejected_findings_without_losing_originals():
    first, duplicate, rejected = finding(), finding('Same premature conclusion'), finding('Unfounded objection')
    dispositions = [
        {'source_id': 'fidelity:1', 'issue_numbers': [1], 'reason': 'Confirmed from the task wording'},
        {'source_id': 'decomposition:1', 'issue_numbers': [1], 'reason': 'Same underlying defect'},
        {'source_id': 'decomposition:2', 'issue_numbers': [], 'reason': 'The original task explicitly requires this comparison'},
    ]
    client = Audits(findings={'fidelity': review(first), 'decomposition': review(duplicate, rejected)},
                    synthesis={'plan_review': review(first), 'dispositions': dispositions})
    result = await audit(client)
    assert len(result['plan_review']['issues']) == 1
    assert result['dispositions'] == dispositions
    assert result['dimension_audits']['decomposition']['plan_review'] == review(duplicate, rejected)


@pytest.mark.asyncio
async def test_protocol_retry_repeats_only_failed_dimension():
    class Retry(Audits):
        async def chat(self, messages, tools=None):
            result = await super().chat(messages, tools)
            if self.calls[-1][0] == 'fidelity' and sum(d == 'fidelity' for d, _ in self.calls) == 1:
                return response('audit_report_fidelity', {'plan_review': {'verdict': 'PASS', 'issues': [finding()]}})
            return result
    client = Retry()
    result = await audit(client)
    assert result['synthesis_skipped']
    assert Counter(d for d, _ in client.calls) == {'delivery': 1, 'fidelity': 2, 'decomposition': 1}
    messages = [m for d, m in client.calls if d == 'fidelity']
    assert messages[1][:-1] == messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('malformation', ['missing_source', 'unknown_target', 'unbacked_issue', 'blank_reason'])
async def test_synthesis_cannot_silently_drop_sources_or_invent_mappings(malformation):
    synthesis = {'plan_review': review(finding()), 'dispositions': [
        {'source_id': 'fidelity:1', 'issue_numbers': [1], 'reason': 'Confirmed'}]}
    if malformation == 'missing_source':
        synthesis['dispositions'] = []
    elif malformation == 'unknown_target':
        synthesis['dispositions'][0]['issue_numbers'] = [2]
    elif malformation == 'unbacked_issue':
        synthesis['dispositions'][0]['issue_numbers'] = []
    else:
        synthesis['dispositions'][0]['reason'] = ' '
    client = Audits(findings={'fidelity': review(finding())}, synthesis=synthesis)
    with pytest.raises(ValueError):
        await audit(client)
    assert Counter(d for d, _ in client.calls) == {'delivery': 1, 'fidelity': 1, 'decomposition': 1, 'synthesis': 3}


@pytest.mark.asyncio
async def test_failed_dimension_cancels_unfinished_peers_instead_of_accepting_partial_results():
    class Failure(Audits):
        def __init__(self):
            super().__init__()
            self.ready = asyncio.Event()
            self.cancelled = set()

        async def chat(self, messages, tools=None):
            dimension = TOOL_DIMENSIONS[tools[0]['function']['name']]
            self.started.add(dimension)
            if len(self.started) == 3:
                self.ready.set()
            await self.ready.wait()
            if dimension == 'fidelity':
                raise ConnectionError('Fidelity unavailable')
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.add(dimension)
                raise
    client = Failure()
    with pytest.raises(ConnectionError, match='Fidelity unavailable'):
        await audit(client)
    assert client.cancelled == {'delivery', 'decomposition'}


@pytest.mark.parametrize('network_enabled', [False, True])
def test_all_audit_roles_obey_access_configuration(network_enabled):
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.topology import ReportPlanningStrategy
    strategy = ReportPlanningStrategy(planner_llm=None, audit_llm=None, revision_llm=None,
                                      planner_system='Planner', network_enabled=network_enabled)
    assert set(strategy.audit_systems) == {*AUDIT_DIMENSIONS, 'synthesis'}
    for prompt in [*strategy.audit_systems.values(), strategy.reviser_system]:
        assert ('Work offline' in prompt) != network_enabled
        assert '[[WEB]]' not in prompt and '[[OFFLINE]]' not in prompt


@pytest.mark.asyncio
async def test_synthesis_ids_are_constrained_per_case_and_retry_names_valid_issue_ids():
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import call_plan_audit

    class SourceCorrection:
        def __init__(self, source_id):
            self.source_id = source_id
            self.calls = []

        async def chat(self, messages, tools=None):
            self.calls.append((deepcopy(messages), deepcopy(tools)))
            await asyncio.sleep(0)
            # Reproduce the replay: an auditor name was supplied instead of an issue ID.
            source = self.source_id.split(':')[0] if len(self.calls) == 1 else self.source_id
            return response('synthesize_report_plan_audit', {
                'plan_review': review(finding()), 'dispositions': [
                    {'source_id': source, 'issue_numbers': [1], 'reason': 'Confirmed from the original task'}],
            })

    clients = [SourceCorrection('fidelity:1'), SourceCorrection('decomposition:2')]
    template = deepcopy(PLAN_AUDIT_TOOLS['synthesis'])
    results = await asyncio.gather(*(
        call_plan_audit(dimension='synthesis', rendered_input='Original context retained',
                        expected_ids=['derive'], llm=c, deadline=time.monotonic() + 2,
                        diagnostics=None, source_ids=[c.source_id]) for c in clients
    ))
    for client, result in zip(clients, results):
        assert result['dispositions'][0]['source_id'] == client.source_id
        assert len(client.calls) == 2
        schema = client.calls[0][1][0]['function']['parameters']['properties']['dispositions']
        assert schema['minItems'] == schema['maxItems'] == 1
        assert schema['items']['properties']['source_id']['enum'] == [client.source_id]
        assert client.source_id in client.calls[1][0][-1]['content']
        assert client.calls[1][0][:-1] == client.calls[0][0]
    assert PLAN_AUDIT_TOOLS['synthesis'] == template


@pytest.mark.asyncio
async def test_duplicate_source_feedback_reports_missing_and_repeated_ids():
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import call_plan_audit

    class DuplicateCorrection:
        def __init__(self):
            self.messages = []

        async def chat(self, messages, tools=None):
            self.messages.append(deepcopy(messages))
            ids = ['decomposition:1', 'decomposition:1' if len(self.messages) == 1 else 'decomposition:2']
            return response('synthesize_report_plan_audit', {
                'plan_review': review(finding()), 'dispositions': [
                    {'source_id': source, 'issue_numbers': [1], 'reason': 'Same underlying defect'}
                    for source in ids],
            })
    client = DuplicateCorrection()
    await call_plan_audit(dimension='synthesis', rendered_input='Unchanged context',
                         expected_ids=['derive'], llm=client, deadline=time.monotonic() + 2,
                         diagnostics=None, source_ids=['decomposition:1', 'decomposition:2'])
    feedback = client.messages[1][-1]['content']
    assert "Missing: ['decomposition:2']" in feedback
    assert "repeated: ['decomposition:1']" in feedback


@pytest.mark.asyncio
async def test_empty_source_inventory_requires_empty_dispositions_and_no_auditor_entries():
    client = Audits(delivery=['draft'])
    plan = deepcopy(PLAN)
    plan['nodes'].append({'id': 'draft', 'title': 'Draft', 'task': 'Write the report', 'depends_on': ['derive']})
    result = await audit(client, plan=plan)
    material = client.calls[-1][1][1]['content']
    assert 'Required disposition count: 0' in material
    assert 'No source issues. Return dispositions = [].' in material
    assert result['dispositions'] == []


@pytest.mark.asyncio
async def test_actual_decomposition_failure_shape_recovers_without_losing_repair_or_repeating_peers(tmp_path):
    class RepairCorrection(Audits):
        async def chat(self, messages, tools=None):
            result = await super().chat(messages, tools)
            if tools[0]['function']['name'] == 'audit_report_decomposition':
                count = sum(d == 'decomposition' for d, _ in self.calls)
                issue = finding('Concrete overlap. Repair: put the coupled work in one milestone.')
                if count == 1:
                    del issue['repair']
                arguments = json.dumps({'plan_review': review(issue)}) + ('}' if count == 2 else '')
                return LLMResponse(tool_calls=[{'function': {'name': 'audit_report_decomposition', 'arguments': arguments}}])
            return result
    issue = finding('Concrete overlap. Repair: put the coupled work in one milestone.')
    client = RepairCorrection(synthesis={'plan_review': review(issue), 'dispositions': [
        {'source_id': 'decomposition:1', 'issue_numbers': [1], 'reason': 'Overlap confirmed'}]})
    result = await audit(client, diagnostics=DiagnosticRecorder(tmp_path))
    assert result['plan_review'] == review(issue)
    assert Counter(d for d, _ in client.calls) == {'delivery': 1, 'fidelity': 1, 'decomposition': 2, 'synthesis': 1}
    retry = [m for d, m in client.calls if d == 'decomposition'][1][-1]['content']
    assert "'repair' is a required property" in retry and 'separate JSON keys' in retry
    assert len(list((tmp_path / 'report_protocol').glob('*-normalized.json'))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('dimension', [*AUDIT_DIMENSIONS, 'synthesis'])
@pytest.mark.parametrize('remaining_seconds', [None, 1800])
async def test_audits_use_question_deadline_without_local_300_second_cap(monkeypatch, dimension, remaining_seconds):
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import call_plan_audit
    observed = []
    original = asyncio.wait_for

    async def observe(awaitable, timeout):
        observed.append(timeout)
        return await original(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio, 'wait_for', observe)

    class Immediate:
        async def chat(self, messages, tools=None):
            payload = {'plan_review': PASS}
            if dimension == 'delivery':
                payload.update(has_delivery=False, nodes=[{'id': 'derive', 'kind': 'research', 'reason': 'Research'}])
            if dimension == 'synthesis':
                payload['dispositions'] = []
            return response(tools[0]['function']['name'], payload)

    await call_plan_audit(dimension=dimension, rendered_input='Task and plan', expected_ids=['derive'],
                         llm=Immediate(), deadline=time.monotonic() + remaining_seconds if remaining_seconds else None,
                         diagnostics=None, source_ids=[] if dimension == 'synthesis' else None)
    assert len(observed) == 1
    if remaining_seconds is None:
        assert observed == [None]
    else:
        assert remaining_seconds - 1 < observed[0] <= remaining_seconds


@pytest.mark.asyncio
@pytest.mark.parametrize('dimension', [*AUDIT_DIMENSIONS, 'synthesis'])
@pytest.mark.parametrize('low_hangs', [False, True])
async def test_audit_allows_high_to_low_but_question_deadline_still_cancels_low(dimension, low_hangs):
    from workflows.dynamic_htn_informed_research.adapters import HighEffortTimeoutFallbackLLMClient
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import call_plan_audit

    class Model:
        model = 'deadline-test'

        def __init__(self, effort):
            self.reasoning_effort = effort
            self.calls = 0
            self.cancelled = False

        async def chat(self, messages, tools=None):
            self.calls += 1
            if self.reasoning_effort == 'high' or low_hangs:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            payload = {'plan_review': PASS}
            if dimension == 'delivery':
                payload.update(has_delivery=False, nodes=[{'id': 'derive', 'kind': 'research', 'reason': 'Research'}])
            if dimension == 'synthesis':
                payload['dispositions'] = []
            return response(tools[0]['function']['name'], payload)

    high, low = Model('high'), Model('low')
    client = HighEffortTimeoutFallbackLLMClient(high, low, timeout_s=0.01)
    pending = call_plan_audit(dimension=dimension, rendered_input='Original task and plan',
                              expected_ids=['derive'], llm=client, deadline=time.monotonic() + 0.15,
                              diagnostics=None, source_ids=[] if dimension == 'synthesis' else None)
    if low_hangs:
        with pytest.raises(TimeoutError):
            await pending
        assert low.cancelled
    else:
        assert (await pending)['plan_review'] == PASS
    assert high.calls == low.calls == 1
    assert high.cancelled
