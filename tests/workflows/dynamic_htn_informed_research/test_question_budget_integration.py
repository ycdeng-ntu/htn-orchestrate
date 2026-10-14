"""Question dispatch quotas retain evidence and final delivery across runtime boundaries."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_native_entry import SMALL_CONFIG, ScriptedModel
from test_report_epoch_integration import (
    _EmptyEvidenceLLM,
    _RecordingDiagnostics,
    _epoch_kwargs,
    _record,
    _requirement,
)
from workflows.dynamic_htn_informed_research.diagnostics import scope_snapshot
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.epoch import run_epoch
from workflows.dynamic_htn_informed_research.htn.orchestrator import OrchestratorResult
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNNodeCompletion,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.nodes.dynamic_solve import dynamic_solve_node
from workflows.dynamic_htn_informed_research.runtime.context import override_runtime
from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
    run_gap_recovery_wave,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import (
    RequirementSet,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import (
    ReportRuntime,
)


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda *_args, **_kwargs: {})
    monkeypatch.delenv("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_TOTAL_WORKERS", raising=False)


@pytest.mark.asyncio
async def test_epoch_quota_drain_preserves_sibling_and_does_not_plan_dependency(
    monkeypatch, tmp_path
):
    """An exhausted internal node must not cancel another admitted leaf's evidence."""
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    budget = QuestionDispatchBudget(limit=2)
    requests = []
    planned = []
    completed = []
    cancelled = []
    both_admitted = asyncio.Event()
    branch_synthesized = asyncio.Event()

    class Planner:
        def __init__(self, **_kwargs):
            pass

        async def decompose(self, tree, node_id):
            planned.append(node_id)
            if node_id == "root":
                return tree.decompose(
                    node_id,
                    [
                        HTNTaskNode(id="branch", needs_decomposition=True),
                        HTNTaskNode(id="sibling", task="Preserve this independent evidence."),
                        HTNTaskNode(
                            id="dependent",
                            depends_on=["branch"],
                            needs_decomposition=True,
                        ),
                    ],
                )
            assert node_id == "branch", "quota exhaustion must stop dependent planning"
            return tree.decompose(node_id, [HTNTaskNode(id="branch-leaf")])

    class QuotaOrchestrator:
        def __init__(self, request):
            requests.append(request)
            self.request = request
            self._results = []

        async def run(self):
            leaf_id = self.request.leaf.id
            assert self.request.dispatch_budget is budget
            assert budget.try_acquire()
            if budget.exhausted:
                both_admitted.set()
            try:
                await both_admitted.wait()
                if leaf_id == "sibling":
                    await branch_synthesized.wait()
                    # Let the other solve_leaf task observe the semantic incident.
                    await asyncio.sleep(0)
                self._results.append(_record(f"worker-{leaf_id}"))
                completed.append(leaf_id)
                return OrchestratorResult(f"Useful report from {leaf_id}.")
            except asyncio.CancelledError:
                cancelled.append(leaf_id)
                raise

        def partial_result(self):
            return OrchestratorResult("Partial evidence.")

    async def summarize(_root, node, _children):
        assert node.id == "branch"
        branch_synthesized.set()
        return HTNNodeCompletion(
            "The branch exhausted its approach but kept its evidence.",
            HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        )

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", QuotaOrchestrator)
    result = await asyncio.wait_for(
        run_epoch(
            **_epoch_kwargs(
                tmp_path,
                max_iterations=3,
                dispatch_budget=budget,
                summarizer=summarize,
                interrupt_on_semantic_exhaustion=True,
            )
        ),
        timeout=2,
    )

    assert len(requests) == 2
    assert budget.used == 2
    assert set(completed) == {"branch-leaf", "sibling"}
    assert not cancelled
    assert planned == ["root", "branch"]
    assert result.dispatch_budget_exhausted is True
    assert result.timeout_incidents == []
    assert result.exhaustion_incidents == []
    assert (
        result.tree.get_node("branch").completion_outcome
        is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    )
    assert result.tree.get_node("sibling").status is HTNTaskStatus.COMPLETED
    assert "Useful report" in result.tree.get_node("sibling").result_summary
    assert result.tree.get_node("dependent").status is HTNTaskStatus.PENDING
    assert "dependent" not in result.orchestrators


@pytest.mark.asyncio
async def test_epochs_share_usage_and_an_exhausted_epoch_does_not_plan(monkeypatch, tmp_path):
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    budget = QuestionDispatchBudget(limit=2)
    seen = []
    planned_epochs = []

    class Planner:
        def __init__(self, **_kwargs):
            pass

        async def decompose(self, tree, node_id):
            planned_epochs.append(tree.epoch_id)
            tree.mark_executable(node_id)
            return []

    class QuotaOrchestrator:
        def __init__(self, request):
            seen.append(request.dispatch_budget)
            self.request = request
            self._results = []

        async def run(self):
            assert self.request.dispatch_budget.try_acquire()
            return OrchestratorResult("Preserved result.")

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", QuotaOrchestrator)
    outcomes = []
    for epoch_id in range(3):
        outcomes.append(
            await run_epoch(
                **_epoch_kwargs(tmp_path, epoch_id=epoch_id, dispatch_budget=budget)
            )
        )

    assert seen == [budget, budget]
    assert planned_epochs == [0, 1]
    assert budget.used == 2
    assert outcomes[0].dispatch_budget_exhausted is False
    assert outcomes[1].dispatch_budget_exhausted is True
    assert outcomes[2].dispatch_budget_exhausted is True
    assert outcomes[2].orchestrators == {}


@pytest.mark.asyncio
async def test_epoch_synthesis_failure_after_quota_exhaustion_preserves_partial(monkeypatch, tmp_path):
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    budget = QuestionDispatchBudget(limit=1)
    partial_summary = "Checkpointed evidence: the published measurement was 42."
    partial_ref = str(tmp_path / "checkpointed-report.md")

    class Planner:
        def __init__(self, **_kwargs):
            pass

        async def decompose(self, tree, node_id):
            tree.mark_executable(node_id)
            return []

    class FailingSynthesisOrchestrator:
        def __init__(self, request):
            self.request = request
            self._results = [_record("admitted-worker", partial_summary)]

        async def run(self):
            assert self.request.dispatch_budget.try_acquire()
            raise RuntimeError("Leaf synthesis failed after research completed.")

        def partial_result(self, *, reason=None):
            assert reason and "exhausted" in reason
            return OrchestratorResult(partial_summary, ref=partial_ref)

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", FailingSynthesisOrchestrator)
    result = await run_epoch(**_epoch_kwargs(tmp_path, dispatch_budget=budget))

    root = result.tree.root()
    assert budget.exhausted
    assert result.dispatch_budget_exhausted is True
    assert root.status is HTNTaskStatus.FAILED
    assert root.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert [incident.node_id for incident in result.execution_failure_incidents] == ["root"]
    assert partial_summary in root.result_summary
    assert root.ref == partial_ref
    assert result.orchestrators["root"]._results[0].result_summary == partial_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling_end", ["completed", "leaf_timeout", "question_timeout"])
async def test_pending_semantic_incident_rechecks_quota_and_drain_keeps_deadlines(
    monkeypatch, tmp_path, sibling_end
):
    """Report observation may yield between checkpointing and delivering an incident."""
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    budget = QuestionDispatchBudget(limit=2)
    diagnostics = _RecordingDiagnostics()
    observation_started = asyncio.Event()
    sibling_admitted = asyncio.Event()
    sibling_cancelled = []
    sibling_completed = []

    class Planner:
        def __init__(self, **_kwargs):
            pass

        async def decompose(self, tree, node_id):
            if node_id == "root":
                return tree.decompose(
                    node_id,
                    [
                        HTNTaskNode(id="branch", needs_decomposition=True),
                        HTNTaskNode(id="sibling", task="Keep independent evidence."),
                    ],
                )
            assert node_id == "branch"
            return tree.decompose(node_id, [HTNTaskNode(id="branch-leaf")])

    class ObservingReportRuntime:
        def delivery_context_for(self, _leaf):
            return ""

        async def refresh_ownership(self, _tree, _llm):
            pass

        async def observe_leaf(self, leaf, _records, _llm, *, deadline):
            if leaf.id == "branch-leaf":
                # record_semantic_exhaustion already raised; the finally block
                # intentionally yields before the epoch can catch that incident.
                assert not budget.exhausted
                assert any(
                    event == "semantic_exhaustion_checkpointed"
                    for event, _ in diagnostics.events
                )
                observation_started.set()
                await sibling_admitted.wait()

    class QuotaOrchestrator:
        def __init__(self, request):
            self.request = request
            self._results = []

        @property
        def results(self):
            return tuple(self._results)

        async def run(self):
            leaf_id = self.request.leaf.id
            if leaf_id == "branch-leaf":
                assert budget.try_acquire()
                self._results.append(_record("branch-worker"))
                return OrchestratorResult("The branch retained a useful partial finding.")
            try:
                await observation_started.wait()
                assert budget.try_acquire()
                assert budget.exhausted
                sibling_admitted.set()
                if sibling_end != "completed":
                    await asyncio.Event().wait()
                # The semantic incident reaches the epoch before this report.
                await asyncio.sleep(0.01)
                self._results.append(_record("sibling-worker", "Independent evidence was saved."))
                sibling_completed.append(leaf_id)
                return OrchestratorResult("Independent evidence was saved.")
            except asyncio.CancelledError:
                sibling_cancelled.append(leaf_id)
                raise

        def partial_result(self):
            return OrchestratorResult("Partial sibling evidence.", ref="partial-sibling.md")

    async def summarize(_root, node, _children):
        return HTNNodeCompletion(
            f"The {node.id} approach exhausted available evidence.",
            HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        )

    if sibling_end == "question_timeout":
        real_wait = asyncio.wait

        async def keep_leaf_deadline_after_question_deadline(awaitables, *, timeout=None):
            tasks = set(awaitables)
            if any(":sibling:orchestrator" in task.get_name() for task in tasks):
                timeout = 10
            return await real_wait(tasks, timeout=timeout)

        # Both runtime deadlines have a one-second minimum. Hold only the
        # sibling's leaf timer later so this case deterministically tests the
        # question deadline while the epoch drains admitted work.
        monkeypatch.setattr(epoch_module.asyncio, "wait", keep_leaf_deadline_after_question_deadline)

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", QuotaOrchestrator)
    started = time.monotonic()
    result = await asyncio.wait_for(
        run_epoch(
            **_epoch_kwargs(
                tmp_path,
                dispatch_budget=budget,
                report_runtime=ObservingReportRuntime(),
                diagnostics=diagnostics,
                summarizer=summarize,
                max_iterations=3,
                leaf_timeout_s=1 if sibling_end == "leaf_timeout" else 30,
                research_deadline=started + (0.05 if sibling_end == "question_timeout" else 60),
                interrupt_on_semantic_exhaustion=True,
            )
        ),
        timeout=3,
    )

    assert result.dispatch_budget_exhausted is True
    assert budget.used == 2
    assert result.exhaustion_incidents == []
    assert any(item.node_id == "branch" for item in result.nonterminal_exhaustion_incidents)
    if sibling_end == "completed":
        assert sibling_completed == ["sibling"]
        assert sibling_cancelled == []
        assert result.tree.get_node("sibling").result_summary == "Independent evidence was saved."
        assert result.timeout_incidents == []
        assert result.question_deadline_exhausted is False
    else:
        assert sibling_cancelled == ["sibling"]
        assert sibling_completed == []
        assert result.tree.get_node("sibling").status is HTNTaskStatus.FAILED
        assert time.monotonic() - started < 2.5
        if sibling_end == "leaf_timeout":
            assert [incident.leaf_id for incident in result.timeout_incidents] == ["sibling"]
            assert result.question_deadline_exhausted is False
        else:
            assert result.timeout_incidents == []
            assert result.question_deadline_exhausted is True


def _gap_runtime():
    return ReportRuntime(
        question="Write a sourced comparison.",
        policy=replace(SCIENTIFIC_REPORT_POLICY, max_recovery_requirements=3, max_recovery_agents=3),
        requirements=RequirementSet(
            "scientific_report", [_requirement(f"req-{i:03d}", priority=1) for i in range(3)]
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_gap_recovery_uses_only_remaining_question_quota_without_refund(fails):
    budget = QuestionDispatchBudget(limit=3)
    assert budget.try_acquire() and budget.try_acquire()
    runtime = _gap_runtime()
    dispatched = []
    evidence_llm = _EmptyEvidenceLLM()

    async def dispatch(agent, _instruction):
        dispatched.append(agent.agent_id)
        if fails:
            raise RuntimeError("Worker failed after admission.")
        return _record(agent.agent_id)

    result = await run_gap_recovery_wave(
        runtime=runtime,
        dispatch=dispatch,
        research_role_id="dynir_report_researcher",
        evidence_llm=evidence_llm,
        question_deadline=time.monotonic() + 30,
        reserve_s=5,
        max_live_agents=3,
        diagnostics=_RecordingDiagnostics(),
        dispatch_budget=budget,
    )

    assert dispatched == ["report-gap-req-000"]
    assert len(result) == (0 if fails else 1)
    assert len(evidence_llm.calls) == (0 if fails else 1)
    assert budget.used == 3
    assert budget.exhausted
    assert runtime.recovery_wave_used is True


@pytest.mark.asyncio
async def test_exhausted_question_skips_gap_dispatch_and_evidence_extraction():
    budget = QuestionDispatchBudget(limit=1)
    assert budget.try_acquire()
    runtime = _gap_runtime()
    evidence_llm = _EmptyEvidenceLLM()
    diagnostics = _RecordingDiagnostics()

    async def unexpected_dispatch(*_args):
        pytest.fail("An exhausted question must not dispatch report gap workers")

    result = await run_gap_recovery_wave(
        runtime=runtime,
        dispatch=unexpected_dispatch,
        research_role_id="dynir_report_researcher",
        evidence_llm=evidence_llm,
        question_deadline=time.monotonic() + 30,
        reserve_s=5,
        max_live_agents=3,
        diagnostics=diagnostics,
        dispatch_budget=budget,
    )

    assert result == ()
    assert budget.used == 1
    assert evidence_llm.calls == []
    assert runtime.recovery_wave_used is False
    assert any(name == "report_gap_recovery_skipped" for name, _ in diagnostics.events)


_DEFAULT = object()


class BudgetFixtureModel(ScriptedModel):
    """Add the existing terminal audit contract to the normal native-entry script."""

    async def chat(self, messages, *, tools=None, **kwargs):
        names = {tool["function"]["name"] for tool in tools or []}
        if "audit_forced_finalization" in names:
            await asyncio.sleep(0)
            self.calls.append((names, scope_snapshot()))
            return self.response(
                "audit_forced_finalization",
                {
                    "requested_output_fields": ["result"],
                    "decisive_constraints": ["Compute six times seven."],
                    "candidates": [
                        {
                            "candidate": "42",
                            "answer_value": "42",
                            "supporting_observations": ["The worker established six times seven."],
                            "contradicting_observations": [],
                            "unresolved_relations": [],
                            "role_binding_issues": [],
                        }
                    ],
                    "unassigned_answer_shaped_values": [],
                },
            )
        return await super().chat(messages, tools=tools, **kwargs)


async def _run_entry(tmp_path, *, limit=_DEFAULT, report=False, model=None):
    model = model or BudgetFixtureModel(
        "report_research" if report else "research"
    )
    config = deepcopy(SMALL_CONFIG)
    config["agent"]["max_rebuilds"] = 1
    if limit is not _DEFAULT:
        config["agent"]["max_total_workers"] = limit
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {
                "question": "Establish the result of six times seven.",
                "metadata": {
                    "profile": "auto",
                    "dynir_config": config,
                    "_trial_dir": str(tmp_path),
                },
            },
            SimpleNamespace(task_id=f"budget-{tmp_path.name}"),
        )
    assert not model.unexpected
    return result, model


@pytest.mark.asyncio
@pytest.mark.parametrize("report", [False, True])
async def test_real_entry_one_worker_cap_still_delivers_and_records_quota(tmp_path, report):
    result, model = await _run_entry(tmp_path, limit=1, report=report)

    assert len(model.worker_roles) == 1
    assert result["dispatch_budget"]["limit"] == 1
    assert result["dispatch_budget"]["used"] == 1
    assert result["dispatch_budget"]["exhausted"] is True
    assert result["final_content"]
    if report:
        # No verifier quota remains: preserve research without claiming verified delivery.
        assert "42" in result["final_content"]
        assert "Report delivery did not complete" in result["final_content"]
        assert not model.finalizer_inputs
        assert result["task_tree"]["report-delivery"]["status"] == "pending"
    else:
        assert result["final_answer"].endswith("# Answer: 42")
        assert any("audit_forced_finalization" in names for names, _ in model.calls)
    artifact_dir = Path(result["artifact_dir"])
    saved = json.loads((artifact_dir / "result.json").read_text())
    assert saved["dispatch_budget"] == result["dispatch_budget"]
    assert json.loads((artifact_dir / "dispatch_budget.json").read_text()) == result[
        "dispatch_budget"
    ]
    assert (artifact_dir / "report.md").read_text() == result["final_content"]
    event_files = list((artifact_dir / "diagnostics").glob("*/events.jsonl"))
    assert len(event_files) == 1
    events = [json.loads(line) for line in event_files[0].read_text().splitlines()]
    epoch_events = [event for event in events if event.get("event") == "epoch_start"]
    assert len(epoch_events) == 1, "an exhausted question must not rebuild even when max_rebuilds=1"
    if report:
        reviews = [event for event in events if event.get("event") == "report_tree_review"]
        assert reviews and all(event["action"] == "partial" for event in reviews)
    else:
        reviews = [event for event in events if event.get("event") == "recovery_review_start"]
        assert reviews and all(event["allow_rebuild"] is False for event in reviews)


@pytest.mark.asyncio
async def test_concurrent_questions_get_independent_dispatch_budgets(tmp_path):
    outcomes = await asyncio.gather(
        *(_run_entry(tmp_path / str(i), limit=1) for i in range(3))
    )

    assert len({result["artifact_dir"] for result, _ in outcomes}) == 3
    assert [result["dispatch_budget"]["used"] for result, _ in outcomes] == [1, 1, 1]
    assert [len(model.worker_roles) for _, model in outcomes] == [1, 1, 1]
    assert all(result["final_answer"].endswith("# Answer: 42") for result, _ in outcomes)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,expected", [(_DEFAULT, 0), (0, 0), ("2", 2)])
async def test_entry_default_disabled_and_string_config(tmp_path, limit, expected):
    result, _model = await _run_entry(tmp_path, limit=limit)
    assert result["dispatch_budget"]["limit"] == expected
    assert result["dispatch_budget"]["used"] == 1
    assert result["dispatch_budget"]["exhausted"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [True, False, -1, 1.5, 2.0, "1.5", "bad", "-1", None])
async def test_entry_rejects_invalid_dispatch_limit_before_model_calls(tmp_path, limit):
    model = ScriptedModel("research")
    with pytest.raises(ValueError, match="max_total_workers"):
        await _run_entry(tmp_path, limit=limit, model=model)
    assert model.calls == []
