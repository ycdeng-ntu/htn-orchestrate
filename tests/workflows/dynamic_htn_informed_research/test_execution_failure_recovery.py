"""Execution failures block dependents and preserve evidence through recovery."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_native_entry import SMALL_CONFIG, ScriptedModel
from test_report_epoch_integration import _RecordingDiagnostics, _epoch_kwargs, _record
from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.epoch import (
    EpochRunResult, ExecutionFailureIncident, run_epoch,
)
from workflows.dynamic_htn_informed_research.htn.orchestrator import OrchestratorResult
from workflows.dynamic_htn_informed_research.htn.recovery import SupportedRecoveryFact
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome, HTNNodeCompletion, HTNTaskNode, HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.nodes import dynamic_solve as driver_module
from workflows.dynamic_htn_informed_research.runtime.context import override_runtime


EVIDENCE = "The instrument recorded 42 units. Source: https://example.test/measurement"


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda *_args, **_kwargs: {})
    monkeypatch.delenv("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_TOTAL_WORKERS", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["runtime_error", "internal_timeout", "protocol_error"])
@pytest.mark.parametrize("epoch_id", [0, 1])
@pytest.mark.parametrize("quota", [2, 64])
async def test_failure_blocks_dependency_but_drains_admitted_sibling(
    monkeypatch, tmp_path, kind, epoch_id, quota,
):
    planned, executed, completed = [], [], []
    sibling_started = asyncio.Event()
    failure_checkpointed = asyncio.Event()
    budget = QuestionDispatchBudget(quota)

    class Diagnostics(_RecordingDiagnostics):
        def emit(self, event, **fields):
            super().emit(event, **fields)
            if event == "node_execution_failure_checkpointed":
                failure_checkpointed.set()

    class Planner:
        def __init__(self, **kwargs):
            pass

        async def decompose(self, tree, node_id):
            planned.append(node_id)
            assert node_id == "root", "failure must stop subsequent planning"
            return tree.decompose(node_id, [
                HTNTaskNode(id="failed", task="Acquire essential evidence"),
                HTNTaskNode(id="sibling", task="Independent evidence"),
                HTNTaskNode(id="dependent", depends_on=["failed"], needs_decomposition=True),
                HTNTaskNode(id="later", depends_on=["sibling"], needs_decomposition=True),
            ])

    class Orchestrator:
        def __init__(self, request):
            self.request = request
            self._results = [_record(f"worker-{request.leaf.id}", EVIDENCE)]

        async def run(self):
            leaf_id = self.request.leaf.id
            executed.append(leaf_id)
            assert budget.try_acquire()
            if leaf_id == "sibling":
                sibling_started.set()
                await failure_checkpointed.wait()
                completed.append(leaf_id)
                return OrchestratorResult("Independent evidence was delivered.")
            await sibling_started.wait()
            if kind == "protocol_error":
                return OrchestratorResult(
                    EVIDENCE, ref=str(tmp_path / "partial.md"),
                    completion_outcome=HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
                    failure_reason="Unknown synthesis tool",
                )
            error = TimeoutError if kind == "internal_timeout" else RuntimeError
            raise error("Backend failed after evidence was checkpointed")

        def partial_result(self, *, reason):
            assert "failed" in reason
            return OrchestratorResult(EVIDENCE, ref=str(tmp_path / "partial.md"))

    diagnostics = Diagnostics()
    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", Orchestrator)
    result = await run_epoch(**_epoch_kwargs(
        tmp_path, epoch_id=epoch_id, max_iterations=1 if epoch_id else 4,
        diagnostics=diagnostics, dispatch_budget=budget,
        interrupt_on_semantic_exhaustion=epoch_id == 0,
    ))
    assert planned == ["root"]
    assert set(executed) == {"failed", "sibling"}
    assert completed == ["sibling"]
    failed = result.tree.get_node("failed")
    assert failed.status is HTNTaskStatus.FAILED
    assert failed.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert failed.result_summary == EVIDENCE
    assert failed.ref == str(tmp_path / "partial.md")
    assert not result.tree.is_ready("dependent")
    assert result.tree.get_node("sibling").status is HTNTaskStatus.COMPLETED
    assert result.tree.root().completion_outcome is not HTNCompletionOutcome.RESOLVED
    assert result.timeout_incidents == result.exhaustion_incidents == []
    assert len(result.execution_failure_incidents) == 1
    incident = result.execution_failure_incidents[0]
    assert incident.node_id == "failed" and incident.report_count == 1
    assert result.dispatch_budget_exhausted is (quota == 2)
    assert any(event == "epoch_end" and fields["status"] == "execution_failure"
               for event, fields in diagnostics.events)


@pytest.mark.asyncio
async def test_failure_incident_survives_later_sibling_leaf_timeout(monkeypatch, tmp_path):
    failure_checkpointed = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    real_wait = asyncio.wait

    class Diagnostics(_RecordingDiagnostics):
        def emit(self, event, **fields):
            super().emit(event, **fields)
            if event == "node_execution_failure_checkpointed":
                failure_checkpointed.set()

    class Planner:
        def __init__(self, **kwargs):
            pass

        async def decompose(self, tree, node_id):
            return tree.decompose(node_id, [HTNTaskNode(id="failed"), HTNTaskNode(id="slow")])

    class Orchestrator:
        def __init__(self, request):
            self.request = request
            self._results = [_record(request.leaf.id, EVIDENCE)]

        async def run(self):
            if self.request.leaf.id == "failed":
                raise RuntimeError("Backend unavailable")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        def partial_result(self, *, reason=None):
            return OrchestratorResult(EVIDENCE)

    async def expire_slow(awaitables, *, timeout=None):
        tasks = set(awaitables)
        if any(":slow:orchestrator" in task.get_name() for task in tasks):
            await failure_checkpointed.wait()
            await asyncio.sleep(0)  # Allow the sibling coroutine to start before expiry.
            return set(), tasks
        return await real_wait(tasks, timeout=timeout)

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", Orchestrator)
    monkeypatch.setattr(epoch_module.asyncio, "wait", expire_slow)
    result = await run_epoch(**_epoch_kwargs(tmp_path, diagnostics=Diagnostics()))
    assert sibling_cancelled.is_set()
    assert [item.leaf_id for item in result.timeout_incidents] == ["slow"]
    assert [item.node_id for item in result.execution_failure_incidents] == ["failed"]
    assert result.tree.get_node("failed").result_summary == EVIDENCE


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_id", ["branch", "root"])
@pytest.mark.parametrize("epoch_id", [0, 1])
@pytest.mark.parametrize("failure_kind", ["exception", "protocol_result"])
async def test_tree_synthesis_failure_belongs_to_parent_not_successful_children(
    monkeypatch, tmp_path, failed_id, epoch_id, failure_kind,
):
    planned, executed = [], []
    failure_checkpointed = asyncio.Event()

    class Diagnostics(_RecordingDiagnostics):
        def emit(self, event, **fields):
            super().emit(event, **fields)
            if event == "node_execution_failure_checkpointed":
                assert fields["node_id"] == failed_id
                failure_checkpointed.set()

    class Planner:
        def __init__(self, **kwargs):
            pass

        async def decompose(self, tree, node_id):
            planned.append(node_id)
            if failed_id == "branch" and node_id == "root":
                return tree.decompose(node_id, [
                    HTNTaskNode(id="branch", needs_decomposition=True),
                    HTNTaskNode(id="sibling"),
                    HTNTaskNode(id="dependent", depends_on=["branch"], needs_decomposition=True),
                ])
            assert node_id == failed_id
            return tree.decompose(node_id, [HTNTaskNode(id="a"), HTNTaskNode(id="b")])

    class Orchestrator:
        def __init__(self, request):
            self.request = request
            self._results = [_record(f"worker-{request.leaf.id}", EVIDENCE)]

        async def run(self):
            executed.append(self.request.leaf.id)
            if self.request.leaf.id == "sibling":
                await failure_checkpointed.wait()
            return OrchestratorResult(EVIDENCE)

        def partial_result(self, **kwargs):
            raise AssertionError("A parent's synthesis failure must not fail its successful child")

    async def summarize(root, parent, children):
        assert parent.id == failed_id
        assert all(child.status is HTNTaskStatus.COMPLETED for child in children)
        if failure_kind == "exception":
            raise TimeoutError("Tree synthesis transport failed")
        return HTNNodeCompletion(
            "\n\n".join(child.result_summary for child in children),
            HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
            failure_reason="Malformed tree synthesis protocol",
        )

    monkeypatch.setattr(epoch_module, "HTNPlanner", Planner)
    monkeypatch.setattr(epoch_module, "Orchestrator", Orchestrator)
    result = await run_epoch(**_epoch_kwargs(
        tmp_path, epoch_id=epoch_id, diagnostics=Diagnostics(),
        summarizer=summarize, interrupt_on_semantic_exhaustion=epoch_id == 0,
    ))
    assert planned == (["root", "branch"] if failed_id == "branch" else ["root"])
    assert "dependent" not in executed
    assert all(result.tree.get_node(node_id).status is HTNTaskStatus.COMPLETED
               for node_id in executed)
    parent = result.tree.get_node(failed_id)
    assert parent.status is HTNTaskStatus.FAILED
    assert parent.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert parent.failure_reason and EVIDENCE in parent.result_summary
    assert [item.node_id for item in result.execution_failure_incidents] == [failed_id]
    assert result.execution_failure_incidents[0].report_count == 2
    assert result.tree.root().completion_outcome is not HTNCompletionOutcome.RESOLVED
    if failed_id == "branch":
        assert not result.tree.is_ready("dependent")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["finalize", "best_effort", "rebuild", "final_epoch_failure", "quota"])
async def test_entry_routes_execution_failure_to_bounded_recovery(monkeypatch, tmp_path, route):
    model = ScriptedModel("research")
    calls, gates, extracted, finalizations = [], [], [], []
    config = deepcopy(SMALL_CONFIG)
    config["agent"]["max_rebuilds"] = 1 if route in {"rebuild", "final_epoch_failure"} else 0
    config["agent"]["max_total_workers"] = 1 if route == "quota" else 64

    async def epoch(**kwargs):
        epoch_id = kwargs["epoch_id"]
        calls.append(epoch_id)
        resolved = route == "rebuild" and epoch_id == 1
        root = HTNTaskNode(
            id="root", task=kwargs["question"],
            status=HTNTaskStatus.COMPLETED if resolved else HTNTaskStatus.FAILED,
            result_summary="# Answer: 42" if resolved else EVIDENCE,
            completion_outcome=HTNCompletionOutcome.RESOLVED if resolved else HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        )
        incidents = [] if resolved else [ExecutionFailureIncident(
            "root", root.task, "RuntimeError", "Backend unavailable", 1, len(EVIDENCE), "/partial.md",
        )]
        if route == "quota":
            assert kwargs["dispatch_budget"].try_acquire()
        return EpochRunResult(
            tree=HTNTree("root", {"root": root}, epoch_id=epoch_id),
            orchestrators={"root": SimpleNamespace(_results=[_record("worker", EVIDENCE)])},
            timeout_incidents=[], execution_failure_incidents=incidents,
            dispatch_budget_exhausted=kwargs["dispatch_budget"].exhausted,
        )

    def check_evidence(text):
        assert "# Failed execution [root]" in text
        assert "# Completed node [root]" not in text
        assert EVIDENCE in text and "Backend unavailable" in text

    async def gate(llm, evidence):
        check_evidence(evidence)
        gates.append(evidence)
        return route == "finalize"

    async def extract(llm, evidence, **kwargs):
        check_evidence(evidence)
        extracted.append(evidence)
        return (SupportedRecoveryFact("The measurement was 42 units.", "recorded 42 units", "worker"),), "Supported measurement: 42 units."

    async def finalize(llm, *, question, evidence_text):
        check_evidence(evidence_text)
        finalizations.append("normal")
        return "The measurement was recorded.\n\n# Answer: 42"

    async def best_effort(**kwargs):
        check_evidence(kwargs["evidence_text"])
        finalizations.append("best_effort")
        return "The measurement was recorded.\n\n# Answer: 42", "completed"

    async def annotate(llm, *, question, facts):
        return facts

    monkeypatch.setattr(driver_module, "run_epoch", epoch)
    monkeypatch.setattr(driver_module, "assess_timeout_finalizability", gate)
    monkeypatch.setattr(driver_module, "extract_supported_recovery_facts", extract)
    monkeypatch.setattr(driver_module, "finalize_root_from_evidence", finalize)
    monkeypatch.setattr(driver_module, "_audit_then_select_root_answer", best_effort)
    monkeypatch.setattr(driver_module, "annotate_recovery_answer_values", annotate)
    with override_runtime(llm_factory=lambda *_: model):
        result = await driver_module.dynamic_solve_node({
            "question": "What measurement did the instrument record?",
            "metadata": {"profile": "research_smoke", "dynir_config": config, "_trial_dir": str(tmp_path)},
        }, SimpleNamespace(task_id="execution-failure-entry"))
    assert result["final_answer"].endswith("# Answer: 42")
    expected_epochs = [0, 1] if route in {"rebuild", "final_epoch_failure"} else [0]
    assert calls == expected_epochs
    assert len(gates) == (0 if route == "quota" else 1)
    assert len(extracted) == (1 if route in {"rebuild", "final_epoch_failure"} else 0)
    assert len(result["execution_failures"]) == (2 if route == "final_epoch_failure" else 1)
    assert [failure["epoch_id"] for failure in result["execution_failures"]] == (
        [0, 1] if route == "final_epoch_failure" else [0]
    )
    if route != "rebuild":
        assert result["task_tree"]["root"]["status"] is HTNTaskStatus.FAILED
        assert finalizations == ["normal" if route == "finalize" else "best_effort"]
    saved = json.loads((Path(result["artifact_dir"]) / "result.json").read_text())
    assert saved["execution_failures"] == result["execution_failures"]
