from __future__ import annotations

import asyncio
import json
import time

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import AgentResultRecord
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies.report_research.finalize import (
    VerificationFinding,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import SCIENTIFIC_REPORT_POLICY
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import (
    Requirement,
    RequirementSet,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import ReportRuntime
from workflows.dynamic_htn_informed_research.task_strategies.report_research.verification_repair import (
    run_verification_repair,
)


class EmptyEvidence:
    async def chat(self, *args, **kwargs):
        return LLMResponse(content=None, tool_calls=[{"function": {
            "name": "extract_evidence", "arguments": '{"cards": []}',
        }}])


def fixture(tmp_path, limit=3):
    child = HTNTaskNode(id="old", parent_id="root", status=HTNTaskStatus.COMPLETED,
                        completion_outcome=HTNCompletionOutcome.RESOLVED,
                        result_summary="Previously collected branch memo")
    root = HTNTaskNode(id="root", children=["old"], status=HTNTaskStatus.COMPLETED,
                       completion_outcome=HTNCompletionOutcome.RESOLVED,
                       result_summary="Previous root answer")
    tree = HTNTree("root", {"root": root, "old": child})
    runtime = ReportRuntime("Check two calculations", SCIENTIFIC_REPORT_POLICY, RequirementSet("scientific_report", [
        Requirement(id=f"req-{i}", description=f"Compute quantity {i}",
                    acceptance_criteria=("Show units",), priority=i, kind="derivation")
        for i in (1, 2)
    ]))
    findings = tuple(VerificationFinding(f"req-{i}", "material", "Units unclear", "Check units")
                     for i in (1, 2))
    return dict(runtime=runtime, tree=tree, findings=findings, draft="Unverified draft",
                dispatch_budget=QuestionDispatchBudget(limit), research_role_id="research",
                evidence_llm=EmptyEvidence(), cutoff=time.monotonic() + 10,
                max_live_agents=3, diagnostics=DiagnosticRecorder(tmp_path))


def record(agent, **overrides):
    return AgentResultRecord(**{
        "agent_id": agent.agent_id, "role_id": agent.role_id,
        "task_summary": agent.task_summary, "result_summary": "Independent unit calculation",
        "execution_quality": "submitted", "submitted": True, **overrides,
    })


def test_append_checks_is_atomic_and_preserves_existing_branches(tmp_path):
    tree = fixture(tmp_path)["tree"]
    before = {nid: n.model_dump() for nid, n in tree.nodes.items()}
    with pytest.raises(ValueError):
        tree.append_report_checks([HTNTaskNode(id="old")])
    with pytest.raises(ValueError):
        tree.append_report_checks([HTNTaskNode(id="check", depends_on=["check"])])
    assert before == {nid: n.model_dump() for nid, n in tree.nodes.items()}
    tree.append_report_checks([HTNTaskNode(id="check")])
    assert tree.get_node("old").model_dump() == before["old"]
    assert tree.root().status is HTNTaskStatus.PENDING
    assert tree.root().completion_outcome is None
    assert tree.root().result_summary is None
    assert tree.root().children == ["old", "check"]
    assert tree.validate() == []


@pytest.mark.asyncio
async def test_shared_remaining_budget_deduplicates_and_caps_actual_dispatch(tmp_path):
    args = fixture(tmp_path, limit=2)
    args["dispatch_budget"].try_acquire()  # Worker already used by original tree.
    args["findings"] += args["findings"]
    called = []

    async def dispatch(agent, instruction):
        called.append(agent.agent_id)
        assert "hypotheses" in instruction
        assert "ground truth" in instruction
        return record(agent)

    result = await run_verification_repair(**args, dispatch=dispatch)
    assert len(result) == 1
    assert called == ["report-check-req-1"]
    assert args["dispatch_budget"].used == 2
    assert args["tree"].validate() == []
    assert args["tree"].get_node(called[0]).completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert (tmp_path / "report_repair/tree-before.json").exists()
    assert args["runtime"].verification_repair_used
    assert await run_verification_repair(**args, dispatch=dispatch) == ()
    assert len(called) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["quota", "time", "rewrite", "unknown", "busy"])
async def test_ineligible_repair_does_not_mutate_tree_or_dispatch(tmp_path, reason):
    args = fixture(tmp_path, limit=1)
    if reason == "quota":
        args["dispatch_budget"].try_acquire()
    elif reason == "time":
        args["cutoff"] = time.monotonic()
    elif reason == "rewrite":
        args["findings"] = (VerificationFinding("req-1", "material", "Missing text", "Restore", "rewrite"),)
    elif reason == "unknown":
        args["findings"] = (VerificationFinding("unknown", "critical", "Missing text", "Check"),)
    else:
        args["tree"].get_node("old").status = HTNTaskStatus.RUNNING
    before = {nid: n.model_dump() for nid, n in args["tree"].nodes.items()}

    async def dispatch(*_):
        pytest.fail("Must not dispatch an ineligible repair")

    assert await run_verification_repair(**args, dispatch=dispatch) == ()
    assert before == {nid: n.model_dump() for nid, n in args["tree"].nodes.items()}
    assert not args["runtime"].verification_repair_used


@pytest.mark.asyncio
async def test_failed_worker_does_not_discard_successful_check(tmp_path):
    args = fixture(tmp_path)

    async def dispatch(agent, _):
        if agent.agent_id.endswith("2"):
            return record(agent, execution_quality="execution_error", submitted=False,
                          result_summary="Transport error, not evidence")
        return record(agent)

    memos = await run_verification_repair(**args, dispatch=dispatch)
    assert len(memos) == 1 and "Transport error" not in memos[0]
    assert args["tree"].get_node("report-check-req-2").status is HTNTaskStatus.FAILED
    assert args["dispatch_budget"].used == 2


@pytest.mark.asyncio
async def test_parent_cancellation_settles_workers_and_preserves_completed_memo(tmp_path):
    args = fixture(tmp_path)
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    async def dispatch(agent, _):
        if agent.agent_id.endswith("1"):
            return record(agent)
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(run_verification_repair(**args, dispatch=dispatch))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert args["tree"].get_node("report-check-req-1").result_summary == "Independent unit calculation"
    assert args["tree"].get_node("report-check-req-2").status is HTNTaskStatus.FAILED
    assert args["dispatch_budget"].used == 2


@pytest.mark.asyncio
async def test_deadline_rejects_late_result_from_cancellation_suppressing_worker(tmp_path):
    args = fixture(tmp_path, limit=1)
    args["cutoff"] = time.monotonic() + 1.01
    stopped = asyncio.Event()

    async def dispatch(agent, _):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return record(agent, result_summary="Late unsupported answer")
        finally:
            stopped.set()

    assert await run_verification_repair(**args, dispatch=dispatch) == ()
    await asyncio.wait_for(stopped.wait(), timeout=1)
    node = args["tree"].get_node("report-check-req-1")
    assert node.status is HTNTaskStatus.FAILED
    assert node.result_summary is None
    assert not args["runtime"].evidence_cards


@pytest.mark.asyncio
async def test_repair_worker_receives_late_sections_and_full_draft(tmp_path):
    args = fixture(tmp_path, limit=1)
    draft = "# Early section\n" + "context " * 6000 + "\n# Late result\nThe tail has the wrong units."
    args["draft"] = draft
    seen = []

    async def dispatch(agent, instruction):
        marker = "# Internal report verification check\n\n"
        payload = json.JSONDecoder().raw_decode(instruction.split(marker, 1)[1])[0]
        seen.append(payload["draft_to_check"])
        return record(agent)

    await run_verification_repair(**args, dispatch=dispatch)
    assert seen == [draft]
