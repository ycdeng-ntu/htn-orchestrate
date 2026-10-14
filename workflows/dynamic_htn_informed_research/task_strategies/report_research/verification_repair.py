"""One additive task-tree wave driven by the report verifier's public findings."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMClient
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentResultRecord,
    AgentTaskSummary,
    DispatchFn,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree

from .finalize import VerificationFinding
from .gap_recovery import _bounded, _cancel_and_settle_insulated, _render_assignment
from .runtime import ReportRuntime


async def run_verification_repair(
    *,
    runtime: ReportRuntime,
    tree: HTNTree,
    findings: Sequence[VerificationFinding],
    draft: str,
    dispatch: DispatchFn,
    dispatch_budget: QuestionDispatchBudget,
    research_role_id: str,
    evidence_llm: LLMClient,
    cutoff: float,
    max_live_agents: int,
    diagnostics: DiagnosticRecorder,
    blocked_source_directive: str = "",
) -> tuple[str, ...]:
    """Append root-level checks and retain completed work within an absolute cutoff.

    A finding challenges prior synthesis; it does not mutate old evidence into
    a new fact. Unrelated branches are untouched. Root synthesis stays invalid
    until the caller rewrites and reverifies the report. No epoch is restarted.
    """
    def skip(reason: str) -> tuple[str, ...]:
        diagnostics.emit("report_verification_repair_skipped", reason=reason)
        return ()

    if runtime.verification_repair_used:
        return skip("wave_already_used")
    if dispatch_budget.exhausted:
        return skip("question_dispatch_budget_exhausted")
    if cutoff - time.monotonic() < 1.0:
        return skip("insufficient_time")

    requirements = {r.id: r for r in runtime.requirements.requirements}
    grouped: dict[str, list[VerificationFinding]] = {}
    for finding in findings:
        if (finding.requirement_id in requirements
                and finding.severity in {"critical", "material"}
                and finding.repair_action == "research"):
            group = grouped.setdefault(finding.requirement_id, [])
            if finding not in group:
                group.append(finding)
    capacity = min(max(0, max_live_agents), max(0, runtime.policy.max_recovery_agents),
                   max(0, runtime.policy.max_recovery_requirements))
    if dispatch_budget.remaining is not None:
        capacity = min(capacity, dispatch_budget.remaining)
    targets = sorted(grouped, key=lambda rid: (
        not any(f.severity == "critical" for f in grouped[rid]),
        requirements[rid].priority, rid,
    ))[:capacity]
    if not targets:
        return skip("no_research_targets_or_capacity")

    planned: list[HTNTaskNode] = []
    for rid in targets:
        related = sorted({
            card.leaf_id for card in runtime.evidence_cards.values()
            if rid in card.requirement_ids and card.leaf_id in tree.nodes
        } | ({requirements[rid].owner_leaf_id}
             if requirements[rid].owner_leaf_id in tree.nodes else set()))
        task = _render_assignment(runtime, requirements[rid],
                                  blocked_source_directive=blocked_source_directive)
        task += "\n\n# Internal report verification check\n\n" + json.dumps({
            "requirement_id": rid,
            "related_node_ids": related,
            "findings": [{
                "severity": f.severity, "finding": _bounded(f.finding, 4000),
                "corrective_instruction": _bounded(f.corrective_instruction, 4000),
            } for f in grouped[rid]],
            "draft_to_check": draft,
        }, ensure_ascii=False)
        task += (
            "\nThese are internal draft-review hypotheses, not benchmark judge feedback or facts. "
            "Verify or refute them against the public question and independently collected "
            "evidence/calculations. Identify the exact claim checked, assumptions and units, "
            "supporting observations, and remaining uncertainty. Do not simply repeat the "
            "old draft or treat the corrective instruction as the answer. "
            "The complete draft is provided: locate the affected sections and all downstream "
            "uses of the disputed claim. Check cross-section consistency, not just the introduction."
        )
        planned.append(HTNTaskNode(id=f"report-check-{rid}",
                                   title=f"Verify report requirement {rid}", task=task))

    before = {nid: node.model_dump() for nid, node in tree.nodes.items()}
    try:
        nodes = tree.append_report_checks(planned)
    except ValueError as exc:
        diagnostics.emit("report_verification_repair_skipped", reason="tree_not_patchable",
                         detail=str(exc))
        return ()
    runtime.verification_repair_used = True
    diagnostics.write_json("report_repair/tree-before.json", before)
    diagnostics.write_json("report_repair/tree-appended.json", {
        nid: node.model_dump() for nid, node in tree.nodes.items()
    })
    diagnostics.emit("report_verification_repair_start", node_ids=[n.id for n in nodes],
                     requirement_ids=targets, budget_s=max(0.0, cutoff - time.monotonic()))
    accept_results = True
    memos: dict[str, str] = {}

    async def check(node: HTNTaskNode, rid: str) -> AgentResultRecord | None:
        agent = AgentTaskSummary(agent_id=node.id, role_id=research_role_id,
                                 task_summary=node.title)
        try:
            record = await dispatch(agent, node.task)
            if not accept_results or time.monotonic() >= cutoff:
                return None
            diagnostics.write_json(f"report_repair/{node.id}-record.json", asdict(record))
            node.result_summary = record.result_summary
            node.ref = record.ref
            if (record.execution_quality not in {"submitted", "best_effort"}
                    or not record.result_summary.strip()):
                node.status = HTNTaskStatus.FAILED
                node.failure_reason = record.quality_reason or "Research check did not deliver usable evidence"
                return None
            # Completion of a worker is not proof that the finding was resolved.
            node.status = HTNTaskStatus.COMPLETED
            node.completion_outcome = HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            memos[node.id] = f"Check {node.id} for {rid}:\n{record.result_summary}"
            await runtime.observe_records(
                leaf_id=node.id, leaf_task=node.task, requirement_ids=(rid,), records=(record,),
                llm=evidence_llm, deadline=cutoff,
                accept_results=lambda: accept_results and time.monotonic() < cutoff,
            )
            return record
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve other successful checks
            if accept_results and time.monotonic() < cutoff:
                if node.status is not HTNTaskStatus.COMPLETED:
                    node.status = HTNTaskStatus.FAILED
                    node.failure_reason = f"Research check failed: {type(exc).__name__}"
                diagnostics.emit("report_verification_repair_target_failed", node_id=node.id,
                                 error_type=type(exc).__name__)
            return None

    tasks: list[asyncio.Task[AgentResultRecord | None]] = []
    cancellation: asyncio.CancelledError | None = None
    try:
        for node, rid in zip(nodes, targets, strict=True):
            if not dispatch_budget.try_acquire():
                node.status = HTNTaskStatus.FAILED
                node.failure_reason = "Question worker dispatch budget exhausted"
                continue
            tree.mark_running(node.id)
            tasks.append(asyncio.create_task(check(node, rid), name=f"report-check:{rid}"))
            diagnostics.emit("question_worker_admitted", phase="report_verification_repair",
                             node_id=node.id, dispatch_budget=dispatch_budget.snapshot())
        if tasks:
            await asyncio.wait(tasks, timeout=max(0.0, cutoff - time.monotonic()))
    except asyncio.CancelledError as exc:
        cancellation = exc
    finally:
        accept_results = False
        unsettled, cancellation, _ = await _cancel_and_settle_insulated(
            tasks, recovery_cutoff=cutoff, cancellation=cancellation,
        )
        for node in nodes:
            if node.status in {HTNTaskStatus.PENDING, HTNTaskStatus.RUNNING}:
                node.status = HTNTaskStatus.FAILED
                node.failure_reason = "Research check cancelled or deadline exhausted"
        diagnostics.write_json("report_repair/tree-after-checks.json", {
            nid: node.model_dump() for nid, node in tree.nodes.items()
        })
        diagnostics.emit("report_verification_repair_end", result_count=len(memos),
                         unsettled_count=unsettled, cancelled=cancellation is not None)
    if cancellation is not None:
        raise cancellation
    return tuple(memos[node.id] for node in nodes if node.id in memos)


__all__ = ["run_verification_repair"]
