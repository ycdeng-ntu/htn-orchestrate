"""A shallow research DAG with one mandatory, dependency-gated delivery leaf."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from ...diagnostics import DiagnosticRecorder
from ...htn.llm_types import LLMClient
from ...htn.planner import DecompositionBudget, DecompositionDecision, DecompositionProfile
from ...htn.task_node import HTNCompletionOutcome, HTNNodeCompletion, HTNTaskNode

from ...htn.task_tree import HTNTree
from .plan_audit import (
    PLAN_AUDIT_SYSTEMS, PLAN_REVISER_SYSTEM, accepted_research_nodes, audit_report_plan,
    plan_constraints, produce_report_plan, revised_plan_issues,
)
from .prompt_access import report_system_prompt
from .review import ReviewState

if TYPE_CHECKING:
    from ...htn.epoch import EpochRunResult

DELIVERY_ID = "report-delivery"


class ReportPlanningStrategy:
    """Planner -> audit -> optional whole-plan revision + mechanical validation -> delivery."""

    profile = DecompositionProfile(name="report_research")

    def __init__(
        self, *, planner_llm: LLMClient, audit_llm: LLMClient, revision_llm: LLMClient,
        planner_system: str, network_enabled: bool = True,
        deadline: float | None = None, diagnostics: DiagnosticRecorder | None = None,
    ) -> None:
        self.planner_llm = planner_llm
        self.audit_llm = audit_llm
        self.revision_llm = revision_llm
        self.planner_system = planner_system
        self.audit_systems = {dimension: report_system_prompt(prompt, network_enabled=network_enabled)
                              for dimension, prompt in PLAN_AUDIT_SYSTEMS.items()}
        self.reviser_system = report_system_prompt(PLAN_REVISER_SYSTEM, network_enabled=network_enabled)
        self.deadline = deadline
        self.diagnostics = diagnostics

    async def decompose(
        self, tree: HTNTree, node: HTNTaskNode, budget: DecompositionBudget
    ) -> DecompositionDecision:
        if node.id != tree.root_id:
            raise ValueError("Report research nodes cannot be decomposed further")
        research_budget = replace(budget, max_children=max(1, budget.max_children - 1))
        # This input is immutable across plan revision, including the full recovery capsule.
        planner_input = {"task": node.task}
        if tree.restart_context:
            planner_input["recovery_capsule"] = tree.restart_context
        constraints = plan_constraints(research_budget)
        children: list[HTNTaskNode] = []
        acceptance_method = "fallback"
        try:
            plan = await produce_report_plan(
                planner_input=planner_input, constraints=constraints, llm=self.planner_llm,
                system=self.planner_system, deadline=self.deadline,
                diagnostics=self.diagnostics, epoch_id=tree.epoch_id,
            )
            audit = await audit_report_plan(
                plan=plan, planner_input=planner_input, budget=research_budget,
                llm=self.audit_llm, systems=self.audit_systems, deadline=self.deadline,
                diagnostics=self.diagnostics, epoch_id=tree.epoch_id,
            )
            if audit["plan_review"]["verdict"] == "PASS":
                children = accepted_research_nodes(plan, audit)
                acceptance_method = "initial_audit"
            else:
                if self.diagnostics:
                    self.diagnostics.emit("report_plan_rejected", epoch_id=tree.epoch_id,
                                          attempt=1, audit=audit, next_action="revise")
                plan = await produce_report_plan(
                    planner_input=planner_input, constraints=constraints, llm=self.revision_llm,
                    system=self.reviser_system, previous_plan=plan, audit=audit,
                    deadline=self.deadline, diagnostics=self.diagnostics, epoch_id=tree.epoch_id,
                )
                # No LLM calls or stale classifications are applied to the replacement tree.
                issues = revised_plan_issues(plan, research_budget, root_id=tree.root_id)
                if self.diagnostics:
                    self.diagnostics.write_json(
                        f"report_delivery/epoch-{tree.epoch_id}/revision-validation.json",
                        {"valid": not issues, "issues": issues, "semantic_audit_performed": False},
                    )
                if issues:
                    raise ValueError("Invalid replacement tree: " + "; ".join(issues))
                children = [HTNTaskNode(**raw) for raw in plan["nodes"]]
                acceptance_method = "revision_mechanical"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.diagnostics:
                self.diagnostics.emit(
                    "report_plan_fallback", epoch_id=tree.epoch_id,
                    error_type=type(exc).__name__,
                    reason="Planning or validation failed; using research-only fallback",
                )
            children = []
        if not children:
            acceptance_method = "fallback"
            children = [
                HTNTaskNode(
                    id="report-research",
                    title="Derive and verify the problem",
                    task="Derive and independently check all scientific results needed for the "
                    "public question below. Return a research evidence memo with calculations, "
                    "assumptions and unresolved gaps. Do not write, assemble or perform final "
                    "review of the complete report; the runtime delivery leaf owns that work."
                    "\n\nPublic question:\n" + node.task,
                )
            ]
        if self.diagnostics:
            acceptance = {
                "accepted": acceptance_method != "fallback",
                "acceptance_method": acceptance_method,
                "accepted_by_audit": acceptance_method == "initial_audit",
                "revision_mechanically_validated": acceptance_method == "revision_mechanical",
                "fallback": acceptance_method == "fallback",
            }
            self.diagnostics.write_json(
                f"report_delivery/epoch-{tree.epoch_id}/plan-acceptance.json", acceptance,
            )
            self.diagnostics.emit("report_plan_acceptance", epoch_id=tree.epoch_id, **acceptance)
        # The delivery identity is owned by the runtime, not inferred from model titles.
        ids = {child.id for child in children}
        delivery_id = DELIVERY_ID
        while delivery_id in ids:
            delivery_id += "-final"
        children = [
            child.model_copy(
                update={"needs_decomposition": False, "execution_kind": "standard"}, deep=True
            )
            for child in children
        ]
        children.append(
            HTNTaskNode(
                id=delivery_id,
                title="Write and independently verify the complete report",
                task="Answer the original ROOT question using all preceding research. Maintain one "
                "complete draft, verify it end to end, and request Planner rebuild for substantive "
                "research gaps or incompatible models. Do not delegate new research outside the tree.",
                depends_on=[child.id for child in children],
                execution_kind="report_delivery",
            )
        )
        return DecompositionDecision(
            action="decompose",
            children=tuple(children),
            rationale="Research leaves followed by mandatory report delivery",
        )


def delivery_root_summary(
    root: HTNTaskNode, parent: HTNTaskNode, children: list[HTNTaskNode]
) -> HTNNodeCompletion:
    """Return the exact checked deliverable, without another model rewrite."""
    delivery = [child for child in children if child.execution_kind == "report_delivery"]
    if parent.id != root.id or len(delivery) != 1:
        raise ValueError("Report tree must have exactly one delivery child of ROOT")
    node = delivery[0]
    return HTNNodeCompletion(
        node.result_summary or "",
        node.completion_outcome or HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        node.failure_reason,
    )


def rebuild_context(run: EpochRunResult) -> str:
    """Preserve full public-task research, draft and actionable review; no lossy capsule."""
    material = []
    review_state = None
    for node in run.tree.nodes.values():
        if node.id == run.tree.root_id:
            continue
        owner = run.orchestrators.get(node.id)
        if getattr(owner, "review_state", None) is not None:
            review_state = owner.review_state
        material.append(
            {
                "id": node.id,
                "task": node.task,
                "status": node.status.value,
                "outcome": node.completion_outcome.value if node.completion_outcome else None,
                "memo": node.result_summary,
                "failure": node.failure_reason,
                "worker_reports": [
                    {
                        "role": record.role_id,
                        "quality": record.execution_quality,
                        "content": record.result_summary,
                    }
                    for record in (owner.results if owner else ())
                ],
                "rebuild_feedback": getattr(owner, "rebuild_feedback", ""),
            }
        )
    if review_state is None:
        # An epoch that never reached delivery must not erase earlier findings.
        review_state = ReviewState(run.tree.restart_context).snapshot()
    return json.dumps(
        {
            "previous_epoch": run.tree.epoch_id,
            "review_state": review_state,
            "instruction": "Reuse supported derivations; prior conclusions and reviewer "
            "claims are evidence to check, not authority. Plan targeted research to resolve "
            "the reported gaps; retain coverage of the full original question.",
            "material": material,
        },
        ensure_ascii=False,
    )
