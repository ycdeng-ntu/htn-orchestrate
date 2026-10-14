"""Independent long-form scientific report strategy."""

from __future__ import annotations

from workflows.dynamic_htn_informed_research.task_strategies.contracts import (
    CoordinationStrategy,
    DeliveryStrategy,
    InstructionStrategy,
    PlanningStrategy,
    RecoveryStrategy,
    SynthesisStrategy,
    TaskStrategy,
    WorkerStrategy,
)

from .coordination import (
    _STRATEGIC_DECISION_SYSTEM,
    _STRATEGIC_DECISION_TOOLS,
    _WAIT_ACTION_GUIDANCE,
)
from .instruction import _INSTRUCTION_GENERATION_SYSTEM, _SCOPE_BOUNDARY
from .planning import (
    _AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL,
    _DECOMPOSITION_REPAIR_SYSTEM,
    _EXECUTE_DIRECT_TOOL,
    _PLANNING_PROTOCOL_RETRY_USER,
    _REBUILD_STRATEGY_REQUIREMENTS,
    _REPORT_DECOMPOSITION_SYSTEM,
    _RESTART_CONTEXT_PREAMBLE,
    _REVISE_SUBTASK_TOOL,
    _SUBTASK_AUDIT_PROTOCOL_RETRY_USER,
    _SUBTASK_FIDELITY_AUDITOR_SYSTEM,
    _SUBTASK_FIDELITY_REVISER_SYSTEM,
)
from .recovery import (
    _ANSWER_VALUE_ANNOTATION_SYSTEM,
    _ANSWER_VALUE_ANNOTATION_TOOL,
    _FINALIZE_GATE_SYSTEM,
    _FINALIZE_GATE_TOOL,
    _MALFORMED_TRACE_RECOVERY_SYSTEM,
    _SUPPORTED_FACTS_SYSTEM,
    _SUPPORTED_FACTS_TOOL,
)
from .synthesis import (
    _EPOCH_0_R_CLOSURE_POLICY,
    _EPOCH_1_BEST_EFFORT_SYNTHESIS_POLICY,
    _FORCED_FINALIZATION_AUDIT_SYSTEM,
    _FORCED_FINALIZATION_AUDIT_TOOL,
    _FORCED_FINALIZATION_SELECTION_SYSTEM,
    _PARTIAL_SYNTHESIS_SYSTEM,
    _ROOT_DIRECT_SYNTHESIS_SYSTEM,
    _ROOT_DIRECT_SYNTHESIS_TOOLS,
    _SYNTHESIS_SYSTEM,
    _SYNTHESIS_TOOLS,
    _TERMINAL_BEST_EFFORT_POLICY,
    _TREE_EPOCH_0_CLOSURE_POLICY,
    _TREE_EPOCH_1_BEST_EFFORT_POLICY,
    _TREE_INTERNAL_SYNTHESIS_SYSTEM,
    _TREE_INTERNAL_USER_GUIDANCE,
    _TREE_ROOT_SYNTHESIS_SYSTEM,
    _TREE_ROOT_USER_GUIDANCE,
)
from .workers import REPORT_RESEARCH_AGENT_DEFS, REPORT_RESEARCH_WORKER_ROLES


_REPORT_RESEARCH_TEMPLATE = TaskStrategy(
    name="report_research",
    planning=PlanningStrategy(
        decomposition_system=_REPORT_DECOMPOSITION_SYSTEM,
        decomposition_repair_system=_DECOMPOSITION_REPAIR_SYSTEM,
        subtask_auditor_system=_SUBTASK_FIDELITY_AUDITOR_SYSTEM,
        subtask_audit_retry_user=_SUBTASK_AUDIT_PROTOCOL_RETRY_USER,
        subtask_reviser_system=_SUBTASK_FIDELITY_REVISER_SYSTEM,
        planning_retry_user=_PLANNING_PROTOCOL_RETRY_USER,
        decompose_tool=_DECOMPOSE_TOOL,
        execute_direct_tool=_EXECUTE_DIRECT_TOOL,
        audit_subtask_tool=_AUDIT_SUBTASK_TOOL,
        revise_subtask_tool=_REVISE_SUBTASK_TOOL,
        restart_context_preamble=_RESTART_CONTEXT_PREAMBLE,
        rebuild_strategy_requirements=_REBUILD_STRATEGY_REQUIREMENTS,
        include_sibling_task_semantics_in_audit=True,
    ),
    coordination=CoordinationStrategy(
        strategic_decision_system=_STRATEGIC_DECISION_SYSTEM,
        wait_action_guidance=_WAIT_ACTION_GUIDANCE,
        decision_tools=tuple(_STRATEGIC_DECISION_TOOLS),
        primary_role_id="dynir_report_researcher",
        require_result_before_verifier=True,
        verifier_role_id="dynir_report_verifier",
        primary_recovery_task=(
            "Research this leaf's uncovered requirements and return an evidence-grounded "
            "memo with explicit remaining gaps. Do not write the final report."
        ),
        verifier_recovery_task=(
            "Independently audit coverage, scientific specificity, calculations, "
            "and upstream identities and assumptions[[WEB]], including citation integrity[[/WEB]] "
            "within this leaf's scope. Reopen decisive premises against ROOT; return local "
            "findings and downstream implications, not a whole-report review."
        ),
        role_aliases=(
            ("researcher", "dynir_report_researcher"),
            ("report_researcher", "dynir_report_researcher"),
            ("verifier", "dynir_report_verifier"),
            ("report_verifier", "dynir_report_verifier"),
        ),
    ),
    instruction=InstructionStrategy(
        generation_system=_INSTRUCTION_GENERATION_SYSTEM,
        scope_boundary=_SCOPE_BOUNDARY,
        attach_evidence_context=True,
    ),
    workers=WorkerStrategy(
        agent_definitions=tuple(REPORT_RESEARCH_AGENT_DEFS),
        role_ids=tuple(REPORT_RESEARCH_WORKER_ROLES),
    ),
    recovery=RecoveryStrategy(
        malformed_trace_system=_MALFORMED_TRACE_RECOVERY_SYSTEM,
        finalize_gate_system=_FINALIZE_GATE_SYSTEM,
        finalize_gate_tool=_FINALIZE_GATE_TOOL,
        supported_facts_system=_SUPPORTED_FACTS_SYSTEM,
        supported_facts_tool=_SUPPORTED_FACTS_TOOL,
        answer_annotation_system=_ANSWER_VALUE_ANNOTATION_SYSTEM,
        answer_annotation_tool=_ANSWER_VALUE_ANNOTATION_TOOL,
    ),
    synthesis=SynthesisStrategy(
        synthesis_system=_SYNTHESIS_SYSTEM,
        root_direct_system=_ROOT_DIRECT_SYNTHESIS_SYSTEM,
        tree_root_system=_TREE_ROOT_SYNTHESIS_SYSTEM,
        tree_internal_system=_TREE_INTERNAL_SYNTHESIS_SYSTEM,
        tree_root_user_guidance=_TREE_ROOT_USER_GUIDANCE,
        tree_internal_user_guidance=_TREE_INTERNAL_USER_GUIDANCE,
        partial_system=_PARTIAL_SYNTHESIS_SYSTEM,
        epoch_0_closure_policy=_EPOCH_0_R_CLOSURE_POLICY,
        epoch_1_best_effort_policy=_EPOCH_1_BEST_EFFORT_SYNTHESIS_POLICY,
        tree_epoch_0_closure_policy=_TREE_EPOCH_0_CLOSURE_POLICY,
        tree_epoch_1_best_effort_policy=_TREE_EPOCH_1_BEST_EFFORT_POLICY,
        terminal_best_effort_policy=_TERMINAL_BEST_EFFORT_POLICY,
        synthesis_tools=tuple(_SYNTHESIS_TOOLS),
        root_direct_tools=tuple(_ROOT_DIRECT_SYNTHESIS_TOOLS),
        forced_audit_system=_FORCED_FINALIZATION_AUDIT_SYSTEM,
        forced_audit_tool=_FORCED_FINALIZATION_AUDIT_TOOL,
        forced_selection_system=_FORCED_FINALIZATION_SELECTION_SYSTEM,
        root_answer_mode="multiline_report",
    ),
    delivery=DeliveryStrategy(output_mode="report"),
)


def report_strategy_for_access(*, network_enabled: bool) -> TaskStrategy:
    from .prompt_access import render_strategy

    return render_strategy(_REPORT_RESEARCH_TEMPLATE, network_enabled=network_enabled)


REPORT_RESEARCH_STRATEGY = report_strategy_for_access(network_enabled=True)

__all__ = ["REPORT_RESEARCH_STRATEGY", "report_strategy_for_access"]
