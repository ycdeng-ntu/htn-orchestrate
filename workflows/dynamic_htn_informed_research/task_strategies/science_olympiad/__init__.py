"""Science-olympiad task strategy assembled independently of research semantics."""

from workflows.dynamic_htn_informed_research.task_strategies.contracts import (
    CoordinationStrategy,
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
from .instruction import (
    _INSTRUCTION_GENERATION_SYSTEM,
    _SCOPE_BOUNDARY,
    _SOLVING_PROTOCOL,
    _WORKER_PROMPT_TEMPLATE,
)
from .planning import (
    _AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL,
    _DECOMPOSITION_REPAIR_SYSTEM,
    _EXECUTE_DIRECT_TOOL,
    _PLANNING_PROTOCOL_RETRY_USER,
    _REBUILD_STRATEGY_REQUIREMENTS,
    _RESTART_CONTEXT_PREAMBLE,
    _REVISE_SUBTASK_TOOL,
    _SCIENCE_DECOMPOSITION_SYSTEM,
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
    _EPOCH_0_CLOSURE_POLICY,
    _EPOCH_1_BEST_EFFORT_POLICY,
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
from .workers import SCIENCE_OLYMPIAD_AGENT_DEFS, SCIENCE_OLYMPIAD_WORKER_ROLES


_SCIENCE_OLYMPIAD_TEMPLATE = TaskStrategy(
    name="science_olympiad",
    planning=PlanningStrategy(
        decomposition_system=_SCIENCE_DECOMPOSITION_SYSTEM,
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
        minimum_decomposition_children=3,
        include_sibling_task_semantics_in_audit=True,
        retry_overlong_subtask_revision=True,
    ),
    coordination=CoordinationStrategy(
        strategic_decision_system=_STRATEGIC_DECISION_SYSTEM,
        wait_action_guidance=_WAIT_ACTION_GUIDANCE,
        decision_tools=tuple(_STRATEGIC_DECISION_TOOLS),
        primary_role_id="dynir_science_solver",
        verifier_role_id="dynir_science_checker",
        primary_recovery_task=(
            "Solve the current science leaf end to end from the authoritative statement; "
            "return the complete candidate answer and reproducible checks."
        ),
        verifier_recovery_task=(
            "Independently recompute the best candidate, audit every requested output field "
            "and condition, and return an explicit verdict plus corrected answer if needed."
        ),
        role_aliases=(
            ("solver", "dynir_science_solver"),
            ("scientist", "dynir_science_solver"),
            ("derive", "dynir_science_solver"),
            ("checker", "dynir_science_checker"),
            ("verifier", "dynir_science_checker"),
            ("reviewer", "dynir_science_checker"),
            ("critic", "dynir_science_checker"),
        ),
        neutralize_initial_primary_assignments=False,
        require_result_before_verifier=True,
        precursor_evidence_unlocks_verifier=True,
    ),
    instruction=InstructionStrategy(
        generation_system=_INSTRUCTION_GENERATION_SYSTEM,
        solving_protocol=_SOLVING_PROTOCOL,
        scope_boundary=_SCOPE_BOUNDARY,
        deterministic_composition=False,
        worker_prompt_template=_WORKER_PROMPT_TEMPLATE,
    ),
    workers=WorkerStrategy(
        agent_definitions=tuple(SCIENCE_OLYMPIAD_AGENT_DEFS),
        role_ids=tuple(SCIENCE_OLYMPIAD_WORKER_ROLES),
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
        epoch_0_closure_policy=_EPOCH_0_CLOSURE_POLICY,
        epoch_1_best_effort_policy=_EPOCH_1_BEST_EFFORT_POLICY,
        tree_epoch_0_closure_policy=_TREE_EPOCH_0_CLOSURE_POLICY,
        tree_epoch_1_best_effort_policy=_TREE_EPOCH_1_BEST_EFFORT_POLICY,
        terminal_best_effort_policy=_TERMINAL_BEST_EFFORT_POLICY,
        synthesis_tools=tuple(_SYNTHESIS_TOOLS),
        root_direct_tools=tuple(_ROOT_DIRECT_SYNTHESIS_TOOLS),
        forced_audit_system=_FORCED_FINALIZATION_AUDIT_SYSTEM,
        forced_audit_tool=_FORCED_FINALIZATION_AUDIT_TOOL,
        forced_selection_system=_FORCED_FINALIZATION_SELECTION_SYSTEM,
        include_child_semantics=True,
    ),
)


_OFFLINE_GUIDANCE = (
    "Worker capabilities for this task are offline computation, readback of task-produced files, "
    "and report submission. Base plans, assignments and verification on ROOT, scientific reasoning "
    "and reproducible local calculations. Do not request external retrieval or try network access "
    "through code. Missing external retrieval alone is not an unresolved requirement; distinguish "
    "uncertain recalled facts from justified conclusions and never invent missing premises."
)


def science_strategy_for_access(*, network_enabled: bool) -> TaskStrategy:
    from ..prompt_access import render_strategy

    return render_strategy(
        _SCIENCE_OLYMPIAD_TEMPLATE, network_enabled=network_enabled,
        system_suffix="" if network_enabled else _OFFLINE_GUIDANCE,
    )


SCIENCE_OLYMPIAD_STRATEGY = science_strategy_for_access(network_enabled=True)

__all__ = ["SCIENCE_OLYMPIAD_STRATEGY", "science_strategy_for_access"]
