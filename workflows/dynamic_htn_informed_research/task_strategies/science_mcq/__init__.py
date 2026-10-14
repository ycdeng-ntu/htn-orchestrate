"""Standalone generic science multiple-choice task strategy."""

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
    _MCQ_STRATEGIC_DECISION_SYSTEM,
    _MCQ_STRATEGIC_DECISION_TOOLS,
    _MCQ_TASK_SUMMARY_MAX_CHARS,
    _MCQ_WAIT_ACTION_GUIDANCE,
)
from .instruction import (
    _MCQ_INSTRUCTION_GENERATION_SYSTEM,
    _MCQ_SCOPE_BOUNDARY,
    _MCQ_SOLVING_PROTOCOL,
    _MCQ_WORKER_PROMPT_TEMPLATE,
)
from .planning import (
    _AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL,
    _EXECUTE_DIRECT_TOOL,
    _MCQ_DECOMPOSITION_REPAIR_SYSTEM,
    _MCQ_DECOMPOSITION_SYSTEM,
    _MCQ_PLANNING_PROTOCOL_RETRY_USER,
    _MCQ_REBUILD_STRATEGY_REQUIREMENTS,
    _MCQ_RESTART_CONTEXT_PREAMBLE,
    _MCQ_SUBTASK_AUDIT_PROTOCOL_RETRY_USER,
    _MCQ_SUBTASK_FIDELITY_AUDITOR_SYSTEM,
    _MCQ_SUBTASK_FIDELITY_REVISER_SYSTEM,
    _REVISE_SUBTASK_TOOL,
)
from .recovery import (
    _ANSWER_VALUE_ANNOTATION_TOOL,
    _FINALIZE_GATE_TOOL,
    _MCQ_ANSWER_VALUE_ANNOTATION_SYSTEM,
    _MCQ_FINALIZE_GATE_SYSTEM,
    _MCQ_MALFORMED_TRACE_RECOVERY_SYSTEM,
    _MCQ_SUPPORTED_FACTS_SYSTEM,
    _SUPPORTED_FACTS_TOOL,
)
from .synthesis import (
    _FORCED_FINALIZATION_AUDIT_TOOL,
    _MCQ_EPOCH_0_CLOSURE_POLICY,
    _MCQ_EPOCH_1_BEST_EFFORT_POLICY,
    _MCQ_FORCED_AUDIT_SYSTEM,
    _MCQ_FORCED_SELECTION_SYSTEM,
    _MCQ_PARTIAL_SYNTHESIS_SYSTEM,
    _MCQ_ROOT_ANSWER_PATTERN,
    _MCQ_ROOT_DIRECT_SYSTEM,
    _MCQ_ROOT_DIRECT_TOOLS,
    _MCQ_SYNTHESIS_SYSTEM,
    _MCQ_TERMINAL_BEST_EFFORT_POLICY,
    _MCQ_TREE_EPOCH_0_CLOSURE_POLICY,
    _MCQ_TREE_EPOCH_1_BEST_EFFORT_POLICY,
    _MCQ_TREE_INTERNAL_SYSTEM,
    _MCQ_TREE_INTERNAL_USER_GUIDANCE,
    _MCQ_TREE_ROOT_SYSTEM,
    _MCQ_TREE_ROOT_USER_GUIDANCE,
    _SYNTHESIS_TOOLS,
)
from .workers import SCIENCE_MCQ_AGENT_DEFS, SCIENCE_MCQ_WORKER_ROLES


_SCIENCE_MCQ_TEMPLATE = TaskStrategy(
    name="science_mcq",
    planning=PlanningStrategy(
        decomposition_system=_MCQ_DECOMPOSITION_SYSTEM,
        decomposition_repair_system=_MCQ_DECOMPOSITION_REPAIR_SYSTEM,
        subtask_auditor_system=_MCQ_SUBTASK_FIDELITY_AUDITOR_SYSTEM,
        subtask_audit_retry_user=_MCQ_SUBTASK_AUDIT_PROTOCOL_RETRY_USER,
        subtask_reviser_system=_MCQ_SUBTASK_FIDELITY_REVISER_SYSTEM,
        planning_retry_user=_MCQ_PLANNING_PROTOCOL_RETRY_USER,
        decompose_tool=_DECOMPOSE_TOOL,
        execute_direct_tool=_EXECUTE_DIRECT_TOOL,
        audit_subtask_tool=_AUDIT_SUBTASK_TOOL,
        revise_subtask_tool=_REVISE_SUBTASK_TOOL,
        restart_context_preamble=_MCQ_RESTART_CONTEXT_PREAMBLE,
        rebuild_strategy_requirements=_MCQ_REBUILD_STRATEGY_REQUIREMENTS,
        minimum_decomposition_children=1,
        include_sibling_task_semantics_in_audit=False,
        retry_overlong_subtask_revision=False,
    ),
    coordination=CoordinationStrategy(
        strategic_decision_system=_MCQ_STRATEGIC_DECISION_SYSTEM,
        wait_action_guidance=_MCQ_WAIT_ACTION_GUIDANCE,
        decision_tools=tuple(_MCQ_STRATEGIC_DECISION_TOOLS),
        primary_role_id="dynir_science_mcq_solver",
        verifier_role_id="dynir_science_mcq_checker",
        primary_recovery_task=(
            "Solve the current science multiple-choice leaf end to end. Derive the facts first, "
            "audit the full option mapping, and return exactly one offered uppercase label."
        ),
        verifier_recovery_task=(
            "Independently recompute the science, audit every atomic statement and the final "
            "fact-to-option mapping, then return a verdict and corrected offered label if needed."
        ),
        role_aliases=(
            ("solver", "dynir_science_mcq_solver"),
            ("scientist", "dynir_science_mcq_solver"),
            ("derive", "dynir_science_mcq_solver"),
            ("checker", "dynir_science_mcq_checker"),
            ("verifier", "dynir_science_mcq_checker"),
            ("reviewer", "dynir_science_mcq_checker"),
            ("critic", "dynir_science_mcq_checker"),
        ),
        neutralize_initial_primary_assignments=False,
        require_result_before_verifier=True,
        precursor_evidence_unlocks_verifier=False,
        task_summary_max_chars=_MCQ_TASK_SUMMARY_MAX_CHARS,
    ),
    instruction=InstructionStrategy(
        generation_system=_MCQ_INSTRUCTION_GENERATION_SYSTEM,
        solving_protocol=_MCQ_SOLVING_PROTOCOL,
        scope_boundary=_MCQ_SCOPE_BOUNDARY,
        deterministic_composition=False,
        assignment_only_composition=True,
        worker_prompt_template=_MCQ_WORKER_PROMPT_TEMPLATE,
    ),
    workers=WorkerStrategy(
        agent_definitions=tuple(SCIENCE_MCQ_AGENT_DEFS),
        role_ids=tuple(SCIENCE_MCQ_WORKER_ROLES),
    ),
    recovery=RecoveryStrategy(
        malformed_trace_system=_MCQ_MALFORMED_TRACE_RECOVERY_SYSTEM,
        finalize_gate_system=_MCQ_FINALIZE_GATE_SYSTEM,
        finalize_gate_tool=_FINALIZE_GATE_TOOL,
        supported_facts_system=_MCQ_SUPPORTED_FACTS_SYSTEM,
        supported_facts_tool=_SUPPORTED_FACTS_TOOL,
        answer_annotation_system=_MCQ_ANSWER_VALUE_ANNOTATION_SYSTEM,
        answer_annotation_tool=_ANSWER_VALUE_ANNOTATION_TOOL,
    ),
    synthesis=SynthesisStrategy(
        synthesis_system=_MCQ_SYNTHESIS_SYSTEM,
        root_direct_system=_MCQ_ROOT_DIRECT_SYSTEM,
        tree_root_system=_MCQ_TREE_ROOT_SYSTEM,
        tree_internal_system=_MCQ_TREE_INTERNAL_SYSTEM,
        tree_root_user_guidance=_MCQ_TREE_ROOT_USER_GUIDANCE,
        tree_internal_user_guidance=_MCQ_TREE_INTERNAL_USER_GUIDANCE,
        partial_system=_MCQ_PARTIAL_SYNTHESIS_SYSTEM,
        epoch_0_closure_policy=_MCQ_EPOCH_0_CLOSURE_POLICY,
        epoch_1_best_effort_policy=_MCQ_EPOCH_1_BEST_EFFORT_POLICY,
        tree_epoch_0_closure_policy=_MCQ_TREE_EPOCH_0_CLOSURE_POLICY,
        tree_epoch_1_best_effort_policy=_MCQ_TREE_EPOCH_1_BEST_EFFORT_POLICY,
        terminal_best_effort_policy=_MCQ_TERMINAL_BEST_EFFORT_POLICY,
        synthesis_tools=tuple(_SYNTHESIS_TOOLS),
        root_direct_tools=tuple(_MCQ_ROOT_DIRECT_TOOLS),
        forced_audit_system=_MCQ_FORCED_AUDIT_SYSTEM,
        forced_audit_tool=_FORCED_FINALIZATION_AUDIT_TOOL,
        forced_selection_system=_MCQ_FORCED_SELECTION_SYSTEM,
        include_child_semantics=False,
        root_answer_pattern=_MCQ_ROOT_ANSWER_PATTERN,
    ),
)


_OFFLINE_GUIDANCE = (
    "Worker capabilities for this task are offline computation, readback of task-produced files, "
    "and report submission. Base plans, assignments and verification on ROOT, scientific reasoning "
    "and reproducible local calculations. Do not request external retrieval or try network access "
    "through code. Missing external retrieval alone is not an unresolved requirement; distinguish "
    "uncertain recalled facts from justified conclusions and never invent missing premises."
)


def mcq_strategy_for_access(*, network_enabled: bool) -> TaskStrategy:
    from ..prompt_access import render_strategy

    return render_strategy(
        _SCIENCE_MCQ_TEMPLATE, network_enabled=network_enabled,
        system_suffix="" if network_enabled else _OFFLINE_GUIDANCE,
    )


SCIENCE_MCQ_STRATEGY = mcq_strategy_for_access(network_enabled=True)

__all__ = ["SCIENCE_MCQ_STRATEGY", "mcq_strategy_for_access"]
