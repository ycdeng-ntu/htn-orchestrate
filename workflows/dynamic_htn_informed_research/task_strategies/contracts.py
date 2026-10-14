"""Narrow contracts separating task semantics from orchestration mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ToolSchema = dict[str, Any]
OutputMode = Literal["short_answer", "report"]
RootAnswerMode = Literal["short_answer", "multiline_report"]
@dataclass(frozen=True)
class DeliveryStrategy:
    output_mode: OutputMode = "short_answer"


@dataclass(frozen=True)
class PlanningStrategy:
    decomposition_system: str
    decomposition_repair_system: str
    subtask_auditor_system: str
    subtask_audit_retry_user: str
    subtask_reviser_system: str
    planning_retry_user: str
    decompose_tool: ToolSchema
    execute_direct_tool: ToolSchema
    audit_subtask_tool: ToolSchema
    revise_subtask_tool: ToolSchema
    restart_context_preamble: str
    rebuild_strategy_requirements: str
    # A task family may reserve a decomposition for a fixed topology.  Values
    # above one reject undersized model plans before they enter the HTN tree.
    minimum_decomposition_children: int = 1
    # Some task families need the semantic auditor to compare complete sibling
    # objectives for shared premise injection or route collapse.  Keep this
    # opt-in so research audits retain their answer-isolated sibling outline.
    include_sibling_task_semantics_in_audit: bool = False
    # A structurally valid local revision that exceeds the compact task limit
    # may be retried once with its actual character count.  When enabled, a
    # structurally valid retry remains authoritative even if it is still over
    # the advisory limit; it is never truncated or replaced by the audited,
    # known-defective original solely because of length.
    retry_overlong_subtask_revision: bool = False
    subtask_revision_max_chars: int = 1_000

    @property
    def planning_tools(self) -> list[ToolSchema]:
        return [self.decompose_tool, self.execute_direct_tool]


@dataclass(frozen=True)
class CoordinationStrategy:
    strategic_decision_system: str
    wait_action_guidance: str
    decision_tools: tuple[ToolSchema, ...]
    primary_role_id: str
    verifier_role_id: str
    primary_recovery_task: str
    verifier_recovery_task: str
    role_aliases: tuple[tuple[str, str], ...] = ()
    neutralize_initial_primary_assignments: bool = False
    require_result_before_verifier: bool = False
    # Some task strategies use completed precursor memos as the candidate
    # evidence for an adjudication leaf.  Keep that capability explicit and
    # strategy-scoped instead of inferring it from a title marker.
    precursor_evidence_unlocks_verifier: bool = False
    # Zero preserves the legacy unrestricted coordinator assignment contract.
    # A positive value is exposed in the tool schema and enforced on the first
    # parse so an oversized assignment gets one explicit compression retry.
    # A structurally valid retry may exceed the advisory limit and is preserved
    # verbatim rather than discarded or truncated.
    task_summary_max_chars: int = 0


@dataclass(frozen=True)
class InstructionStrategy:
    generation_system: str
    solving_protocol: str = ""
    scope_boundary: str = ""
    deterministic_composition: bool = False
    # Skip the instruction-generation LLM and pass only the coordinator's
    # scoped assignment into ``worker_prompt_template``.  This is distinct from
    # deterministic_composition, which intentionally renders the full C/R/live
    # context for legacy strategies.
    assignment_only_composition: bool = False
    worker_prompt_template: str = ""
    # Attach original evidence after generation, including its failure fallback.
    # Scoped to this leaf and its dependencies; other strategies remain unchanged.
    attach_evidence_context: bool = False


@dataclass(frozen=True)
class WorkerStrategy:
    agent_definitions: tuple[Any, ...]
    role_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryStrategy:
    malformed_trace_system: str
    finalize_gate_system: str
    finalize_gate_tool: ToolSchema
    supported_facts_system: str
    supported_facts_tool: ToolSchema
    answer_annotation_system: str
    answer_annotation_tool: ToolSchema


@dataclass(frozen=True)
class SynthesisStrategy:
    synthesis_system: str
    root_direct_system: str
    tree_root_system: str
    tree_internal_system: str
    tree_root_user_guidance: str
    tree_internal_user_guidance: str
    partial_system: str
    epoch_0_closure_policy: str
    epoch_1_best_effort_policy: str
    tree_epoch_0_closure_policy: str
    tree_epoch_1_best_effort_policy: str
    terminal_best_effort_policy: str
    synthesis_tools: tuple[ToolSchema, ...]
    root_direct_tools: tuple[ToolSchema, ...]
    forced_audit_system: str
    forced_audit_tool: ToolSchema
    forced_selection_system: str
    # Expose child objectives and dependency edges to tree synthesis when the
    # strategy assigns semantic meaning to alternative-route portfolios.
    include_child_semantics: bool = False
    # Optional full-response normalization contract for root answers.  When
    # set, the first capture group is serialized as the answer; non-matches are
    # rejected as protocol failures.  Empty preserves the legacy free-text
    # behavior used by research and science_olympiad.
    root_answer_pattern: str = ""
    # Report roots carry their complete Markdown deliverable in the answer tool
    # field. The default preserves the legacy single-line marker contract.
    root_answer_mode: RootAnswerMode = "short_answer"


@dataclass(frozen=True)
class TaskStrategy:
    name: str
    planning: PlanningStrategy
    coordination: CoordinationStrategy
    instruction: InstructionStrategy
    workers: WorkerStrategy
    recovery: RecoveryStrategy
    synthesis: SynthesisStrategy
    delivery: DeliveryStrategy = DeliveryStrategy()
