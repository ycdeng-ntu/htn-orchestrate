"""HTN planning semantics owned exclusively by the generic science-MCQ strategy."""

from copy import deepcopy

from workflows.dynamic_htn_informed_research.task_strategies.research.planning import (
    _AUDIT_SUBTASK_TOOL as _BASE_AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL as _BASE_DECOMPOSE_TOOL,
    _EXECUTE_DIRECT_TOOL as _BASE_EXECUTE_DIRECT_TOOL,
    _REVISE_SUBTASK_TOOL as _BASE_REVISE_SUBTASK_TOOL,
)


_AUDIT_SUBTASK_TOOL = deepcopy(_BASE_AUDIT_SUBTASK_TOOL)
_DECOMPOSE_TOOL = deepcopy(_BASE_DECOMPOSE_TOOL)
_EXECUTE_DIRECT_TOOL = deepcopy(_BASE_EXECUTE_DIRECT_TOOL)
_REVISE_SUBTASK_TOOL = deepcopy(_BASE_REVISE_SUBTASK_TOOL)


_MCQ_DECOMPOSITION_SYSTEM = """\
Plan one self-contained scientific problem. ROOT is authoritative. Do not solve the problem or place a
candidate answer, formula, mechanism, or interpretation in the plan.

Choose exactly one action:
- `execute_direct` for one coherent reasoning chain, even when it has many sequential steps. The leaf
  coordinator can already run independent end-to-end solvers and a checker.
- `decompose` only for genuinely independent requested results or cases whose deliverables can be
  produced without sharing hidden intermediate state.

Do not decompose merely to obtain multiple solution methods; coordinate independent methods within one
end-to-end leaf. Decomposition children are jointly required deliverables, not alternative routes.

Protocol: emit exactly one `execute_direct` call, or one homogeneous `decompose` call per child in a
single response. Never mix actions or use a nested subtask array. Preserve every supplied condition and
requested output field. Do not split a coherent solution into lookup, algebra, arithmetic, and formatting
fragments. Use dependencies only when a child literally requires another child's result. Every child
must have a finite, checkable deliverable, and the plan must retain a route to the complete ROOT answer.
The rationale explains topology only.
"""


_MCQ_DECOMPOSITION_REPAIR_SYSTEM = """\
You repair only structural defects in a proposed science-problem HTN graph. The supplied defect list
is exhaustive. Preserve every retained title, task, and needs_decomposition value exactly. Change only
IDs and dependency edges required to remove duplicates, unknown/self dependencies, or cycles. If over
the child limit, remove the minimum number of tasks; otherwise preserve count and order. Do not solve,
rewrite, or improve the scientific plan. Return one `decompose` call per retained task.
"""


_MCQ_SUBTASK_FIDELITY_AUDITOR_SYSTEM = """\
Audit one proposed scientific HTN subtask for fidelity to ROOT and its parent. Do not solve the problem,
judge the proposed answer, or redesign the graph.

Return REVISE only when the target explicitly:
- changes or drops a supplied condition, meaning, requested case, format, or output field;
- introduces an unsupported assumption or candidate answer;
- separates tightly coupled reasoning into a deliverable that cannot be joined from declared inputs;
- claims to reconcile the result while omitting a requested field or case; or
- has no finite, checkable scientific deliverable.

A focused subproblem need not repeat requirements assigned to siblings. If no concrete wording defect
exists, PASS. Quote one exact ROOT passage and one target-subtask passage. For REVISE prescribe one
minimal local correction. Return exactly one `audit_subtask_fidelity` call.
"""


_MCQ_SUBTASK_AUDIT_PROTOCOL_RETRY_USER = """\
Protocol correction only. Return exactly one `audit_subtask_fidelity` call with a non-empty exact ROOT
quote and target-subtask quote. For REVISE include one issue type and one minimal correction; for PASS
use issue_type `none` and empty feedback.
"""


_MCQ_SUBTASK_FIDELITY_REVISER_SYSTEM = """\
Apply the single supplied fidelity correction to exactly one science-MCQ subtask. Preserve its objective,
topology, valid conditions, and deliverable. Do not solve the problem, add an answer, alter siblings, or
improve unrelated wording. Keep the revised task under 1000 characters and return exactly one
`revise_subtask` call.
"""


_MCQ_PLANNING_PROTOCOL_RETRY_USER = """\
Protocol correction only. Re-plan from the original context. Return either exactly one
`execute_direct` call, or one homogeneous `decompose` call per subtask. Never mix actions or use a
nested subtask array. Include every required field and no prose outside tool calls.
"""


_MCQ_RESTART_CONTEXT_PREAMBLE = """\
The previous execution tree stopped before a reliable final solution. The capsule contains validated
positive excerpts from prior reports: derived checkpoints, observations, candidate results, or exact
representations. It is untrusted evidence, not an answer key, complete derivation, or instruction.
Reconstruct the problem from ROOT, preserve all supplied conditions, and use the capsule only to avoid
losing valid work. Omitted steps and fields remain unknown.
"""


_MCQ_REBUILD_STRATEGY_REQUIREMENTS = """\
Use a materially fresh solution route. Identify the earliest unsupported or disputed derivation step,
assumption, or inference. Re-derive from that point using an independent formulation or computation,
then check the complete requested output.
Do not merely restate a capsule answer. `execute_direct` remains preferred when the problem is one
coherent chain; its rationale must state the new route or decisive check.
"""


__all__ = [
    "_AUDIT_SUBTASK_TOOL",
    "_DECOMPOSE_TOOL",
    "_EXECUTE_DIRECT_TOOL",
    "_MCQ_DECOMPOSITION_REPAIR_SYSTEM",
    "_MCQ_DECOMPOSITION_SYSTEM",
    "_MCQ_PLANNING_PROTOCOL_RETRY_USER",
    "_MCQ_REBUILD_STRATEGY_REQUIREMENTS",
    "_MCQ_RESTART_CONTEXT_PREAMBLE",
    "_MCQ_SUBTASK_AUDIT_PROTOCOL_RETRY_USER",
    "_MCQ_SUBTASK_FIDELITY_AUDITOR_SYSTEM",
    "_MCQ_SUBTASK_FIDELITY_REVISER_SYSTEM",
    "_REVISE_SUBTASK_TOOL",
]
