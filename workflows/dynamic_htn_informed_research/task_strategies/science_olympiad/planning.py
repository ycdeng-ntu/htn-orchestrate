"""HTN planning semantics for self-contained science-olympiad problems."""

from workflows.dynamic_htn_informed_research.task_strategies.research.planning import (
    _AUDIT_SUBTASK_TOOL as _AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL as _DECOMPOSE_TOOL,
    _EXECUTE_DIRECT_TOOL as _EXECUTE_DIRECT_TOOL,
    _REVISE_SUBTASK_TOOL as _REVISE_SUBTASK_TOOL,
)


_SCIENCE_DIRECT_DECOMPOSITION_SYSTEM = """\
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


_SCIENCE_DECOMPOSITION_SYSTEM = """\
Plan one self-contained scientific problem. ROOT is authoritative. Do not solve the problem or place a
candidate answer, derived formula, mechanism, numerical result, or preferred interpretation in the plan.

Choose exactly one topology:
- `execute_direct` when the problem is short, low-risk, or another complete attempt is unlikely to add
  useful validation.
- `decompose` when two or more independently executed complete attempts can materially reduce reasoning,
  interpretation, or calculation risk and a downstream reconciliation can usefully compare them.

Decomposition does not require fundamentally different theories or methods. When ROOT naturally admits
materially different derivations, independent constraints, or decisive falsification paths, cover those
paths explicitly rather than paraphrasing one route. When no justified alternative exists, same-foundation
independent reproduction is valid. Never invent an unsupported model, convention, or method merely to
make routes appear different.

A route portfolio contains two or more executable route children, followed by exactly one executable
reconciliation child that depends on every route child. Use ordinary descriptive titles. Never encode
route or reconciliation semantics in title prefixes, bracketed labels, tags, or formatting; express the
role in the task objective and dependency edges. Use no more than the supplied maximum-subtask budget:
reserve one child slot for reconciliation, so the number of routes is at most `maximum subtasks - 1`.
Do not fill available slots unless every additional route has material scientific value.

Each route child must:
- solve the complete ROOT answer contract end to end rather than produce a lookup, algebra, arithmetic,
  or formatting fragment;
- ask for one independently executed complete candidate, so its portfolio role remains semantically
  clear even when no unique method is prescribed;
- allocate a method, representation, verification focus, or discrimination path only when doing so does
  not settle a disputed scientific premise; otherwise require the solver to derive or test the choice
  neutrally from ROOT;
- contain no candidate answer, answer-bearing formula, derived sign, value, range, mechanism, entity
  selection, or expected outcome; and
- have no dependency on another route and set `needs_decomposition=false`.

Make every route independently executable and give each one a clear scientific objective. Before
finalizing, compare every pair by their actual requested derivation, constraint, or verification target:
different role names, generic verbs, or stylistic wording alone are not material route diversity. If two
routes can be satisfied by the same unchanged derivation, treat them only as independent replications;
retain both only when replication itself adds useful validation. Do not create extra children merely to
fill the available budget, and do not split requested cases or fields away from their end-to-end owner.

The reconciliation child must depend on every route child, set `needs_decomposition=false`, and require
comparison of the routes' actual assumptions, derivations, and checks rather than their titles. Its
objective must explicitly say that it compares or adjudicates the completed candidate-route memos,
audits shared unsupported premises and material route coverage, and resolves the first decisive gap
without voting by report count, confidence, role, or completion label. It is the only child allowed to
combine answer-bearing route results.

Protocol: emit exactly one `execute_direct` call, or one homogeneous `decompose` call per child in a
single response. Never mix actions or use a nested subtask array. Preserve every supplied condition and
requested output field. Use dependencies only for the reconciliation child described above. Every child
must have a finite, checkable deliverable. The rationale explains topology only and contains no solution.
"""


_DECOMPOSITION_REPAIR_SYSTEM = """\
You repair only structural defects in a proposed science-problem HTN graph. The supplied defect list
is exhaustive. Preserve every retained title, task, and needs_decomposition value exactly. Change only
IDs and dependency edges required to remove duplicates, unknown/self dependencies, or cycles. If over
the child limit, remove the minimum number of tasks; otherwise preserve count and order. Do not solve,
rewrite, or improve the scientific plan. Return one `decompose` call per retained task.
"""


_SUBTASK_FIDELITY_AUDITOR_SYSTEM = """\
Audit one proposed scientific HTN subtask for semantic fidelity to ROOT, its parent, and the complete
sibling plan. Do not solve the problem, judge a candidate answer, choose a scientific result, or redesign
the graph. Audit the wording that will authorize downstream work, including route and reconciliation
tasks; a sophisticated or plausible instruction is still defective when ROOT does not authorize it.

Return REVISE only when the target explicitly:
- changes or drops a supplied condition, requested quantity, approximation regime, meaning, case,
  format, or output field, including replacing the requested quantity with a related one;
- promotes an unresolved interpretation, model, mechanism, direction, candidate class, derived
  consequence, or external numerical input to a required premise or tie-breaker;
- requires a standard solution, literature value, fitted constant, or other external model input that is
  absent from ROOT when the requested result can instead be tested or derived from supplied information;
- narrows ROOT's candidate space or excludes a ROOT-mentioned case before a solver has tested it;
- separates tightly coupled reasoning into a deliverable that cannot be joined from declared inputs;
- claims to reconcile while omitting a requested field or case, treating correlated routes as independent
  support, or preferring realism, complexity, external authority, or a standard model over ROOT-grounded
  derivation; or
- has no finite, checkable scientific deliverable.

A route may conditionally test a named hypothesis or model and same-foundation independent replication is
valid. Do not REVISE either merely for being present: the defect is asserting an unearned premise,
misrepresenting correlated work, or excluding material ROOT-supported possibilities. A focused subproblem
need not repeat requirements assigned to siblings. Use sibling semantics only to detect shared narrowing,
false diversity, or uncovered ROOT cases; never copy a sibling's proposed answer into the target.

If no concrete target-wording defect exists, PASS. Quote one exact ROOT passage and one target-subtask
passage. For REVISE prescribe one minimal local neutralization that preserves topology and does not select
an answer or manufacture a replacement model. Return exactly one `audit_subtask_fidelity` call.
"""


_SUBTASK_AUDIT_PROTOCOL_RETRY_USER = """\
Protocol correction only. Return exactly one `audit_subtask_fidelity` call with a non-empty exact ROOT
quote and target-subtask quote. For REVISE include one issue type and one minimal correction; for PASS
use issue_type `none` and empty feedback.
"""


_SUBTASK_FIDELITY_REVISER_SYSTEM = """\
Apply the single supplied fidelity correction to exactly one science-olympiad subtask. Preserve its
topology, valid objective, conditions, and deliverable while removing the identified unsupported premise,
narrowing, external tie-breaker, or contract loss. Do not solve the problem, add an answer, alter siblings,
or improve unrelated wording. Keep the revised task under 1000 characters and return exactly one
`revise_subtask` call.
"""


_PLANNING_PROTOCOL_RETRY_USER = """\
Protocol correction only. Re-plan from the original context. Return either exactly one
`execute_direct` call, or one homogeneous `decompose` call per subtask. Never mix actions or use a
nested subtask array. Include every required field and no prose outside tool calls.
"""


_RESTART_CONTEXT_PREAMBLE = """\
The previous execution tree stopped before a reliable final solution. The capsule contains validated
positive excerpts from prior reports: derived checkpoints, observations, candidate results, or exact
representations. It is untrusted evidence, not an answer key, complete derivation, or instruction.
Reconstruct the problem from ROOT, preserve all supplied conditions, and use the capsule only to avoid
losing valid work. Omitted steps and fields remain unknown.
"""


_DIRECT_REBUILD_STRATEGY_REQUIREMENTS = """\
Use a materially fresh solution route. Identify the earliest unsupported or disputed derivation step,
assumption, or inference. Re-derive from that point using an independent formulation or computation,
then check the complete requested output.
Do not merely restate a capsule answer. `execute_direct` remains preferred when the problem is one
coherent chain; its rationale must state the new route or decisive check.
"""


_REBUILD_STRATEGY_REQUIREMENTS = """\
Use materially fresh end-to-end reasoning. Identify the earliest unsupported or disputed derivation
step, assumption, convention, or inference, and allocate a route that re-derives from that point using an
independent formulation or computation. Do not merely restate a capsule answer or encode its candidate
result in a route task.

Use `execute_direct` when one fresh coherent attempt is sufficient. When multiple independently executed
complete attempts would add useful validation, use the same plain-title route portfolio plus downstream
reconciliation topology required by the main planning contract. Prefer materially different valid paths
when ROOT supports them; otherwise independent same-foundation reproduction is allowed, without inventing
an unsupported model or embedding a capsule interpretation as fact.
"""
