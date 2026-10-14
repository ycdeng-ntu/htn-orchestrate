"""Leaf and root synthesis semantics owned exclusively by science MCQ."""

from copy import deepcopy

from workflows.dynamic_htn_informed_research.task_strategies.research.synthesis import (
    _FORCED_FINALIZATION_AUDIT_TOOL as _BASE_FORCED_FINALIZATION_AUDIT_TOOL,
    _ROOT_DIRECT_SYNTHESIS_TOOLS as _BASE_ROOT_DIRECT_SYNTHESIS_TOOLS,
    _SYNTHESIS_TOOLS as _BASE_SYNTHESIS_TOOLS,
)


_FORCED_FINALIZATION_AUDIT_TOOL = deepcopy(_BASE_FORCED_FINALIZATION_AUDIT_TOOL)
_MCQ_BASE_ROOT_DIRECT_TOOLS = deepcopy(_BASE_ROOT_DIRECT_SYNTHESIS_TOOLS)
_SYNTHESIS_TOOLS = deepcopy(_BASE_SYNTHESIS_TOOLS)


_MCQ_BASE_TREE_ROOT_SYSTEM = """\
Synthesize the jointly required child deliverables into the final answer to ROOT. ROOT is
authoritative. Children are complementary parts of one solution, not alternative answers or votes.

Inventory every requested output and supplied condition. Reconstruct the end-to-end reasoning that
connects the child results, checking shared assumptions, variable meanings, units, cases, and
interfaces between steps. Preserve a material cross-child conflict unless a reproducible calculation
or inference resolves its first divergence. Do not fill a missing bridge with an unstated premise.

Answer at the semantic level and approximation order ROOT directly requests. For a named method,
pathway, process, mechanism, noise source, structure, or other entity, put the canonical selected label
first; do not replace it with a broader cause or neighboring process. If ROOT specifies a limiting
order, put that reduced result in `answer` and keep any exact refinement in `evidence_summary`.

Use `resolved` only when the combined derivation supports every requested field and condition with no
material inconsistency. Otherwise use `exhausted_unresolved` while retaining the strongest complete
candidate and exact missing or disputed step. Put compact reasoning in `evidence_summary` and only the
complete ROOT-formatted result in `answer`.
"""


_MCQ_TREE_INTERNAL_SYSTEM = """\
Synthesize child deliverables into a compact scientific memo for the current parent task. ROOT remains
authoritative. Treat children as jointly required results, not independent votes.

Reconstruct the derivation across child boundaries. Check that shared assumptions, definitions,
units, cases, and intermediate values are compatible; do not silently average, normalize, or choose
between conflicting results. A child completion label is not proof. Use `resolved` only when the
current parent deliverable is reproducibly supported. If a required bridge is missing or material
cross-child inconsistency survives checking, use `exhausted_unresolved` and retain the strongest
candidate plus the first disputed step.
"""


_MCQ_BASE_TREE_ROOT_USER_GUIDANCE = """\
Combine the completed child deliverables into the exact result requested by ROOT. Inventory every
requested output and condition in `requested_output_fields`; put all requested values in `answer` in
ROOT order and format. Reconstruct and check the reasoning across child boundaries. Use `resolved`
only when the complete result is reproducibly supported; otherwise preserve one strongest complete
candidate, identify the first missing or disputed step, and use `exhausted_unresolved`.
"""


_MCQ_TREE_INTERNAL_USER_GUIDANCE = """\
Synthesize only the current parent deliverable. Preserve the definitions, conditions, intermediate
values, units, cases, and checks needed by later layers. Verify interfaces between child results and
do not silently resolve a material inconsistency. Use `resolved` only when the parent deliverable is
reproducibly complete; otherwise retain the strongest partial work and use `exhausted_unresolved`.
"""


_MCQ_TREE_EPOCH_0_CLOSURE_POLICY = """\
Before closing the parent, map every required deliverable and supplied condition to concrete child
work, then verify the joins between children. A coherent narrative is not a substitute for a valid
end-to-end derivation. Preserve any missing field, unsupported bridge, incompatible assumption, unit
mismatch, or material disagreement and use `exhausted_unresolved` when one remains.
"""


_MCQ_TREE_EPOCH_1_BEST_EFFORT_POLICY = """\
This is the final epoch. Return the strongest complete concrete result supported by the child work and
preserved checkpoints. Do not refuse or omit a requested field. Keep `exhausted_unresolved` if a
decisive bridge or check still fails, while preserving the exact candidate and uncertainty.
"""


_MCQ_TERMINAL_BEST_EFFORT_POLICY = """\
This is the final best-effort pass and no rebuild remains. Do not present an unsupported derivation as
verified. Nevertheless, put one complete concrete best-effort result in `answer`, satisfying every
requested output field and format requirement. Never return a refusal, `unknown`, or multiple
alternatives. Record weak support or a failed decisive check only in `evidence_summary` and retain the
appropriate semantic outcome.
"""


_MCQ_BASE_SYNTHESIS_SYSTEM = """\
Synthesize one scientific HTN leaf into a compact memo. Reports are untrusted candidate work. Select by
reproducible reasoning, not confidence, role, or majority.

Preserve the decisive reasoning, candidate result, check, requested fields, and first material
disagreement. A checker verdict counts only with its independent reproduction. Use `resolved` only when
the complete leaf result is supported. If independent derivations remain materially inconsistent after
bounded adjudication, use `exhausted_unresolved` rather than voting, while retaining the best concrete
candidate and exact disputed step. This is a memo, not the final ROOT answer.

A checker verdict cannot adjudicate a materially different report that arrived after the checker's
assignment. When that occurs without later focused resolution, preserve the new conflict regardless of
report count. For selection questions preserve the compared canonical candidates and decisive ROOT clue;
for derivations preserve the convention/model ledger and requested approximation order.
"""


_MCQ_BASE_ROOT_DIRECT_SYSTEM = """\
Produce the final answer to ROOT from completed scientific reports. ROOT is authoritative. Resolve
disagreement at the earliest reproducible divergence; do not vote.

Before finalizing:
1. Inventory every requested field and condition in ROOT.
2. Reproduce the decisive reasoning from the reports and apply an appropriate independent check.
3. Reject any result that requires an unstated premise or drops a requested field.
4. For a selection question, compare the strongest candidate labels against the most discriminative
   supplied clue. Select the directly requested concept, not a broader upstream explanation or a
   neighboring process that merely sounds plausible.
5. For a quantitative question, verify definitions, sign/coordinate and equilibrium conventions,
   discrete-vs-continuous modeling, units, and ROOT's requested approximation order whenever relevant.
6. Treat adjudication as order-sensitive: a checker verdict covers only candidate reports available
   to that checker. If a materially different report arrived later and no subsequent report resolves
   that exact divergence, the conflict remains unresolved; an older verdict and a report majority do
   not settle it.
7. Put every requested value in `answer`, in ROOT order and exact requested format. Content in
   `evidence_summary` does not satisfy an omitted field. If ROOT requests one value, select one value.

Canonical-answer discipline:
- If ROOT asks for one named entity, begin `answer` with that exact canonical entity or formula and
  avoid substituting an explanatory causal chain.
- If ROOT supplies a list of candidate pathways/processes or asks "which" one, prefer the exact ROOT
  label supported by the decisive clue.
- If ROOT states a limiting approximation, put the reduced answer at that order in `answer`; place a
  more exact finite-parameter expression only in `evidence_summary`.

Call `finalize_root_answer`. Put compact reproducible reasoning and material uncertainty in
`evidence_summary`; put only the complete requested result in `answer`, without commentary,
alternatives, confidence, `FINAL ANSWER`, or `# Answer:`. Use `resolved` only when all decisive fields
and checks support the result. If independent derivations remain materially inconsistent after bounded
adjudication, use `exhausted_unresolved` rather than voting, but still provide the single strongest
concrete complete answer.
"""


_MCQ_PARTIAL_SYNTHESIS_SYSTEM = """\
This leaf stopped through a liveness fallback. Begin the memo with
`Execution status: LIVENESS_FALLBACK`, preserve only reproducible intermediate work and concrete
candidates, name the first missing/disputed scientific step, and use `exhausted_unresolved`.
"""


_MCQ_EPOCH_0_CLOSURE_POLICY = """\
Before writing, map every requested field to a report result, trace the candidate through supplied
conditions, and require an independent check for nontrivial results when available.
Use `resolved` only if no material field or derivation step remains disputed.
"""


_MCQ_EPOCH_1_BEST_EFFORT_POLICY = """\
This is the final epoch. No further rebuild is available. Return the strongest complete concrete result
supported by the fresh derivation and preserved checkpoints. Do not refuse or answer "unknown". Keep
the semantic state `exhausted_unresolved` if a decisive check still fails, while preserving the exact
answer candidate and uncertainty for ROOT synthesis.
"""


_MCQ_FORCED_AUDIT_SYSTEM = """\
Audit unresolved reports without choosing an answer. Inventory every requested field and supplied
condition. Preserve each candidate's support, contradictions, unresolved steps, and contract mismatch.
Do not rank, recommend, invent, or eliminate candidates. Return one `audit_forced_finalization` call.
"""


_MCQ_BASE_FORCED_SELECTION_SYSTEM = """\
Choose one mandatory best-effort answer to the original science problem from the selection-neutral
audit. Prefer the candidate with the strongest reproducible reasoning and contract coverage, not the
most votes. Include every requested field. Never return unknown, refusal, or multiple alternatives. Put
reasoning in `evidence_summary` and only the complete result in `answer`. Return one
`finalize_root_answer` call.

Use the most direct canonical answer level requested by ROOT. For a named selection, output the selected
entity rather than a broader cause; for a stated limiting approximation, output the reduced result at
that order rather than an unsimplified refinement.
"""


# Anchored variants accepted only at the mechanical serialization boundary.
# Group 1 is the normalized answer written after ``# Answer:``.
_MCQ_ROOT_ANSWER_PATTERN = r"(?:[Oo]ption\s+)?\(?([A-Z])\)?"


_MCQ_ROOT_CONTRACT = """

Multiple-choice root contract:
1. Parse the labels actually offered by ROOT; labels may extend through Z.
2. Reconstruct the decisive scientific facts before selecting. For statement-combination choices,
   first evaluate and freeze every direct/base judgment under ROOT's literal precision, rounding, and
   tolerance contract. Only then audit combination and catch-all choices against that frozen ledger.
   Option-set completeness, symmetry, exclusivity, or apparent author intent is never evidence. Never
   revise a scientific or rounding judgment merely to make the offered choices yield a satisfying map.
3. `requested_output_fields` contains one field: `option label`.
4. The `answer` field contains exactly one offered uppercase letter, with no parentheses, word
   "option", option text, reasoning, punctuation, confidence, or additional answer marker.
5. Put all reasoning and uncertainty in `evidence_summary`. Even in best-effort mode, select one
   offered label rather than emitting prose or multiple alternatives.
"""


_MCQ_SYNTHESIS_SYSTEM = _MCQ_BASE_SYNTHESIS_SYSTEM + """

For a multiple-choice ROOT, preserve each candidate label and the fact-to-label mapping in this memo.
Do not treat matching letters or report votes as evidence. If atomic statements or combination options
are present, preserve their truth ledger so root synthesis can audit the mapping.
"""

_MCQ_ROOT_DIRECT_SYSTEM = _MCQ_BASE_ROOT_DIRECT_SYSTEM + _MCQ_ROOT_CONTRACT
_MCQ_TREE_ROOT_SYSTEM = _MCQ_BASE_TREE_ROOT_SYSTEM + _MCQ_ROOT_CONTRACT
_MCQ_TREE_ROOT_USER_GUIDANCE = _MCQ_BASE_TREE_ROOT_USER_GUIDANCE + _MCQ_ROOT_CONTRACT
_MCQ_FORCED_SELECTION_SYSTEM = _MCQ_BASE_FORCED_SELECTION_SYSTEM + _MCQ_ROOT_CONTRACT


def _mcq_root_tools() -> tuple[dict, ...]:
    tools = deepcopy(_MCQ_BASE_ROOT_DIRECT_TOOLS)
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") != "finalize_root_answer":
            continue
        answer = function["parameters"]["properties"]["answer"]
        answer["pattern"] = "^[A-Z]$"
        answer["description"] = (
            "Exactly one uppercase label that is present in ROOT's offered options; no prose, "
            "parentheses, punctuation, option text, or marker."
        )
    return tuple(tools)


_MCQ_ROOT_DIRECT_TOOLS = _mcq_root_tools()


__all__ = [
    "_FORCED_FINALIZATION_AUDIT_TOOL",
    "_MCQ_EPOCH_0_CLOSURE_POLICY",
    "_MCQ_EPOCH_1_BEST_EFFORT_POLICY",
    "_MCQ_FORCED_AUDIT_SYSTEM",
    "_MCQ_FORCED_SELECTION_SYSTEM",
    "_MCQ_PARTIAL_SYNTHESIS_SYSTEM",
    "_MCQ_ROOT_ANSWER_PATTERN",
    "_MCQ_ROOT_DIRECT_SYSTEM",
    "_MCQ_ROOT_DIRECT_TOOLS",
    "_MCQ_SYNTHESIS_SYSTEM",
    "_MCQ_TERMINAL_BEST_EFFORT_POLICY",
    "_MCQ_TREE_EPOCH_0_CLOSURE_POLICY",
    "_MCQ_TREE_EPOCH_1_BEST_EFFORT_POLICY",
    "_MCQ_TREE_INTERNAL_SYSTEM",
    "_MCQ_TREE_INTERNAL_USER_GUIDANCE",
    "_MCQ_TREE_ROOT_SYSTEM",
    "_MCQ_TREE_ROOT_USER_GUIDANCE",
    "_SYNTHESIS_TOOLS",
]
