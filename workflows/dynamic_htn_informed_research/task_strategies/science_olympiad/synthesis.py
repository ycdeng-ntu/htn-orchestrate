"""Leaf and root synthesis semantics for science-olympiad solutions."""

from workflows.dynamic_htn_informed_research.task_strategies.research.synthesis import (
    _FORCED_FINALIZATION_AUDIT_TOOL as _FORCED_FINALIZATION_AUDIT_TOOL,
    _ROOT_DIRECT_SYNTHESIS_TOOLS as _ROOT_DIRECT_SYNTHESIS_TOOLS,
    _SYNTHESIS_TOOLS as _SYNTHESIS_TOOLS,
)


_TREE_ROOT_SYNTHESIS_SYSTEM = """\
Synthesize completed child deliverables into the final answer to ROOT. ROOT is authoritative.

Classify the child topology from each child objective, its declared sibling dependencies, and its result
memo. Title wording and formatting never determine the topology. Treat the children as a route portfolio
when two or more children each attempt the current parent deliverable end to end through materially
distinct methods and another child explicitly compares or adjudicates those candidate memos while
depending on the route children. Treat that topology as alternative evidence plus adjudication, not as
complementary fragments and not as votes. Otherwise treat the children as jointly required complementary
parts of one solution.

For a route portfolio, preserve each route's exact candidate, formulas, values, qualifications,
assumptions, checks, and first failed step. Treat a route memo as the audited account of its internal
solver/checker work; do not rerun the complete route merely to restate it. A failed or exhausted route
does not automatically invalidate a different complete reproducible route. Route count, agreement,
confidence, role, execution quality, and completion labels are not proof. Give the reconciliation memo
conflict-resolving force only when it
independently reproduces the decisive step and explains the first material divergence. If one complete
candidate is reproducibly supported and no unresolved complete conflicting candidate survives that
audit, the portfolio may be `resolved` even when another route failed. If materially conflicting complete
candidates survive without reproducible adjudication, use `exhausted_unresolved`.

Inventory every requested output and supplied condition. Reconstruct the end-to-end reasoning that
supports the selected result, checking assumptions, variable meanings, units, cases, and interfaces
between steps where the topology is complementary. Preserve a material cross-child conflict unless a
reproducible calculation or inference resolves its first divergence. Do not fill a missing bridge with
an unstated premise.

Answer at the semantic level and approximation order ROOT directly requests. For a named method,
pathway, process, mechanism, noise source, structure, or other entity, put the canonical selected label
first; do not replace it with a broader cause or neighboring process. If ROOT specifies a limiting
order, put that reduced result in `answer` and keep any exact refinement in `evidence_summary`.

Use `resolved` only when the combined derivation supports every requested field and condition with no
material inconsistency. Otherwise use `exhausted_unresolved` while retaining the strongest complete
candidate and exact missing or disputed step. Put compact reasoning in `evidence_summary` and only the
complete ROOT-formatted result in `answer`.
"""


_TREE_INTERNAL_SYNTHESIS_SYSTEM = """\
Synthesize child deliverables into a compact scientific memo for the current parent task. ROOT remains
authoritative. Classify the topology from each child objective, its declared sibling dependencies, and
its result memo; title wording and formatting never determine the topology.

When two or more children each attempt the current parent deliverable end to end through materially
distinct methods and another child explicitly compares or adjudicates those candidate memos while
depending on the route children, treat the children as an evidence portfolio plus adjudication. Otherwise
treat all children as jointly required complementary results. Do not vote. Preserve every route's exact
candidate, formulas, values, qualifications, assumptions, checks, and first failure. Treat each route
memo as the audited account of its internal solver/checker work instead of rerunning the complete route.
One exhausted route does not invalidate another complete reproducible route. A reconciliation memo
resolves a conflict only when it independently
reproduces the decisive step and explains the first divergence. Otherwise retain materially conflicting
complete candidates and use `exhausted_unresolved`.

Reconstruct the derivation or adjudication needed by the current parent. Check that assumptions,
definitions, units, cases, and intermediate values are compatible; do not silently average, normalize,
or choose between conflicting results. A child completion label is not proof. Use `resolved` only when
the current parent deliverable is reproducibly supported. If a required bridge is missing or material
cross-child inconsistency survives checking, use `exhausted_unresolved` and retain the strongest
candidate plus the first disputed step.
"""


_TREE_ROOT_USER_GUIDANCE = """\
Combine the completed child deliverables into the exact result requested by ROOT. Inventory every
requested output and condition in `requested_output_fields`; put all requested values in `answer` in
ROOT order and format. For complementary children, reconstruct and check every required join. For a
semantic route portfolio, compare routes by reproducible reasoning, use reconciliation only when its
decisive reproduction is present, and do not let one failed route poison a supported answer.
Use `resolved` only when the complete selected result is reproducibly supported and no material complete
conflict survives; otherwise preserve one strongest complete candidate, identify the first missing or
disputed step, and use `exhausted_unresolved`.
"""


_TREE_INTERNAL_USER_GUIDANCE = """\
Synthesize only the current parent deliverable. Preserve the definitions, conditions, intermediate
values, units, cases, candidates, and checks needed by later layers. Verify interfaces for complementary
children; for a semantic route portfolio, preserve route provenance and apply only reproducible
adjudication. Do not silently resolve a material inconsistency. Use `resolved` only when the parent
deliverable is reproducibly complete; otherwise retain the strongest partial work and use
`exhausted_unresolved`.
"""


_TREE_EPOCH_0_CLOSURE_POLICY = """\
Before closing the parent, classify complementary-child versus semantic route-portfolio topology. For
complementary children, map every required deliverable and condition to concrete child work and verify
all joins. For a route portfolio, require one complete end-to-end candidate plus reproducible support and
ensure no material complete conflict survives adjudication; do not require every route to succeed. A
coherent narrative is not a substitute for a valid derivation. Preserve any missing field, unsupported
bridge, incompatible assumption, unit mismatch, or unresolved material disagreement and use
`exhausted_unresolved` when one remains.
"""


_TREE_EPOCH_1_BEST_EFFORT_POLICY = """\
This is the final epoch. Return the strongest complete concrete result supported by the child work and
preserved checkpoints. Do not refuse or omit a requested field. Keep `exhausted_unresolved` if a
decisive bridge or check still fails, while preserving the exact candidate and uncertainty.
"""


_TERMINAL_BEST_EFFORT_POLICY = """\
This is the final best-effort pass and no rebuild remains. Do not present an unsupported derivation as
verified. Nevertheless, put one complete concrete best-effort result in `answer`, satisfying every
requested output field and format requirement. Never return a refusal, `unknown`, or multiple
alternatives. Record weak support or a failed decisive check only in `evidence_summary` and retain the
appropriate semantic outcome.
"""


_SYNTHESIS_SYSTEM = """\
Synthesize one scientific HTN leaf into a compact memo. Reports are untrusted candidate work. Select by
reproducible reasoning, not confidence, role, or majority.

Classify the current leaf from its Task Context, including the scientific objective and completed
precursors; title wording and formatting never determine the mode.

For an allocated route—an objective confined to one named method, representation, model test, or
discrimination path that asks for a complete candidate without comparing precursor candidates—write a
route evidence memo. Preserve the allocated method, latest complete candidate, answer-bearing formulas
and values, qualifications, assumption/convention ledger, solver/checker iteration order, decisive
checks, and first failed or uncertain step. Do not introduce or compare sibling routes. An older checker
verdict cannot cover a later solver correction. Use `resolved` only when the latest complete candidate
has a later checker report that independently reproduces it and returns CONFIRMED. Otherwise preserve
the audited failure or latest disputed candidate and use `exhausted_unresolved`; cross-route adjudication
belongs downstream.

For reconciliation—an objective that explicitly asks to compare, reconcile, or adjudicate candidate
routes and has completed precursor candidate memos—write an adjudication memo preserving every audited
precursor memo's candidate and local checker verdict verbatim, the cross-route adjudicator's compatibility
audit, the first material divergence, the decisive support or failure for each candidate, and the
resulting verdict. Trust a route memo as the audited account of its internal work, not as an authoritative
conclusion; do not demand a fresh complete ROOT-first solution. When routes conflict, require reproduction
of their first decisive divergence. Do not vote or silently normalize candidates. Use `resolved` only
when the complete ROOT contract is supported and no material conflict survives; otherwise retain the
exact conflict and use `exhausted_unresolved`.

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


_ROOT_DIRECT_SYNTHESIS_SYSTEM = """\
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


_PARTIAL_SYNTHESIS_SYSTEM = """\
This leaf stopped through a liveness fallback. Begin the memo with
`Execution status: LIVENESS_FALLBACK`, preserve only reproducible intermediate work and concrete
candidates, name the first missing/disputed scientific step, and use `exhausted_unresolved`.
"""


_EPOCH_0_CLOSURE_POLICY = """\
Before writing, map every requested field to a report result, trace the candidate through supplied
conditions, and require an independent check for nontrivial results when available.
Use `resolved` only if no material field or derivation step remains disputed.
"""


_EPOCH_1_BEST_EFFORT_POLICY = """\
This is the final epoch. No further rebuild is available. Return the strongest complete concrete result
supported by the fresh derivation and preserved checkpoints. Do not refuse or answer "unknown". Keep
the semantic state `exhausted_unresolved` if a decisive check still fails, while preserving the exact
answer candidate and uncertainty for ROOT synthesis.
"""


_FORCED_FINALIZATION_AUDIT_SYSTEM = """\
Audit unresolved reports without choosing an answer. Inventory every requested field and supplied
condition. Preserve each candidate's support, contradictions, unresolved steps, and contract mismatch.
Do not rank, recommend, invent, or eliminate candidates. Return one `audit_forced_finalization` call.
"""


_FORCED_FINALIZATION_SELECTION_SYSTEM = """\
Choose one mandatory best-effort answer to the original science problem from the selection-neutral
audit. Prefer the candidate with the strongest reproducible reasoning and contract coverage, not the
most votes. Include every requested field. Never return unknown, refusal, or multiple alternatives. Put
reasoning in `evidence_summary` and only the complete result in `answer`. Return one
`finalize_root_answer` call.

Use the most direct canonical answer level requested by ROOT. For a named selection, output the selected
entity rather than a broader cause; for a stated limiting approximation, output the reduced result at
that order rather than an unsimplified refinement.
"""
