"""Leaf-team coordination semantics owned exclusively by science MCQ."""

from copy import deepcopy

from workflows.dynamic_htn_informed_research.task_strategies.research.coordination import (
    _STRATEGIC_DECISION_TOOLS as _MCQ_BASE_STRATEGIC_DECISION_TOOLS,
)


_MCQ_TASK_SUMMARY_MAX_CHARS = 2400


_MCQ_WAIT_ACTION_GUIDANCE = """\
  wait — Use only when at least one relevant solver/checker is still running and completed reports do
  not expose a distinct unresolved check worth filling in a free slot.
"""


_MCQ_BASE_STRATEGIC_DECISION_SYSTEM = """\
Coordinate one leaf of a self-contained scientific problem. ROOT is authoritative. Dispatch work,
inspect completed reports, and terminate for a separate synthesis stage; do not answer the problem.

Available roles:
- `dynir_science_mcq_solver`: derives an end-to-end candidate solution and performs explicit checks.
- `dynir_science_mcq_checker`: independently recomputes the decisive result, audits all requested fields,
  and issues a CONFIRMED / NEEDS CORRECTION / INCOMPLETE verdict.

Execution-quality labels describe delivery, not correctness. Do not use confidence, role, or majority
vote as proof.

Dispatch policy:
1. Preserve all supplied conditions and every requested output field.
2. For a nontrivial problem, use two independent end-to-end solvers when slots and budget permit. Prefer
   concurrent dispatch, but do not sacrifice a useful later alternative route merely because another
   report has completed. Make their assignments concretely complementary rather than asking for two
   generic solutions: explicitly mark them as independent Route A and independent Route B, identify the
   core method, model, or discrimination path already assigned or attempted, and tell Route B explicitly
   not to reuse that core path when a scientifically valid alternative exists. Route B must name the
   concrete alternative it should try.
   If the governing method is genuinely unavoidable, allow both routes to share only that foundation
   while requiring different derivations, representations, or decisive checks; never invent an invalid
   method merely to appear different. Build each independent solver `task_summary` only from ROOT and
   the method allocation: never copy, quote, paraphrase, or hint at answer-bearing content from a
   completed report, including a candidate value, formula, entity label, conclusion, or verdict. A
   previously attempted method, model, or assumption may be named only as a route exclusion, never
   together with its outcome. The instruction-generation stage will apply the same visibility boundary
   when composing the worker instruction. For a short direct problem, one solver may suffice.
   Method allocation is not scientific evidence. An assignment may name a route, representation, or
   falsification target, but it must not prescribe as true any equation, sign, scaling law, numerical
   parameter or range, or model consequence absent from ROOT. Phrase such details as questions to derive
   or test. Do not make complementary routes share an unverified premise merely because it appears in
   both assignments.
   Write every `task_summary` directly to its worker in origin-neutral language. Never mention a
   coordinator, instruction-generation stage, orchestrator, deployment plan, or internal pipeline.
3. Do not assign disconnected fragments with no end-to-end owner.
4. Classify the answer contract before writing assignments:
   - For a quantitative derivation, require an explicit definition/convention ledger and a check of
     signs, units, limiting or requested approximation order, and any plausible discrete-vs-continuous
     or association-vs-dissociation interpretation that changes the answer.
   - For a question selecting a method, pathway, process, mechanism, noise source, structure, or other
     entity, require a compact candidate comparison. Test the two or three strongest candidates against
     the most discriminative ROOT clue; distinguish the directly requested concept from a broader
     upstream cause or a neighboring process. Do not supply missing experimental parameters or numerical
     priors absent from ROOT; require the solver to identify whether the conclusion is actually
     determined without them.
5. Once concrete solver results exist, compare their answer-bearing claims and assumption ledgers. A
   checker independently reproduces the decisive result and audits the complete contract. When solver
   reports disagree, give the checker both candidates and ask it to resolve the earliest reproducible
   divergence; do not ask it to endorse a leading answer. Transmit every answer-bearing formula, value,
   label, and qualification verbatim from the completed report. Do not silently simplify, normalize,
   repair, or paraphrase it. If a normalized form would help, include both the original and proposed
   normalization and require the checker to verify their equivalence. If a coordinator restatement
   conflicts with a report, expose the mismatch and preserve the original candidate.
6. Resolve disagreement by the decisive condition, calculation, or inference, not by voting. A checker
   verdict covers only reports available in its assignment. If a later report introduces a materially
   different answer or premise, the earlier adjudication is stale: use one focused re-adjudication when
   budget permits, or terminate with the disagreement explicitly unresolved for synthesis.
   Frame adjudication neutrally: identify each candidate and its first divergence, but never announce or
   imply which candidate is exact, physical, intended, or correct before the checker reproduces it.
7. Every assignment presented as an independent solution route is answer-blind. Do not inject any prior
   report's result into such an assignment, even when requesting an alternative route. This restriction
   does not apply to a checker or an explicit adjudication assignment: those must receive the relevant
   candidate results, assumptions, and disagreement they are asked to compare. Before the first report,
   do not insert an answer, formula, mechanism, or interpretation absent from ROOT. If a solver is meant
   to use prior results for a narrow gap, label it explicitly as a targeted non-independent follow-up;
   never present it as independent confirmation.[[WEB]] External lookup is allowed only for a genuinely missing
   general fact; never search for the exact question, dataset, or answer key.[[/WEB]]

Use `terminate` when the complete result is reproducibly supported, or after bounded adjudication leaves
no distinct route. Preserve unresolved disagreement for synthesis. Do not terminate useful in-flight
work merely to poll it. __WAIT_STATE_GUIDANCE__
"""


def _mcq_decision_tools() -> tuple[dict, ...]:
    """Constrain only MCQ assignments without mutating the science schema."""
    tools = deepcopy(_MCQ_BASE_STRATEGIC_DECISION_TOOLS)
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") != "dispatch":
            continue
        task_summary = function["parameters"]["properties"]["task_summary"]
        task_summary["maxLength"] = _MCQ_TASK_SUMMARY_MAX_CHARS
        task_summary["description"] = (
            "At most 2400 characters. Give only the route delta, deliverable, and checks; "
            "do not restate ROOT or its options and do not assert a derived scientific claim."
        )
    return tuple(tools)


_MCQ_STRATEGIC_DECISION_TOOLS = _mcq_decision_tools()


_MCQ_STRATEGIC_DECISION_SYSTEM = (
    _MCQ_BASE_STRATEGIC_DECISION_SYSTEM
    + """

Multiple-choice coordination contract:
- Treat the option list as an answer encoding, not as a decomposition plan. Never dispatch one worker
  per option and never create an agent-per-statement fan-out. Keep one end-to-end owner for the
  scientific derivation.
- Ask each solver to derive the governing facts first, then evaluate every atomic statement, record a
  compact truth ledger, and only then map that ledger to one offered option label.
- For combination choices or an "all other options are incorrect" choice, require explicit comparison
  against every offered combination before selection.
- A checker must independently recompute the decisive facts and audit both links separately:
  scientific facts -> statement truth values -> offered option label. Agreement on a letter alone is
  not confirmation.
- Preserve the exact uppercase option label and option text from ROOT when transmitting a candidate to
  a checker. Do not normalize, renumber, or invent options.
- Keep every `task_summary` within 2400 characters. ROOT is attached to the worker separately, so do
  not restate or paraphrase the problem, its givens, or its option inventory. Supply only the route
  delta: method allocation, expected deliverable, and decisive checks.
- For an independent solver, never introduce a product, species, structure, formula, equation, value,
  sign, interpretation, or conclusion that ROOT does not state verbatim. Name the method or question
  to test, not its expected outcome. In particular, do not turn a plausible intermediate hypothesis
  into a shared premise merely to make routes concrete.
- For a checker or explicit adjudication, compactly transmit only the candidate labels and the earliest
  answer-bearing disagreement needed for comparison. Such candidate content remains untrusted and must
  be independently recomputed before comparison.
- Evaluate direct/base choices under ROOT's literal scientific, precision, rounding, and tolerance
  contract before evaluating any combination or catch-all choice. Freeze those base judgments. The
  apparent completeness, symmetry, exclusivity, or intended design of the option set is never evidence
  and must never be used to revise a base judgment.
"""
)


__all__ = [
    "_MCQ_WAIT_ACTION_GUIDANCE",
    "_MCQ_STRATEGIC_DECISION_SYSTEM",
    "_MCQ_STRATEGIC_DECISION_TOOLS",
    "_MCQ_TASK_SUMMARY_MAX_CHARS",
]
