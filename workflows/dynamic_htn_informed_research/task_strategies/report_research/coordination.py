"""Leaf-team coordination policy for report research."""

from __future__ import annotations

from typing import Any


_WAIT_ACTION_GUIDANCE = """\
Use wait only when relevant report research remains in flight and completed memos expose no distinct
uncovered requirement that can use the free slot.
"""

_STRATEGIC_DECISION_SYSTEM = """\
Coordinate one report-research milestone. The public ROOT question is authoritative. Start from the
leaf's requested result and completion criteria. Give researchers coherent result-producing work
with derivation, calculation and self-checking together; do not automatically split these into
serial micro-assignments. Dispatch an independent verifier once candidate evidence is available.
Never access a private rubric, reference answer, judge feedback, benchmark dataset, or answer key.
Keep dispatches within this leaf's research contract; other leaves own other scopes, and the runtime
delivery leaf owns final writing and whole-report verification. Do not dispatch a final-report writer,
create new HTN nodes, or initiate a separate report workflow.

The researcher must return exact claims, [[WEB]]source URLs, verbatim excerpts, [[/WEB]]reproducible reasoning and calculations,
conflicts, and an explicit coverage ledger. Cover each assigned scientific sub-question with relevant
equations, named methods, assumptions, units and intermediate values. Treat generic textbook prose as
a gap, but do not invent additional deliverables merely to make the memo more elaborate.

Dispatch independent work without duplicating in-flight assignments. For FIRST verification of a
result, explicitly reopen load-bearing identities, organism/system choices, methods and assumptions
from precursor memos. Require a question-based check of the model itself, not just confirmation of a
predecessor's answer or arithmetic under an inherited model. For FOLLOW-UP verification after repair,
identify the earlier report/check, the changed claims and affected dependent conclusions in the
assignment. Check that change and its scientific impact. Reopen an already checked, unchanged claim
only for a concrete new counterexample, evidence, faulty earlier check or changed dependency; state
the reason. New or previously unchecked decisive claims still require independent verification.
Verification must use [[WEB]]fresh queries and [[/WEB]]independent reasoning or computation and distinguish unsupported from contradicted.
For an upstream error, correct affected conclusions within this leaf and identify affected external
results for delivery and Planner. Do not claim to have modified another leaf's results.

After each report, distinguish three cases:
- A correction is already supported by the available evidence (for example a corrected label,
  qualification, or independently computed value): leave it for synthesis to integrate consistently.
  If no other decisive gap needs research or verification, terminate. Do not launch another
  researcher/verifier pair just to restate that correction. An untested proposed fix is not a
  supported correction; disputed or newly introduced scientific claims still need checking.
- A decisive requirement needs a feasible new derivation, calculation or evidence check: dispatch
  that specific work. In task_summary and rationale identify the gap, the new check or distinct
  route, and which requested result it could change. Preserve supported results outside that scope.
- Available capabilities and evidence cannot resolve the decisive gap, and no distinct feasible
  route remains: terminate with the missing evidence, competing supported conclusions and their
  impact preserved. Repeated opinions or another wording of the same investigation are not a new
  route. Offline retrieval limitations alone do not invalidate a justified derivation.

Terminate when the milestone's decisive requirements are supported (including supported corrections
ready for synthesis), or when distinct feasible routes are exhausted. Preserve warranted conditional
conclusions; leave unmet decisive requirements explicitly unresolved for delivery and possible rebuild.
Do not claim that reaching the budget, report count or agreement establishes correctness.
__WAIT_STATE_GUIDANCE__
"""

_STRATEGIC_DECISION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Finish coordination and pass completed memos to synthesis.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for relevant in-flight report work.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch",
            "description": "Dispatch one bounded report-research assignment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role_id": {"type": "string"},
                    "task_summary": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["role_id", "task_summary", "rationale"],
                "additionalProperties": False,
            },
        },
    },
)


__all__ = [
    "_STRATEGIC_DECISION_SYSTEM",
    "_STRATEGIC_DECISION_TOOLS",
    "_WAIT_ACTION_GUIDANCE",
]
