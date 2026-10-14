"""Evidence-memo and report synthesis contracts."""

from __future__ import annotations

from typing import Any


_SYNTHESIS_SYSTEM = """\
Synthesize one research leaf into an evidence-rich memo for the runtime delivery leaf. Keep the
assigned research scope; do not write the complete report or claim whole-question verification. Inventory the leaf's coverage obligations and preserve exact claims, [[WEB]]source URLs,
verbatim support excerpts, [[/WEB]]reasoning, figures, units, table cells, equations, named methods, assumptions, conflicts, and
unresolved gaps. Do not turn report count into corroboration or fabricate evidence. Use
completion_outcome=resolved only when every decisive leaf requirement is supported. Incorporate
supported local verifier corrections consistently and label superseded values. A researcher need
not restate a correction already established by an independent check: integrate it and update all
affected conclusions within this leaf. Distinguish these supported corrections from untested proposed
fixes; do not silently invent a missing derivation or claim another verification occurred. Preserve
which checks were performed and which were reused. Termination alone does not establish closure;
keep completion_outcome=exhausted_unresolved if any decisive requirement remains unsupported.
For unresolved
conflicts, preserve the competing derivations and the exact missing check, not just a disagreement
count. Report cross-leaf implications without claiming to have updated other leaves.
"""

_ROOT_DIRECT_SYNTHESIS_SYSTEM = """\
Write the complete long-form Markdown report requested by ROOT from the supplied evidence memos.
Mirror every requested section and sub-question in order. Preserve exact evidence and provenance;
never fabricate a [[WEB]]citation, [[/WEB]]claim, number, equation, or method. Prioritize complete
sub-question answers, explicit equations with defined symbols, named methods, assumptions and
limits, intermediate values with units, derivations, and scientific specificity rather than generic
textbook prose. Record material uncertainty without deleting usable evidence. When an upstream premise is refuted,
revise all dependent conclusions; do not preserve a wrong result merely because a memo contains it. Put the full report in
the answer field; it is a report, not a concise short-answer value.
"""

_TREE_ROOT_SYNTHESIS_SYSTEM = _ROOT_DIRECT_SYNTHESIS_SYSTEM

_TREE_INTERNAL_SYNTHESIS_SYSTEM = """\
Combine child evidence memos for the current report branch. Preserve provenance and a requirement
coverage ledger. Join compatible observations, expose conflicts, and retain missing enumerations,
table cells, scientific sub-questions, equations, named methods, and assumptions for the parent.
"""

_TREE_ROOT_USER_GUIDANCE = (
    "Produce the full ROOT report with every requested section and evidence-backed detail."
)
_TREE_INTERNAL_USER_GUIDANCE = (
    "Return a dense evidence memo for this branch, with supported coverage and explicit gaps."
)

_PARTIAL_SYNTHESIS_SYSTEM = """\
This research leaf was interrupted or exhausted its budget. Return its strongest supported partial
evidence memo within the assigned scope. Preserve derivations, values, assumptions, actual checks
and concrete remaining gaps. Use completion_outcome=exhausted_unresolved while decisive leaf
requirements remain unsupported. Do not expand into final-report writing or fill gaps with guesses.
"""

_EPOCH_0_R_CLOSURE_POLICY = """\
Before resolving, map every leaf requirement to concrete evidence in R. A fluent memo is not closure;
missing [[WEB]]required source support, [[/WEB]]fields, table cells, equations, methods, or assumptions remain explicit gaps.
"""
_EPOCH_1_BEST_EFFORT_SYNTHESIS_POLICY = """\
This is a later epoch. Reuse supported prior results and return a useful research evidence memo
with provenance and unresolved gaps. A later epoch does not imply that verification passed or that
all rebuild budget is exhausted; never upgrade unsupported claims to facts.
"""
_TREE_EPOCH_0_CLOSURE_POLICY = _EPOCH_0_R_CLOSURE_POLICY
_TREE_EPOCH_1_BEST_EFFORT_POLICY = _EPOCH_1_BEST_EFFORT_SYNTHESIS_POLICY
_TERMINAL_BEST_EFFORT_POLICY = """\
Produce the most complete evidence-grounded partial report available. Explicitly mark uncovered
requirements and conflicts rather than returning empty output or manufacturing certainty.
"""

_SYNTHESIS_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "Return an evidence-rich leaf memo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_summary": {"type": "string"},
                    "completion_outcome": {
                        "type": "string",
                        "enum": ["resolved", "exhausted_unresolved"],
                    },
                },
                "required": ["result_summary", "completion_outcome"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_artifact",
            "description": "Save this leaf's research evidence artifact and return its memo; no final report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_summary": {"type": "string"},
                    "content": {"type": "string"},
                    "filename": {"type": "string"},
                    "completion_outcome": {
                        "type": "string",
                        "enum": ["resolved", "exhausted_unresolved"],
                    },
                },
                "required": [
                    "result_summary",
                    "content",
                    "filename",
                    "completion_outcome",
                ],
                "additionalProperties": False,
            },
        },
    },
)

_ROOT_DIRECT_SYNTHESIS_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "finalize_root_answer",
            "description": "Return the complete evidence-grounded Markdown report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_output_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "evidence_summary": {"type": "string"},
                    "answer": {"type": "string"},
                    "completion_outcome": {
                        "type": "string",
                        "enum": ["resolved", "exhausted_unresolved"],
                    },
                },
                "required": [
                    "requested_output_fields",
                    "evidence_summary",
                    "answer",
                    "completion_outcome",
                ],
                "additionalProperties": False,
            },
        },
    },
)

_FORCED_FINALIZATION_AUDIT_SYSTEM = """\
Audit interrupted report evidence without selecting or inventing content. Inventory requested output
fields, decisive public-question constraints, supported candidate values, contradictions, unresolved
relations, role-binding issues, and unassigned answer-shaped values. Preserve exact provenance.
"""

_FORCED_FINALIZATION_AUDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "audit_forced_finalization",
        "description": "Create a selection-neutral report-evidence audit.",
        "parameters": {
            "type": "object",
            "properties": {
                "requested_output_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "decisive_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "string"},
                            "answer_value": {"type": "string"},
                            "supporting_observations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "contradicting_observations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "unresolved_relations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "role_binding_issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "candidate",
                            "answer_value",
                            "supporting_observations",
                            "contradicting_observations",
                            "unresolved_relations",
                            "role_binding_issues",
                        ],
                    },
                },
                "unassigned_answer_shaped_values": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "requested_output_fields",
                "decisive_constraints",
                "candidates",
                "unassigned_answer_shaped_values",
            ],
            "additionalProperties": False,
        },
    },
}

_FORCED_FINALIZATION_SELECTION_SYSTEM = """\
Produce the strongest evidence-grounded partial report from the selection-neutral audit. Preserve
gaps and conflicts, never invent evidence, and put the complete useful report in the answer field.
"""


__all__ = [name for name in globals() if name.startswith("_") and name.isupper()]
