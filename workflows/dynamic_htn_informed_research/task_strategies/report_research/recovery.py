"""Recovery prompts and schemas for report evidence."""

from __future__ import annotations


_MALFORMED_TRACE_RECOVERY_SYSTEM = """\
Recover a compact evidence memo from a malformed report-worker trace. Treat ROOT and assignments as
scope only; the trace is the sole evidence source. Preserve exact claims, [[WEB]]observed source URLs and verbatim excerpts, [[/WEB]]reasoning,
enumeration/table-cell coverage, equations, named methods, assumptions, conflicts, and uncovered
requirements. Never invent support[[WEB]], citations[[/WEB]], or completion.
"""

_FINALIZE_GATE_SYSTEM = """\
Decide whether checkpointed report evidence is sufficient to synthesize a useful evidence-grounded
report. Return true only when it contains concrete support for the requested core sections and no
load-bearing requirement is represented solely by a guess. Do not write or score the report.
"""

_FINALIZE_GATE_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_finalizability",
        "description": "Report whether available evidence supports report synthesis.",
        "parameters": {
            "type": "object",
            "properties": {"can_finalize": {"type": "boolean"}},
            "required": ["can_finalize"],
            "additionalProperties": False,
        },
    },
}

_SUPPORTED_FACTS_SYSTEM = """\
Extract only positive atomic facts directly supported by the checkpointed report evidence. Preserve
exact values, units, equations, method names, qualifications, [[WEB]]source URLs and source excerpts, [[/WEB]]report references, and short quotes from
the supplied worker records. These record quotes may support a derivation or observed calculation.
Do not infer missing table cells, fabricate support, or select answers.[[WEB]] Do not turn failed
search into a world claim.[[/WEB]]
"""

_SUPPORTED_FACTS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_supported_facts",
        "description": "Extract directly supported report facts with provenance.",
        "parameters": {
            "type": "object",
            "properties": {
                "supported_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "supporting_quote": {"type": "string"},
                            "source_or_report_ref": {"type": "string"},
                        },
                        "required": ["claim", "supporting_quote"],
                    },
                }
            },
            "required": ["supported_facts"],
            "additionalProperties": False,
        },
    },
}

_ANSWER_VALUE_ANNOTATION_SYSTEM = """\
Annotate exact public-question output values already present in validated report facts. Copy only a
continuous substring from its supporting quote. Do not normalize, rank, endorse, or invent values.
"""

_ANSWER_VALUE_ANNOTATION_TOOL = {
    "type": "function",
    "function": {
        "name": "annotate_answer_shaped_values",
        "description": "Mark exact report-output substrings in supported facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_index": {"type": "integer", "minimum": 1},
                            "answer_shaped_value": {"type": "string", "minLength": 1},
                        },
                        "required": ["fact_index", "answer_shaped_value"],
                    },
                }
            },
            "required": ["annotations"],
            "additionalProperties": False,
        },
    },
}


__all__ = [name for name in globals() if name.startswith("_") and name.isupper()]
