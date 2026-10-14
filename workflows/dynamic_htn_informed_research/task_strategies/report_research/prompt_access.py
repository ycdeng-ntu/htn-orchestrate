"""Report evidence guidance rendered from task-local benchmark capabilities."""
from __future__ import annotations

from ..contracts import TaskStrategy
from ..prompt_access import render_strategy as _render_strategy, render_web_sections

def evidence_guidance(*, network_enabled: bool) -> str:
    if network_enabled:
        return (
            "Web access is available. For externally sourced claims, record observed source URLs "
            "and verbatim supporting excerpts. Tool-enabled researchers and verifiers independently "
            "verify load-bearing claims online; planning and writing stages use supplied provenance "
            "and must not claim retrieval they did not perform. Do not invent citations or treat "
            "failed retrieval as a contradiction."
        )
    return (
        "Work offline using the supplied question, explicit reasoning, local files produced by "
        "this task, and calculations actually executed with available tools. Judge support by "
        "question constraints, derivations, assumptions and observed results. Missing external "
        "retrieval alone is not a coverage gap or a reason to reject a conclusion. Distinguish "
        "established results from uncertain recalled facts; do not lower scientific correctness "
        "standards or claim checks that were not performed. Preserve explicit public-question "
        "requirements and state any capability limitation without inventing evidence."
    )


def report_system_prompt(template: str, *, network_enabled: bool) -> str:
    text = render_web_sections(template, network_enabled=network_enabled).strip()
    return text + "\n\n" + evidence_guidance(network_enabled=network_enabled)


def render_strategy(template: TaskStrategy, *, network_enabled: bool) -> TaskStrategy:
    return _render_strategy(
        template, network_enabled=network_enabled,
        system_suffix=evidence_guidance(network_enabled=network_enabled),
    )
