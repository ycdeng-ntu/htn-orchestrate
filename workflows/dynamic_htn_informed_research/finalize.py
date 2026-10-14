"""Safe parsing helpers for the final ``# Answer:`` contract.

Dynamic HTN Informed deliberately has no independent final-verifier or
verification-driven revision stage. Semantic selection belongs to root
synthesis or its audit/select fallback; these helpers only clean and preserve
an explicitly serialized upstream answer.
"""

from __future__ import annotations

import re

from workflows.dynamic_htn_informed_research.markup import strip_leaked_tool_markup


# Junk the answer-marker step must never emit as the answer: leftover template
# placeholders / tags, the leaf-timeout sentinel, or empty-ish markers.
_PLACEHOLDER_RE = re.compile(
    r"^\s*(<\s*answer\s*>|<\s*the answer\s*>|<[^>]{0,30}>|n/?a|none|unknown|"
    r"\(?\s*no answer (?:could be determined|was found|available)\s*\)?|"
    r"no\s+(?:qualifying\s+)?(?:author|band|song|series|candidate|person|place|date|"
    r"number|game|film|book|album|track|company|organization)s?"
    r"(?:\s+(?:was\s+)?(?:found|identified|determined|matched|verified|qualified))?|"
    r"cannot be determined(?: from available evidence)?|not found|"
    r"\(leaf timed out[^)]*\)|answer|tbd|\.*)\s*$",
    re.I,
)


def _clean_answer(value: str, *, answer_pattern: str = "") -> str:
    """Strip the marker and surrounding tags; return ``""`` for junk."""
    text = strip_leaked_tool_markup(value or "")
    text = re.sub(r"(?im)^\s*#\s*answer\s*:\s*", "", text).strip()
    text = re.sub(
        r"^<\s*answer\s*>|</\s*answer\s*>$",
        "",
        text,
        flags=re.I,
    ).strip()
    if _PLACEHOLDER_RE.match(text):
        return ""
    if not answer_pattern:
        return text
    match = re.fullmatch(answer_pattern, text)
    if match is None:
        return ""
    return (match.group(1) if match.lastindex else match.group(0)).strip()


def _best_fallback_answer(candidate: str, *, answer_pattern: str = "") -> str:
    """Return the last valid answer explicitly marked by upstream synthesis."""
    matches = re.findall(
        r"(?im)^\s*#\s*answer\s*:\s*([^\n]+)\s*$",
        candidate or "",
    )
    for raw in reversed(matches):
        cleaned = _clean_answer(raw, answer_pattern=answer_pattern)
        if cleaned:
            return cleaned
    return ""


def _answer_from_model_output(
    text: str,
    *,
    answer_pattern: str = "",
    allow_multiline: bool = False,
) -> str:
    """Accept a report body, requested marker line, or single-line response."""
    if allow_multiline:
        report = strip_leaked_tool_markup(text or "").strip()
        return "" if _PLACEHOLDER_RE.fullmatch(report) else report
    explicit = _best_fallback_answer(text, answer_pattern=answer_pattern)
    if explicit:
        return explicit
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return ""
    return _clean_answer(lines[0], answer_pattern=answer_pattern)


__all__ = [
    "_answer_from_model_output",
    "_best_fallback_answer",
]
