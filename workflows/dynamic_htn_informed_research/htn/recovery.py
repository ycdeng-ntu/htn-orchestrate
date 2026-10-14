"""Checkpointed evidence routing for interrupted or unresolved HTN epochs.

The active HTN tree is disposable.  After execution fails, a deadline expires,
or an approach is exhausted, a conservative gate decides whether checkpointed evidence
is ready for the normal root finalizer.  If it is not, a separate fresh call
extracts positive, directly quoted facts for a new planning epoch.  Neither
call has research tools or mutates an HTN tree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from workflows.dynamic_htn_informed_research.htn.llm_types import (
    LLMClient,
    system_msg,
    user_msg,
)
from workflows.dynamic_htn_informed_research.htn.orchestrator import AgentResultRecord
from workflows.dynamic_htn_informed_research.task_strategies.contracts import (
    RecoveryStrategy,
)
from workflows.dynamic_htn_informed_research.task_strategies.registry import (
    get_active_task_strategy,
)

_TRUNCATION_MARKER = "\n...[recovery evidence truncated]...\n"


def _bounded_text(text: str, limit: int) -> str:
    clean = str(text or "").strip()
    if not clean or limit <= 0:
        return ""
    if len(clean) <= limit:
        return clean
    if limit <= len(_TRUNCATION_MARKER) + 2:
        return clean[:limit]
    payload = limit - len(_TRUNCATION_MARKER)
    head = max(1, payload * 2 // 3)
    return clean[:head] + _TRUNCATION_MARKER + clean[-(payload - head) :]


@dataclass(frozen=True)
class CompletedNodeEvidence:
    node_id: str
    title: str
    result_summary: str
    ref: str | None = None


@dataclass(frozen=True)
class FailedNodeEvidence:
    """Preserved evidence from a failed execution, never a completed node."""

    node_id: str
    task: str
    error_type: str
    error: str
    partial_summary: str
    ref: str | None = None


@dataclass
class TimeoutRecoveryEvidence:
    question: str
    epoch_id: int
    timed_out_leaf_ids: list[str]
    timed_out_tasks: list[str]
    trigger_kind: str = "timeout"
    completed_nodes: list[CompletedNodeEvidence] = field(default_factory=list)
    agent_results: list[AgentResultRecord] = field(default_factory=list)
    failed_nodes: list[FailedNodeEvidence] = field(default_factory=list)

    def unique_agent_results(self) -> list[AgentResultRecord]:
        """Return one record per agent, preferring the first checkpoint."""
        seen: set[str] = set()
        unique: list[AgentResultRecord] = []
        for record in self.agent_results:
            if record.agent_id in seen:
                continue
            seen.add(record.agent_id)
            unique.append(record)
        return unique


_NEGATIVE_FACT_RE = re.compile(
    r"\b(?:not|no candidate|not found|did not|could not|unable to|failed to|without|"
    r"exhaustive|unlikely|unknown|unresolved|ruled out|no evidence|not determinable)\b",
    re.IGNORECASE,
)
_SOURCE_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_REPORT_ID_RE = re.compile(r"\bdyni_[A-Za-z0-9_-]+\b")
_EVIDENCE_BLOCK_ID_RE = re.compile(
    r"^# (?:Agent report|Completed node|Failed execution) \[([^\]]+)\]",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SupportedRecoveryFact:
    claim: str
    supporting_quote: str
    source_or_report_ref: str
    answer_shaped_value: str = ""


def render_recovery_evidence(
    evidence: TimeoutRecoveryEvidence,
    *,
    max_chars: int = 500_000,
    max_item_chars: int | None = None,
) -> str:
    """Render unique evidence verbatim unless the total exceeds the safety cap."""
    trigger_label = (
        "Semantic exhaustion"
        if evidence.trigger_kind == "semantic_exhaustion"
        else "Tree completed unresolved"
        if evidence.trigger_kind == "tree_complete_unresolved"
        else "Resolved root omitted the required answer"
        if evidence.trigger_kind == "root_answer_contract_failure"
        else "Question deadline"
        if evidence.trigger_kind == "question_deadline"
        else "Execution failure (not a finding about the question)"
        if evidence.trigger_kind == "execution_failure"
        else "Question worker dispatch budget exhausted"
        if evidence.trigger_kind == "question_dispatch_budget_exhausted"
        else "Leaf timeout"
    )
    header = (
        f"# Original question\n{evidence.question}\n\n"
        f"# Recovery trigger\nType: {trigger_label}\nEpoch: {evidence.epoch_id}\n"
        f"Trigger nodes/leaves: {', '.join(evidence.timed_out_leaf_ids) or '(unknown)'}\n"
        "Trigger tasks:\n- "
        + "\n- ".join(evidence.timed_out_tasks or ["(unknown)"])
    )
    raw_blocks: list[str] = []
    for node in evidence.completed_nodes:
        ref = f"\nFull ref: {node.ref}" if node.ref else ""
        raw_blocks.append(
            f"# Completed node [{node.node_id}] {node.title}\n"
            f"{node.result_summary}{ref}"
        )

    for node in evidence.failed_nodes:
        ref = f"\nPartial evidence ref: {node.ref}" if node.ref else ""
        raw_blocks.append(
            f"# Failed execution [{node.node_id}]\nTask: {node.task}\n"
            f"Error type: {node.error_type}\nError detail (operational metadata, not evidence): {node.error}\n"
            "This execution did not resolve the task. Preserve supported findings below, "
            "but do not infer a factual conclusion from the execution failure.\n"
            f"Partial checkpoint:\n{node.partial_summary or '(no partial evidence)'}{ref}"
        )

    quality_rank = {"submitted": 0, "best_effort": 1, "thin": 2, "malformed": 3}
    records = list(enumerate(evidence.unique_agent_results()))
    records.sort(key=lambda pair: (quality_rank.get(pair[1].execution_quality, 4), pair[0]))
    for _, record in records:
        metadata = (
            f"\nObserved source metadata:\n{record.observed_source_index}"
            if record.observed_source_index
            else ""
        )
        ref = f"\nFull ref: {record.ref}" if record.ref else ""
        raw_blocks.append(
            f"# Agent report [{record.agent_id}] role={record.role_id}\n"
            f"Task: {record.task_summary}\n"
            f"Delivery quality: {record.execution_quality}\n"
            f"{record.result_summary}{metadata}{ref}"
        )

    # Preserve the full raw reports whenever they fit.  The 500K default is a
    # guard against pathological growth, not a latency optimization.  Only an
    # over-cap epoch is fair-shared so every returned route remains represented.
    separator_chars = len("\n\n---\n\n") * len(raw_blocks)
    if (
        len(header) + separator_chars + sum(len(block) for block in raw_blocks)
        <= max_chars
    ):
        return "\n\n---\n\n".join([header, *raw_blocks])

    remaining = max(0, max_chars - len(header) - separator_chars)
    blocks = [header]
    for index, block in enumerate(raw_blocks):
        items_left = len(raw_blocks) - index
        fair_share = remaining // max(1, items_left)
        item_limit = max(240, fair_share)
        if max_item_chars is not None:
            item_limit = min(max_item_chars, item_limit)
        bounded = _bounded_text(block, item_limit)
        if bounded:
            blocks.append(bounded)
            remaining = max(0, remaining - len(bounded))

    rendered = "\n\n---\n\n".join(blocks)
    return _bounded_text(rendered, max_chars)


def _parse_tool_args(response: object, expected_name: str) -> dict[str, object] | None:
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return None
    try:
        call = tool_calls[0]
        if str(call["function"]["name"]) != expected_name:
            return None
        args = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return args if isinstance(args, dict) else None


async def assess_timeout_finalizability(
    llm: LLMClient,
    rendered_evidence: str,
    strategy: RecoveryStrategy | None = None,
) -> bool | None:
    """Return a conservative routing decision without extracting an answer."""
    strategy = strategy or get_active_task_strategy().recovery
    response = await llm.chat(
        [
            system_msg(strategy.finalize_gate_system),
            user_msg(rendered_evidence),
        ],
        tools=[strategy.finalize_gate_tool],
    )
    args = _parse_tool_args(response, "assess_finalizability")
    if args is None or not isinstance(args.get("can_finalize"), bool):
        return None
    return bool(args["can_finalize"])


def _normalized_inline(text: object) -> str:
    return " ".join(str(text or "").split())


def _validated_ref(ref: str, normalized_evidence: str) -> str:
    """Return one exact observed pointer, never model-authored ref prose."""
    if ref in normalized_evidence:
        return ref
    candidates = [*_SOURCE_URL_RE.findall(ref), *_REPORT_ID_RE.findall(ref)]
    for candidate in candidates:
        candidate = candidate.rstrip(".,;:)]}")
        if candidate and candidate in normalized_evidence:
            return candidate
    return ""


def _containing_block_ref(quote: str, rendered_evidence: str) -> str:
    """Bind an exact quote to the report/node block that contains it."""
    for block in rendered_evidence.split("\n\n---\n\n"):
        if quote not in _normalized_inline(block):
            continue
        match = _EVIDENCE_BLOCK_ID_RE.search(block)
        if match:
            return _normalized_inline(match.group(1))
    return ""


def _validated_supported_facts(
    value: object,
    rendered_evidence: str,
) -> tuple[SupportedRecoveryFact, ...]:
    if not isinstance(value, list):
        return ()
    normalized_evidence = _normalized_inline(rendered_evidence)
    facts: list[SupportedRecoveryFact] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = _normalized_inline(item.get("claim"))
        quote = _normalized_inline(item.get("supporting_quote"))
        proposed_ref = _normalized_inline(item.get("source_or_report_ref"))
        if not claim or not quote or _NEGATIVE_FACT_RE.search(claim):
            continue
        # Quotes and refs are evidence pointers, not free-form model prose.  A
        # normalized exact match tolerates line wrapping but rejects invention.
        ref = _validated_ref(proposed_ref, normalized_evidence) if proposed_ref else ""
        if not ref:
            ref = _containing_block_ref(quote, rendered_evidence)
        if quote not in normalized_evidence or not ref:
            continue
        key = (claim.casefold(), ref.casefold())
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            SupportedRecoveryFact(
                claim=claim,
                supporting_quote=quote,
                source_or_report_ref=ref,
            )
        )
    return tuple(facts)


def render_supported_fact_capsule(
    facts: tuple[SupportedRecoveryFact, ...],
    *,
    max_chars: int = 24_000,
    view: Literal["planning", "answer"] = "answer",
) -> str:
    """Render one view over the same validated recovery facts.

    The planning view preserves extractor order and gives every fact identical
    visual weight.  The answer view prioritizes exact substrings of the field
    requested by the root question so they survive the independent character
    budget used by root finalization.
    """
    if not facts or max_chars <= 0:
        return ""
    if view not in {"planning", "answer"}:
        raise ValueError(f"Unknown recovery capsule view: {view!r}")
    marker_indices: set[int] = set()
    seen_answer_values: set[str] = set()
    if view == "answer":
        for index, fact in enumerate(facts):
            value_key = fact.answer_shaped_value.casefold()
            if not value_key or value_key in seen_answer_values:
                continue
            seen_answer_values.add(value_key)
            marker_indices.add(index)
    ordered_items = (
        [
            *((facts[index], True) for index in sorted(marker_indices)),
            *((fact, False) for index, fact in enumerate(facts) if index not in marker_indices),
        ]
        if view == "answer"
        else [(fact, False) for fact in facts]
    )
    blocks = ["## Confirmed facts from prior epoch"]
    if view == "answer" and marker_indices:
        blocks.append(
            "Answer-field markers below are unranked retrieval aids, not candidate "
            "endorsements. Repeated values receive only one marker."
        )
    for fact, show_answer_marker in ordered_items:
        if show_answer_marker:
            block = (
                "- Observed answer-field value (unranked; not endorsed): "
                f"{fact.answer_shaped_value}\n"
                f"  Claim: {fact.claim}\n"
                f'  Support: "{fact.supporting_quote}"\n'
                f"  Ref: {fact.source_or_report_ref}"
            )
        else:
            block = (
                f"- Claim: {fact.claim}\n"
                f'  Support: "{fact.supporting_quote}"\n'
                f"  Ref: {fact.source_or_report_ref}"
            )
        candidate = "\n\n".join([*blocks, block])
        if len(candidate) > max_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks) if len(blocks) > 1 else ""


async def extract_supported_recovery_facts(
    llm: LLMClient,
    rendered_evidence: str,
    *,
    max_capsule_chars: int = 24_000,
    strategy: RecoveryStrategy | None = None,
) -> tuple[tuple[SupportedRecoveryFact, ...], str] | None:
    """Fresh-call positive fact extraction with deterministic validation."""
    strategy = strategy or get_active_task_strategy().recovery
    response = await llm.chat(
        [
            system_msg(strategy.supported_facts_system),
            user_msg(rendered_evidence),
        ],
        tools=[strategy.supported_facts_tool],
    )
    args = _parse_tool_args(response, "extract_supported_facts")
    if args is None:
        return None
    facts = _validated_supported_facts(args.get("supported_facts"), rendered_evidence)
    return facts, render_supported_fact_capsule(
        facts,
        max_chars=max_capsule_chars,
        view="planning",
    )


def _render_answer_annotation_user(
    question: str,
    facts: tuple[SupportedRecoveryFact, ...],
) -> str:
    payload = [
        {
            "fact_index": index,
            "claim": fact.claim,
            "supporting_quote": fact.supporting_quote,
            "source_or_report_ref": fact.source_or_report_ref,
        }
        for index, fact in enumerate(facts, start=1)
    ]
    return (
        f"# ROOT question\n{question}\n\n"
        "# Validated positive recovery facts\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


async def annotate_recovery_answer_values(
    llm: LLMClient,
    *,
    question: str,
    facts: tuple[SupportedRecoveryFact, ...],
    strategy: RecoveryStrategy | None = None,
) -> tuple[SupportedRecoveryFact, ...] | None:
    """Add exact answer-field substrings without changing the validated fact set."""
    if not facts:
        return facts
    strategy = strategy or get_active_task_strategy().recovery
    response = await llm.chat(
        [
            system_msg(strategy.answer_annotation_system),
            user_msg(_render_answer_annotation_user(question, facts)),
        ],
        tools=[strategy.answer_annotation_tool],
    )
    args = _parse_tool_args(response, "annotate_answer_shaped_values")
    if args is None or not isinstance(args.get("annotations"), list):
        return None
    values: dict[int, str] = {}
    for item in args["annotations"]:
        if not isinstance(item, dict):
            return None
        fact_index = item.get("fact_index")
        value = _normalized_inline(item.get("answer_shaped_value"))
        if (
            not isinstance(fact_index, int)
            or isinstance(fact_index, bool)
            or fact_index < 1
            or fact_index > len(facts)
            or fact_index in values
            or not value
            or value not in facts[fact_index - 1].supporting_quote
        ):
            return None
        values[fact_index] = value
    return tuple(
        replace(fact, answer_shaped_value=values.get(index, ""))
        for index, fact in enumerate(facts, start=1)
    )
