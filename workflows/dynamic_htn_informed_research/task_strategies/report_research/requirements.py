"""Public-question requirement extraction for report research.

This layer turns only the public question and immutable report policy guidance
into a compact, locally identified delivery contract.  Runtime evidence and
mutable coverage ownership deliberately live elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import (
    LLMClient,
    LLMResponse,
    system_msg,
    user_msg,
)

from .prompt_access import report_system_prompt

from .policy import ReportPolicy
from .protocol import report_call


class RequirementStatus(StrEnum):
    PENDING = "pending"
    GROUNDED = "grounded"
    EXHAUSTED = "exhausted"


@dataclass
class Requirement:
    id: str
    description: str
    acceptance_criteria: tuple[str, ...]
    priority: int
    kind: str
    owner_leaf_id: str | None = None
    status: RequirementStatus = RequirementStatus.PENDING
    evidence_ids: list[str] = field(default_factory=list)
    covered_criterion_ids: list[str] = field(default_factory=list)
    sub_obligations: tuple[str, ...] = ()

    disputed_evidence_ids: list[str] = field(default_factory=list)
    reopen_reasons: list[str] = field(default_factory=list)

    def reopen(self, reason: str) -> None:
        """Withdraw current coverage for review without erasing its provenance."""
        if not reason.strip():
            raise ValueError("reopening a requirement needs a reason")
        for evidence_id in self.evidence_ids:
            if evidence_id not in self.disputed_evidence_ids:
                self.disputed_evidence_ids.append(evidence_id)
        self.evidence_ids.clear()
        self.covered_criterion_ids.clear()
        self.status = RequirementStatus.PENDING
        if reason not in self.reopen_reasons:
            self.reopen_reasons.append(reason)

    def transition_to(self, status: RequirementStatus) -> None:
        """Record coverage or exhaustion; use reopen() when support is challenged."""
        if not isinstance(status, RequirementStatus):
            raise ValueError("requirement status must be a RequirementStatus")
        if self.status is not RequirementStatus.PENDING:
            raise ValueError("requirement status is terminal")
        if status is RequirementStatus.PENDING:
            raise ValueError("requirement status transition must be terminal")
        self.status = status


@dataclass
class RequirementSet:
    policy_key: str
    requirements: list[Requirement]


_TOOL_NAME = "derive_requirements"
_MAX_PRIORITY = 5
_REQUIREMENT_FIELDS = frozenset(
    {
        "id",
        "description",
        "acceptance_criteria",
        "priority",
        "kind",
        "entities",
        "fields",
    }
)

_REQUIREMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": "Return the complete, source-neutral delivery requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string", "minLength": 1},
                            "acceptance_criteria": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                            "kind": {"type": "string", "minLength": 1},
                            "entities": {
                                "description": (
                                    "Optional entity labels. For table expansion supply fields too "
                                    "and both {entity}/{field} placeholders in description; "
                                    "otherwise retain as scope metadata."
                                ),
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "fields": {
                                "description": (
                                    "Optional field labels. Expand a table only with entities "
                                    "and both description placeholders; otherwise retain "
                                    "as scope metadata."
                                ),
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                        "required": [
                            "description",
                            "acceptance_criteria",
                            "priority",
                            "kind",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["requirements"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = """\
Extract a source-neutral report delivery contract from the public user question.
Use only that question and the supplied immutable policy guidance. Do not solve the
question, infer an answer, or consult any private evaluation material. Keep one requirement
per explicit part/scenario; put its requested attributes in acceptance criteria. Do not add
a comparison matrix, candidate inventory, numerical experiment, or presentation obligation
unless the public question asks for it. Do not repeat the same obligation both per scenario
and in an entity-by-field matrix. Return exactly
one derive_requirements tool call. Each requirement must be concrete, testable, and
include acceptance criteria. Acceptance criteria restate what the question demands;
they must never assert a solution value, numeric convention, unit conversion, formula
choice, or method selection that the question itself does not state — deciding those
is the researcher's job, not the contract's. Where a request is a finite
entity-by-field table, use entities and fields with {entity} and {field} placeholders
in the description so cells can be independently covered. Omit these arrays for ordinary
sub-questions; labels without placeholders are scope metadata, not a table to expand. Model-provided IDs are
checked only for duplicates; local IDs are assigned after validation.
""".strip()


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError(f"requirement {field_name} must be a non-empty string")
    return text


def _clean_text_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"requirement {field_name} must be a non-empty list")
    return tuple(_clean_text(item, field_name) for item in value)


def _parse_call_arguments(response: LLMResponse) -> dict[str, Any]:
    calls = response.tool_calls if isinstance(response, LLMResponse) else None
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("response must contain exactly one requirement tool call")
    call = calls[0]
    if not isinstance(call, dict):
        raise ValueError("requirement tool call must be an object")
    if set(call) != {"function"}:
        raise ValueError("requirement tool call has unknown fields")
    function = call.get("function")
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        raise ValueError("requirement tool call function is malformed")
    if function["name"] != _TOOL_NAME:
        raise ValueError("unexpected requirement tool name")
    arguments = function["arguments"]
    if not isinstance(arguments, str):
        raise ValueError("requirement tool arguments must be JSON text")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("requirement tool arguments are invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("requirement tool arguments must be an object")
    return payload


def _validated_requirement(raw: Any, seen_model_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("requirement must be an object")
    unknown = set(raw) - _REQUIREMENT_FIELDS
    if unknown:
        raise ValueError("requirement has unknown fields")
    required = {"description", "acceptance_criteria", "priority", "kind"}
    if set(raw) & required != required:
        raise ValueError("requirement is missing required fields")

    model_id = raw.get("id")
    if model_id is not None:
        model_id = _clean_text(model_id, "id")
        if model_id in seen_model_ids:
            raise ValueError("duplicate model requirement id")
        seen_model_ids.add(model_id)

    priority = raw["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= _MAX_PRIORITY:
        raise ValueError("requirement priority must be an integer from 1 through 5")

    description = _clean_text(raw["description"], "description")
    cleaned_entities = _clean_text_list(raw["entities"], "entities") if "entities" in raw else ()
    cleaned_fields = _clean_text_list(raw["fields"], "fields") if "fields" in raw else ()
    if len(set(cleaned_entities)) != len(cleaned_entities) or len(
        set(cleaned_fields)
    ) != len(cleaned_fields):
        raise ValueError("duplicate entity or field values are not allowed")

    # The advertised schema permits optional labels. A model often supplies
    # them for an ordinary scientific sub-question, not a Cartesian table.
    # Preserve that complete obligation and its labels instead of discarding
    # every requirement because optional metadata was not an expansion template.
    templated = "{entity}" in description or "{field}" in description
    sub_obligations: tuple[str, ...] = ()
    if templated:
        if "{entity}" not in description or "{field}" not in description:
            raise ValueError("entity and field placeholders are required for expansion")
        if not cleaned_entities or not cleaned_fields:
            raise ValueError("table expansion requires both non-empty entities and fields")
    else:
        sub_obligations = tuple(
            f"{label}: {', '.join(values)}"
            for label, values in (("Entities in scope", cleaned_entities),
                                  ("Fields in scope", cleaned_fields))
            if values
        )
        cleaned_entities = cleaned_fields = ()
    return {
        "description": description,
        "acceptance_criteria": _clean_text_list(
            raw["acceptance_criteria"], "acceptance_criteria"
        ),
        "priority": priority,
        "kind": _clean_text(raw["kind"], "kind"),
        "entities": cleaned_entities,
        "fields": cleaned_fields,
        "sub_obligations": sub_obligations,
    }


def _expanded_rows(item: dict[str, Any], *, expand_fields: bool = True) -> list[dict[str, Any]]:
    entities = item["entities"]
    fields = item["fields"]
    if not entities:
        return [item]
    if not expand_fields:
        # Scientific reports keep one entity/part together while retaining every
        # requested cell as a testable criterion and source obligation.
        return [{
            **item,
            "description": item["description"].replace("{entity}", entity).replace(
                "{field}", "requested attributes (" + ", ".join(fields) + ")"),
            "acceptance_criteria": tuple(dict.fromkeys(
                criterion.replace("{entity}", entity).replace("{field}", field)
                for field in fields for criterion in item["acceptance_criteria"]
            )) + tuple(f"Address the requested {field} for {entity}." for field in fields),
            "sub_obligations": tuple(item["description"].replace("{entity}", entity).replace(
                "{field}", field) for field in fields),
            "entities": (), "fields": (),
        } for entity in entities]
    return [
        {
            **item,
            "description": item["description"].replace("{entity}", entity).replace(
                "{field}", field
            ),
            "kind": f"{item['kind']}_cell",
            "entities": (),
            "fields": (),
        }
        for entity in entities
        for field in fields
    ]


def _make_requirement(item: dict[str, Any], index: int) -> Requirement:
    return Requirement(
        id=f"req-{index:03d}",
        description=item["description"],
        acceptance_criteria=item["acceptance_criteria"],
        priority=item["priority"],
        kind=item["kind"],
        sub_obligations=tuple(item.get("sub_obligations") or ()),
    )


def _aggregate_obligations(items: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = tuple(
        criterion for item in items for criterion in item["acceptance_criteria"]
    )
    return {
        "acceptance_criteria": criteria,
        "description": (
            f"Cover all {len(items)} remaining public delivery obligations "
            "listed in the structured sub-obligation contract."
        ),
        "priority": min(item["priority"] for item in items),
        "kind": "aggregate",
        "entities": (),
        "fields": (),
        "sub_obligations": tuple(
            text for item in items
            for text in (item["description"], *item.get("sub_obligations", ()))
        ),
    }


def _requirements_from_payload(payload: dict[str, Any], policy: ReportPolicy) -> list[Requirement]:
    if set(payload) != {"requirements"}:
        raise ValueError("requirement payload has unknown or missing fields")
    raw_requirements = payload["requirements"]
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError("requirement payload must contain a non-empty requirements list")
    if policy.max_requirements < 1:
        raise ValueError("report policy max_requirements must be positive")

    seen_model_ids: set[str] = set()
    items = []
    for index, raw in enumerate(raw_requirements):
        try:
            items.append(_validated_requirement(raw, seen_model_ids))
        except ValueError as exc:
            raise ValueError(f"requirements[{index}]: {exc}") from exc
    rows = [row for item in items for row in _expanded_rows(
        item, expand_fields=policy.enumerative_coverage,
    )]
    # Merge exact repeated obligations only. Never fuzzy-merge names or formulas.
    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["description"], row["kind"])
        if key not in distinct:
            distinct[key] = dict(row)
        else:
            prior = distinct[key]
            prior["priority"] = min(prior["priority"], row["priority"])
            for field in ("acceptance_criteria", "sub_obligations"):
                prior[field] = tuple(dict.fromkeys((*prior.get(field, ()), *row.get(field, ()))))
    obligations = list(distinct.values())
    if len(obligations) > policy.max_requirements:
        obligations = [
            *obligations[: policy.max_requirements - 1],
            _aggregate_obligations(obligations[policy.max_requirements - 1 :]),
        ]
    return [_make_requirement(item, index) for index, item in enumerate(obligations, start=1)]


def parse_requirement_tool_call(
    response: LLMResponse,
    *,
    question: str,
    policy: ReportPolicy,
) -> RequirementSet:
    """Strictly parse one requirement extraction tool call into local IDs."""
    _clean_text(question, "question")
    payload = _parse_call_arguments(response)
    return RequirementSet(
        policy_key=policy.key,
        requirements=_requirements_from_payload(payload, policy),
    )


def aggregate_requirement_fallback(question: str, policy: ReportPolicy) -> RequirementSet:
    """Return the deterministic, coverage-preserving fallback contract."""
    public_question = _clean_text(question, "question")
    return RequirementSet(
        policy_key=policy.key,
        requirements=[
            Requirement(
                id="req-001",
                description=f"Address the complete public question: {public_question}",
                acceptance_criteria=(
                    "Address every explicitly requested deliverable in the public question.",
                    "State any remaining uncertainty or unavailable evidence explicitly.",
                ),
                priority=1,
                kind="aggregate",
            )
        ],
    )


def _messages(question: str, policy: ReportPolicy) -> list[dict[str, str]]:
    return [
        system_msg(report_system_prompt(
            f"{_SYSTEM_PROMPT}\n\nImmutable policy guidance:\n{policy.requirement_prompt}",
            network_enabled=policy.network_enabled,
        )),
        user_msg(f"Public question:\n{_clean_text(question, 'question')}"),
    ]


def _emit(
    diagnostics: DiagnosticRecorder | None,
    *,
    attempt: int,
    parse_status: str,
    requirement_count: int,
    fallback: bool,
    error: str | None = None,
) -> None:
    if diagnostics is not None:
        diagnostics.emit(
            "report_requirement_derivation",
            attempt=attempt,
            parse_status=parse_status,
            requirement_count=requirement_count,
            fallback=fallback,
            **({"error": error} if error else {}),
        )


async def derive_requirement_set(
    *,
    question: str,
    policy: ReportPolicy,
    llm: LLMClient,
    diagnostics: DiagnosticRecorder | None = None,
    question_deadline: float | None = None,
) -> RequirementSet:
    """Derive requirements with one protocol retry and a deterministic fallback."""
    _clean_text(question, "question")
    fallback = aggregate_requirement_fallback(question, policy)
    last_attempt = 0

    def record(attempt: int, status: str, error: str | None) -> None:
        nonlocal last_attempt
        last_attempt = attempt
        if status != "valid":
            _emit(diagnostics, attempt=attempt, parse_status=status,
                  requirement_count=0, fallback=False, error=error)

    try:
        _, result = await report_call(
            llm=llm, messages=_messages(question, policy),
            tool=_REQUIREMENT_TOOL, stage="requirements", deadline=question_deadline,
            diagnostics=diagnostics, on_attempt=record,
            parse=lambda response: parse_requirement_tool_call(response, question=question, policy=policy),
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - preserve deterministic public-question fallback
        _emit(diagnostics, attempt=last_attempt, parse_status="fallback",
              requirement_count=len(fallback.requirements), fallback=True)
        return fallback
    _emit(diagnostics, attempt=last_attempt, parse_status="valid",
          requirement_count=len(result.requirements), fallback=False)
    return result


__all__ = [
    "Requirement",
    "RequirementSet",
    "RequirementStatus",
    "aggregate_requirement_fallback",
    "derive_requirement_set",
    "parse_requirement_tool_call",
]
