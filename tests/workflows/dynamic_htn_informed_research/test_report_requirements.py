from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
from workflows.dynamic_htn_informed_research.task_strategies.report_research.finalize import (
    _bounded_inputs,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import (
    Requirement,
    RequirementSet,
    RequirementStatus,
    aggregate_requirement_fallback,
    derive_requirement_set,
    parse_requirement_tool_call,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import (
    ReportRuntime,
)


def _response(payload: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "derive_requirements",
                    "arguments": json.dumps(payload),
                }
            }
        ]
    )


def _payload(*requirements: dict[str, Any]) -> dict[str, Any]:
    return {"requirements": list(requirements)}


def _requirement(**overrides: Any) -> dict[str, Any]:
    requirement = {
        "description": "Compare each listed country by renewable-energy share.",
        "acceptance_criteria": ["Each requested country has one sourced value."],
        "priority": 2,
        "kind": "comparison",
    }
    requirement.update(overrides)
    return requirement


class FakeLLM:
    model = "fake"

    def __init__(self, *responses: LLMResponse) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, Any]] | None]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        return self._responses.pop(0)


def test_parser_generates_stable_local_ids_across_refreshes() -> None:
    response = _response(
        _payload(
            _requirement(id="model-one"),
            _requirement(
                id="model-two",
                description="Give the methodology used for every value.",
                acceptance_criteria=["Methodology is stated for every value."],
                priority=1,
                kind="method",
            ),
        )
    )

    first = parse_requirement_tool_call(
        response, question="Compare renewable energy shares.", policy=SCIENTIFIC_REPORT_POLICY
    )
    second = parse_requirement_tool_call(
        response, question="Compare renewable energy shares.", policy=SCIENTIFIC_REPORT_POLICY
    )

    assert [requirement.id for requirement in first.requirements] == ["req-001", "req-002"]
    assert [requirement.id for requirement in second.requirements] == ["req-001", "req-002"]


def test_requirement_status_transitions_are_terminal() -> None:
    requirement = Requirement(
        id="req-001",
        description="State the requested conclusion.",
        acceptance_criteria=("Conclusion is explicit.",),
        priority=1,
        kind="conclusion",
    )

    requirement.transition_to(RequirementStatus.GROUNDED)

    assert requirement.status is RequirementStatus.GROUNDED
    with pytest.raises(ValueError, match="terminal"):
        requirement.transition_to(RequirementStatus.EXHAUSTED)
    with pytest.raises(ValueError, match="terminal"):
        requirement.transition_to(RequirementStatus.PENDING)


@pytest.mark.parametrize(
    "payload",
    [
        _payload(_requirement(id="duplicate"), _requirement(id="duplicate")),
        _payload(_requirement(description="   ")),
        _payload(_requirement(priority=0)),
        _payload(_requirement(unexpected="not allowed")),
    ],
)
def test_parser_rejects_invalid_requirement_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_requirement_tool_call(
            _response(payload), question="Public question", policy=SCIENTIFIC_REPORT_POLICY
        )


@pytest.mark.asyncio
async def test_derivation_retries_one_malformed_tool_response_and_keeps_prompt_public() -> None:
    llm = FakeLLM(
        LLMResponse(content="not a tool call"),
        _response(_payload(_requirement())),
    )

    result = await derive_requirement_set(
        question="Compare annual solar generation in 2023.",
        policy=SCIENTIFIC_REPORT_POLICY,
        llm=llm,
    )

    assert [requirement.id for requirement in result.requirements] == ["req-001"]
    assert len(llm.calls) == 2
    prompt_text = "\n".join(
        message["content"] for messages, _ in llm.calls for message in messages
    ).lower()
    assert "compare annual solar generation in 2023" in prompt_text
    assert SCIENTIFIC_REPORT_POLICY.requirement_prompt.lower() in prompt_text
    for forbidden in ("ground truth", "private rubric", "answer key", "judge feedback"):
        assert forbidden not in prompt_text


@pytest.mark.asyncio
async def test_two_malformed_responses_use_deterministic_aggregate_fallback() -> None:
    llm = FakeLLM(LLMResponse(), LLMResponse())

    result = await derive_requirement_set(
        question="Explain the current methods used to measure sea level.",
        policy=SCIENTIFIC_REPORT_POLICY,
        llm=llm,
    )

    assert result == aggregate_requirement_fallback(
        "Explain the current methods used to measure sea level.", SCIENTIFIC_REPORT_POLICY
    )
    assert result.requirements[0].id == "req-001"
    assert "Explain the current methods" in result.requirements[0].description
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_derivation_deadline_cancels_hung_attempt_and_returns_prebuilt_fallback(
    tmp_path: Path,
) -> None:
    question = "Explain the public methods used to measure sea level."

    class HangingLLM:
        model = "hanging"

        def __init__(self) -> None:
            self.calls = 0
            self.cancelled = 0

        async def chat(self, _messages, tools=None):  # noqa: ANN001, ARG002
            self.calls += 1
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return _response(_payload(_requirement()))

    llm = HangingLLM()
    diagnostics = DiagnosticRecorder(tmp_path)
    started = time.monotonic()

    result = await derive_requirement_set(
        question=question,
        policy=SCIENTIFIC_REPORT_POLICY,
        llm=llm,
        diagnostics=diagnostics,
        question_deadline=time.monotonic() + 0.02,
    )

    assert time.monotonic() - started < 0.25
    assert result == aggregate_requirement_fallback(question, SCIENTIFIC_REPORT_POLICY)
    assert llm.calls == 1
    assert llm.cancelled == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "report_requirement_derivation"
    assert events[-1]["parse_status"] == "fallback"
    assert events[-1]["fallback"] is True


def test_entity_field_expansion_respects_cap_and_keeps_aggregate_remainder() -> None:
    policy = replace(SCIENTIFIC_REPORT_POLICY, enumerative_coverage=True, max_requirements=3)
    result = parse_requirement_tool_call(
        _response(
            _payload(
                _requirement(
                    description="Report {field} for {entity}.",
                    acceptance_criteria=["Every requested cell has a source."],
                    kind="table",
                    entities=["A", "B"],
                    fields=["revenue", "employees"],
                )
            )
        ),
        question="Provide a company table.",
        policy=policy,
    )

    assert len(result.requirements) == 3
    assert [requirement.id for requirement in result.requirements] == [
        "req-001",
        "req-002",
        "req-003",
    ]
    assert result.requirements[-1].kind == "aggregate"
    assert "remaining" in result.requirements[-1].description.lower()
    assert result.requirements[-1].sub_obligations == (
        "Report revenue for B.",
        "Report employees for B.",
    )


def test_overflow_aggregate_preserves_omitted_table_cells_and_later_requirements() -> None:
    policy = replace(SCIENTIFIC_REPORT_POLICY, enumerative_coverage=True, max_requirements=3)
    result = parse_requirement_tool_call(
        _response(
            _payload(
                _requirement(
                    description="Define the reporting period.",
                    acceptance_criteria=["The reporting period is explicit."],
                    kind="definition",
                ),
                _requirement(
                    description="Report {field} for {entity}.",
                    acceptance_criteria=["Every requested cell has a source."],
                    kind="table",
                    entities=["A", "B"],
                    fields=["revenue", "employees"],
                ),
                _requirement(
                    description="Explain the table caveats.",
                    acceptance_criteria=["Caveats are explicit."],
                    kind="caveat",
                ),
            )
        ),
        question="Provide a company table.",
        policy=policy,
    )

    assert len(result.requirements) == 3
    assert result.requirements[-1].kind == "aggregate"
    assert result.requirements[-1].sub_obligations == (
        "Report employees for A.",
        "Report revenue for B.",
        "Report employees for B.",
        "Explain the table caveats.",
    )


def test_last_overflow_obligation_reaches_worker_and_finalizer_inputs() -> None:
    policy = replace(SCIENTIFIC_REPORT_POLICY, enumerative_coverage=True, max_requirements=2)
    result = parse_requirement_tool_call(
        _response(
            _payload(
                _requirement(
                    description="Report {field} for {entity}.",
                    acceptance_criteria=["Every requested cell has a source."],
                    kind="table",
                    entities=["Alpha", "Omega"],
                    fields=["revenue", "employees"],
                )
            )
        ),
        question="Provide the complete company table.",
        policy=policy,
    )
    aggregate = result.requirements[-1]
    aggregate.owner_leaf_id = "leaf-report"
    runtime = ReportRuntime(
        question="Provide the complete company table.",
        policy=policy,
        requirements=RequirementSet(policy.key, result.requirements),
    )
    leaf = HTNTaskNode(id="leaf-report", task="Complete the overflow table cells.")

    worker_context = runtime.delivery_context_for(leaf)
    finalizer_input = _bounded_inputs(
        question=runtime.question,
        runtime=runtime,
        root_summary="",
        completed_node_summaries=(),
    )

    tail = "Report employees for Omega."
    assert tail in worker_context
    aggregate_snapshot = next(
        item for item in finalizer_input["requirements"] if item["id"] == aggregate.id
    )
    assert tail in aggregate_snapshot["sub_obligations"]


def test_entity_field_expansion_at_exact_cap_keeps_every_obligation_separate() -> None:
    policy = replace(SCIENTIFIC_REPORT_POLICY, enumerative_coverage=True, max_requirements=5)
    result = parse_requirement_tool_call(
        _response(
            _payload(
                _requirement(
                    description="Define the reporting period.",
                    acceptance_criteria=["The reporting period is explicit."],
                    kind="definition",
                ),
                _requirement(
                    description="Report {field} for {entity}.",
                    acceptance_criteria=["Every requested cell has a source."],
                    kind="table",
                    entities=["A", "B"],
                    fields=["revenue", "employees"],
                ),
            )
        ),
        question="Provide a company table.",
        policy=policy,
    )

    assert len(result.requirements) == 5
    assert all(requirement.kind != "aggregate" for requirement in result.requirements)
    assert [requirement.description for requirement in result.requirements] == [
        "Define the reporting period.",
        "Report revenue for A.",
        "Report employees for A.",
        "Report revenue for B.",
        "Report employees for B.",
    ]


@pytest.mark.parametrize(
    "description",
    ["Report {field} for every company.", "Report revenue for {entity}."],
)
def test_entity_field_expansion_requires_both_cell_placeholders(description: str) -> None:
    with pytest.raises(ValueError, match="entity and field placeholders"):
        parse_requirement_tool_call(
            _response(
                _payload(
                    _requirement(
                        description=description,
                        entities=["A"],
                        fields=["revenue"],
                    )
                )
            ),
            question="Provide a company table.",
            policy=SCIENTIFIC_REPORT_POLICY,
        )


@pytest.mark.parametrize(
    ("entities", "fields"),
    [(["A", "A"], ["revenue"]), (["A"], ["revenue", "revenue"])],
)
def test_entity_field_expansion_rejects_duplicate_cell_dimensions(
    entities: list[str], fields: list[str]
) -> None:
    with pytest.raises(ValueError, match="duplicate entity or field"):
        parse_requirement_tool_call(
            _response(
                _payload(
                    _requirement(
                        description="Report {field} for {entity}.",
                        entities=entities,
                        fields=fields,
                    )
                )
            ),
            question="Provide a company table.",
            policy=SCIENTIFIC_REPORT_POLICY,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [
    {'entities': ['sample', 'solvent']},
    {'fields': ['method', 'reason']},
    {'entities': ['sample'], 'fields': ['method', 'reason']},
])
async def test_optional_scope_labels_preserve_individual_scientific_questions(metadata):
    """Regression: valid labeled sub-questions must not become one fallback."""
    payload = _payload(*[
        _requirement(
            description=f'Explain experimental choice {i}.',
            acceptance_criteria=[f'Explain the consequence of choice {i}.'],
            **metadata,
        ) for i in range(1, 6)
    ])
    llm = FakeLLM(_response(payload))
    result = await derive_requirement_set(
        question='Explain the five experimental choices.',
        policy=SCIENTIFIC_REPORT_POLICY, llm=llm,
    )
    assert len(llm.calls) == 1
    assert len(result.requirements) == 5
    for i, requirement in enumerate(result.requirements, 1):
        assert requirement.description == f'Explain experimental choice {i}.'
        assert requirement.acceptance_criteria == (f'Explain the consequence of choice {i}.',)
        assert requirement.sub_obligations
        for labels in metadata.values():
            assert all(label in ' '.join(requirement.sub_obligations) for label in labels)


@pytest.mark.asyncio
async def test_protocol_repair_includes_indexed_error_previous_output_and_full_contract(tmp_path):
    malformed = _payload(_requirement(description='Report {entity}.', entities=['A']))
    corrected = _payload(_requirement(description='Report on A.', entities=['A']))
    llm = FakeLLM(_response(malformed), _response(corrected))
    result = await derive_requirement_set(
        question='Report on A.', policy=SCIENTIFIC_REPORT_POLICY,
        llm=llm, diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert result.requirements[0].description == 'Report on A.'
    second_messages = llm.calls[1][0]
    assert 'Extract a source-neutral' in second_messages[0]['content']
    feedback = second_messages[-1]['content']
    assert 'requirements[0]: entity and field placeholders' in feedback
    assert 'Report {entity}.' in feedback
    assert 'complete corrected contract' in feedback
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert events[0]['error'].startswith('requirements[0]:')
    assert events[-1]['parse_status'] == 'valid'


def test_overflow_keeps_scope_labels_without_inventing_cartesian_rows():
    result = parse_requirement_tool_call(
        _response(_payload(
            _requirement(description='Explain the first part.'),
            _requirement(description='Compare A with B.', entities=['A', 'B'], fields=['stability']),
        )), question='Compare two systems.',
        policy=replace(SCIENTIFIC_REPORT_POLICY, max_requirements=1),
    )
    aggregate = result.requirements[0]
    assert aggregate.sub_obligations == (
        'Explain the first part.', 'Compare A with B.',
        'Entities in scope: A, B', 'Fields in scope: stability',
    )


def test_report_groups_fields_without_losing_requested_table_cells():
    result = parse_requirement_tool_call(_response(_payload(_requirement(
        description="Report {field} for {entity}.", entities=["North", "South"],
        fields=["cost", "units", "uncertainty"],
        acceptance_criteria=["Explain {field} for {entity}."],
    ))), question="Compare North and South in a table.", policy=SCIENTIFIC_REPORT_POLICY)
    assert len(result.requirements) == 2
    for entity, requirement in zip(["North", "South"], result.requirements, strict=True):
        for field in ["cost", "units", "uncertainty"]:
            assert f"Report {field} for {entity}." in requirement.sub_obligations
            assert f"Explain {field} for {entity}." in requirement.acceptance_criteria
    assert all("{" not in c for r in result.requirements for c in r.acceptance_criteria)


def test_exact_duplicates_merge_criteria_but_distinct_formulas_do_not_merge():
    result = parse_requirement_tool_call(_response(_payload(
        _requirement(description="Derive a = b.", acceptance_criteria=["Give units."], priority=3),
        _requirement(description="Derive a = b.", acceptance_criteria=["Give units.", "Check limits."], priority=1),
        _requirement(description="Derive A = b.", acceptance_criteria=["Preserve upper case A."]),
    )), question="Derive a and A.", policy=SCIENTIFIC_REPORT_POLICY)
    assert len(result.requirements) == 2
    assert result.requirements[0].acceptance_criteria == ("Give units.", "Check limits.")
    assert result.requirements[0].priority == 1
    assert result.requirements[1].description == "Derive A = b."
