from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.task_strategies.report_research import (
    finalize as report_finalize,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.evidence import (
    EvidenceCard,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.finalize import (
    ReportFinalizationResult,
    finalize_report,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import (
    Requirement,
    RequirementSet,
    RequirementStatus,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import (
    ReportRuntime,
)


def _tool_response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            {
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            }
        ],
    )


class _LLM:
    model = "test-report-finalizer"
    reasoning_effort = "low"

    def __init__(
        self,
        responses: list[LLMResponse | BaseException],
        *,
        delay_s: float = 0.0,
    ) -> None:
        self.responses = list(responses)
        self.delay_s = delay_s
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, Any]] | None]] = []
        self.premise_calls: list[Any] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001
        if tools and tools[0]["function"]["name"] == "derive_premise_checks":
            # Legacy finalization scenarios script the comparison stage. Keep
            # their successful blind setup explicit and separately inspectable;
            # premise-specific tests below use fully scripted responses.
            self.premise_calls.append((messages, tools))
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            return _tool_response("derive_premise_checks", {"checks": [{
                "question_quote": payload["question"], "constraint": "Respect the public question.",
                "independent_reasoning": "The requested result must follow its stated assumptions.",
                "test": "Check the derivation against the supplied conditions.",
            }]})
        self.calls.append((messages, tools))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _requirement(
    requirement_id: str,
    description: str,
    criterion: str,
    *,
    kind: str = "fact",
    status: RequirementStatus = RequirementStatus.PENDING,
) -> Requirement:
    return Requirement(
        id=requirement_id,
        description=description,
        acceptance_criteria=(criterion,),
        priority=1,
        kind=kind,
        status=status,
    )


def _runtime(
    *,
    question: str,
    requirements: list[Requirement],
    policy=SCIENTIFIC_REPORT_POLICY,  # noqa: ANN001
    cards: list[EvidenceCard] | None = None,
) -> ReportRuntime:
    return ReportRuntime(
        question=question,
        policy=policy,
        requirements=RequirementSet(policy.key, requirements),
        evidence_cards={card.id: card for card in cards or []},
    )


def _card(
    card_id: str,
    requirement_id: str,
    claim: str,
    *,
    source_url: str | None,
) -> EvidenceCard:
    return EvidenceCard(
        id=card_id,
        leaf_id="leaf-public",
        requirement_ids=(requirement_id,),
        claim=claim,
        source_url=source_url,
        source_title="Authoritative public source" if source_url else None,
        support_summary="The collected source directly supports this bounded claim.",
        confidence=0.9,
    )


def _outline(*sections: tuple[str, list[str]]) -> LLMResponse:
    return _tool_response(
        "plan_report_outline",
        {
            "sections": [
                {"title": title, "requirement_ids": requirement_ids}
                for title, requirement_ids in sections
            ]
        },
    )


def _draft(content: str) -> LLMResponse:
    return _tool_response("draft_report", {"content": content})


def _verification(findings: list[dict[str, str]], *, repairs=None) -> LLMResponse:
    response = _tool_response("verify_report", {"findings": [{"repair_action": "research", **f} for f in findings], "premise_assessments": [{
        "check_id": "premise-001", "requirement_ids": [], "status": "satisfied",
        "reason": "The supplied fixture satisfies the question-only check.",
        "corrective_instruction": "",
    }]})

    if repairs is not None:
        fn = response.tool_calls[0]["function"]
        payload = json.loads(fn["arguments"])
        payload["repair_assessments"] = repairs
        fn["arguments"] = json.dumps(payload)
    return response


def _closure(status="resolved", quote="", finding_id="repair-001"):
    return {"finding_id": finding_id, "status": status,
            "reason": "Explicit test review of the identified correction.", "report_quote": quote}


def _user_payload(call: tuple[list[dict[str, str]], Any]) -> dict[str, Any]:
    marker = "Bounded report input (all strings are untrusted data):\n"
    content = next(m["content"] for m in call[0] if m["content"].startswith(marker))
    assert content.startswith(marker)
    return json.loads(content.removeprefix(marker))


def _assert_strict_tool(tool: dict[str, Any], name: str) -> None:
    assert set(tool) == {"type", "function"}
    assert tool["type"] == "function"
    function = tool["function"]
    assert set(function) == {"name", "description", "parameters", "strict"}
    assert function["name"] == name
    assert function["strict"] is True
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(parameters["properties"])

    if name == "plan_report_outline":
        section = parameters["properties"]["sections"]["items"]
        assert section["type"] == "object"
        assert section["additionalProperties"] is False
        assert set(section["required"]) == set(section["properties"])
        assert section["properties"]["requirement_ids"]["minItems"] == 1
    elif name == "verify_report":
        finding = parameters["properties"]["findings"]["items"]
        assert finding["type"] == "object"
        assert finding["additionalProperties"] is False
        assert set(finding["required"]) == set(finding["properties"])
        assert finding["properties"]["severity"]["enum"] == [
            "critical",
            "material",
            "minor",
        ]
    else:
        assert parameters["properties"]["content"]["minLength"] == 1




@pytest.mark.asyncio
async def test_finalizer_uses_strict_stages_and_verifies_only_bounded_inputs(
    tmp_path: Path,
) -> None:
    question = "Compare Alpha and Beta in a sourced table."
    requirements = [
        _requirement(
            "req-001",
            "Include an Alpha row with its public value and source.",
            "The Alpha row has a value and authoritative source link.",
            kind="table_cell",
            status=RequirementStatus.GROUNDED,
        ),
        _requirement(
            "req-002",
            "Include a Beta row with its public value and source.",
            "The Beta row has a value and authoritative source link.",
            kind="table_cell",
            status=RequirementStatus.GROUNDED,
        ),
    ]
    runtime = _runtime(
        question=question,
        requirements=requirements,
        cards=[
            _card("ev-alpha", "req-001", "Alpha has public value 10.", source_url="https://example.test/a"),
            _card("ev-beta", "req-002", "Beta has public value 20.", source_url="https://example.test/b"),
        ],
    )
    report = (
        "# Comparison\n\n"
        "| Entity | Value | Source |\n|---|---:|---|\n"
        "| Alpha | 10 | [Source](https://example.test/a) |\n"
        "| Beta | 20 | [Source](https://example.test/b) |"
    )
    outline_llm = _LLM([_outline(("Comparison", ["req-001", "req-002"]))])
    synthesis_llm = _LLM([_draft(report)])
    verification_llm = _LLM([_verification([])])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="Root synthesis retained both sourced findings.",
        completed_node_summaries=("Alpha memo.", "Beta memo."),
        outline_llm=outline_llm,
        synthesis_llm=synthesis_llm,
        verification_llm=verification_llm,
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result == ReportFinalizationResult(report, "verified", ())
    _assert_strict_tool(outline_llm.calls[0][1][0], "plan_report_outline")
    _assert_strict_tool(synthesis_llm.calls[0][1][0], "draft_report")
    _assert_strict_tool(verification_llm.calls[0][1][0], "verify_report")
    verification_input = _user_payload(verification_llm.calls[0])
    assert set(verification_input) == {
        "question",
        "policy",
        "requirements",
        "evidence_cards",
        "root_summary",
        "completed_node_summaries",
        "outline",
        "draft",
        "independent_premise_checks",
    }
    assert verification_input["question"] == question
    assert [item["id"] for item in verification_input["requirements"]] == [
        "req-001",
        "req-002",
    ]
    assert [item["id"] for item in verification_input["evidence_cards"]] == [
        "ev-alpha",
        "ev-beta",
    ]
    assert verification_input["root_summary"] == "Root synthesis retained both sourced findings."
    assert verification_input["completed_node_summaries"] == ["Alpha memo.", "Beta memo."]


@pytest.mark.asyncio
async def test_policy_delivery_directive_reaches_every_finalization_stage(
    tmp_path: Path,
) -> None:
    question = "Derive the requested rate constant and name the mechanism."
    requirements = [
        _requirement(
            "req-001",
            "Derive the rate constant.",
            "States the value with units.",
        )
    ]
    outline_llm = _LLM([_outline(("Derivation", ["req-001"]))])
    synthesis_llm = _LLM([_draft("# Derivation\n\nk = 1.0 s^-1 via an SN2 mechanism.")])
    verification_llm = _LLM([_verification([])])

    await finalize_report(
        question=question,
        runtime=_runtime(
            question=question,
            requirements=requirements,
            policy=SCIENTIFIC_REPORT_POLICY,
        ),
        root_summary="Root synthesis derived the constant.",
        completed_node_summaries=("Derivation memo.",),
        outline_llm=outline_llm,
        synthesis_llm=synthesis_llm,
        verification_llm=verification_llm,
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    for llm in (outline_llm, synthesis_llm, verification_llm):
        policy_payload = _user_payload(llm.calls[0])["policy"]
        assert policy_payload["delivery_directive"] == (
            SCIENTIFIC_REPORT_POLICY.delivery_directive
        )



@pytest.mark.asyncio
async def test_malformed_outline_falls_back_to_requirement_order(tmp_path: Path) -> None:
    question = "Explain first and second public requirements."
    runtime = _runtime(
        question=question,
        requirements=[
            _requirement("req-001", "First public requirement.", "Explain the first."),
            _requirement("req-002", "Second public requirement.", "Explain the second."),
        ],
    )
    outline_llm = _LLM([LLMResponse(content="prose outline", tool_calls=[])] * 2)
    synthesis_llm = _LLM([_draft("# First\n\nOne.\n\n# Second\n\nTwo.")])
    verification_llm = _LLM([_verification([])])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="",
        completed_node_summaries=(),
        outline_llm=outline_llm,
        synthesis_llm=synthesis_llm,
        verification_llm=verification_llm,
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    draft_input = _user_payload(synthesis_llm.calls[0])
    assert draft_input["outline"] == [
        {"title": "First public requirement.", "requirement_ids": ["req-001"]},
        {"title": "Second public requirement.", "requirement_ids": ["req-002"]},
    ]


@pytest.mark.asyncio
async def test_malformed_first_draft_is_repaired_once(tmp_path: Path) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    valid_draft = "# Supported section\n\nThe bounded evidence supports this section."
    synthesis_llm = _LLM(
        [LLMResponse(content="unstructured prose", tool_calls=[]), _draft(valid_draft)]
    )

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Supported section", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([_verification([])]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result == ReportFinalizationResult(valid_draft, "partial", ())
    assert len(synthesis_llm.calls) == 2
    assert _user_payload(synthesis_llm.calls[1]) == _user_payload(synthesis_llm.calls[0])
    assert "Bounded protocol retry" in synthesis_llm.calls[1][0][-1]["content"]


@pytest.mark.asyncio
async def test_transport_first_draft_is_not_retried_as_a_protocol_error(
    tmp_path: Path,
) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    valid_draft = "# Supported section\n\nThe retry stayed within bounded evidence."
    synthesis_llm = _LLM([RuntimeError("transport unavailable"), _draft(valid_draft)])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Supported section", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([_verification([])]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    assert result.content.startswith("# Partial Research Report")
    assert result.failure_reason == "draft:stage_unavailable"
    assert len(synthesis_llm.calls) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    failed = next(
        event
        for event in events
        if event["event"] == "report_finalization_stage" and event["stage"] == "draft"
    )
    assert failed["status"] == "error"
    assert failed["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_two_failed_draft_attempts_return_deterministic_partial(
    tmp_path: Path,
) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    synthesis_llm = _LLM(
        [
            LLMResponse(content="unstructured prose", tool_calls=[]),
            RuntimeError("retry transport unavailable"),
        ]
    )

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Supported section", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    assert result.content.startswith("# Partial Research Report")
    assert len(synthesis_llm.calls) == 2


@pytest.mark.asyncio
async def test_draft_retry_with_insufficient_deadline_makes_no_extra_llm_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    synthesis_llm = _LLM([LLMResponse(content="unstructured prose", tool_calls=[])])
    stages: list[tuple[str, int]] = []
    original_chat_stage = report_finalize._chat_stage

    async def deadline_exhausting_chat_stage(**kwargs: Any) -> LLMResponse:
        stage = kwargs["stage"]
        stages.append((stage, kwargs["remaining_stage_count"]))
        if stage == "outline":
            return _outline(("Supported section", ["req-001"]))
        assert stage == "draft"
        return await original_chat_stage(**{**kwargs, "question_deadline": 0.0})

    monkeypatch.setattr(report_finalize, "_chat_stage", deadline_exhausting_chat_stage)

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    assert stages == [("outline", 6), ("draft", 5)]
    assert len(synthesis_llm.calls) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(
        event["event"] == "report_finalization_stage"
        and event["stage"] == "draft"
        and event["status"] == "skipped_no_time"
        for event in events
    )


@pytest.mark.asyncio
async def test_malformed_verification_preserves_draft_without_rewrite(tmp_path: Path) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    draft = "# Supported section\n\nThe evidence remains bounded."
    synthesis_llm = _LLM([_draft(draft)])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Supported section", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([LLMResponse(content="looks fine", tool_calls=[])]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.content == draft
    assert result.status == "verification_unavailable"
    assert result.findings == ()
    assert len(synthesis_llm.calls) == 1


@pytest.mark.asyncio
async def test_non_string_verification_severity_preserves_draft_without_rewrite(
    tmp_path: Path,
) -> None:
    question = "Write one supported section."
    runtime = _runtime(
        question=question,
        requirements=[_requirement("req-001", "Supported section.", "Include support.")],
    )
    draft = "# Supported section\n\nThe evidence remains bounded."
    synthesis_llm = _LLM([_draft(draft)])
    invalid_verification = _tool_response(
        "verify_report",
        {
            "findings": [
                {
                    "requirement_id": "req-001",
                    "severity": [],
                    "finding": "Malformed severity must not escape parsing.",
                    "corrective_instruction": "Preserve the draft.",
                }
            ]
        },
    )

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="One evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Supported section", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([invalid_verification, invalid_verification]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result == ReportFinalizationResult(
        draft, "verification_unavailable", (), "verification:invalid_tool_payload")
    saved = json.loads((tmp_path / "report_delivery/result.json").read_text())
    assert saved["failure_reason"] == result.failure_reason
    assert (tmp_path / "report_delivery/final.md").read_text() == draft
    assert len(synthesis_llm.calls) == 1


@pytest.mark.asyncio
async def test_actionable_findings_trigger_exactly_one_corrective_rewrite(
    tmp_path: Path,
) -> None:
    question = "Compare Alpha and Beta in a sourced table."
    runtime = _runtime(
        question=question,
        requirements=[
            _requirement("req-001", "Include Alpha and Beta rows.", "Both rows are present."),
            _requirement("req-002", "Link every public value.", "Every value has a source link."),
        ],
        cards=[
            _card(
                "ev-alpha",
                "req-002",
                "Alpha has collected source support.",
                source_url="https://example.test/a",
            )
        ],
    )
    initial = "# Comparison\n\nAlpha only."
    corrected = (
        "# Comparison\n\n| Entity | Finding | Source |\n|---|---|---|\n"
        "| Alpha | Supported | [A](https://example.test/a) |\n"
        "| Beta | Uncovered | No supported source collected |"
    )
    synthesis_llm = _LLM([_draft(initial), _draft(corrected)])
    findings = [
        {
            "requirement_id": "req-001",
            "severity": "critical",
            "finding": "The requested Beta row is absent.",
            "corrective_instruction": "Add the Beta row and label it uncovered without inventing data.",
        },
        {
            "requirement_id": "req-002",
            "severity": "minor",
            "finding": "The source label could be clearer.",
            "corrective_instruction": "Use descriptive source labels.",
        },
    ]
    diagnostics = DiagnosticRecorder(tmp_path)

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="Partial comparison memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Comparison", ["req-001", "req-002"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM([_verification(findings), _verification([findings[1]], repairs=[_closure(quote=corrected)])]),
        question_deadline=time.monotonic() + 10,
        diagnostics=diagnostics,
    )

    assert result.content == corrected
    assert result.status == "partial"
    assert [finding.severity for finding in result.findings] == ["minor"]
    assert len(synthesis_llm.calls) == 2
    _assert_strict_tool(synthesis_llm.calls[1][1][0], "draft_report")
    rewrite_input = _user_payload(synthesis_llm.calls[1])
    assert rewrite_input["draft"] == initial
    assert rewrite_input["actionable_findings"] == [findings[0]]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    final_event = next(event for event in events if event["event"] == "report_finalization_end")
    assert final_event["rewrite_count"] == 1
    assert final_event["critical_finding_count"] == 0
    assert final_event["minor_finding_count"] == 1


@pytest.mark.asyncio
async def test_deadline_returns_nonempty_supported_partial_with_uncovered_sections(
    tmp_path: Path,
) -> None:
    question = "Report Alpha and Beta with sources."
    runtime = _runtime(
        question=question,
        requirements=[
            _requirement(
                "req-001",
                "Report Alpha.",
                "Cite Alpha.",
                status=RequirementStatus.GROUNDED,
            ),
            _requirement("req-002", "Report Beta.", "Cite Beta."),
        ],
        cards=[
            _card(
                "ev-alpha",
                "req-001",
                "Alpha is supported by the public record.",
                source_url="https://example.test/alpha",
            )
        ],
    )
    outline_llm = _LLM(
        [_outline(("Alpha", ["req-001"]), ("Beta", ["req-002"]))],
        delay_s=0.05,
    )
    synthesis_llm = _LLM([_draft("unexpected complete draft")])
    verification_llm = _LLM([_verification([])])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="Root retained a partial memo.",
        completed_node_summaries=("Completed public evidence memo.",),
        outline_llm=outline_llm,
        synthesis_llm=synthesis_llm,
        verification_llm=verification_llm,
        question_deadline=time.monotonic() + 0.02,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    assert result.content.strip()
    assert "Alpha is supported by the public record." in result.content
    assert "https://example.test/alpha" in result.content
    assert "Report Beta." in result.content
    assert "Uncovered" in result.content
    assert not synthesis_llm.calls
    assert not verification_llm.calls


@pytest.mark.asyncio
async def test_report_rewrite_restores_explicitly_requested_table_and_source_links(tmp_path: Path) -> None:
    question = "Enumerate the organizations in a table with source links."
    runtime = _runtime(
        question=question,
        requirements=[
            _requirement(
                "req-001",
                "Provide the requested organization table.",
                "Every requested organization has a row and source link.",
                kind="table",
                status=RequirementStatus.GROUNDED,
            )
        ],
        cards=[
            _card(
                "ev-org",
                "req-001",
                "Organization One is in the requested set.",
                source_url="https://example.test/org-one",
            )
        ],
    )
    corrected = (
        "# Organizations\n\n| Organization | Finding | Source |\n|---|---|---|\n"
        "| Organization One | Requested entry | [Public source](https://example.test/org-one) |"
    )
    synthesis_llm = _LLM([_draft("# Organizations\n\nOrganization One."), _draft(corrected)])

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary="Organization evidence memo.",
        completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Organizations", ["req-001"]))]),
        synthesis_llm=synthesis_llm,
        verification_llm=_LLM(
            [
                _verification(
                    [
                        {
                            "requirement_id": "req-001",
                            "severity": "material",
                            "finding": "The requested table and source link are absent.",
                            "corrective_instruction": "Render the enumerated row as a Markdown table with its collected source URL.",
                        }
                    ]
                ),
                _verification([], repairs=[_closure(quote=corrected)]),
            ]
        ),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "partial"
    assert "| Organization | Finding | Source |" in result.content
    assert "https://example.test/org-one" in result.content








@pytest.mark.asyncio
async def test_fsr_draft_preserves_subquestions_equations_methods_and_assumptions(
    tmp_path: Path,
) -> None:
    question = "Derive the speed and name the numerical method and assumptions."
    runtime = _runtime(
        question=question,
        policy=SCIENTIFIC_REPORT_POLICY,
        requirements=[
            _requirement(
                "req-001",
                "Derive the speed with units.",
                "Show equation and units.",
                status=RequirementStatus.GROUNDED,
            ),
            _requirement(
                "req-002",
                "Name the method and assumptions.",
                "State method and assumptions.",
                status=RequirementStatus.GROUNDED,
            ),
        ],
        cards=[
            _card(
                "ev-speed",
                "req-001",
                (
                    "The supported derivation is v=d/t with d=10 m and t=2 s, "
                    "giving v=5 m s^-1."
                ),
                source_url=None,
            ),
            _card(
                "ev-method",
                "req-002",
                (
                    "The supported named method is fourth-order Runge–Kutta (RK4), "
                    "under the stated constant-distance and elapsed-time assumption."
                ),
                source_url=None,
            ),
        ],
    )
    report = (
        "# 1. Speed derivation\n\n"
        "Using $v = d/t$, $v = 10\\,\\mathrm{m}/2\\,\\mathrm{s} = 5\\,\\mathrm{m\\,s^{-1}}$.\n\n"
        "# 2. Method and assumptions\n\n"
        "Named method: fourth-order Runge–Kutta (RK4). Assumption: constant distance and elapsed time."
    )

    result = await finalize_report(
        question=question,
        runtime=runtime,
        root_summary=(
            "Scientific memo derives v=d/t from 10 m and 2 s and reports 5 m s^-1."
        ),
        completed_node_summaries=(
            "Method memo names RK4 and states constant distance and elapsed time.",
        ),
        outline_llm=_LLM(
            [
                _outline(
                    ("1. Speed derivation", ["req-001"]),
                    ("2. Method and assumptions", ["req-002"]),
                )
            ]
        ),
        synthesis_llm=_LLM([_draft(report)]),
        verification_llm=_LLM([_verification([])]),
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
    )

    assert result.status == "verified"
    assert "# 1. Speed derivation" in result.content
    assert "$v = d/t$" in result.content
    assert "\\mathrm{m\\,s^{-1}}" in result.content
    assert "fourth-order Runge–Kutta (RK4)" in result.content
    assert "Assumption:" in result.content



@pytest.mark.parametrize('url', [
    'https://www.cell.com/cell/fulltext/S0092-8674(21)00003-9',
    'https://example.test/a_(b_(c))',
    'https://example.test/a%28b%29',
])
@pytest.mark.parametrize('template', [
    '[paper]({url})',
    '[paper]({url} "Paper title")',
    '[paper](<{url}>)',
    '![figure]({url})',
    '[paper][1]\n[1]: {url} "Paper title"',
    '<a href="{url}">paper</a>',
    '<a href={url}>paper</a>',
    '<{url}>',
    'Source: {url}.',
    '(see {url}).',
])
def test_parenthesized_observed_citations_preserve_complete_report(url, template):
    runtime = _runtime(
        question='Write a sourced scientific report.', policy=SCIENTIFIC_REPORT_POLICY,
        requirements=[_requirement('req-001', 'Explain the result.', 'Cite the paper.')],
        cards=[_card('ev-1', 'req-001', 'Observed paper.', source_url=url)],
    )
    report = '# Full report\n\nDerivation and results.\n' + template.format(url=url)
    assert report_finalize._parse_draft(_draft(report)) == report





@pytest.mark.asyncio
async def test_fsr_parenthesized_citation_draft_reaches_verification_without_retry(tmp_path):
    url = 'https://www.cell.com/cell/fulltext/S0092-8674(21)00003-9'
    runtime = _runtime(
        # Scientific report delivery must preserve complete parenthesized citations.
        question='Explain the scientific result.', policy=SCIENTIFIC_REPORT_POLICY,
        requirements=[_requirement('req-001', 'Explain.', 'Cite.', status=RequirementStatus.GROUNDED)],
        cards=[_card('ev-1', 'req-001', 'Supported result.', source_url=url)],
    )
    content = '# Complete report\n\nSupported result. [Paper](' + url + ')'
    outline = _LLM([_outline(('Result', ['req-001']))])
    synthesis = _LLM([_draft(content)])
    verification = _LLM([_verification([])])
    result = await finalize_report(
        question=runtime.question, runtime=runtime, root_summary='',
        completed_node_summaries=[], outline_llm=outline, synthesis_llm=synthesis,
        verification_llm=verification, question_deadline=time.monotonic() + 30,
        diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert result.content == content
    assert result.status == 'verified'
    assert len(synthesis.calls) == 1
    assert len(verification.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_action", ["rewrite", "research"])
async def test_rewrite_reverifies_with_updated_evidence_and_retains_remaining_findings(
    tmp_path, repair_action,
):
    runtime = _runtime(question="Find a sourced value", requirements=[
        _requirement("req-001", "Find the value", "Cite its source")], cards=[])
    finding = {"requirement_id": "req-001", "severity": "material",
               "finding": "Value needs checking", "corrective_instruction": "Check public source",
               "repair_action": repair_action}
    repairs = []

    async def repair(findings, draft, cutoff):
        repairs.append(findings)
        assert cutoff < deadline
        assert draft == "# Initial\nUncertain value"
        runtime.evidence_cards["new"] = _card("new", "req-001", "Value is 42",
                                             source_url="https://example.test/new")
        return ("Scientific repair calculation memo",)

    rewritten = "# Result\nValue is still uncertain"
    if repair_action == "research":
        rewritten += " [Source](https://example.test/new)"
    synthesis = _LLM([_draft("# Initial\nUncertain value"), _draft(rewritten)])
    verifier = _LLM([_verification([finding]), _verification([finding], repairs=[_closure("unresolved")])])
    deadline = time.monotonic() + 10
    result = await finalize_report(
        question=runtime.question, runtime=runtime, root_summary="Old root",
        completed_node_summaries=(), outline_llm=_LLM([_outline(("Result", ["req-001"]))]),
        synthesis_llm=synthesis, verification_llm=verifier, question_deadline=deadline,
        diagnostics=DiagnosticRecorder(tmp_path), research_repair=repair,
    )
    assert len(repairs) == (1 if repair_action == "research" else 0)
    assert result.status == "partial" and result.findings[0].severity == "material"
    assert len(verifier.calls) == 2
    assert _user_payload(verifier.calls[1])["draft"] == rewritten
    if repair_action == "research":
        updated = _user_payload(synthesis.calls[1])
        assert updated["evidence_cards"][0]["source_url"] == "https://example.test/new"
        assert updated["research_check_memos"] == ["Scientific repair calculation memo"]


@pytest.mark.asyncio
async def test_repair_exception_preserves_draft_and_still_attempts_reverification(tmp_path):
    runtime = _runtime(question="Check a calculation", policy=SCIENTIFIC_REPORT_POLICY,
                       requirements=[_requirement("req-001", "Compute", "Show arithmetic")])
    finding = {"requirement_id": "req-001", "severity": "critical",
               "finding": "Calculation unsupported", "corrective_instruction": "Calculate"}

    async def repair(*_):
        raise RuntimeError("Synthetic repair failure")

    result = await finalize_report(
        question=runtime.question, runtime=runtime, root_summary="", completed_node_summaries=(),
        outline_llm=_LLM([_outline(("Calculation", ["req-001"]))]),
        synthesis_llm=_LLM([_draft("# Draft"), _draft("# Draft\nCalculation remains unverified")]),
        verification_llm=_LLM([_verification([finding]), RuntimeError("Synthetic verifier failure")]),
        question_deadline=time.monotonic() + 10, diagnostics=DiagnosticRecorder(tmp_path),
        research_repair=repair,
    )
    assert result.content == "# Draft\nCalculation remains unverified"
    assert result.status == "verification_unavailable"
    assert result.findings[0].severity == "critical"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_action", ["research", "rewrite"])
async def test_review_withdraws_disputed_coverage_even_without_research_budget(tmp_path, repair_action):
    requirement = _requirement("req-001", "Compute quantity", "Show correct units",
                               status=RequirementStatus.GROUNDED)
    card = _card("old-evidence", "req-001", "Original unverified value", source_url=None)
    runtime = _runtime(question="Compute quantity", requirements=[requirement], cards=[card])
    requirement.evidence_ids = [card.id]
    requirement.covered_criterion_ids = ["req-001-criterion-001"]
    finding = {"requirement_id": "req-001", "severity": "critical",
               "finding": "The units need an independent check",
               "corrective_instruction": "Check units", "repair_action": repair_action}
    synthesis = _LLM([_draft("# Initial\nOriginal value"), _draft("# Revised\nConditional result")])
    result = await finalize_report(
        question=runtime.question, runtime=runtime, root_summary="Old result",
        completed_node_summaries=(), outline_llm=_LLM([_outline(("Result", ["req-001"]))]),
        synthesis_llm=synthesis,
        verification_llm=_LLM([_verification([finding]), _verification([finding], repairs=[_closure("unresolved")])]),
        question_deadline=time.monotonic() + 10, diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert result.status == "partial"
    payload = _user_payload(synthesis.calls[1])
    if repair_action == "research":
        assert requirement.status is RequirementStatus.PENDING
        assert requirement.evidence_ids == []
        assert payload["evidence_cards"][0]["disputed_requirement_ids"] == ["req-001"]
    else:
        assert requirement.status is RequirementStatus.GROUNDED
        assert payload["evidence_cards"][0]["disputed_requirement_ids"] == []


def test_partial_fallback_does_not_present_disputed_evidence_as_supported():
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.finalize import (
        _deterministic_partial,
    )
    requirement = _requirement("req-001", "Compute quantity", "Show units",
                               status=RequirementStatus.GROUNDED)
    runtime = _runtime(question="Compute", requirements=[requirement], cards=[
        _card("old", "req-001", "Disputed numerical value", source_url=None),
    ])
    runtime.reopen_requirement("req-001", "Independent unit check failed")
    content = _deterministic_partial(runtime=runtime, root_summary="", completed_node_summaries=())
    assert "0 of 1" in content
    assert "Uncovered" in content
    assert "Disputed numerical value" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize('coverage, expected', [
    (RequirementStatus.PENDING, 'partial'),
    (RequirementStatus.EXHAUSTED, 'partial'),
    (RequirementStatus.GROUNDED, 'verified'),
])
async def test_empty_verifier_findings_cannot_override_uncovered_requirements(tmp_path, coverage, expected):
    runtime = _runtime(
        question='State the justified result.',
        requirements=[_requirement('req-001', 'State the result.', 'Justify it.', status=coverage)],
    )
    result = await finalize_report(
        question=runtime.question, runtime=runtime, root_summary='', completed_node_summaries=(),
        outline_llm=_LLM([_outline(('Result', ['req-001']))]),
        synthesis_llm=_LLM([_draft('# Result\n\nA conditional result.')]),
        verification_llm=_LLM([_verification([])]),
        question_deadline=time.monotonic() + 5, diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert result.status == expected


# These scenarios script every call, including the independent first pass.
class _PremiseLLM(_LLM):
    async def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _premises(quote="Calculate six times seven."):
    return {"checks": [{
        "question_quote": quote,
        "constraint": "Use the operands stated in the question.",
        "independent_reasoning": "Six groups of seven give 7 + 7 + 7 + 7 + 7 + 7 = 42.",
        "test": "Check the proposed answer against repeated addition of seven six times.",
    }]}


def _assessment(status="satisfied", **overrides):
    return {
        "check_id": "premise-001", "requirement_ids": ["req-001"],
        "status": status, "reason": "Independent arithmetic comparison.",
        "corrective_instruction": "Recompute the sum." if status in {"violated", "unresolved"} else "",
        **overrides,
    }


async def _premise_scenario(tmp_path, verifier, *, question="Calculate six times seven.", repair=None):
    runtime = _runtime(question=question, requirements=[_requirement(
        "req-001", "PLANNER_ONLY assumption", "PLANNER_ONLY criterion",
        status=RequirementStatus.GROUNDED,
    )], cards=[_card("ev-private", "req-001", "EVIDENCE_ONLY", source_url=None)])
    runtime.requirements.requirements[0].evidence_ids = ["ev-private"]
    runtime.requirements.requirements[0].covered_criterion_ids = ["criterion-001"]
    synthesis = _LLM([_draft("# Draft\nDRAFT_ONLY: 41."), _draft("# Repaired\n42.")])

    async def callback(findings, draft, cutoff):
        return await repair(runtime, findings, draft, cutoff)

    result = await finalize_report(
        question=question, runtime=runtime, root_summary="ROOT_ONLY",
        completed_node_summaries=("MEMO_ONLY",),
        outline_llm=_LLM([_outline(("OUTLINE_ONLY", ["req-001"]))]),
        synthesis_llm=synthesis, verification_llm=verifier,
        question_deadline=time.monotonic() + 10,
        diagnostics=DiagnosticRecorder(tmp_path),
        research_repair=callback if repair else None,
    )
    return result, runtime, synthesis


@pytest.mark.asyncio
async def test_independent_pass_sees_only_full_public_question_and_access_policy(tmp_path):
    # A decisive condition beyond the ordinary report context truncation survives.
    question = "Calculate six times seven. " + "x" * 31000 + " FINAL_PUBLIC_CONDITION"
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises("FINAL_PUBLIC_CONDITION")),
        _verification([]),
    ])
    result, _, _ = await _premise_scenario(tmp_path, verifier, question=question)
    assert result.status == "verified"
    blind, comparison = verifier.calls
    assert len(blind[0]) == 2
    assert [m["role"] for m in blind[0]] == ["system", "user"]
    assert _user_payload(blind) == {
        "question": question, "policy": {"network_enabled": SCIENTIFIC_REPORT_POLICY.network_enabled},
    }
    for secret in ("PLANNER_ONLY", "EVIDENCE_ONLY", "ROOT_ONLY", "MEMO_ONLY", "OUTLINE_ONLY", "DRAFT_ONLY"):
        assert secret not in json.dumps(blind)
    assert blind[1][0]["function"]["name"] == "derive_premise_checks"
    checks = json.loads((tmp_path / "report_delivery/premise-checks.json").read_text())
    assert _user_payload(comparison)["independent_premise_checks"] == checks
    assert checks[0]["question_quote"] == "FINAL_PUBLIC_CONDITION"
    assert (tmp_path / "report_delivery/premise-assessments.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["violated", "unresolved"])
@pytest.mark.parametrize("disposition", ["satisfied", "not_applicable"])
async def test_premise_failure_forces_research_and_same_checks_are_reverified(tmp_path, status, disposition):
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises()),
        _tool_response("verify_report", {"findings": [], "premise_assessments": [_assessment(status)]}),
        _tool_response("verify_report", {"findings": [], "premise_assessments": [
            _assessment(disposition, reason="The independent check is refuted by an explicit conditional derivation.")
        ], "repair_assessments": [_closure(quote="# Repaired\n42.")]}),
    ])
    repairs = []

    async def repair(runtime, findings, draft, cutoff):
        repairs.append(findings)
        requirement = runtime.requirements.requirements[0]
        assert requirement.status is RequirementStatus.PENDING
        assert requirement.evidence_ids == []
        assert requirement.covered_criterion_ids == []
        assert requirement.disputed_evidence_ids == ["ev-private"]
        assert findings[0].severity == "critical"
        assert findings[0].repair_action == "research"
        assert "repeated addition" in findings[0].corrective_instruction
        assert draft == "# Draft\nDRAFT_ONLY: 41."
        assert time.monotonic() < cutoff < time.monotonic() + 10
        # Emulate the research worker grounding the reopened requirement.
        requirement.transition_to(RequirementStatus.GROUNDED)
        runtime.evidence_cards["ev-repaired"] = _card("ev-repaired", "req-001", "NEW_EVIDENCE", source_url=None)
        return ("Independent recomputation returned 42.",)

    result, _, synthesis = await _premise_scenario(tmp_path, verifier, repair=repair)
    assert result.status == "verified"
    assert result.content == "# Repaired\n42."
    assert len(repairs) == 1
    assert [call[1][0]["function"]["name"] for call in verifier.calls] == [
        "derive_premise_checks", "verify_report", "verify_report",
    ]
    comparison, recheck = map(_user_payload, verifier.calls[1:])
    assert recheck["independent_premise_checks"] == comparison["independent_premise_checks"]
    assert recheck["draft"] == result.content
    assert "NEW_EVIDENCE" in json.dumps(recheck["evidence_cards"])
    assert _user_payload(synthesis.calls[-1])["research_check_memos"] == ["Independent recomputation returned 42."]
    assert (tmp_path / "report_repair/premise-assessments.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("assessments", [
    [], [_assessment(check_id="unknown")],
    [_assessment(), _assessment()], [_assessment("unresolved", requirement_ids=[])],
    [_assessment("violated", corrective_instruction="")],
    [_assessment(requirement_ids=["unknown"])], [_assessment(reason=" ")],
    [_assessment(status="ignored")],
])
async def test_incomplete_or_invalid_premise_assessments_cannot_verify(tmp_path, assessments):
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises()),
        _tool_response("verify_report", {"findings": [], "premise_assessments": assessments}),
    ])
    result, _, synthesis = await _premise_scenario(tmp_path, verifier)
    assert result.status == "verification_unavailable"
    assert result.content == "# Draft\nDRAFT_ONLY: 41."
    assert len(synthesis.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {}, {"checks": []}, _premises("Not in the public question"),
    {"checks": _premises()["checks"] * 13},
    {"checks": [{**_premises()["checks"][0], "test": ""}]},
    {"checks": [{**_premises()["checks"][0], "unexpected": "field"}]},
])
async def test_invalid_independent_pass_preserves_draft_without_comparison(tmp_path, payload):
    verifier = _PremiseLLM([_tool_response("derive_premise_checks", payload)] * 2)
    result, _, _ = await _premise_scenario(tmp_path, verifier)
    assert result.status == "verification_unavailable"
    assert result.failure_reason.startswith("independent_premises:")
    assert result.content == "# Draft\nDRAFT_ONLY: 41."
    assert len(verifier.calls) == 2


@pytest.mark.asyncio
async def test_independent_pass_cancellation_propagates(tmp_path):
    with pytest.raises(asyncio.CancelledError):
        await _premise_scenario(tmp_path, _PremiseLLM([asyncio.CancelledError()]))


@pytest.mark.asyncio
async def test_independent_pass_obeys_stage_deadline_and_preserves_draft(tmp_path, monkeypatch):
    original = report_finalize._chat_stage

    async def short_deadline(**kwargs):
        if kwargs["stage"] == "independent_premises":
            kwargs["question_deadline"] = time.monotonic() + 0.04
        return await original(**kwargs)

    monkeypatch.setattr(report_finalize, "_chat_stage", short_deadline)
    result, _, _ = await _premise_scenario(tmp_path, _PremiseLLM([], delay_s=1))
    assert result.status == "verification_unavailable"
    assert result.content == "# Draft\nDRAFT_ONLY: 41."
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e.get("stage") == "independent_premises" and e.get("status") == "timeout" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_assessments", [[], [_assessment(check_id="unknown")]])
async def test_verification_retries_missing_or_unknown_check_before_accepting(tmp_path, bad_assessments):
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises()),
        _tool_response("verify_report", {"findings": [], "premise_assessments": bad_assessments}),
        _verification([]),
    ])
    result, _, synthesis = await _premise_scenario(tmp_path, verifier)
    assert result.status == "verified"
    assert len(verifier.calls) == 3
    assert len(synthesis.calls) == 1  # Protocol correction does not dispatch research/rewrite.
    assert verifier.calls[2][0][:-1] == verifier.calls[1][0]
    assert "Validation error:" in verifier.calls[2][0][-1]["content"]


@pytest.mark.asyncio
async def test_escaped_math_quote_reaches_verification_with_original_excerpt(tmp_path):
    question = r"Use \\(b\_\\omega\\) in the derivation."
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises(r"\(b_\omega\)")),
        _verification([]),
    ])
    result, _, _ = await _premise_scenario(tmp_path, verifier, question=question)
    assert result.status == "verified"
    assert len(verifier.calls) == 2
    comparison = _user_payload(verifier.calls[1])
    quote = comparison["independent_premise_checks"][0]["question_quote"]
    assert quote == r"\\(b\_\\omega\\)"
    assert quote in question


@pytest.mark.parametrize("assessments", [
    [], [_closure(finding_id="unknown", quote="Fixed")],
    [_closure(quote="Not present in the revision")],
    [_closure("resolved")],
    [{**_closure(quote="Fixed"), "reason": ""}],
])
def test_reverification_rejects_missing_or_unsubstantiated_closure(assessments):
    finding = report_finalize.VerificationFinding("req-001", "critical", "Wrong sign", "Fix the sign")
    response = _tool_response("verify_report", {"findings": [], "repair_assessments": assessments})
    with pytest.raises(ValueError):
        report_finalize._parse_verification(response, [_requirement("req-001", "Derive result", "Check sign")],
                                            repair_findings=[finding], draft="Fixed")


def test_unresolved_repair_cannot_disappear_or_be_downgraded_to_minor():
    finding = report_finalize.VerificationFinding("req-001", "critical", "Wrong sign", "Fix the sign")
    response = _tool_response("verify_report", {"findings": [{
        "requirement_id": "req-001", "severity": "minor", "finding": "A sign caveat remains",
        "corrective_instruction": "Mention the caveat", "repair_action": "rewrite",
    }], "repair_assessments": [_closure("unresolved")]})
    parsed = report_finalize._parse_verification(
        response, [_requirement("req-001", "Derive result", "Check sign")],
        repair_findings=[finding], draft="Still wrong")
    assert finding in parsed
    assert any(f.severity == "critical" for f in parsed)


@pytest.mark.parametrize("status", ["resolved", "refuted"])
def test_review_can_close_a_fix_or_refute_an_incorrect_finding(status):
    finding = report_finalize.VerificationFinding("req-001", "material", "Check sign", "Recompute")
    response = _tool_response("verify_report", {"findings": [], "repair_assessments": [
        _closure(status, quote="The result is positive." if status == "resolved" else ""),
    ]})
    assert report_finalize._parse_verification(
        response, [_requirement("req-001", "Derive result", "Check sign")],
        repair_findings=[finding], draft="# Result\nThe result is positive.") == ()


def test_repair_ids_must_cover_distinct_findings_even_for_the_same_requirement():
    findings = [report_finalize.VerificationFinding("req-001", "material", text, "Fix it")
                for text in ["Wrong sign", "Wrong unit"]]
    response = _tool_response("verify_report", {"findings": [], "repair_assessments": [
        _closure(quote="Fixed"), _closure(quote="Fixed"),
    ]})
    with pytest.raises(ValueError, match="repeated"):
        report_finalize._parse_verification(response, [_requirement("req-001", "Derive result", "Check")],
                                            repair_findings=findings, draft="Fixed")


@pytest.mark.asyncio
async def test_reverification_corrects_missing_ledger_and_preserves_ids(tmp_path):
    finding = {"requirement_id": "req-001", "severity": "material", "finding": "Wrong sum",
               "corrective_instruction": "Use the computed sum", "repair_action": "rewrite"}
    verifier = _PremiseLLM([
        _tool_response("derive_premise_checks", _premises()),
        _verification([finding]),
        _verification([]),  # Invalid: omission is not resolution.
        _verification([], repairs=[_closure(quote="42.")]),
    ])
    result, _, synthesis = await _premise_scenario(tmp_path, verifier)
    assert result.status == "verified"
    assert len(synthesis.calls) == 2
    first = _user_payload(verifier.calls[2])
    retry = _user_payload(verifier.calls[3])
    assert first["repair_findings"] == retry["repair_findings"]
    assert first["repair_findings"][0]["finding_id"] == "repair-001"
    saved = json.loads((tmp_path / "report_repair/repair-assessments.json").read_text())
    assert saved[0]["report_quote"] == "42."
