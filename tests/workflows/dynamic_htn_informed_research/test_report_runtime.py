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
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    ExecutionBudget,
    Orchestrator,
    OrchestratorRequest,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies.report_research.evidence import (
    _EVIDENCE_TOOL,
    EvidenceCard,
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


class _SequenceLLM:
    model = "test"

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, Any]] | None]] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001
        self.calls.append((messages, tools))
        if not self.responses:
            raise AssertionError("deterministic local behavior must not call the LLM")
        return self.responses.pop(0)


_OBSERVED_REVENUE_INDEX = (
    '- [exact_observed] "Revenue filing" — https://example.test/revenue'
)


def _requirement(
    requirement_id: str,
    description: str,
    criterion: str,
    *,
    priority: int = 1,
    owner: str | None = None,
    status: RequirementStatus = RequirementStatus.PENDING,
    evidence_ids: list[str] | None = None,
) -> Requirement:
    return Requirement(
        id=requirement_id,
        description=description,
        acceptance_criteria=(criterion,),
        priority=priority,
        kind="fact",
        owner_leaf_id=owner,
        status=status,
        evidence_ids=list(evidence_ids or []),
    )


def _runtime(*requirements: Requirement) -> ReportRuntime:
    return ReportRuntime(
        question="Compare annual revenue and headquarters location.",
        policy=SCIENTIFIC_REPORT_POLICY,
        requirements=RequirementSet(
            policy_key="scientific_report",
            requirements=list(requirements),
        ),
    )


def _tree_with_leaves(*leaves: HTNTaskNode) -> HTNTree:
    root = HTNTaskNode(
        id="root",
        title="Comparison",
        task="Compare annual revenue and headquarters location.",
        children=[leaf.id for leaf in leaves],
    )
    normalized = [leaf.model_copy(update={"parent_id": "root"}) for leaf in leaves]
    return HTNTree(
        root_id="root",
        nodes={"root": root, **{leaf.id: leaf for leaf in normalized}},
    )


def _evidence_response(*cards: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            {
                "function": {
                    "name": "extract_evidence",
                    "arguments": json.dumps({"cards": list(cards), "challenges": []}),
                }
            }
        ],
    )


def _card_payload(
    *,
    record_id: str,
    requirement_id: str,
    criterion: str | None = None,
    criterion_id: str | None = None,
    claim: str = "Revenue was USD 12 million in 2025.",
    source_url: str | None = "https://example.test/revenue",
    source_title: str | None = "Revenue filing",
    support_summary: str = "The filing reports USD 12 million for 2025.",
    confidence: float = 0.91,
) -> dict[str, Any]:
    support = {"requirement_id": requirement_id}
    if criterion_id is not None:
        support["acceptance_criterion_id"] = criterion_id
    elif criterion is not None:
        support["acceptance_criterion"] = criterion
    else:  # pragma: no cover - test fixture misuse
        raise ValueError("criterion or criterion_id is required")
    return {
        "record_id": record_id,
        "supports": [support],
        "claim": claim,
        "source_url": source_url,
        "source_title": source_title,
        "support_summary": support_summary,
        "confidence": confidence,
    }


def _record(
    agent_id: str,
    summary: str,
    *,
    quality: str = "submitted",
    submitted: bool = True,
    observed_source_index: str | None = None,
) -> AgentResultRecord:
    return AgentResultRecord(
        agent_id=agent_id,
        role_id="dynir_report_researcher",
        task_summary="Research one report requirement.",
        result_summary=summary,
        execution_quality=quality,  # type: ignore[arg-type]
        submitted=submitted,
        observed_source_index=observed_source_index,
    )


@pytest.mark.asyncio
async def test_ownership_is_semantic_repeatable_and_uses_leaf_id_for_ties() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
        ),
        _requirement(
            "req-002",
            "Report headquarters city and country.",
            "Give the headquarters location.",
        ),
        _requirement(
            "req-003",
            "Summarize the shared market context.",
            "Give the market context.",
        ),
    )
    revenue = HTNTaskNode(
        id="leaf-z",
        title="Revenue",
        task="Find annual revenue in USD and its year. Summarize the shared market context.",
    )
    location = HTNTaskNode(
        id="leaf-a",
        title="Headquarters",
        task="Find headquarters city and country. Summarize the shared market context.",
    )
    first_tree = _tree_with_leaves(revenue, location)
    reversed_tree = _tree_with_leaves(location, revenue)
    llm = _SequenceLLM()

    await runtime.refresh_ownership(first_tree, llm)
    first = {
        requirement.id: requirement.owner_leaf_id
        for requirement in runtime.requirements.requirements
    }
    await runtime.refresh_ownership(reversed_tree, llm)
    second = {
        requirement.id: requirement.owner_leaf_id
        for requirement in runtime.requirements.requirements
    }

    assert first == {
        "req-001": "leaf-z",
        "req-002": "leaf-a",
        "req-003": "leaf-a",
    }
    assert second == first
    assert llm.calls == []


@pytest.mark.asyncio
async def test_ownership_refresh_after_decomposition_preserves_requirement_ids() -> None:
    runtime = _runtime(
        _requirement(
            "req-stable-revenue",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
        ),
        _requirement(
            "req-stable-location",
            "Report headquarters city and country.",
            "Give the headquarters location.",
        ),
    )
    root = HTNTaskNode(
        id="root",
        task="Compare annual revenue and headquarters location.",
        needs_decomposition=True,
    )
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _SequenceLLM()

    await runtime.refresh_ownership(tree, llm)
    assert [requirement.owner_leaf_id for requirement in runtime.requirements.requirements] == [
        None,
        None,
    ]

    tree.decompose(
        "root",
        [
            HTNTaskNode(
                id="leaf-revenue",
                task="Find annual revenue in USD and its year.",
            ),
            HTNTaskNode(
                id="leaf-location",
                task="Find headquarters city and country.",
            ),
        ],
    )
    await runtime.refresh_ownership(tree, llm)

    assert [requirement.id for requirement in runtime.requirements.requirements] == [
        "req-stable-revenue",
        "req-stable-location",
    ]
    assert [requirement.owner_leaf_id for requirement in runtime.requirements.requirements] == [
        "leaf-revenue",
        "leaf-location",
    ]


@pytest.mark.asyncio
async def test_unrelated_requirements_remain_unassigned_for_recovery() -> None:
    runtime = _runtime(
        *[
            _requirement(
                f"req-{index:03d}",
                f"Obligation {index} with unrelated vocabulary.",
                f"Criterion {index}.",
            )
            for index in range(1, 9)
        ]
    )
    tree = _tree_with_leaves(
        HTNTaskNode(id="leaf-a", task="Alpha topic."),
        HTNTaskNode(id="leaf-b", task="Beta topic."),
    )

    await runtime.refresh_ownership(tree, _SequenceLLM())

    owners = [item.owner_leaf_id for item in runtime.requirements.requirements]
    assert owners == [None] * 8
    assert len(runtime.uncovered_requirements()) == 8


@pytest.mark.asyncio
async def test_ownership_preserves_completed_scope_and_excludes_failed_leaves() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
        )
    )
    tree = _tree_with_leaves(
        HTNTaskNode(
            id="leaf-completed-best-match",
            task="Find annual revenue in USD and its year.",
            status=HTNTaskStatus.COMPLETED,
        ),
        HTNTaskNode(
            id="leaf-failed-best-match",
            task="Verify annual revenue in USD and its year.",
            status=HTNTaskStatus.FAILED,
        ),
        HTNTaskNode(
            id="leaf-pending-frontier",
            task="Complete the remaining report delivery work.",
        ),
    )

    assert [leaf.id for leaf in tree.ready_for_execution()] == ["leaf-pending-frontier"]

    await runtime.refresh_ownership(tree, _SequenceLLM())

    assert runtime.requirements.requirements[0].owner_leaf_id == "leaf-completed-best-match"


@pytest.mark.asyncio
async def test_observe_leaf_does_not_blanket_bind_owned_requirements() -> None:
    revenue_criterion = "Give the annual revenue and year."
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            revenue_criterion,
            owner="leaf-revenue",
        ),
        _requirement(
            "req-002",
            "Report revenue growth percentage.",
            "Give the year-over-year growth percentage.",
            owner="leaf-revenue",
        ),
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion=revenue_criterion,
                )
            )
        ]
    )
    leaf = HTNTaskNode(
        id="leaf-revenue",
        task="Find annual revenue and revenue growth.",
    )

    await runtime.observe_leaf(
        leaf,
        [
            _record(
                "worker-good",
                "The filing reports revenue, but not growth.",
                observed_source_index=_OBSERVED_REVENUE_INDEX,
            )
        ],
        llm,
    )

    first, second = runtime.requirements.requirements
    assert first.status is RequirementStatus.GROUNDED
    assert len(first.evidence_ids) == 1
    assert second.status is RequirementStatus.PENDING
    assert second.evidence_ids == []
    card = runtime.evidence_cards[first.evidence_ids[0]]
    assert card.requirement_ids == ("req-001",)


@pytest.mark.asyncio
async def test_requirement_is_grounded_only_after_every_acceptance_criterion() -> None:
    requirement = Requirement(
        id="req-001",
        description="Report revenue with year and currency.",
        acceptance_criteria=(
            "Give the annual revenue and year.",
            "State the reported currency.",
        ),
        priority=1,
        kind="fact",
        owner_leaf_id="leaf-revenue",
    )
    runtime = _runtime(requirement)
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-year",
                    requirement_id="req-001",
                    criterion_id="req-001-criterion-001",
                )
            ),
            _evidence_response(
                _card_payload(
                    record_id="worker-currency",
                    requirement_id="req-001",
                    criterion_id="req-001-criterion-002",
                )
            ),
        ]
    )

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find the revenue year.",
        requirement_ids=("req-001",),
        records=[
            _record(
                "worker-year",
                "The filing gives the revenue year.",
                observed_source_index=_OBSERVED_REVENUE_INDEX,
            )
        ],
        llm=llm,
    )

    assert requirement.status is RequirementStatus.PENDING
    assert requirement.covered_criterion_ids == ["req-001-criterion-001"]

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find the revenue currency.",
        requirement_ids=("req-001",),
        records=[
            _record(
                "worker-currency",
                "The filing gives the reported currency.",
                observed_source_index=_OBSERVED_REVENUE_INDEX,
            )
        ],
        llm=llm,
    )

    assert requirement.status is RequirementStatus.GROUNDED
    assert requirement.covered_criterion_ids == [
        "req-001-criterion-001",
        "req-001-criterion-002",
    ]


@pytest.mark.asyncio
async def test_wrong_ids_and_unsupported_criteria_leave_requirements_pending() -> None:
    criterion = "Give the annual revenue and year."
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            criterion,
            owner="leaf-revenue",
        )
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-secret-answer-key",
                    criterion=criterion,
                ),
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion="Mention revenue without a year.",
                ),
            )
        ]
    )

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find annual revenue.",
        requirement_ids=("req-001",),
        records=[_record("worker-good", "A plausible but insufficient memo.")],
        llm=llm,
    )

    requirement = runtime.requirements.requirements[0]
    assert runtime.evidence_cards == {}
    assert requirement.status is RequirementStatus.PENDING
    assert requirement.evidence_ids == []


@pytest.mark.asyncio
async def test_malformed_extraction_leaves_requirement_unbound() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
            owner="leaf-revenue",
        )
    )
    llm = _SequenceLLM([LLMResponse(content="The memo looks supported.", tool_calls=[])])

    await runtime.observe_leaf(
        HTNTaskNode(id="leaf-revenue", task="Find annual revenue."),
        [_record("worker-good", "Revenue may have been USD 12 million.")],
        llm,
    )

    assert runtime.evidence_cards == {}
    assert runtime.requirements.requirements[0].status is RequirementStatus.PENDING


@pytest.mark.asyncio
async def test_long_acceptance_criterion_uses_stable_id_and_can_be_grounded() -> None:
    criterion = (
        "Canonically support the annual revenue claim with all requested detail: "
        + "revenue year currency audited scope " * 18
        + "and preserve this canonical ending."
    )
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            criterion,
            owner="leaf-revenue",
        )
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion_id="req-001-criterion-001",
                )
            )
        ]
    )

    await runtime.observe_leaf(
        HTNTaskNode(id="leaf-revenue", task="Find annual revenue."),
        [
            _record(
                "worker-good",
                "The audited filing reports USD 12 million in 2025.",
                observed_source_index=_OBSERVED_REVENUE_INDEX,
            )
        ],
        llm,
    )

    requirement = runtime.requirements.requirements[0]
    assert requirement.status is RequirementStatus.GROUNDED
    assert len(requirement.evidence_ids) == 1
    prompt = llm.calls[0][0][-1]["content"]
    assert '"id": "req-001-criterion-001"' in prompt
    assert criterion not in prompt
    assert len(prompt) < 10_000


@pytest.mark.asyncio
async def test_extraction_uses_only_readable_delivery_records_and_observed_sources() -> None:
    criterion = "Give the annual revenue and year."
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            criterion,
            owner="leaf-revenue",
        )
    )
    observed_index = (
        "Automatically captured source metadata (untrusted evidence; titles and observed URLs "
        "come from this worker's actual tool results):\n"
        '- [exact_observed] "Observed annual filing" — https://example.test/revenue'
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-submitted",
                    requirement_id="req-001",
                    criterion=criterion,
                    source_title="Model-invented title",
                ),
                _card_payload(
                    record_id="worker-thin",
                    requirement_id="req-001",
                    criterion=criterion,
                    claim="Thin output must not become evidence.",
                ),
            )
        ]
    )

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find annual revenue.",
        requirement_ids=("req-001",),
        records=[
            _record(
                "worker-submitted",
                "USABLE SUBMITTED MEMO",
                observed_source_index=observed_index,
            ),
            _record(
                "worker-best-effort",
                "USABLE BEST EFFORT MEMO",
                quality="best_effort",
                submitted=False,
            ),
            _record("worker-thin", "REJECT THIN MEMO", quality="thin", submitted=False),
            _record("worker-malformed", "REJECT MALFORMED MEMO", quality="malformed"),
            _record("worker-blank", "   ", quality="submitted"),
        ],
        llm=llm,
    )

    assert len(runtime.evidence_cards) == 1
    card = next(iter(runtime.evidence_cards.values()))
    assert card.source_url == "https://example.test/revenue"
    assert card.source_title == "Observed annual filing"
    prompt = llm.calls[0][0][-1]["content"]
    assert "USABLE SUBMITTED MEMO" in prompt
    assert "USABLE BEST EFFORT MEMO" in prompt
    assert "REJECT THIN MEMO" not in prompt
    assert "REJECT MALFORMED MEMO" not in prompt
    assert "worker-blank" not in prompt


@pytest.mark.asyncio
async def test_observed_source_title_with_quotes_is_reused_with_its_url() -> None:
    criterion = "Give the annual revenue and year."
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            criterion,
            owner="leaf-revenue",
        )
    )
    observed_index = (
        "Automatically captured source metadata (untrusted evidence; titles and observed URLs "
        "come from this worker's actual tool results):\n"
        '- [exact_observed] "Annual "Audited" Filing" — https://example.test/revenue'
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion=criterion,
                    source_title="Model-invented title",
                )
            )
        ]
    )

    await runtime.observe_leaf(
        HTNTaskNode(id="leaf-revenue", task="Find annual revenue."),
        [
            _record(
                "worker-good",
                "The audited filing reports USD 12 million in 2025.",
                observed_source_index=observed_index,
            )
        ],
        llm,
    )

    card = next(iter(runtime.evidence_cards.values()))
    assert card.source_url == "https://example.test/revenue"
    assert card.source_title == 'Annual "Audited" Filing'


@pytest.mark.asyncio
async def test_scientific_report_removes_unobserved_model_url_from_evidence() -> None:
    criterion = "Give the annual revenue and year."
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            criterion,
            owner="leaf-revenue",
        )
    )
    observed_index = (
        '- [exact_observed] "Observed annual filing" — '
        "https://example.test/observed-revenue"
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion=criterion,
                    source_url="https://invented.test/unobserved",
                )
            )
        ]
    )

    await runtime.observe_leaf(
        HTNTaskNode(id="leaf-revenue", task="Find annual revenue."),
        [
            _record(
                "worker-good",
                "The audited filing reports USD 12 million in 2025.",
                observed_source_index=observed_index,
            )
        ],
        llm,
    )

    assert len(runtime.evidence_cards) == 1
    assert all(card.source_url is None for card in runtime.evidence_cards.values())


@pytest.mark.asyncio
async def test_scientific_report_accepts_source_less_derivation_evidence() -> None:
    criterion = "Show the derivation and assumptions."
    requirement = _requirement(
        "req-001",
        "Derive the requested scientific result.",
        criterion,
        owner="leaf-derivation",
    )
    runtime = ReportRuntime(
        question="Derive the result in a scientific report.",
        policy=SCIENTIFIC_REPORT_POLICY,
        requirements=RequirementSet("scientific_report", [requirement]),
    )
    llm = _SequenceLLM(
        [
            _evidence_response(
                _card_payload(
                    record_id="worker-good",
                    requirement_id="req-001",
                    criterion=criterion,
                    source_url=None,
                    source_title=None,
                )
            )
        ]
    )

    await runtime.observe_leaf(
        HTNTaskNode(id="leaf-derivation", task="Derive the result."),
        [_record("worker-good", "A complete derivation with explicit assumptions.")],
        llm,
    )

    card = next(iter(runtime.evidence_cards.values()))
    assert card.source_url is None
    assert requirement.status is RequirementStatus.GROUNDED


@pytest.mark.asyncio
async def test_expired_evidence_deadline_skips_llm_boundary() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
            owner="leaf-revenue",
        )
    )
    llm = _SequenceLLM()

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find annual revenue.",
        requirement_ids=("req-001",),
        records=[_record("worker-good", "A readable submitted memo.")],
        llm=llm,
        deadline=time.monotonic() - 1,
    )

    assert llm.calls == []
    assert runtime.evidence_cards == {}


@pytest.mark.asyncio
async def test_evidence_extraction_is_cancelled_at_its_deadline() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
            owner="leaf-revenue",
        )
    )

    class _NeverLLM:
        model = "never"

        async def chat(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    started = time.monotonic()
    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find annual revenue.",
        requirement_ids=("req-001",),
        records=[_record("worker-good", "A readable submitted memo.")],
        llm=_NeverLLM(),
        deadline=time.monotonic() + 0.02,
    )

    assert time.monotonic() - started < 0.5
    assert runtime.evidence_cards == {}


def test_evidence_tool_requests_strict_schema_validation() -> None:
    assert _EVIDENCE_TOOL["function"]["strict"] is True


@pytest.mark.asyncio
async def test_no_readable_records_skip_the_external_extraction_boundary() -> None:
    runtime = _runtime(
        _requirement(
            "req-001",
            "Report annual revenue in USD.",
            "Give the annual revenue and year.",
            owner="leaf-revenue",
        )
    )
    llm = _SequenceLLM()

    await runtime.observe_records(
        leaf_id="leaf-revenue",
        leaf_task="Find annual revenue.",
        requirement_ids=("req-001",),
        records=[
            _record("worker-thin", "Some text.", quality="thin", submitted=False),
            _record("worker-empty", "", quality="submitted"),
        ],
        llm=llm,
    )

    assert runtime.evidence_cards == {}
    assert llm.calls == []


def test_delivery_context_includes_every_owned_requirement_and_grounded_precursors() -> None:
    owned = [
        _requirement(
            f"req-{index:03d}",
            f"Owned delivery requirement {index}.",
            f"Acceptance criterion {index}.",
            owner="leaf-target",
        )
        for index in range(1, 10)
    ]
    owned[0].acceptance_criteria = tuple(
        f"Acceptance criterion 1.{index}." for index in range(1, 6)
    )
    precursor = _requirement(
        "req-010",
        "Grounded precursor requirement.",
        "Grounded precursor criterion.",
        owner="leaf-precursor",
        status=RequirementStatus.GROUNDED,
        evidence_ids=["ev-precursor"],
    )
    unrelated = _requirement(
        "req-011",
        "Grounded unrelated requirement.",
        "Grounded unrelated criterion.",
        owner="leaf-unrelated",
        status=RequirementStatus.GROUNDED,
        evidence_ids=["ev-unrelated"],
    )
    runtime = _runtime(*owned, precursor, unrelated)
    runtime.evidence_cards = {
        "ev-precursor": EvidenceCard(
            id="ev-precursor",
            leaf_id="leaf-precursor",
            requirement_ids=("req-010",),
            claim="Grounded precursor fact.",
            source_url="https://example.test/precursor",
            source_title="Precursor source",
            support_summary="This fact satisfies the precursor criterion.",
            confidence=0.9,
        ),
        "ev-unrelated": EvidenceCard(
            id="ev-unrelated",
            leaf_id="leaf-unrelated",
            requirement_ids=("req-011",),
            claim="Grounded but unrelated fact.",
            source_url="https://example.test/unrelated",
            source_title="Unrelated source",
            support_summary="This fact belongs to an unrelated leaf.",
            confidence=0.9,
        ),
    }
    target = HTNTaskNode(
        id="leaf-target",
        task="Research the remaining owned requirements.",
        depends_on=["leaf-precursor"],
    )

    context = runtime.delivery_context_for(target)

    assert "Assigned: 9 owned requirements" in context
    assert "[req-001]" in context
    assert "[req-008]" in context
    assert "[req-009]" in context
    assert "Acceptance criterion 1.5." in context
    assert "Grounded precursor fact." in context
    assert "Grounded but unrelated fact." not in context


def test_delivery_context_policy_directive_is_policy_gated_and_bounded() -> None:
    requirement = _requirement(
        "req-001",
        "Derive the requested rate constant.",
        "States the value with units.",
        owner="leaf-target",
    )
    target = HTNTaskNode(id="leaf-target", task="Derive the requested rate constant.")

    assert "Policy directive:" in _runtime(requirement).delivery_context_for(target)
    assert "Contract precedence:" in _runtime(requirement).delivery_context_for(target)

    fsr_runtime = ReportRuntime(
        question="Derive the requested rate constant.",
        policy=SCIENTIFIC_REPORT_POLICY,
        requirements=RequirementSet(policy_key="scientific_report", requirements=[requirement]),
    )
    context = fsr_runtime.delivery_context_for(target)
    assert "Policy directive:" in context
    assert "Develop every requested part fully" in context

    overlong_runtime = ReportRuntime(
        question="Derive the requested rate constant.",
        policy=replace(SCIENTIFIC_REPORT_POLICY, delivery_directive="directive " * 1_000),
        requirements=RequirementSet(policy_key="scientific_report", requirements=[requirement]),
    )
    directive_line = next(
        line
        for line in overlong_runtime.delivery_context_for(target).splitlines()
        if line.startswith("Policy directive:")
    )
    assert len(directive_line) <= len("Policy directive: ") + 2_001


def test_snapshot_is_json_serializable_and_exposes_coverage_without_private_inputs() -> None:
    grounded = _requirement(
        "req-001",
        "Report annual revenue in USD.",
        "Give the annual revenue and year.",
        owner="leaf-revenue",
        status=RequirementStatus.GROUNDED,
        evidence_ids=["ev-revenue"],
    )
    pending = _requirement(
        "req-002",
        "Report headquarters city and country.",
        "Give the headquarters location.",
        priority=2,
    )
    runtime = _runtime(grounded, pending)
    runtime.evidence_cards["ev-revenue"] = EvidenceCard(
        id="ev-revenue",
        leaf_id="leaf-revenue",
        requirement_ids=("req-001",),
        claim="Revenue was USD 12 million in 2025.",
        source_url="https://example.test/revenue",
        source_title="Annual filing",
        support_summary="The filing reports USD 12 million for 2025.",
        confidence=0.91,
    )

    snapshot = runtime.snapshot()

    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["policy"]["key"] == "scientific_report"
    assert snapshot["ownership"] == {
        "req-001": "leaf-revenue",
        "req-002": None,
    }
    assert snapshot["evidence"][0]["id"] == "ev-revenue"
    assert [row["id"] for row in snapshot["uncovered_requirements"]] == ["req-002"]
    assert [requirement.id for requirement in runtime.uncovered_requirements()] == ["req-002"]
    assert "ground_truth" not in json.dumps(snapshot).lower()


@pytest.mark.asyncio
async def test_create_returns_task_local_mutable_runtime_state(tmp_path: Path) -> None:
    def requirement_response(description: str) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                {
                    "function": {
                        "name": "derive_requirements",
                        "arguments": json.dumps(
                            {
                                "requirements": [
                                    {
                                        "description": description,
                                        "acceptance_criteria": ["Cover the requested fact."],
                                        "priority": 1,
                                        "kind": "fact",
                                    }
                                ]
                            }
                        ),
                    }
                }
            ],
        )

    llm = _SequenceLLM(
        [
            requirement_response("First task requirement."),
            requirement_response("Second task requirement."),
        ]
    )
    diagnostics = DiagnosticRecorder(tmp_path)

    first = await ReportRuntime.create(
        question="First public question?",
        policy=replace(SCIENTIFIC_REPORT_POLICY, max_requirements=4),
        llm=llm,
        diagnostics=diagnostics,
    )
    second = await ReportRuntime.create(
        question="Second public question?",
        policy=replace(SCIENTIFIC_REPORT_POLICY, max_requirements=4),
        llm=llm,
        diagnostics=diagnostics,
    )
    first.evidence_cards["ev-local"] = EvidenceCard(
        id="ev-local",
        leaf_id="leaf-first",
        requirement_ids=("req-001",),
        claim="First-task-only fact.",
        source_url=None,
        source_title=None,
        support_summary="First-task-only support.",
        confidence=0.5,
    )

    assert first.question == "First public question?"
    assert first.requirements.requirements[0].description == "First task requirement."
    assert second.question == "Second public question?"
    assert second.requirements.requirements[0].description == "Second task requirement."
    assert second.evidence_cards == {}


@pytest.mark.asyncio
async def test_create_forwards_question_deadline_to_requirement_derivation(
    tmp_path: Path,
) -> None:
    question = "Explain every requested public report field."

    runtime = await ReportRuntime.create(
        question=question,
        policy=SCIENTIFIC_REPORT_POLICY,
        llm=_SequenceLLM(),
        diagnostics=DiagnosticRecorder(tmp_path),
        question_deadline=time.monotonic() - 1,
    )

    assert runtime.requirements.requirements[0].description == (
        f"Address the complete public question: {question}"
    )


async def _unused_dispatch(*_args):  # pragma: no cover - construction only
    raise AssertionError("dispatch must not run")


def test_orchestrator_results_is_an_immutable_tuple_snapshot(tmp_path: Path) -> None:
    leaf = HTNTaskNode(id="root", task="Research one fact.")
    tree = HTNTree(root_id="root", nodes={"root": leaf})
    orchestrator = Orchestrator(
        OrchestratorRequest(
            tree=tree,
            leaf=leaf,
            available_agents=[AgentDescriptor("dynir_report_researcher")],
            execution_budget=ExecutionBudget(max_rounds=1, max_agents_per_round=1),
            output_dir=tmp_path,
            llm=_SequenceLLM(),
            dispatch=_unused_dispatch,
        )
    )
    first = _record("worker-one", "First delivered memo.")
    second = _record("worker-two", "Second delivered memo.")

    orchestrator._record_result(first)
    snapshot = orchestrator.results
    orchestrator._record_result(second)

    assert isinstance(snapshot, tuple)
    assert snapshot == (first,)
    assert orchestrator.results == (first, second)
    with pytest.raises(AttributeError):
        snapshot.append(second)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_decomposition", [False, True])
async def test_dependencies_do_not_move_future_or_completed_requirements_between_waves(needs_decomposition):
    model = _requirement("req-model", "Derive governing equations", "Derive governing equations")
    numeric = _requirement("req-numeric", "Compute numerical convergence", "Compute convergence")
    runtime = _runtime(model, numeric)
    tree = _tree_with_leaves(
        HTNTaskNode(id="model", task="Derive governing equations"),
        HTNTaskNode(id="numeric", task="Compute numerical convergence", depends_on=["model"],
                    needs_decomposition=needs_decomposition),
    )
    await runtime.refresh_ownership(tree, _SequenceLLM())
    assert model.owner_leaf_id == "model"
    assert numeric.owner_leaf_id == "numeric"
    assert "[req-numeric]" not in runtime.delivery_context_for(tree.get_node("model"))

    tree.get_node("model").status = HTNTaskStatus.COMPLETED
    model.transition_to(RequirementStatus.GROUNDED)
    await runtime.refresh_ownership(tree, _SequenceLLM())
    assert model.owner_leaf_id == "model"
    assert numeric.owner_leaf_id == "numeric"
    assert "[req-model]" not in runtime.delivery_context_for(tree.get_node("numeric"))


@pytest.mark.asyncio
async def test_decomposed_owner_transfers_scope_to_its_descendants():
    requirement = _requirement("req-001", "Compute numerical convergence", "Check convergence",
                               owner="branch")
    runtime = _runtime(requirement)
    tree = _tree_with_leaves(
        HTNTaskNode(id="branch", task="Compute numerical convergence", needs_decomposition=True),
        HTNTaskNode(id="unrelated", task="Check convergence of a different model"),
    )
    tree.decompose("branch", [
        HTNTaskNode(id="compute", task="Compute numerical convergence"),
        HTNTaskNode(id="interpret", task="Interpret model results", depends_on=["compute"]),
    ])
    await runtime.refresh_ownership(tree, _SequenceLLM())
    assert requirement.owner_leaf_id == "compute"


@pytest.mark.asyncio
async def test_counterevidence_reopens_coverage_and_new_checked_evidence_can_restore_it():
    requirement = _requirement("req-001", "Compute the result", "Show correct units", owner="leaf")
    unaffected = _requirement("req-002", "Explain mechanism", "Explain mechanism",
                              status=RequirementStatus.GROUNDED)
    runtime = _runtime(requirement, unaffected)

    async def observe(response, record_id):
        await runtime.observe_records(
            leaf_id="leaf", leaf_task="Check result", requirement_ids=("req-001",),
            records=[_record(record_id, "Independent calculation and unit check")],
            llm=_SequenceLLM([response]),
        )

    await observe(_evidence_response(_card_payload(
        record_id="initial", requirement_id="req-001", criterion="Show correct units",
    )), "initial")
    old_id = requirement.evidence_ids[0]
    assert requirement.status is RequirementStatus.GROUNDED
    challenge = _evidence_response()
    challenge.tool_calls[0]["function"]["arguments"] = json.dumps({
        "cards": [], "challenges": [
            {"requirement_id": "req-001", "record_id": "checker",
             "reason": "Recomputation shows a dimensional inconsistency"},
            {"requirement_id": "req-002", "record_id": "checker",
             "reason": "An out-of-scope claim must not mutate another requirement"},
            {"requirement_id": "req-001", "record_id": "unknown",
             "reason": "An unknown record must not enter the ledger"},
        ],
    })
    await observe(challenge, "checker")
    assert requirement.status is RequirementStatus.PENDING
    assert not requirement.evidence_ids and not requirement.covered_criterion_ids
    assert requirement.disputed_evidence_ids == [old_id]
    assert len(requirement.reopen_reasons) == 1
    assert old_id in runtime.evidence_cards  # Preserve the original evidence for inspection.
    assert unaffected.status is RequirementStatus.GROUNDED
    assert requirement in runtime.uncovered_requirements()
    child = HTNTaskNode(id="child", depends_on=["leaf"])
    assert "Revenue was USD" not in runtime.delivery_context_for(child)

    await observe(_evidence_response(_card_payload(
        record_id="rechecked", requirement_id="req-001", criterion="Show correct units",
        claim="Corrected result with consistent units", support_summary="Independent recomputation",
    )), "rechecked")
    assert requirement.status is RequirementStatus.GROUNDED
    assert old_id not in requirement.evidence_ids
    assert old_id in requirement.disputed_evidence_ids


@pytest.mark.asyncio
async def test_rejected_late_challenge_cannot_reopen_coverage():
    requirement = _requirement("req-001", "Compute", "Check", status=RequirementStatus.GROUNDED)
    runtime = _runtime(requirement)
    response = _evidence_response()
    response.tool_calls[0]["function"]["arguments"] = json.dumps({"cards": [], "challenges": [
        {"requirement_id": "req-001", "record_id": "late", "reason": "Late conflict"},
    ]})
    await runtime.observe_records(
        leaf_id="leaf", leaf_task="Check", requirement_ids=("req-001",),
        records=[_record("late", "A disputed result")], llm=_SequenceLLM([response]),
        accept_results=lambda: False,
    )
    assert requirement.status is RequirementStatus.GROUNDED
    assert not requirement.reopen_reasons


@pytest.mark.asyncio
async def test_long_memo_conclusion_reaches_evidence_extraction():
    runtime = _runtime(_requirement("req-001", "Compute", "Check convergence"))
    llm = _SequenceLLM([_evidence_response()])
    await runtime.observe_records(
        leaf_id="leaf", leaf_task="Compute", requirement_ids=("req-001",),
        records=[_record("worker", "Derivation step. " * 600 + "FINAL CONVERGENCE RESULT")],
        llm=llm,
    )
    assert "FINAL CONVERGENCE RESULT" in llm.calls[0][0][-1]["content"]


@pytest.mark.asyncio
async def test_late_verifier_is_retained_when_evidence_records_exceed_capacity():
    runtime = _runtime(_requirement("req-001", "Compute", "Check convergence"))
    llm = _SequenceLLM([_evidence_response()])
    records = [_record(f"worker-{i}", f"Early investigation {i}") for i in range(14)]
    records.append(_record("last-verifier", "Decisive correction after independent check"))
    await runtime.observe_records(
        leaf_id="leaf", leaf_task="Compute", requirement_ids=("req-001",),
        records=records, llm=llm,
    )
    payload = json.loads(llm.calls[0][0][-1]["content"].split("\n", 1)[1])
    assert len(payload["worker_records"]) == 12
    assert payload["worker_records"][-1]["record_id"] == "last-verifier"


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [False, True])
async def test_missing_cards_retry_cannot_ground_coverage_until_valid_evidence(corrected):
    requirement = _requirement("req-001", "Derive result", "Show derivation", owner="leaf")
    runtime = _runtime(requirement)
    bad = _evidence_response()
    bad.tool_calls[0]["function"]["arguments"] = json.dumps({"challenges": []})
    good = _evidence_response(_card_payload(
        record_id="worker-good", requirement_id="req-001",
        criterion_id="req-001-criterion-001", source_url=None, source_title=None,
    ))

    class CheckingLLM(_SequenceLLM):
        async def chat(self, messages, tools=None):
            assert requirement.status is RequirementStatus.PENDING
            assert not runtime.evidence_cards
            return await super().chat(messages, tools=tools)

    llm = CheckingLLM([bad, good if corrected else bad])
    await runtime.observe_leaf(
        HTNTaskNode(id="leaf", task="Derive result"),
        [_record("worker-good", "A complete derivation with explicit assumptions.")], llm,
    )
    assert len(llm.calls) == 2
    assert "'cards' is a required property" in llm.calls[1][0][-1]["content"]
    assert requirement.status is (RequirementStatus.GROUNDED if corrected else RequirementStatus.PENDING)
    assert bool(runtime.evidence_cards) == corrected
