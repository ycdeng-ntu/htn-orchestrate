from __future__ import annotations

import pytest

from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.task_router import route_task


def _route_response(
    strategy_name: str,
    rationale: str = "boundary",
    *,
    report_kind: str | None = None,
) -> LLMResponse:
    arguments = {
        "strategy_name": strategy_name,
        "rationale": rationale,
    }
    if report_kind is not None:
        arguments["report_kind"] = report_kind
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "route_task",
                    "arguments": arguments,
                }
            }
        ]
    )


class _SequenceLLM:
    model = "router-test"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def chat(self, _messages, tools=None):  # noqa: ANN001
        assert tools
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_route_task_reports_fresh_attempt_failures_before_success() -> None:
    llm = _SequenceLLM(
        [
            LLMResponse(content="no tool"),
            RuntimeError("provider failure"),
            _route_response("science_mcq", "explicit finite choices"),
        ]
    )

    route = await route_task(llm, "Question with choices", timeout_s=1)

    assert route.strategy_name == "science_mcq"
    assert route.rationale == "explicit finite choices"
    assert route.attempt_count == 3
    assert route.attempt_failure_reasons == (
        "attempt 1: invalid response",
        "attempt 2: RuntimeError",
    )
    assert route.used_fallback is False
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_route_task_falls_back_to_research_after_three_invalid_responses() -> None:
    llm = _SequenceLLM([LLMResponse(), LLMResponse(), LLMResponse()])

    route = await route_task(llm, "ambiguous task", timeout_s=1)

    assert route.strategy_name == "research"
    assert route.attempt_count == 3
    assert route.attempt_failure_reasons == (
        "attempt 1: invalid response",
        "attempt 2: invalid response",
        "attempt 3: invalid response",
    )
    assert route.used_fallback is True
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_route_task_selects_single_report_strategy() -> None:
    llm = _SequenceLLM([_route_response("report_research", "A developed scientific analysis.")])
    route = await route_task(llm, "Produce a scientific report.", timeout_s=1)
    assert route.strategy_name == "report_research"
    assert not hasattr(route, "report_kind")
    assert not route.used_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_kind", ["sourced_report", "scientific_report"])
async def test_report_route_rejects_removed_kind_field(legacy_kind) -> None:
    llm = _SequenceLLM([
        _route_response("report_research", report_kind=legacy_kind),
        _route_response("report_research"),
    ])
    route = await route_task(llm, "Produce a scientific report.", timeout_s=1)
    assert route.strategy_name == "report_research" and route.attempt_count == 2
    assert route.attempt_failure_reasons == ("attempt 1: invalid response",)


def test_route_schema_and_prompt_have_only_four_flat_strategies():
    from workflows.dynamic_htn_informed_research.task_router import _ROUTE_TOOL, _ROUTER_SYSTEM
    schema = _ROUTE_TOOL["function"]["parameters"]
    assert set(schema["properties"]) == {"strategy_name", "rationale"}
    assert set(schema["required"]) == {"strategy_name", "rationale"}
    assert "report_kind" not in _ROUTER_SYSTEM and "sourced_report" not in _ROUTER_SYSTEM
    assert "General external-fact reports belong to research" in _ROUTER_SYSTEM


@pytest.mark.asyncio
async def test_router_rejects_multiple_calls_even_when_one_is_valid() -> None:
    duplicate = _route_response("research")
    duplicate.tool_calls.append(_route_response("science_mcq").tool_calls[0])
    llm = _SequenceLLM([duplicate, _route_response("research")])

    route = await route_task(llm, "Find one external fact.", timeout_s=1)

    assert route.strategy_name == "research"
    assert route.attempt_count == 2
    assert route.attempt_failure_reasons == ("attempt 1: invalid response",)


@pytest.mark.asyncio
async def test_router_rejects_unknown_argument_fields() -> None:
    invalid = _route_response("research")
    invalid.tool_calls[0]["function"]["arguments"]["benchmark"] = "drb"
    llm = _SequenceLLM([invalid, _route_response("science_olympiad")])

    route = await route_task(llm, "Solve the supplied equation.", timeout_s=1)

    assert route.strategy_name == "science_olympiad"
    assert route.attempt_count == 2
