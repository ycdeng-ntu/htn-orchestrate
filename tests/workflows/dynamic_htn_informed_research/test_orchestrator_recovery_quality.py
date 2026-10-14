from __future__ import annotations

import json

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    BoundedExpansionDecisionLoop,
    _DispatchDecision,
    _PARTIAL_SYNTHESIS_SYSTEM,
    _PartialTerminateDecision,
    _WaitDecision,
    _ensure_partial_status,
    _render_synthesis_user,
    _safe_recovery_decision,
    _synthesize_from_r,
    build_stable_context,
)
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree


def _record(
    *,
    role: str,
    quality: str,
    content: str,
    stop_reason: str = "max_turns",
) -> AgentResultRecord:
    return AgentResultRecord(
        agent_id=f"{role}-1",
        role_id=role,
        task_summary="Investigate the unresolved gap.",
        result_summary=content,
        execution_quality=quality,  # type: ignore[arg-type]
        stop_reason=stop_reason,
    )


def _agents() -> list[AgentDescriptor]:
    return [AgentDescriptor("dynir_researcher"), AgentDescriptor("dynir_verifier")]


@pytest.mark.parametrize("quality", ["thin", "malformed", "execution_error"])
def test_failed_verifier_delivery_never_causes_safe_terminate(quality: str) -> None:
    decision = _safe_recovery_decision(
        r=[_record(role="dynir_verifier", quality=quality, content="")],
        in_flight=[],
        available_agents=_agents(),
    )

    assert isinstance(decision, _DispatchDecision)
    assert decision.plan.agents[0].role_id == "dynir_researcher"


@pytest.mark.parametrize("quality", ["submitted", "best_effort"])
def test_readable_verifier_delivery_becomes_explicit_partial_not_confirmation(
    quality: str,
) -> None:
    decision = _safe_recovery_decision(
        r=[
            _record(
                role="dynir_verifier",
                quality=quality,
                content="Verdict: INCOMPLETE — publication evidence is still missing.",
            )
        ],
        in_flight=[],
        available_agents=_agents(),
    )

    assert isinstance(decision, _PartialTerminateDecision)
    assert "does not prove" in decision.reason


class _SequenceLLM:
    model = "test"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _dispatch_response() -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "dispatch",
                    "arguments": json.dumps(
                        {
                            "role_id": "dynir_researcher",
                            "task_summary": "Search the unresolved gap.",
                            "rationale": "recover protocol",
                        }
                    ),
                }
            }
        ]
    )


def _dispatch_call(task_summary: str, rationale: str) -> dict:
    return {
        "function": {
            "name": "dispatch",
            "arguments": json.dumps(
                {
                    "role_id": "dynir_researcher",
                    "task_summary": task_summary,
                    "rationale": rationale,
                }
            ),
        }
    }


def _legacy_agents_array_dispatch_response() -> LLMResponse:
    return LLMResponse(
        content="I should dispatch a focused researcher.",
        tool_calls=[
            {
                "function": {
                    "name": "dispatch",
                    "arguments": json.dumps(
                        {
                            "agents": [
                                {
                                    "role_id": "dynir_researcher",
                                    "task_summary": "Search the unresolved gap.",
                                }
                            ],
                            "rationale": "continue research",
                        }
                    ),
                }
            }
        ],
    )


def _context() -> tuple[HTNTree, HTNTaskNode]:
    root = HTNTaskNode(id="root", task="Which exact title?")
    return HTNTree(root_id="root", nodes={"root": root}), root


@pytest.mark.asyncio
async def test_coordinator_protocol_retry_is_bounded_and_can_recover() -> None:
    tree, root = _context()
    llm = _SequenceLLM([LLMResponse(content="I should search."), _dispatch_response()])

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[],
    )

    assert isinstance(decision, _DispatchDecision)
    assert len(llm.calls) == 2
    assert "Protocol correction" in llm.calls[1][-1]["content"]
    assert [message["role"] for message in llm.calls[1]] == [
        "system",
        "user",
        "user",
    ]


@pytest.mark.asyncio
async def test_parallel_dispatch_tool_calls_merge_without_retry(tmp_path) -> None:
    tree, root = _context()
    response = LLMResponse(
        tool_calls=[
            _dispatch_call("Search route A.", "Route A"),
            _dispatch_call("Search route B.", "Route B"),
            _dispatch_call("Search route C.", "Route C"),
        ]
    )
    llm = _SequenceLLM([response])
    diagnostics = DiagnosticRecorder(tmp_path)

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=3,
        in_flight=[],
        diagnostics=diagnostics,
        decision_call=1,
    )

    assert isinstance(decision, _DispatchDecision)
    assert [agent.task_summary for agent in decision.plan.agents] == [
        "Search route A.",
        "Search route B.",
        "Search route C.",
    ]
    assert decision.plan.rationale == "Route A\nRoute B\nRoute C"
    assert len(llm.calls) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    merged = next(
        event
        for event in events
        if event["event"] == "orchestrator_dispatch_calls_merged"
    )
    assert merged["original_tool_call_count"] == 3
    assert merged["merged_agent_count"] == 3
    outcome = next(
        event
        for event in events
        if event["event"] == "orchestrator_protocol_outcome"
    )
    assert outcome["outcome"] == "initial_valid"


@pytest.mark.asyncio
async def test_parallel_dispatch_merge_still_respects_free_slots() -> None:
    tree, root = _context()
    llm = _SequenceLLM(
        [
            LLMResponse(
                tool_calls=[
                    _dispatch_call("Search route A.", "Route A"),
                    _dispatch_call("Search route B.", "Route B"),
                    _dispatch_call("Search route C.", "Route C"),
                ]
            )
        ]
    )

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=2,
        in_flight=[],
    )

    assert isinstance(decision, _DispatchDecision)
    assert [agent.task_summary for agent in decision.plan.agents] == [
        "Search route A.",
        "Search route B.",
    ]


@pytest.mark.asyncio
async def test_mixed_parallel_coordinator_actions_still_retry() -> None:
    tree, root = _context()
    llm = _SequenceLLM(
        [
            LLMResponse(
                tool_calls=[
                    _dispatch_call("Search route A.", "Route A"),
                    {
                        "function": {
                            "name": "terminate",
                            "arguments": json.dumps({"rationale": "done"}),
                        }
                    },
                ]
            ),
            _dispatch_response(),
        ]
    )

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=2,
        in_flight=[],
    )

    assert isinstance(decision, _DispatchDecision)
    assert len(llm.calls) == 2
    assert "Protocol correction" in llm.calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_one_malformed_parallel_dispatch_makes_batch_retry() -> None:
    tree, root = _context()
    llm = _SequenceLLM(
        [
            LLMResponse(
                tool_calls=[
                    _dispatch_call("Search route A.", "Route A"),
                    {
                        "function": {
                            "name": "dispatch",
                            "arguments": '{"agents": [',
                        }
                    },
                ]
            ),
            _dispatch_response(),
        ]
    )

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=2,
        in_flight=[],
    )

    assert isinstance(decision, _DispatchDecision)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_legacy_dispatch_agents_array_receives_fresh_retry(tmp_path) -> None:
    tree, root = _context()
    llm = _SequenceLLM([_legacy_agents_array_dispatch_response(), _dispatch_response()])
    diagnostics = DiagnosticRecorder(tmp_path)

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[],
        diagnostics=diagnostics,
        decision_call=1,
    )

    assert isinstance(decision, _DispatchDecision)
    assert decision.plan.agents[0].task_summary == "Search the unresolved gap."
    assert len(llm.calls) == 2
    assert all(
        "I should dispatch a focused researcher." not in message["content"]
        for message in llm.calls[1]
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempts = [
        event
        for event in events
        if event["event"] == "orchestrator_protocol_attempt"
    ]
    assert [event["attempt"] for event in attempts] == [0, 1]
    assert attempts[0]["valid"] is False
    assert "unsupported legacy agents array" in attempts[0]["issue"]
    assert attempts[1]["valid"] is True
    outcome = next(
        event
        for event in events
        if event["event"] == "orchestrator_protocol_outcome"
    )
    assert outcome["outcome"] == "retry_recovered"
    assert outcome["successful_attempt"] == 1


@pytest.mark.asyncio
async def test_repeated_protocol_failure_does_not_loop_or_confirm_verifier() -> None:
    tree, root = _context()
    llm = _SequenceLLM([LLMResponse(), LLMResponse()])

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[
            _record(
                role="dynir_verifier",
                quality="submitted",
                content="Verdict: CONFIRMED",
                stop_reason="final_answer",
            )
        ],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[],
    )

    assert isinstance(decision, _PartialTerminateDecision)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_repeated_protocol_failure_waits_when_work_is_live() -> None:
    tree, root = _context()
    llm = _SequenceLLM([LLMResponse(), LLMResponse()])

    decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=llm,
        available_agents=_agents(),
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[],
    )

    # With no report and no live work, recovery dispatches research. A separate
    # direct assertion covers the live-work branch without constructing tasks.
    assert isinstance(decision, _DispatchDecision)
    direct = _safe_recovery_decision(
        r=[],
        in_flight=[
            # The recovery function only needs the list to be non-empty.
            object()  # type: ignore[list-item]
        ],
        available_agents=_agents(),
    )
    assert isinstance(direct, _WaitDecision)
    assert direct.reason == "coordinator_protocol_recovery_with_live_work"


def test_partial_fallback_preserves_execution_metadata() -> None:
    body = _synthesize_from_r(
        [
            _record(
                role="dynir_researcher",
                quality="best_effort",
                content="Candidate X matches the date; the author clue remains unresolved.",
            )
        ]
    )
    partial = _ensure_partial_status(body, "Leaf timeout")

    assert partial.startswith("Execution status: LIVENESS_FALLBACK")
    assert "Reason: Leaf timeout" in partial
    assert "**Execution quality:** best_effort" in partial
    assert "stopped_by=max_turns" in partial
    assert "author clue remains unresolved" in partial


def test_partial_synthesis_prompt_forbids_claiming_completion() -> None:
    rendered = _render_synthesis_user(
        "task context",
        "result context",
        force_partial=True,
        partial_reason="protocol failure",
    )

    assert "LIVENESS_FALLBACK execution path" in rendered
    assert "Semantic completion outcome: exhausted_unresolved" in rendered
    assert "protocol failure" in rendered
    assert (
        "MUST begin with\n``Execution status: LIVENESS_FALLBACK``"
        in _PARTIAL_SYNTHESIS_SYSTEM
    )
    assert "do not invent a third semantic state" in _PARTIAL_SYNTHESIS_SYSTEM
    assert "do not claim" in _PARTIAL_SYNTHESIS_SYSTEM
