from __future__ import annotations

import json

import pytest

from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentTaskSummary,
    BoundedExpansionDecisionLoop,
    _PartialTerminateDecision,
    _WaitDecision,
    _effective_synthesis_outcome,
    _synthesize,
    build_stable_context,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree


class _RecordingLLM:
    model = "test"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], list[dict] | None]] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001
        self.calls.append((messages, tools))
        return self.response


class _SequenceLLM:
    model = "test"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], list[dict] | None]] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def _tool_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            }
        ]
    )


def _tool_names(tools: list[dict] | None) -> set[str]:
    return {
        tool["function"]["name"]
        for tool in tools or []
    }


@pytest.mark.asyncio
async def test_direct_root_owns_exact_answer_contract(tmp_path) -> None:  # noqa: ANN001
    root = HTNTaskNode(id="root", task="What exact classification was listed?")
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _RecordingLLM(
        _tool_response(
            "finalize_root_answer",
            {
                "requested_output_fields": ["exact classification"],
                "evidence_summary": (
                    "The official profile says category alpha-two.\n\n"
                    "# Answer: Category alpha"
                ),
                "answer": "category alpha-two",
                "completion_outcome": "resolved",
            },
        )
    )

    result = await _synthesize(
        build_stable_context(tree, root),
        [],
        tree,
        llm,
        tmp_path,
    )

    assert result.result_summary == (
        "The official profile says category alpha-two.\n\n"
        "# Answer: category alpha-two"
    )
    messages, tools = llm.calls[0]
    system_prompt = " ".join(messages[0]["content"].split())
    assert _tool_names(tools) == {"finalize_root_answer"}
    assert "final answer stage" in system_prompt
    assert "type or category rather than a more specific member" in system_prompt
    assert "unless the question explicitly asks for a subtype" in system_prompt
    assert "Do not append a location, hierarchy" in system_prompt
    assert "every output field explicitly requested" in system_prompt
    assert "Preserving one only" in system_prompt
    assert "does not satisfy the answer contract" in system_prompt
    assert "provide X with the" in system_prompt
    assert "inside an entity description" in system_prompt
    assert "closure procedure over the CURRENT TASK and R" in system_prompt
    assert "locate the concrete supporting observation in R" in system_prompt
    assert "coherent candidate narrative is not" in system_prompt
    assert "complete concise answer containing every output field" in next(
        tool["function"]["parameters"]["properties"]["answer"]["description"]
        for tool in tools or []
        if tool["function"]["name"] == "finalize_root_answer"
    )
    root_tool = next(
        tool for tool in tools or [] if tool["function"]["name"] == "finalize_root_answer"
    )
    assert "requested_output_fields" in root_tool["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_non_root_leaf_keeps_evidence_memo_contract(tmp_path) -> None:  # noqa: ANN001
    root = HTNTaskNode(id="root", task="Which exact title?", children=["lookup"])
    child = HTNTaskNode(
        id="lookup",
        parent_id="root",
        title="Find title",
        task="Locate the exact source title.",
    )
    tree = HTNTree(root_id="root", nodes={"root": root, "lookup": child})
    llm = _RecordingLLM(
        _tool_response(
            "finalize_answer",
            {"result_summary": "The source title is Example.", "completion_outcome": "resolved"},
        )
    )

    result = await _synthesize(
        build_stable_context(tree, child),
        [],
        tree,
        llm,
        tmp_path,
    )

    assert result.result_summary == "The source title is Example."
    messages, tools = llm.calls[0]
    assert _tool_names(tools) == {"finalize_answer", "finalize_artifact"}
    assert "evidence memo" in messages[0]["content"]
    assert "closure procedure over the CURRENT TASK and R" in messages[0]["content"]


@pytest.mark.asyncio
async def test_epoch1_leaf_synthesis_preserves_semantic_outcome(tmp_path) -> None:  # noqa: ANN001
    root = HTNTaskNode(id="root", task="What exact title?")
    tree = HTNTree(root_id="root", nodes={"root": root}, epoch_id=1)
    llm = _RecordingLLM(
        _tool_response(
            "finalize_root_answer",
            {
                "requested_output_fields": ["exact title"],
                "evidence_summary": "Only one candidate remains.",
                "answer": "Example Work",
                "completion_outcome": "exhausted_unresolved",
            },
        )
    )

    result = await _synthesize(
        build_stable_context(tree, root),
        [],
        tree,
        llm,
        tmp_path,
    )

    messages, _tools = llm.calls[0]
    assert "This is Epoch 1" in messages[0]["content"]
    assert "still returning the mandatory best-effort result" in messages[0]["content"]
    assert "closure procedure over the CURRENT TASK and R" not in messages[0]["content"]
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert (
        _effective_synthesis_outcome(
            epoch_id=1,
            strict_partial=False,
            proposed=HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        )
        is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    )


@pytest.mark.asyncio
async def test_direct_root_retries_prose_through_structured_outcome_tool(
    tmp_path,  # noqa: ANN001
) -> None:
    root = HTNTaskNode(id="root", task="What exact date?")
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _SequenceLLM(
        [
            LLMResponse(content="The supported date is March 19."),
            _tool_response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["exact date"],
                    "evidence_summary": "Two direct sources support the date.",
                    "answer": "March 19",
                    "completion_outcome": "resolved",
                },
            ),
        ]
    )

    result = await _synthesize(
        build_stable_context(tree, root),
        [],
        tree,
        llm,
        tmp_path,
    )

    assert len(llm.calls) == 2
    assert "Protocol correction only" in llm.calls[1][0][-1]["content"]
    assert result.result_summary.endswith("# Answer: March 19")
    assert result.completion_outcome.value == "resolved"


@pytest.mark.asyncio
async def test_direct_root_retries_resolved_tool_call_with_empty_answer(tmp_path) -> None:
    root = HTNTaskNode(id="root", task="What exact title?")
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _SequenceLLM(
        [
            _tool_response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["exact title"],
                    "evidence_summary": "The evidence confirms the title.",
                    "answer": "",
                    "completion_outcome": "resolved",
                },
            ),
            _tool_response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["exact title"],
                    "evidence_summary": "The evidence confirms the title.",
                    "answer": "Exact Work Title",
                    "completion_outcome": "resolved",
                },
            ),
        ]
    )

    result = await _synthesize(
        build_stable_context(tree, root),
        [],
        tree,
        llm,
        tmp_path,
    )

    assert len(llm.calls) == 2
    assert "non-empty `answer` field" in llm.calls[1][0][-1]["content"]
    assert result.result_summary.endswith("# Answer: Exact Work Title")


@pytest.mark.asyncio
async def test_explicit_wait_requires_live_work() -> None:
    root = HTNTaskNode(id="root", task="Which exact title?")
    tree = HTNTree(root_id="root", nodes={"root": root})
    response = _tool_response("wait", {})
    agents = [AgentDescriptor("dynir_researcher")]

    live_llm = _RecordingLLM(response)
    live_decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=live_llm,
        available_agents=agents,
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[
            AgentTaskSummary(
                agent_id="dynir_researcher-live",
                role_id="dynir_researcher",
                task_summary="Search the remaining source.",
            )
        ],
    )

    assert isinstance(live_decision, _WaitDecision)
    assert live_decision.reason == "coordinator_explicit_wait"
    assert "wait" in _tool_names(live_llm.calls[0][1])

    idle_llm = _RecordingLLM(response)
    idle_decision = await BoundedExpansionDecisionLoop().run(
        c=build_stable_context(tree, root),
        r=[],
        tree=tree,
        llm=idle_llm,
        available_agents=agents,
        rounds_remaining=2,
        max_agents_per_round=1,
        in_flight=[],
    )

    assert isinstance(idle_decision, _PartialTerminateDecision)
    assert "no sub-agent work was in flight" in idle_decision.reason
    idle_messages, idle_tools = idle_llm.calls[0]
    assert "wait" not in _tool_names(idle_tools)
    assert "Call one of: `terminate`, `dispatch`." in idle_messages[1]["content"]
    assert "No agents are currently in flight" in idle_messages[0]["content"]
    assert "  wait —" not in idle_messages[0]["content"]
