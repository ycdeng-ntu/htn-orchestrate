from __future__ import annotations

import asyncio
import json

import pytest

from workflows.dynamic_htn_informed_research.adapters import build_summarizer
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNNodeCompletion,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy


class _LLM:
    def __init__(self, *responses: LLMResponse | BaseException) -> None:
        self.responses = iter(responses)
        self.calls = 0
        self.requests = []

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        self.requests.append(_kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def _response(args: dict, *, name: str = "complete_node", count: int = 1) -> LLMResponse:
    call = {"function": {"name": name, "arguments": json.dumps(args)}}
    return LLMResponse(tool_calls=[call] * count)


def _valid_args(*, is_root: bool, outcome: str = "resolved") -> dict:
    if is_root:
        return {
            "requested_output_fields": ["name"],
            "evidence_summary": "A verified observation.",
            "answer": "Alice" if outcome == "resolved" else "",
            "completion_outcome": outcome,
        }
    return {"result_summary": "A verified observation.", "completion_outcome": outcome}


def _completed_children(*, is_root: bool):
    root = HTNTaskNode(id="root", task="Return the person's name.")
    parent = root if is_root else HTNTaskNode(id="parent", parent_id="root", task="Find name.")
    child = HTNTaskNode(
        id="child",
        parent_id=parent.id,
        status=HTNTaskStatus.COMPLETED,
        result_summary="Original child evidence: Alice, observed at https://example.test/source.",
        completion_outcome=HTNCompletionOutcome.RESOLVED,
    )
    parent.children = [child.id]
    return root, parent, [child]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_root", [False, True])
async def test_tree_api_error_preserves_children_without_claiming_resolution(is_root):
    root, parent, children = _completed_children(is_root=is_root)
    llm = _LLM(TimeoutError("synthesis API timeout"))
    result = await build_summarizer(llm)(root, parent, children)

    assert llm.calls == 1
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert "TimeoutError" in result.failure_reason
    assert children[0].result_summary in result.result_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("is_root", [False, True])
@pytest.mark.parametrize(
    "invalid", ["prose", "missing_outcome", "invalid_outcome", "multiple_calls"]
)
async def test_two_invalid_tree_responses_are_a_protocol_failure(is_root, invalid):
    root, parent, children = _completed_children(is_root=is_root)
    args = _valid_args(is_root=is_root)
    if invalid == "missing_outcome":
        args.pop("completion_outcome")
    elif invalid == "invalid_outcome":
        args["completion_outcome"] = "partial"
    response = (
        LLMResponse(content="Unsupported replacement evidence.")
        if invalid == "prose"
        else _response(args, count=2 if invalid == "multiple_calls" else 1)
    )
    llm = _LLM(response, response)
    result = await build_summarizer(llm)(root, parent, children)

    assert llm.calls == 2
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert "protocol" in result.failure_reason
    assert children[0].result_summary in result.result_summary
    assert "Unsupported replacement evidence" not in result.result_summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_root", "field", "value"),
    [
        (True, "answer", ""),
        (True, "answer", 123),
        (True, "evidence_summary", {"text": "unsupported"}),
        (True, "requested_output_fields", ["name", None]),
        (False, "result_summary", ["unsupported"]),
        (False, "result_summary", ""),
    ],
)
async def test_tree_protocol_rejects_wrong_field_types_and_empty_results(is_root, field, value):
    root, parent, children = _completed_children(is_root=is_root)
    args = _valid_args(is_root=is_root)
    args[field] = value
    llm = _LLM(_response(args), _response(args))

    result = await build_summarizer(llm)(root, parent, children)

    assert llm.calls == 2
    assert result.failure_reason
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert children[0].result_summary in result.result_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("is_root", [False, True])
@pytest.mark.parametrize("outcome", ["resolved", "exhausted_unresolved"])
async def test_one_protocol_correction_accepts_a_valid_semantic_result(is_root, outcome):
    root, parent, children = _completed_children(is_root=is_root)
    llm = _LLM(
        LLMResponse(content="use tool next"),
        _response(
            _valid_args(
                is_root=is_root,
                outcome=outcome,
            )
        ),
    )

    result = await build_summarizer(llm)(root, parent, children)

    assert llm.calls == 2
    assert result.failure_reason is None
    assert result.completion_outcome is HTNCompletionOutcome(outcome)
    assert "verified observation" in result.result_summary


@pytest.mark.asyncio
async def test_failed_internal_synthesis_keeps_successful_children_and_blocks_dependency():
    calls = []

    async def summarize(_root, parent, children):
        calls.append(parent.id)
        return HTNNodeCompletion(
            result_summary="\n".join(child.result_summary for child in children),
            # A failure reason always wins over a contradictory semantic label.
            completion_outcome=HTNCompletionOutcome.RESOLVED,
            failure_reason="synthesis protocol failed",
        )

    nodes = {
        "root": HTNTaskNode(id="root", children=["parent", "sibling", "dependent"]),
        "parent": HTNTaskNode(id="parent", parent_id="root", children=["a", "b"]),
        "a": HTNTaskNode(id="a", parent_id="parent"),
        "b": HTNTaskNode(id="b", parent_id="parent"),
        "sibling": HTNTaskNode(id="sibling", parent_id="root"),
        "dependent": HTNTaskNode(id="dependent", parent_id="root", depends_on=["parent"]),
    }
    tree = HTNTree(root_id="root", nodes=nodes, summarizer=summarize)
    for node_id in ("sibling", "a", "b"):
        tree.mark_running(node_id)
    await tree.mark_completed("sibling", "successful sibling", "/sibling.md")
    await tree.mark_completed("a", "successful a", "/a.md")
    trigger = await tree.mark_completed(
        "b", "successful b", "/b.md", interrupt_on_semantic_exhaustion=False
    )

    assert trigger is nodes["parent"]
    assert trigger.status is HTNTaskStatus.FAILED
    assert trigger.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert trigger.failure_reason == "synthesis protocol failed"
    assert trigger.result_summary == "successful a\nsuccessful b"
    assert calls == ["parent"]
    assert not tree.is_ready("dependent")
    assert tree.root().status is HTNTaskStatus.PENDING
    for node_id in ("sibling", "a", "b"):
        assert nodes[node_id].status is HTNTaskStatus.COMPLETED
        assert nodes[node_id].ref == f"/{node_id}.md"
        assert nodes[node_id].failure_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_adapter", [False, True])
async def test_root_synthesis_exception_is_attributed_to_root_not_successful_leaf(use_adapter):
    def raises(*_args):
        raise RuntimeError("custom synthesizer failed")

    root = HTNTaskNode(id="root", children=["child"])
    child = HTNTaskNode(id="child", parent_id="root")
    tree = HTNTree(
        root_id="root",
        nodes={"root": root, "child": child},
        summarizer=build_summarizer(_LLM(RuntimeError("API failed"))) if use_adapter else raises,
    )
    tree.mark_running("child")

    trigger = await tree.mark_completed("child", "retained observation", "/child.md")

    assert trigger is root
    assert root.status is HTNTaskStatus.FAILED
    assert root.failure_reason
    assert root.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert "retained observation" in root.result_summary
    assert child.status is HTNTaskStatus.COMPLETED
    assert child.failure_reason is None
    assert child.ref == "/child.md"
    assert tree._summarizing_nodes == set()


@pytest.mark.asyncio
async def test_absent_summarizer_still_allows_deterministic_aggregation():
    root = HTNTaskNode(id="root", children=["child"])
    child = HTNTaskNode(id="child", parent_id="root")
    tree = HTNTree(root_id="root", nodes={"root": root, "child": child})
    tree.mark_running("child")

    trigger = await tree.mark_completed("child", "retained observation")

    assert trigger is None
    assert root.status is HTNTaskStatus.COMPLETED
    assert root.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert root.failure_reason is None


@pytest.mark.asyncio
async def test_synthesis_cancellation_propagates_and_releases_tree_reservation():
    root = HTNTaskNode(id="root", children=["child"])
    child = HTNTaskNode(id="child", parent_id="root")
    tree = HTNTree(
        root_id="root",
        nodes={"root": root, "child": child},
        summarizer=build_summarizer(_LLM(asyncio.CancelledError())),
    )
    tree.mark_running("child")

    with pytest.raises(asyncio.CancelledError):
        await tree.mark_completed("child", "retained observation")

    assert root.status is HTNTaskStatus.PENDING
    assert root.failure_reason is None
    assert child.status is HTNTaskStatus.COMPLETED
    assert tree._summarizing_nodes == set()


_SCIENTIFIC_REPORT = """# Scientific report

## (a) Derivation
For the stated system, define the displacement x and restoring constant k.
The potential energy is
\\[
U(x) = \\frac{1}{2} k x^2.
\\]

## (b) Specific result
For k = 4 N/m and x = 3 m, U = 18 J.
Source: https://example.test/experiment
""".strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("epoch_id", [0, 1])
@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("outcome", ["resolved", "exhausted_unresolved"])
async def test_scientific_report_root_preserves_full_multiline_answer(
    epoch_id,
    correction,
    outcome,
):
    strategy = get_task_strategy("report_research").synthesis
    assert strategy.root_answer_mode == "multiline_report"
    args = {
        "requested_output_fields": ["(a) Derivation", "(b) Specific result"],
        "evidence_summary": "Internal synthesis memo, separate from the deliverable.",
        "answer": _SCIENTIFIC_REPORT,
        "completion_outcome": outcome,
    }
    responses = [LLMResponse(content=_SCIENTIFIC_REPORT)] if correction else []
    llm = _LLM(*responses, _response(args))
    root = HTNTaskNode(id="root", task="Derive (a) and calculate (b).", children=["child"])
    child = HTNTaskNode(id="child", parent_id="root")
    tree = HTNTree(
        root_id="root",
        nodes={"root": root, "child": child},
        epoch_id=epoch_id,
        summarizer=build_summarizer(llm, strategy=strategy, epoch_id=epoch_id),
    )
    tree.mark_running("child")

    trigger = await tree.mark_completed(
        "child",
        "Original observations and equations.",
        interrupt_on_semantic_exhaustion=epoch_id == 0,
    )

    assert llm.calls == (2 if correction else 1)
    assert root.status is HTNTaskStatus.COMPLETED
    assert root.failure_reason is None
    assert root.completion_outcome is HTNCompletionOutcome(outcome)
    assert root.result_summary == _SCIENTIFIC_REPORT
    assert "# Answer:" not in root.result_summary
    assert trigger is (root if outcome == "exhausted_unresolved" else None)
    for request in llm.requests:
        function = request["tools"][0]["function"]
        assert function["name"] == "complete_node"
        description = function["parameters"]["properties"]["answer"]["description"]
        assert "complete Markdown report" in description
        assert "concise" not in description


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_name", ["research", "science_mcq"])
@pytest.mark.parametrize("epoch_id", [0, 1])
async def test_short_answer_strategies_still_reject_unmarked_multiline_answer(
    strategy_name,
    epoch_id,
):
    root, parent, children = _completed_children(is_root=True)
    args = _valid_args(is_root=True)
    args["answer"] = "A\nUnrequested second line"
    llm = _LLM(_response(args), _response(args))

    result = await build_summarizer(
        llm,
        strategy=get_task_strategy(strategy_name).synthesis,
        epoch_id=epoch_id,
    )(root, parent, children)

    assert llm.calls == 2
    assert result.failure_reason
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert children[0].result_summary in result.result_summary
    assert (
        "concise"
        in llm.requests[0]["tools"][0]["function"]["parameters"]["properties"]["answer"][
            "description"
        ]
    )


@pytest.mark.asyncio
async def test_mcq_root_still_requires_the_exact_requested_option():
    root, parent, children = _completed_children(is_root=True)
    args = _valid_args(is_root=True)
    args["answer"] = "The answer is A"
    llm = _LLM(_response(args), _response(args))
    strategy = get_task_strategy("science_mcq").synthesis

    invalid = await build_summarizer(llm, strategy=strategy)(root, parent, children)
    assert invalid.failure_reason

    args["answer"] = "A"
    valid = await build_summarizer(_LLM(_response(args)), strategy=strategy)(root, parent, children)
    assert valid.failure_reason is None
    assert valid.result_summary.endswith("# Answer: A")


@pytest.mark.asyncio
async def test_resolved_report_root_still_requires_nonempty_answer():
    root, parent, children = _completed_children(is_root=True)
    args = _valid_args(is_root=True)
    args["answer"] = " \n "
    llm = _LLM(_response(args), _response(args))

    result = await build_summarizer(
        llm,
        strategy=get_task_strategy("report_research").synthesis,
    )(root, parent, children)

    assert llm.calls == 2
    assert result.failure_reason
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
