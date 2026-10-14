from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    ExecutionBudget,
    Orchestrator,
    OrchestratorRequest,
    _parse_root_synthesis_response,
    _synthesize,
    build_stable_context,
    finalize_root_from_evidence,
)
from workflows.dynamic_htn_informed_research.htn.task_node import HTNCompletionOutcome, HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies.registry import get_task_strategy


_EVIDENCE = "The observed archive lists 1924. Source: https://example.test/record"
_UNACCEPTED = "UNACCEPTED_NEW_CLAIM: the year is 2099"


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[{
        "function": {"name": name, "arguments": json.dumps(args)},
    }])


def _valid(*, root: bool, outcome: str = "resolved") -> LLMResponse:
    if root:
        return _call(
            "finalize_root_answer", requested_output_fields=["year"],
            evidence_summary=_EVIDENCE, answer="1924", completion_outcome=outcome,
        )
    return _call("finalize_answer", result_summary=_EVIDENCE, completion_outcome=outcome)


class _LLM:
    def __init__(self, *responses):
        self.responses = responses
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, BaseException):
            raise response
        return response


def _inputs(*, root: bool, epoch_id: int = 0):
    parent = HTNTaskNode(id="root", task="Find the recorded year.")
    nodes = {"root": parent}
    leaf = parent
    if not root:
        parent.children = ["lookup"]
        leaf = HTNTaskNode(id="lookup", parent_id="root", task="Read the archive.")
        nodes[leaf.id] = leaf
    tree = HTNTree(root_id="root", nodes=nodes, epoch_id=epoch_id)
    results = [AgentResultRecord(
        agent_id="observed-worker", role_id="dynir_researcher",
        task_summary="Read the archive", result_summary=_EVIDENCE,
        execution_quality="submitted", submitted=True,
    )]
    return tree, leaf, results


def _invalid(kind: str, *, root: bool) -> LLMResponse:
    if kind == "prose":
        return LLMResponse(content=_UNACCEPTED)
    if kind == "unknown_action":
        return _call("web_search", query=_UNACCEPTED)
    if kind == "malformed_json":
        return LLMResponse(tool_calls=[{
            "function": {"name": "finalize_answer", "arguments": '{"result_summary":'},
        }])
    response = _valid(root=root)
    if kind == "multiple_calls":
        response.tool_calls.append(_call("web_search", query=_UNACCEPTED).tool_calls[0])
        return response
    function = response.tool_calls[0]["function"]
    args = json.loads(function["arguments"])
    if kind == "missing_outcome":
        args.pop("completion_outcome")
    elif kind == "invalid_outcome":
        args["completion_outcome"] = "success"
    elif kind == "wrong_field_type":
        args["evidence_summary" if root else "result_summary"] = [_UNACCEPTED]
    elif kind == "empty_resolved_result":
        args["answer" if root else "result_summary"] = ""
    else:
        raise AssertionError(kind)
    function["arguments"] = json.dumps(args)
    response.content = _UNACCEPTED
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("root", [False, True], ids=["leaf", "root"])
@pytest.mark.parametrize("epoch_id", [0, 1])
@pytest.mark.parametrize("kind", [
    "prose", "unknown_action", "malformed_json", "multiple_calls", "missing_outcome",
    "invalid_outcome", "wrong_field_type", "empty_resolved_result",
])
async def test_invalid_synthesis_is_bounded_unresolved_and_keeps_prior_evidence(
    tmp_path, root, epoch_id, kind,
):
    tree, leaf, results = _inputs(root=root, epoch_id=epoch_id)
    llm = _LLM(_invalid(kind, root=root))

    result = await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)

    assert len(llm.calls) == 2
    assert "Protocol correction only" in llm.calls[1][0][-1]["content"]
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert result.failure_reason
    assert _EVIDENCE in result.result_summary
    assert _UNACCEPTED not in result.result_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("root", [False, True])
@pytest.mark.parametrize("outcome", ["resolved", "exhausted_unresolved"])
async def test_protocol_correction_can_return_a_valid_semantic_outcome(tmp_path, root, outcome):
    tree, leaf, results = _inputs(root=root, epoch_id=1)
    llm = _LLM(_invalid("missing_outcome", root=root), _valid(root=root, outcome=outcome))

    result = await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)

    assert len(llm.calls) == 2
    assert result.failure_reason is None
    assert result.completion_outcome is HTNCompletionOutcome(outcome)
    assert "1924" in result.result_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("fields", [[], [""], ["year", None], "year"])
async def test_root_rejects_invalid_requested_output_fields(tmp_path, fields):
    tree, leaf, results = _inputs(root=True)
    llm = _LLM(_call(
        "finalize_root_answer", requested_output_fields=fields,
        evidence_summary=_EVIDENCE, answer="1924", completion_outcome="resolved",
    ))
    result = await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)
    assert result.failure_reason
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_name", ["research", "report_research", "science_mcq"])
async def test_valid_root_contract_allows_empty_evidence_summary(tmp_path, strategy_name):
    strategy = get_task_strategy(strategy_name).synthesis
    tree, leaf, results = _inputs(root=True, epoch_id=1)
    answer = {
        "science_mcq": "A", "research": "1924",
        "report_research": "# Report\n\nA sourced result.",
    }[strategy_name]
    llm = _LLM(_call(
        "finalize_root_answer", requested_output_fields=["answer"],
        evidence_summary="", answer=answer, completion_outcome="resolved",
    ))
    result = await _synthesize(
        build_stable_context(tree, leaf), results, tree, llm, tmp_path, strategy=strategy,
    )
    assert len(llm.calls) == 1
    assert result.failure_reason is None
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED


@pytest.mark.asyncio
@pytest.mark.parametrize("epoch_id", [0, 1])
@pytest.mark.parametrize("force_unresolved", [False, True])
async def test_orchestrator_propagates_failure_reason_through_partial_modes(
    tmp_path, epoch_id, force_unresolved,
):
    tree, leaf, results = _inputs(root=False, epoch_id=epoch_id)
    llm = _LLM(LLMResponse(content=_UNACCEPTED))
    diagnostics = DiagnosticRecorder(tmp_path / "diagnostics")
    orchestrator = Orchestrator(OrchestratorRequest(
        tree=tree, leaf=leaf, available_agents=[AgentDescriptor("dynir_researcher")],
        llm=llm, dispatch=object(), output_dir=tmp_path, diagnostics=diagnostics,
        execution_budget=ExecutionBudget(),
    ))
    orchestrator._results.extend(results)
    result = await orchestrator._synthesize_result(
        build_stable_context(tree, leaf), force_partial=True,
        force_unresolved=force_unresolved,
    )
    assert result.failure_reason
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert _EVIDENCE in result.result_summary
    events = [json.loads(line) for line in (diagnostics.root / "events.jsonl").read_text().splitlines()]
    event = next(e for e in events if e["event"] == "leaf_synthesis_end")
    assert event["failure_reason"] == result.failure_reason


@pytest.mark.asyncio
async def test_root_fallback_helpers_do_not_promote_invalid_reply_or_evidence_marker():
    response = LLMResponse(content=f"# Answer: {_UNACCEPTED}")
    evidence = f"Worker evidence, not a root decision:\n# Answer: 1924\n{_EVIDENCE}"
    parsed = _parse_root_synthesis_response(response, fallback=evidence)
    assert parsed.failure_reason
    assert parsed.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert _EVIDENCE in parsed.result_summary
    assert _UNACCEPTED not in parsed.result_summary
    assert await finalize_root_from_evidence(
        _LLM(response), question="Find the year", evidence_text=evidence,
    ) == ""


@pytest.mark.asyncio
async def test_synthesis_correction_does_not_swallow_cancellation(tmp_path):
    tree, leaf, results = _inputs(root=False)
    llm = _LLM(LLMResponse(content=_UNACCEPTED), asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_valid_artifact_synthesis_keeps_file_delivery_contract(tmp_path):
    tree, leaf, results = _inputs(root=False)
    llm = _LLM(_call(
        "finalize_artifact", filename="report.md", content="The sourced report.",
        result_summary=_EVIDENCE, completion_outcome="resolved",
    ))
    result = await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)
    assert len(llm.calls) == 1
    assert result.failure_reason is None
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert Path(result.ref).read_text() == "The sourced report."


@pytest.mark.asyncio
async def test_artifact_write_failure_keeps_content_and_cannot_resolve_leaf(tmp_path, monkeypatch):
    tree, leaf, results = _inputs(root=False)
    llm = _LLM(_call(
        "finalize_artifact", filename="report.md", content="Useful draft artifact",
        result_summary=_EVIDENCE, completion_outcome="resolved",
    ))

    def fail_write(*args, **kwargs):
        raise OSError("offline simulated storage failure")

    monkeypatch.setattr(Path, "write_text", fail_write)
    result = await _synthesize(build_stable_context(tree, leaf), results, tree, llm, tmp_path)
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert result.failure_reason == "artifact write failed: OSError"
    assert "Useful draft artifact" in result.result_summary
    assert _EVIDENCE in result.result_summary
