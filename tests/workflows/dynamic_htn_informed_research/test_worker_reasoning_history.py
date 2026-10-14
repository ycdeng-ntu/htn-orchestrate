"""Native worker tool continuations preserve reasoning on the actual SDK wire."""
from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from agent_harness.infra.openai_client import OpenAIClient
from workflows.dynamic_htn_informed_research.htn import AgentTaskSummary
from workflows.dynamic_htn_informed_research.runtime.context import TaskRuntime, override_runtime
from workflows.dynamic_htn_informed_research.runtime.worker import build_dispatch
from workflows.dynamic_htn_informed_research.tools import build_research_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
async def test_worker_echoes_all_reasoning_without_upstream_cache(tmp_path, monkeypatch, reasoning_field):
    monkeypatch.setattr("dotenv.dotenv_values", lambda *args: {})
    artifact_dir = tmp_path / "question"
    refs = artifact_dir / "refs"
    refs.mkdir(parents=True)
    evidence = refs / "evidence.txt"
    evidence.write_text("41")
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        history = [m for m in body["messages"] if m["role"] == "assistant"]
        turn = len(history)
        assert turn <= 2
        for index, message in enumerate(history):
            assert message["reasoning_content"] == f"reasoning turn {index}"
            assert message["reasoning_details"] == [
                {"type": "reasoning.text", "text": f"reasoning turn {index}"},
            ]
            assert "reasoning turn" not in message["content"]
        tool_results = [m for m in body["messages"] if m["role"] == "tool"]
        assert len(tool_results) == turn
        assert [m["tool_call_id"] for m in tool_results] == [f"call_{i}" for i in range(turn)]
        name = "submit_report" if turn == 2 else "read_file"
        args = {"content": "42", "confidence": 0.8} if turn == 2 else {"path": str(evidence)}
        return httpx.Response(200, json={
            "id": f"response_{turn}", "object": "chat.completion", "created": 0,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                reasoning_field: f"reasoning turn {turn}",
                "reasoning_details": [{"type": "reasoning.text", "text": f"reasoning turn {turn}"}],
                "tool_calls": [{"id": f"call_{turn}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                }}],
            }}],
        })

    def make_sdk(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr("agent_harness.infra.openai_client.AsyncOpenAI", make_sdk)
    client = OpenAIClient("deepseek/deepseek-v4.1-flash", api_key="offline-test", base_url="https://offline.invalid/v1")

    def no_sandbox():
        raise AssertionError("This test uses only local report reads")

    def tools(role, **kwargs):
        return build_research_tools(role, **kwargs, sandbox_factory=no_sandbox)

    try:
        with override_runtime(llm_factory=lambda *_: client, tool_factory=tools):
            # Exercise compaction too: archived tool outputs must not remove reasoning.
            async with TaskRuntime({"agent": {"max_llm_retries": 1, "keep_tool_results": 0}}, "q", artifact_dir):
                dispatch = build_dispatch("q", max_turns=5, sampling={}, provider="openrouter", model=client.model)
                for worker in ("first", "second"):
                    result = await dispatch(AgentTaskSummary(worker, "dynir_researcher", "read evidence"), "Read evidence and submit 42")
                    assert result.submitted and result.result_summary == "42"
    finally:
        await client._client.close()

    assert len(requests) == 6
    # A newly dispatched worker starts its own history.
    assert not any(m["role"] == "assistant" for m in requests[3]["messages"])
    checkpoints = list((artifact_dir / "workers").glob("*/checkpoint.json"))
    assert len(checkpoints) == 2
    # Accepted submission stops before checkpointing the terminal turn.
    for path in checkpoints:
        history = [m for m in json.loads(path.read_text())["messages"] if m["role"] == "assistant"]
        assert [m["reasoning_content"] for m in history] == [f"reasoning turn {i}" for i in range(2)]
