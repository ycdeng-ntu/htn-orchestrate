"""Real loop/tool/compaction boundaries, with no provider or sandbox traffic."""
from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent_harness.core.llm import LLMResponse
from agent_harness.core.loop_types import LoopConfig, LoopPolicy, ToolResult
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.loop.tool_exec import execute_tools
from agent_harness.core.tool import Tool, ToolExecutionError, ToolOutput
from workflows.dynamic_htn_informed_research.runtime.evidence_memory import EvidenceMemory
from workflows.dynamic_htn_informed_research.runtime.llm import ResearchToolCallParser
from workflows.dynamic_htn_informed_research.runtime.worker import SubmissionObserver
from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig, build_research_tools
from workflows.react_base.compactors import AlwaysCompactPolicy


def _config(**kwargs):
    return ResearchToolConfig(
        serper_api_key="fake-search", jina_api_key="fake-fetch", e2b_api_key="",
        read_roots=(), summary_use_environment=False, retry_delay=0,
        direct_fetch_enabled=False, **kwargs,
    )


def _native(name, args=None):
    # Reused provider IDs also exercise the host loop's ID normalization.
    return LLMResponse(content="", usage={}, tool_calls=[{
        "id": "provider-id", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }])


def _read_args(index):
    match = re.search(r"Full tool result; read_file: (.+)", index)
    assert match is not None, index
    return json.loads(match.group(1))


class _Sandbox:
    def __init__(self, content, exit_code):
        self.writes = {}
        self.kills = 0
        self.files = SimpleNamespace(write=self.write, read=self.read)
        self.commands = SimpleNamespace(run=self.run)
        self.result = SimpleNamespace(stdout=content, stderr="", exit_code=exit_code)

    def write(self, path, content):
        self.writes[path] = content

    def read(self, path):
        raise AssertionError(f"Archived evidence must be read locally: {path}")

    def run(self, command, timeout):
        return self.result

    def kill(self):
        self.kills += 1


@pytest.mark.asyncio
async def test_execute_tools_preserves_trusted_metadata_and_error_status():
    refs = {"artifact_refs": [{"path": "/artifact.json", "kind": "tool_output"}]}
    successful = ToolOutput("complete output", metadata=refs)
    failed = ToolExecutionError(ToolOutput("failed calculation", metadata=refs))
    refs["artifact_refs"][0]["path"] = "/mutated.json"

    class UntrustedString(str):
        metadata = {"artifact_refs": [{"path": "/untrusted.json"}]}

    async def success():
        return successful

    async def failure():
        raise failed

    async def untrusted():
        return UntrustedString("ordinary output")

    tools = {name: Tool(name, name, {}, function) for name, function in (
        ("success", success), ("failure", failure), ("untrusted", untrusted),
    )}
    results = await execute_tools(
        [{"name": name, "args": {}, "id": name} for name in tools],
        tools, timeout=1, turn=1, count_offset=0,
    )
    assert [result.is_error for result in results] == [False, True, False]
    assert [result.tool_call_id for result in results] == list(tools)
    assert results[0].result == "complete output"
    assert "failed calculation" in results[1].result
    assert results[0].metadata["artifact_refs"][0]["path"] == "/artifact.json"
    assert results[1].metadata == results[0].metadata
    assert results[2].metadata == {}
    first = ToolResult("x", {}, "", 0, "a", False)
    second = ToolResult("x", {}, "", 0, "b", False)
    first.metadata["changed"] = True
    assert second.metadata == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("size,exit_code", [(9_000, 0), (24_000, 0), (24_000, 2)])
async def test_loop_recovers_old_output_tail_after_compaction(tmp_path, size, exit_code):
    marker = "RECOVERED_VALUE=42.875"
    content = "execution details " + "x" * size + marker
    sandbox = _Sandbox(content, exit_code)
    checkpoints = []

    def provider(request):
        assert request.url.path == "/search"
        return httpx.Response(200, json={"organic": []})

    class ReadbackLLM:
        model = "fake"

        def __init__(self):
            self.requests = []
            self.path = None
            self.recovered = False

        async def chat(self, messages, **kwargs):
            self.requests.append(copy.deepcopy(messages))
            turn = len(self.requests)
            if turn == 1:
                return _native("run_python_code", {"code": "print('controlled fixture')"})
            if turn == 2:
                return _native("web_search", {"q": "continue with another tool"})
            if turn == 3:
                original = next(message for message in messages if message.get("role") == "tool")
                index = original["content"]
                assert "Archived tool evidence:" in index
                assert marker not in index
                assert f"Execution status: {'error' if exit_code else 'success'}" in index
                self.path = _read_args(index)["path"]
                return _native("read_file", {"path": self.path, "start_char": 0})
            page = next(message["content"] for message in reversed(messages)
                        if message.get("role") == "tool")
            if marker not in page:
                next_page = re.search(r"next_start_char=(\d+)", page)
                assert next_page is not None, page
                assert int(next_page.group(1)) > 0
                return _native("read_file", {
                    "path": self.path, "start_char": int(next_page.group(1)),
                })
            self.recovered = True
            return _native("submit_report", {"content": marker, "confidence": 0.6})

    async def checkpoint(turn, messages, metadata):
        checkpoints.append(copy.deepcopy(messages))

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools(
            "dynir_researcher", task_id="readback", artifact_dir=tmp_path,
            config=_config(), http_client=http, sandbox_factory=lambda: sandbox,
        ) as runtime:
            memory = EvidenceMemory(runtime.artifact_dir, keep_tool_results=1)
            llm = ReadbackLLM()
            result = await run_agent_loop(
                system_prompt="Recover the exact diagnostic value before submitting.",
                user_message="Read the earlier result and report its final value.",
                llm=llm, tools=runtime.tools, parser=ResearchToolCallParser(),
                config=LoopConfig(
                    max_turns=7, max_llm_retries=1, task_id="readback",
                    role_id="dynir_researcher", compactor=memory,
                    compaction_policy=AlwaysCompactPolicy(),
                    loop_policy=LoopPolicy(terminal_tool_names=("submit_report",)),
                ),
                observers=[memory, SubmissionObserver(runtime)],
                on_turn_complete=checkpoint,
            )
            assert result.stopped_by == "submitted_report"
            assert result.final_content == marker and llm.recovered
            assert len(llm.requests) == (4 if size < 10_000 else 5)
            assert runtime.report_latch.report == marker
            artifact = Path(llm.path)
            stored = json.loads(artifact.read_text())
            assert marker in stored["content"]
            if size < 10_000:
                assert stored["content"] == content
                assert not any(item["tool_name"] == "run_python_code" for item in runtime.provenance)
            else:
                assert artifact.parent == runtime.artifact_dir
                assert len(stored["content"]) > 15_000
            archives = [path for path in runtime.artifact_dir.rglob("*.json")
                        if "evidence_id" in json.loads(path.read_text())]
            # Reading a page must reuse its source reference instead of
            # creating nested archives for successive read_file results.
            assert len(archives) == (2 if size < 10_000 else 1)
            original_call_id = next(message["tool_call_id"] for message in checkpoints[0]
                                    if message.get("role") == "tool")
            original_indices = [next(message["content"] for message in snapshot
                                     if message.get("tool_call_id") == original_call_id)
                                for snapshot in checkpoints[1:]]
            assert len(set(original_indices)) == 1
            call_ids = [call["id"] for message in result.messages
                        for call in message.get("tool_calls", [])]
            reply_ids = [message["tool_call_id"] for message in result.messages
                         if message.get("role") == "tool"]
            assert call_ids == reply_ids and len(call_ids) == len(set(call_ids))
    assert sandbox.kills == 1


@pytest.mark.asyncio
async def test_parallel_fetches_keep_source_refs_and_batch_outputs_separate(tmp_path):
    started = set()
    all_started = asyncio.Event()

    async def provider(request):
        url = json.loads(request.content)["url"]
        started.add(url)
        if len(started) == 4:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        return httpx.Response(200, json={"data": {"content": "# " + url + "\n" + "body " * 200}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools(
            "dynir_researcher", task_id="parallel", artifact_dir=tmp_path,
            config=_config(max_inline_chars=128), http_client=http,
        ) as runtime:
            calls = [{
                "name": "web_fetch", "id": group,
                "args": {"url": [f"https://paper.example/{group}1", f"https://paper.example/{group}2"]},
            } for group in ("a", "b")]
            results = await execute_tools(
                calls, {tool.name: tool for tool in runtime.tools},
                timeout=5, turn=1, count_offset=0,
            )
            memory = EvidenceMemory(runtime.artifact_dir, keep_tool_results=0)
            for call, result in zip(calls, results):
                assert not result.is_error, result.result
                refs = result.metadata["artifact_refs"]
                sources = [ref for ref in refs if ref["kind"] == "source"]
                canonical = [ref for ref in refs if ref["kind"] == "tool_output"]
                assert len(sources) == 2 and len(canonical) == 1
                source_urls = {json.loads(Path(ref["path"]).read_text())["requested_url"] for ref in sources}
                assert source_urls == set(call["args"]["url"])
                batch = json.loads(Path(canonical[0]["path"]).read_text())["content"]
                assert all(url in batch for url in source_urls)
                other_group = "b" if call["id"] == "a" else "a"
                assert f"https://paper.example/{other_group}" not in batch
                await memory.on_tool_result(SimpleNamespace(), result)
                compacted = await memory.compact([
                    {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.result},
                ], keep_recent=0)
                assert _read_args(compacted[0]["content"])["path"] == canonical[0]["path"]
                assert compacted[0]["content"].count("Related source") == 2
