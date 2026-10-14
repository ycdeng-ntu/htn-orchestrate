"""Worker reports are shared locally without exposing unrelated host artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent_harness.core.llm import LLMResponse
from workflows.dynamic_htn_informed_research.htn import AgentTaskSummary
from workflows.dynamic_htn_informed_research.runtime.context import TaskRuntime, override_runtime
from workflows.dynamic_htn_informed_research.runtime.worker import build_dispatch
from workflows.dynamic_htn_informed_research.tools import build_research_tools


def call(name, args):
    return LLMResponse(tool_calls=[{
        "id": "provider-call", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }])


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_ref_dir", [False, True])
async def test_following_worker_reads_complete_report_locally_with_paging(tmp_path, monkeypatch, custom_ref_dir):
    monkeypatch.setattr("dotenv.dotenv_values", lambda *args: {})
    report = "报告正文\n" + "测量值🧪" * 8000 + "\nDECISIVE_TAIL=42"
    report_ref = None
    configs = []
    sandbox_attempts = []
    recovered = []
    artifact_dir = tmp_path / "question-a"
    ref_dir = tmp_path / "explicit-reports" if custom_ref_dir else None

    def sandbox():
        sandbox_attempts.append(True)
        raise AssertionError("No sandbox is available in this offline readback test")

    def tools(role, **kwargs):
        configs.append(kwargs["config"])
        return build_research_tools(role, **kwargs, sandbox_factory=sandbox)

    class Model:
        model = "offline"

        async def chat(self, messages, **kwargs):
            if report_ref is None:
                return call("submit_report", {"content": report, "confidence": 0.8})
            reads = [m["content"] for m in messages if m["role"] == "tool"]
            if not reads:
                return call("read_file", {"path": report_ref, "start_char": 0})
            page = reads[-1]
            header, body = page.split("\n", 1)
            assert header.startswith("[read_file chars ")
            recovered.append(body)
            cursor = re.search(r"next_start_char=(\d+)", header)
            if cursor:
                return call("read_file", {"path": report_ref, "start_char": int(cursor.group(1))})
            assert "".join(recovered) == report
            return call("submit_report", {"content": "Read decisive report evidence: 42", "confidence": 0.8})

    with override_runtime(llm_factory=lambda *_: Model(), tool_factory=tools):
        async with TaskRuntime({"agent": {"max_llm_retries": 1}}, "a", artifact_dir):
            dispatch = build_dispatch("a", max_turns=10, sampling={}, provider="fake", model="fake", ref_dir=ref_dir)
            first = await dispatch(AgentTaskSummary("first", "dynir_researcher", "produce report"), "produce report")
            assert first.submitted
            report_ref = first.ref
            assert Path(report_ref).is_absolute() and Path(report_ref).read_text() == report
            second = await dispatch(AgentTaskSummary("second", "dynir_researcher", "read report"), f"Read {report_ref}")
            assert second.submitted and second.result_summary == "Read decisive report evidence: 42"
    assert len(recovered) > 1
    assert not sandbox_attempts
    expected_root = (ref_dir or artifact_dir / "refs").resolve()
    assert all(expected_root in cfg.read_roots for cfg in configs)
    assert all(artifact_dir.resolve() not in cfg.read_roots for cfg in configs)

    # A report-root symlink must not grant access to another question or to
    # diagnostics outside that root. Existing sandbox fallback cannot read host files.
    foreign = tmp_path / "question-b" / "refs" / "other.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("UNRELATED_REPORT")
    private = artifact_dir / "diagnostics.json"
    private.write_text("PRIVATE_DIAGNOSTICS")
    escaped = expected_root / "escape.md"
    escaped.symlink_to(foreign)
    async with build_research_tools("dynir_researcher", task_id="a", artifact_dir=artifact_dir / "check",
                                    config=configs[-1], sandbox_factory=sandbox) as reader:
        own = await reader.read_file(report_ref, start_char=len(report) - 16)
        assert "DECISIVE_TAIL=42" in own
        assert not sandbox_attempts
        for path in (foreign, private, escaped):
            with pytest.raises(AssertionError, match="No sandbox"):
                await reader.read_file(str(path))
