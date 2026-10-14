from __future__ import annotations

import asyncio
import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_harness.core import atomic_files
from agent_harness.core.llm_errors import LLMQuotaError
from benchmarks.runner import run_subprocess as runner
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget


@pytest.mark.parametrize("stage", ["fsync", "replace"])
def test_failed_atomic_write_preserves_previous_result(tmp_path, monkeypatch, stage):
    path = tmp_path / "result.json"
    path.write_text('{"previous": true}')

    def fail(*args):
        raise OSError(28, "Synthetic storage failure")

    monkeypatch.setattr(atomic_files.os, stage, fail)
    with pytest.raises(OSError):
        atomic_files.atomic_write_text(path, '{"new": true}')
    assert path.read_text() == '{"previous": true}'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["quota", "cancel", "success"])
async def test_storage_failure_cannot_skip_cleanup_or_mask_primary(tmp_path, monkeypatch, primary):
    module = importlib.import_module("workflows.dynamic_htn_informed_research.nodes.dynamic_solve")
    original = module.TaskRuntime.write_json
    closed = AsyncMock()

    def write(self, name, payload):
        if name == "dispatch_budget.json":
            raise OSError(28, "Synthetic disk full")
        original(self, name, payload)

    class Driver:
        def __init__(self, *args):
            self.dispatch_budget = QuestionDispatchBudget()

        async def run(self, *args):
            if primary == "quota":
                raise LLMQuotaError()
            if primary == "cancel":
                raise asyncio.CancelledError()
            return {"final_content": "Preserved final report"}

    monkeypatch.setattr(module, "_QuestionDriver", Driver)
    monkeypatch.setattr(module.TaskRuntime, "write_json", write)
    monkeypatch.setattr(module.TaskRuntime, "aclose", closed)
    expected = {"quota": LLMQuotaError, "cancel": asyncio.CancelledError, "success": OSError}[primary]
    with pytest.raises(expected):
        await module.dynamic_solve_node(
            {"question": "Offline", "metadata": {"profile": "research_smoke", "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id="storage-test"),
        )
    closed.assert_awaited_once()
    if primary == "success":
        assert next(tmp_path.rglob("report.md")).read_text() == "Preserved final report"


@pytest.mark.asyncio
@pytest.mark.parametrize("old", [
    '{"question_id":"q", "error":"disk full", "is_correct":false}',
    '', '{broken', '[]', '{}', '{"question_id":"other","is_correct":true}',
])
async def test_resume_retries_failed_or_corrupt_results_and_archives_them(tmp_path, monkeypatch, old):
    trial = tmp_path / "trials/q"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(old)
    (trial / "agent.log").write_text("previous log\n")
    worker = tmp_path / "worker.py"
    worker.write_text(
        'import sys,json\nfrom pathlib import Path\n'
        'p=Path(sys.argv[2])/"trials"/sys.argv[1]/"result.json"\n'
        'p.write_text(json.dumps({"question_id":sys.argv[1],"is_correct":True}))\n'
        'print("new attempt",flush=True)\n'
    )
    monkeypatch.setattr(runner, "_WORKER_SCRIPT", worker)
    result = await runner.run_task_subprocess("q", tmp_path, "browsecomp_200", asyncio.Semaphore(1))
    assert result["is_correct"] is True
    assert next((tmp_path / "previous_attempts/q").glob("*/result.json")).read_text() == old
    assert (trial / "agent.log").read_text() == "previous log\nnew attempt\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"question_id": "q", "is_correct": False, "error": None},
    {"question_id": "q", "predicted_answer": "42", "is_correct": None, "scoring_error": "judge outage"},
])
async def test_resume_keeps_completed_solving_even_if_wrong_or_judge_failed(tmp_path, monkeypatch, result):
    trial = tmp_path / "trials/q"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(result))
    spawn = AsyncMock(side_effect=AssertionError("Must not rerun completed solving"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await runner.run_task_subprocess("q", tmp_path, "browsecomp_200", asyncio.Semaphore(1)) == result
    spawn.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["cancel", "log_failure", "timeout"])
async def test_runner_stops_real_child_on_cancellation_or_log_failure(tmp_path, monkeypatch, cause):
    worker = tmp_path / "worker.py"
    worker.write_text(
        'import sys,os,time\nfrom pathlib import Path\n'
        '(Path(sys.argv[2])/"child.pid").write_text(str(os.getpid()))\n'
        'print("ready",flush=True)\ntime.sleep(60)\n'
    )
    monkeypatch.setattr(runner, "_WORKER_SCRIPT", worker)
    if cause == "log_failure":
        async def tee(reader, *args, **kwargs):
            await reader.readline()
            raise OSError(28, "Synthetic log disk full")
        monkeypatch.setattr(runner, "_tee", tee)
    if cause == "timeout":
        monkeypatch.setattr(runner, "_QUESTION_TIMEOUT", 0.3)
    task = asyncio.create_task(runner.run_task_subprocess("q", tmp_path, "browsecomp_200", asyncio.Semaphore(1)))

    async def started():
        while not (tmp_path / "child.pid").exists():
            await asyncio.sleep(0.01)
    try:
        await asyncio.wait_for(started(), 5)
        if cause == "cancel":
            task.cancel()
        if cause == "timeout":
            result = await asyncio.wait_for(task, 5)
            assert result["error"].startswith("subprocess_timeout")
        else:
            with pytest.raises(asyncio.CancelledError if cause == "cancel" else OSError):
                await asyncio.wait_for(task, 5)
        pid = int((tmp_path / "child.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
