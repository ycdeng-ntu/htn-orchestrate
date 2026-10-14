import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.runner import adopt


@pytest.mark.asyncio
async def test_active_pilot_occupies_slot_until_result_is_copied(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "full"
    trial = source / "trials/7"
    (trial / "agent").mkdir(parents=True)
    monkeypatch.setattr(adopt, "process_token", lambda _: "11:start")
    plan = adopt.Adoption(source, frozenset({"7"}), 11, "11:start")
    sem = asyncio.Semaphore(1)
    pending = asyncio.create_task(adopt.adopt_result(plan, "7", out, sem, poll_seconds=.01))
    await asyncio.sleep(.02)
    assert sem.locked() and not pending.done()
    acquired = asyncio.Event()
    async def new_job():
        async with sem:
            acquired.set()
    other = asyncio.create_task(new_job())
    await asyncio.sleep(.02)
    assert not acquired.is_set()
    original = json.dumps(dict(question_id="7", error=None, scoring_error=None,
                              judge_method="rubric", rubric_score=75, is_correct=None))
    (trial / "agent/final_content.md").write_text("Complete physics report")
    (trial / "result.json").write_text(original)
    result = await asyncio.wait_for(pending, 1)
    await other
    assert result["rubric_score"] == 75 and acquired.is_set()
    assert (out / "trials/7/agent/final_content.md").read_text() == "Complete physics report"
    assert (trial / "result.json").read_text() == original


@pytest.mark.asyncio
async def test_failed_or_dead_source_allows_local_retry(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "full"
    plan = adopt.Adoption(source, frozenset({"7"}), 11, "11:start")
    monkeypatch.setattr(adopt, "process_token", lambda _: None)
    sem = asyncio.Semaphore(1)
    assert await adopt.adopt_result(plan, "7", out, sem) is None
    assert not sem.locked()
    trial = source / "trials/7"; trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(dict(question_id="7", error="failed")))
    assert await adopt.adopt_result(plan, "7", out, sem) is None
    assert not (out / "trials/7/result.json").exists()
