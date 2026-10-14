"""Exercise native failure recovery with local child processes and no model calls."""
import argparse
import asyncio
import json
from pathlib import Path

import pytest

from benchmarks import watch_drb
from benchmarks.runner import run_subprocess as runner

DRB = "deepresearch_bench_ii"


class CountingSemaphore(asyncio.Semaphore):
    def __init__(self, value):
        super().__init__(value)
        self.acquisitions = 0
        self.active = 0
        self.peak = 0

    async def __aenter__(self):
        await super().__aenter__()
        self.acquisitions += 1
        self.active += 1
        self.peak = max(self.peak, self.active)

    async def __aexit__(self, *args):
        self.active -= 1
        await super().__aexit__(*args)


def fake_worker(tmp_path, monkeypatch, *, signal_number=6, always_crash=False,
                publish_before_crash=False, normal_exit=False):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(f'''
import faulthandler,json,os,resource,signal,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
assert faulthandler.is_enabled()
assert os.environ['PYTHONUNBUFFERED']=='1'
qid,root=sys.argv[1],Path(sys.argv[2])
counter=root/(qid+'.attempts')
attempt=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(attempt))
trial=root/'trials'/qid
(trial/'agent').mkdir(exist_ok=True)
(trial/'agent'/'artifact.txt').write_text('attempt '+str(attempt))
(root/('.worker_kernel_'+qid+'.db')).write_text('database '+str(attempt))
print('attempt '+str(attempt),flush=True)
time.sleep(.03)
result={{'question_id':qid,'predicted_answer':'Completed report',
         'is_correct':None,'reward':None,'judge_method':'not_graded','error':None}}
if {publish_before_crash!r}:
    (trial/'result.json').write_text(json.dumps(result))
if attempt==1 or {always_crash!r}:
    if {normal_exit!r}:
        sys.exit(1)
    os.kill(os.getpid(), {signal_number})
(trial/'result.json').write_text(json.dumps(result))
''')
    monkeypatch.setattr(runner, "_WORKER_SCRIPT", worker)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_number", [6, 11])
async def test_native_crash_retries_once_and_preserves_artifacts(tmp_path, monkeypatch, signal_number):
    fake_worker(tmp_path, monkeypatch, signal_number=signal_number)
    sem = CountingSemaphore(1)
    result = await runner.run_task_subprocess("q", tmp_path, DRB, sem)
    assert result["predicted_answer"] == "Completed report" and result["error"] is None
    assert (tmp_path / "q.attempts").read_text() == "2"
    assert sem.acquisitions == 2 and sem.peak == 1 and sem.active == 0
    [archive] = (tmp_path / "previous_attempts/q").iterdir()
    assert (archive / "agent/artifact.txt").read_text() == "attempt 1"
    assert (archive / ".worker_kernel_q.db").read_text() == "database 1"
    assert "Fatal Python error" in (archive / "agent.log").read_text()
    diagnostic = json.loads((archive / "subprocess.json").read_text())
    assert diagnostic["exit_code"] == -signal_number and diagnostic["native_crash"]
    assert diagnostic["native_root_cause"] == "unknown"
    failed = json.loads((archive / "result.json").read_text())
    assert failed["worker_exit_code"] == -signal_number and failed["reward"] is None
    assert (tmp_path / "trials/q/agent/artifact.txt").read_text() == "attempt 2"


@pytest.mark.asyncio
async def test_repeated_native_crash_stops_after_two_attempts(tmp_path, monkeypatch):
    fake_worker(tmp_path, monkeypatch, always_crash=True)
    result = await runner.run_task_subprocess("q", tmp_path, DRB, asyncio.Semaphore(1))
    assert (tmp_path / "q.attempts").read_text() == "2"
    assert result["error"] == "worker_exit_-6_no_result"
    assert result["is_correct"] is None and result["reward"] is None
    assert json.loads((tmp_path / "trials/q/subprocess.json").read_text())["exit_code"] == -6
    assert len(list((tmp_path / "previous_attempts/q").iterdir())) == 1


@pytest.mark.asyncio
async def test_committed_report_is_reused_even_after_native_exit(tmp_path, monkeypatch):
    fake_worker(tmp_path, monkeypatch, publish_before_crash=True)
    result = await runner.run_task_subprocess("q", tmp_path, DRB, asyncio.Semaphore(1))
    assert result["predicted_answer"] == "Completed report"
    assert (tmp_path / "q.attempts").read_text() == "1"
    assert json.loads((tmp_path / "trials/q/subprocess.json").read_text())["native_crash"]

    async def forbidden(*args, **kwargs):
        pytest.fail("An existing completed report must not be regenerated")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    assert await runner.run_task_subprocess("q", tmp_path, DRB, asyncio.Semaphore(1)) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark,normal_exit", [(DRB, True), ("browsecomp_200", False),
                                                       ("frontier_science_research", False), ("prl_bench", False)])
async def test_only_drb_native_failures_retry(tmp_path, monkeypatch, benchmark, normal_exit):
    fake_worker(tmp_path, monkeypatch, normal_exit=normal_exit)
    result = await runner.run_task_subprocess("q", tmp_path, benchmark, asyncio.Semaphore(1))
    assert result["error"]
    assert (tmp_path / "q.attempts").read_text() == "1"


@pytest.mark.asyncio
async def test_many_retrying_questions_share_the_same_worker_limit(tmp_path, monkeypatch):
    fake_worker(tmp_path, monkeypatch)
    sem = CountingSemaphore(2)
    results = await asyncio.gather(*(runner.run_task_subprocess(str(i), tmp_path, DRB, sem)
                                     for i in range(5)))
    assert all(not result["error"] for result in results)
    assert sem.peak == 2 and sem.acquisitions == 10 and sem.active == 0


@pytest.mark.asyncio
async def test_runner_completion_reads_existing_watcher_scores(tmp_path, monkeypatch):
    from benchmarks.core import registry
    from benchmarks.core.question import BenchmarkQuestion

    question = BenchmarkQuestion(id="q", question="Offline test", ground_truth="{}",
                                 answer_type="report", metadata={"idx": 1})
    monkeypatch.setattr(registry, "load_questions", lambda *a, **kw: [question])

    async def worker(qid, out_dir, benchmark, sem, mirror):
        result = dict(question_id=qid, predicted_answer="Completed report", reward=None,
                      is_correct=None, error=None, judge_method="not_graded")
        trial = out_dir / "trials" / qid
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps(result))
        watch_drb.refresh_run_summary(out_dir, grading_status={
            "stage": "watching", "total": 1, "scored": 1, "grading": 0, "failed": 0,
            "aggregate": {"mean_score100": 0.0, "scored_reports": 1, "source": "official.csv"}})
        return result

    monkeypatch.setattr(runner, "run_task_subprocess", worker)
    args = argparse.Namespace(benchmark=DRB, pipeline="react_base", profile="fixture", limit=None,
                              offset=0, answer_type=None, category=None, no_shuffle=True,
                              concurrency=1, judge_variant=None)
    report = await runner.run_eval(args, out_dir=tmp_path, seed=42)
    assert report["graded"] == 1 and report["not_graded"] == 0
    assert report["official_mean_score100"] == 0.0
    assert report["generation"]["stage"] == "completed"
    assert json.loads((tmp_path / "results.json").read_text()) == report
    assert "0.0000" in (tmp_path / "summary.txt").read_text()
