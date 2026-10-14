"""Streaming grading must preserve completed work and exclude failed scores."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from benchmarks import watch_drb
from benchmarks.core.harbor_task_generator import generate_task_dirs


def setup_run(tmp_path, monkeypatch, count=3):
    run, official = tmp_path / "run", tmp_path / "official"
    official.mkdir()
    rubric = {"info_recall": ["fact"], "analysis": ["reason"], "presentation": ["format"]}
    generate_task_dirs([
        dict(id=str(i), question=f"Prompt {i}", answer=json.dumps({"rubric": rubric}),
             answer_type="report", benchmark="deepresearch_bench_ii",
             evaluation_metadata=dict(idx=i, judge_task="Judge task"))
        for i in range(1, count + 1)
    ], run / "tasks")
    (run / "trials").mkdir()
    (official / "gpt_client.py").touch()
    (official / "run_evaluation.py").write_text('''
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['pdf_dir','out_jsonl','tasks_jsonl','log_file']: p.add_argument('--'+k)
a=p.parse_args()
time.sleep(0.15)
task=json.loads(Path(a.tasks_jsonl).read_text())
model=next(Path(a.pdf_dir).iterdir()).name
result={'scores':{d:{v:{'score':0 if task['idx']==2 else 1} for v in items} for d,items in task['content']['rubric'].items()}}
if task['idx']==3: result={'error':'simulated official API failure'}
Path(a.out_jsonl).write_text(json.dumps({'idx':task['idx'],'model':model,'result':result})+'\\n')
''')
    (official / "aggregate_scores.py").write_text('''
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['input','output-prefix','tasks-file']: p.add_argument('--'+k)
a=p.parse_args()
rows=[json.loads(l) for l in Path(a.input).read_text().splitlines()]
assert all(not r['result'].get('error') for r in rows)
Path(a.output_prefix+'.csv').write_text(str(len(rows)))
model=rows[0]['model']
with Path(a.output_prefix+'_total.csv').open('w', newline='') as out:
    writer=csv.writer(out)
    writer.writerow(['idx','language','theme',model,'avg'])
    values=[]
    for row in rows:
        scores=[v['score'] for dim in row['result']['scores'].values() for v in dim.values()]
        value=sum(scores)/len(scores)
        values.append(value)
        writer.writerow([row['idx'],'','',value,value])
    mean=sum(values)/len(values)
    writer.writerow(['avg','','',mean,mean])
''')
    monkeypatch.setattr(watch_drb.grade_drb, "resolve_benchmark_judge", lambda *_, **kw: {
        "provider": "openrouter", "model": "gpt-5.5", "reasoning_effort": "medium"})
    monkeypatch.setattr(watch_drb.grade_drb, "official_environment", lambda *_: dict(os.environ))
    return run, official


def complete(run, idx, *, error=None):
    trial = run / "trials" / str(idx)
    (trial / "agent").mkdir(parents=True, exist_ok=True)
    (trial / "agent" / "final_content.md").write_text("Full report\n<!-- citation-map: {} -->")
    (trial / "result.json").write_text(json.dumps(dict(
        question_id=str(idx), predicted_answer="short", error=error, judge_method="not_graded")))


def drain(grader):
    for _ in range(100):
        grader.step()
        if not any(r["status"] == "running" for r in grader.records.values()):
            return
        time.sleep(.02)
    pytest.fail("fake evaluator did not finish")


def test_grade_during_generation_then_resume_without_duplicate_calls(tmp_path, monkeypatch):
    run, official = setup_run(tmp_path, monkeypatch)
    complete(run, 1)
    # A partially written report is not eligible until result.json is committed.
    partial = run / "trials/2/agent"
    partial.mkdir(parents=True)
    (partial / "final_content.md").write_text("Still writing")
    grader = watch_drb.LiveGrader(run, tmp_path / "live", official, tmp_path / "profile",
                                  "test", 2, os.getpid())
    grader.step()
    assert set(grader.records) == {"1"}
    drain(grader)
    assert grader.records["1"]["status"] == "scored"
    status = json.loads((grader.out / "status.json").read_text())
    assert status["generation_running"] and status["scored"] == 1
    complete(run, 2)
    complete(run, 3)
    grader.step()
    assert sum(r["status"] == "running" for r in grader.records.values()) == 2
    drain(grader)
    assert grader.records["3"]["status"] == "failed"
    rows = [json.loads(l) for l in (grader.out / "result.jsonl").read_text().splitlines()]
    assert {r["idx"] for r in rows} == {1, 2}
    assert (grader.out / "agg_scores.csv").read_text() == "2"
    assert json.loads((run / "trials/1/result.json").read_text())["judge_method"] == "not_graded"
    monkeypatch.setattr(watch_drb.subprocess, "Popen", lambda *_a, **_k: pytest.fail("duplicate judging"))
    restarted = watch_drb.LiveGrader(run, grader.out, official, tmp_path / "profile", "test", 2, os.getpid())
    restarted.last_merged = tuple(q.id for q in restarted.questions if restarted.records[q.id]["status"] == "scored")
    restarted.step()
    assert restarted.records == grader.records


def test_failed_generation_is_skipped_and_dead_producer_finishes(tmp_path, monkeypatch):
    run, official = setup_run(tmp_path, monkeypatch, count=1)
    complete(run, 1, error="worker_exit_-11_no_result")
    grader = watch_drb.LiveGrader(run, tmp_path / "live", official, tmp_path / "profile", "test", 2, 999999999)
    assert grader.step()
    assert not grader.records
    assert json.loads((grader.out / "status.json").read_text())["stage"] == "completed_with_missing_or_failed_reports"


def test_incomplete_official_scores_are_rejected(tmp_path):
    from benchmarks.core.question import BenchmarkQuestion
    q = BenchmarkQuestion(id="1", question="prompt", answer_type="report",
                          ground_truth=json.dumps({"rubric": {"analysis": ["a", "b"]}}),
                          metadata={"idx": 1})
    (tmp_path / "result.jsonl").write_text(json.dumps({"idx": 1, "model": "test", "result": {
        "scores": {"analysis": {"a": {"score": 1}}}}}))
    with pytest.raises(ValueError, match="coverage"):
        watch_drb.validated_result(tmp_path, q, "test")


@pytest.mark.parametrize("generation_first", [True, False])
def test_shared_summary_keeps_latest_grading_in_both_completion_orders(tmp_path, monkeypatch, generation_first):
    run, official = setup_run(tmp_path, monkeypatch)
    for idx in (1, 2, 3):
        complete(run, idx)
    generation = dict(benchmark="deepresearch_bench_ii", total=3, pipeline="fixture",
                      generation_stage="completed", graded=0, not_graded=3,
                      results=[json.loads(path.read_text()) for path in sorted((run / "trials").glob("*/result.json"))])
    if generation_first:
        watch_drb.refresh_run_summary(run, generation_report=generation)
    grader = watch_drb.LiveGrader(run, tmp_path / "live", official, tmp_path / "profile", "test", 2, 999999999)
    drain(grader)
    if not generation_first:
        watch_drb.refresh_run_summary(run, generation_report=generation)
    # The second report has a valid official zero; the third failed without a score.
    report = json.loads((run / "results.json").read_text())
    assert report["graded"] == 2 and report["not_graded"] == 1
    assert report["grading"]["failed"] == 1
    assert report["generation"]["failed"] == 0
    assert report["official_mean_score100"] == 50
    assert report["grading"]["aggregate"]["scored_reports"] == 2
    assert report["correct"] is None and report["accuracy"] is None
    assert all(row["judge_method"] == "not_graded" for row in report["results"])
    text = (run / "summary.txt").read_text()
    assert "2/3 scored" in text and "denominator: 2 scored reports" in text
    assert "50.0000" in text
    assert json.loads((run / "generation_results.json").read_text())["graded"] == 0
    # A stale generation-only snapshot cannot reset the joined scoring view.
    again = watch_drb.refresh_run_summary(run, generation_report=generation)
    assert again["graded"] == 2 and again["official_mean_score100"] == 50


def test_official_aggregate_zero_is_a_score_and_missing_is_not_zero(tmp_path):
    aggregate = tmp_path / "agg_scores_total.csv"
    aggregate.write_text("idx,language,theme,test,avg\n2,en,science,0,0\navg,,,0,0\n")
    assert watch_drb.official_aggregate_summary(tmp_path, "test", 1)["mean_score100"] == 0
    assert watch_drb.official_aggregate_summary(tmp_path, "test", 0)["mean_score100"] is None
    mismatch = watch_drb.official_aggregate_summary(tmp_path, "test", 2)
    assert mismatch["mean_score100"] is None and mismatch["errors"]
    aggregate.unlink()
    assert watch_drb.official_aggregate_summary(tmp_path, "test", 1)["mean_score100"] is None


def test_failed_generation_summary_does_not_fabricate_scores(tmp_path, monkeypatch):
    run, official = setup_run(tmp_path, monkeypatch, count=1)
    complete(run, 1, error="worker_exit_-11_no_result")
    grader = watch_drb.LiveGrader(run, tmp_path / "live", official, tmp_path / "profile", "test", 2, 999999999)
    assert grader.step()
    report = json.loads((run / "results.json").read_text())
    assert report["graded"] == 0 and report["not_graded"] == 1
    assert report["official_mean_score100"] is None
    assert report["generation"]["failed"] == 1
