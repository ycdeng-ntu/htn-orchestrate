"""Offline DRB official-format export and deferred-grading accounting."""

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks import grade_drb
from benchmarks.core.harbor_task_generator import generate_task_dirs
from benchmarks.core.question import BenchmarkQuestion
from benchmarks.runner import check_progress, rejudge


def make_run(root: Path, *, original_id=False):
    metadata = {"idx": 7, "task_id": "task7", "language": "en", "theme": "science",
                "judge_task": "Official judge context"}
    generate_task_dirs([{
        "id": "7", "question": "Long solver prompt with blocked sources",
        "answer": json.dumps({"rubric": {"quality": ["check"]}, "blocked": {"site": "private"}}),
        "answer_type": "report", "evaluation_metadata": metadata,
        "benchmark": "deepresearch_bench_ii",
    }], root / "tasks")
    trial = root / "trials" / ("task7" if original_id else "7")
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "final_content.md").write_text(
        '# Full report\nEvidence [1]\n<!-- citation-map: {"1":"https://example.org"} -->',
        encoding="utf-8",
    )
    (trial / "result.json").write_text(json.dumps({
        "question_id": "7", "predicted_answer": "truncated legacy output",
        "is_correct": None, "reward": None, "judge_method": "not_graded",
        "rubric_score": None,
    }), encoding="utf-8")
    return trial


def test_export_prefers_saved_inputs_and_raw_report(tmp_path, monkeypatch):
    run = tmp_path / "run"
    make_run(run, original_id=True)
    monkeypatch.setattr(grade_drb, "load_questions", lambda *_: pytest.fail("must use saved inputs"))
    bundle = tmp_path / "bundle"
    manifest = grade_drb.export_bundle(run, bundle, model_name="test_model")
    assert manifest["status"] == "not_graded"
    assert manifest["reports"] == 1
    content = (bundle / "report" / "test_model" / "idx-7.md").read_text()
    assert content == "# Full report\nEvidence [1]\n"
    task = json.loads((bundle / "tasks_and_rubrics.jsonl").read_text())
    assert task == {
        "idx": 7, "language": "en", "theme": "science",
        "content": {"task": "Official judge context", "rubric": {"quality": ["check"]},
                    "blocked": {"site": "private"}},
    }
    assert "--tasks_jsonl" in manifest["commands"][0]
    assert "--tasks-file" in manifest["commands"][1]
    with pytest.raises(ValueError, match="already contains"):
        grade_drb.export_bundle(run, bundle)


def test_multi_run_exports_separate_model_buckets(tmp_path):
    run = tmp_path / "run"
    make_run(run / "run_1")
    make_run(run / "run_2")
    bundle = tmp_path / "bundle"
    manifest = grade_drb.export_bundle(run, bundle, model_name="test")
    assert manifest["reports"] == 2
    assert (bundle / "report" / "test_run_1" / "idx-7.md").is_file()
    assert (bundle / "report" / "test_run_2" / "idx-7.md").is_file()


def test_missing_judge_task_requires_explicit_nonofficial_fallback(tmp_path):
    q = BenchmarkQuestion(id="1", question="solver prompt", ground_truth='{"rubric": {}}',
                          answer_type="report", metadata={"idx": 1})
    dest = tmp_path / "tasks.jsonl"
    with pytest.raises(ValueError, match="no judge_task"):
        grade_drb.build_tasks_file("deepresearch_bench_ii", dest, [q])
    assert not dest.exists()
    grade_drb.build_tasks_file("deepresearch_bench_ii", dest, [q], allow_prompt_fallback=True)
    assert json.loads(dest.read_text())["content"]["task"] == "solver prompt"


def test_official_interface_waits_for_nonempty_scoring_output(tmp_path, monkeypatch):
    run = tmp_path / "run"
    make_run(run)
    official = tmp_path / "official"
    official.mkdir()
    for filename in ("run_evaluation.py", "aggregate_scores.py", "gpt_client.py"):
        (official / filename).write_text("# external evaluator fixture\n")
    bundle = tmp_path / "bundle"
    grade_drb.export_bundle(run, bundle, official_dir=official)
    calls = []

    def no_output(argv, **kwargs):
        calls.append(argv)

    monkeypatch.setattr(grade_drb.subprocess, "run", no_output)
    monkeypatch.setattr(grade_drb, "official_environment", lambda *_: {})
    with pytest.raises(RuntimeError, match="no scores"):
        grade_drb.run_official(official, bundle)
    assert len(calls) == 1
    assert json.loads((bundle / "manifest.json").read_text())["status"] == "not_graded"


def test_monitor_does_not_count_deferred_result_as_wrong(tmp_path, capsys):
    make_run(tmp_path)
    stats = check_progress._scan_trials_dir(tmp_path / "trials", "test")
    assert stats.n_done == 1 and stats.n_not_graded == 1
    assert stats.n_wrong == 0 and stats.wrong_samples == []
    check_progress.analyse(tmp_path, verbose=True)
    output = capsys.readouterr().out
    assert "N/A (not graded)" in output
    assert "Sample wrong answers" not in output


@pytest.mark.asyncio
async def test_rejudge_defers_drb_without_model_or_dataset(tmp_path):
    trial = make_run(tmp_path)
    outcome = await rejudge._rejudge_one(
        trial / "result.json", "deepresearch_bench_ii", {}, asyncio.Semaphore(1), True,
    )
    assert outcome == "not_graded"
    result = json.loads((trial / "result.json").read_text())
    assert result["is_correct"] is None and result["reward"] is None


def test_gpt_judge_selects_one_credential_pair_without_mutating_parent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "parent-key")
    monkeypatch.setenv("OPENAI_MODEL", "parent-model")
    source = {
        "OPENROUTER_API_KEY": "selected-key",
        "OPENROUTER_BASE_URL": "https://judge.example/v1/chat/completions/",
        "OPENAI_API_KEY": "unrelated-key", "OPENAI_API_URL": "https://other.example/v1/chat/completions",
    }
    judge = {"provider": "openrouter", "model": "gpt-5.5", "reasoning_effort": "medium"}
    env = grade_drb.official_environment(judge, source=source)
    assert env["OPENAI_API_KEY"] == "selected-key"
    assert env["OPENAI_API_URL"] == "https://judge.example/v1/chat/completions"
    assert env["OPENAI_MODEL"] == "gpt-5.5"
    assert env["OPENAI_REASONING_EFFORT"] == "medium"
    assert grade_drb.os.environ["OPENAI_API_KEY"] == "parent-key"
    assert grade_drb.os.environ["OPENAI_MODEL"] == "parent-model"
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        grade_drb.official_environment(judge, source={"OPENAI_API_KEY": "unrelated-key"})


def test_grade_uses_exported_judge_snapshot_and_keeps_secrets_out_of_bundle(tmp_path, monkeypatch):
    import yaml

    run = tmp_path / "run"
    make_run(run)
    official = tmp_path / "official"
    official.mkdir()
    for name in ("run_evaluation.py", "aggregate_scores.py", "gpt_client.py"):
        (official / name).touch()
    profile = tmp_path / "judge.yaml"
    config = {
        "benchmark": "deepresearch_bench_ii",
        "judge": {"provider": "openrouter", "model": "gpt-5.5", "reasoning_effort": "medium"},
    }
    profile.write_text(yaml.safe_dump(config))
    bundle = tmp_path / "bundle"
    manifest = grade_drb.export_bundle(run, bundle, official_dir=official, benchmark_profile=profile)
    assert manifest["judge"] == {
        "provider": "openrouter", "model": "gpt-5.5", "reasoning_effort": "medium",
    }
    config["judge"]["model"] = "changed-after-export"
    profile.write_text(yaml.safe_dump(config))
    calls = []

    def environment(judge):
        assert judge == manifest["judge"]
        return grade_drb.official_environment_original(judge, source={
            "OPENROUTER_API_KEY": "credential-canary",
            "OPENROUTER_BASE_URL": "https://judge.example/v1",
        })

    monkeypatch.setattr(grade_drb, "official_environment_original", grade_drb.official_environment,
                        raising=False)
    monkeypatch.setattr(grade_drb, "official_environment", environment)

    def evaluate(argv, **kwargs):
        calls.append((argv, kwargs))
        if Path(argv[1]).name == "run_evaluation.py":
            assert kwargs["env"]["OPENAI_MODEL"] == "gpt-5.5"
            assert kwargs["env"]["OPENAI_API_KEY"] == "credential-canary"
            (bundle / "result.jsonl").write_text('{"idx":7,"result":{}}\n')

    monkeypatch.setattr(grade_drb.subprocess, "run", evaluate)
    grade_drb.run_official(official, bundle)
    assert len(calls) == 2
    saved = (bundle / "manifest.json").read_text()
    assert "secret-canary" not in saved and "credential-canary" not in saved
    assert json.loads(saved)["status"] == "official_scores_exported"
    original = json.loads((run / "trials/7/result.json").read_text())
    assert original["judge_method"] == "not_graded" and original["is_correct"] is None


def test_explicit_drb_judge_variant_and_default():
    assert grade_drb.resolve_benchmark_judge("drb_gpt55_medium") == {
        "provider": "openrouter", "model": "gpt-5.5", "reasoning_effort": "medium"}
    assert grade_drb.resolve_benchmark_judge() == {
        "provider": "openrouter", "model": "openai/gpt-5.4", "reasoning_effort": "high"}
