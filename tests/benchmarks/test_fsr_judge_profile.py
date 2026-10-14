from __future__ import annotations

import asyncio
import json
import tomllib
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.judges import frontier_science, score_answer
from benchmarks.runner import rejudge, run_one_task, run_subprocess
from benchmarks.runner.judge_config import resolve_judge_config


CONFIG = {"provider": "openrouter", "model": "openai/gpt-5.4", "reasoning_effort": "high"}
SNAPSHOT = {"configuration": {"benchmark": "frontier_science_research", "judge": CONFIG}}
HTN_PIPELINES = ["dynamic_htn_informed_research"]


def _task(tmp_path: Path, *, profile: str = "report_research",
          pipeline: str = "dynamic_htn_informed_research") -> Path:
    path = tmp_path / "tasks" / "q1"
    path.mkdir(parents=True)
    (path / "task.toml").write_text(
        f'[metadata]\npipeline_id = "{pipeline}"\n'
        f'profile = "{profile}"\n', encoding="utf-8",
    )
    (path / "instruction.md").write_text("A scientific problem", encoding="utf-8")
    return path


@pytest.mark.parametrize("pipeline", HTN_PIPELINES + ["react_base"])
def test_judge_uses_evaluation_snapshot_independent_of_solver(tmp_path, pipeline):
    task = _task(tmp_path, profile="does-not-exist", pipeline=pipeline)
    (task / "tests").mkdir()
    (task / "tests/evaluation.json").write_text(json.dumps({"benchmark_profile": SNAPSHOT}))
    assert resolve_judge_config("frontier_science_research", task) == CONFIG
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_judge_config("frontier_science_olympiad", task)
    saved = {**CONFIG, "api_key": "secret-canary", "base_url": "private-endpoint"}
    assert resolve_judge_config("frontier_science_research", task, saved_config=saved) == CONFIG


def test_judge_does_not_read_solver_overrides(tmp_path):
    task = _task(tmp_path)
    with (task / "task.toml").open("a") as handle:
        handle.write("dynir_config = '{\"judge\": {\"model\": \"solver-owned\"}}'\n")
    assert resolve_judge_config("frontier_science_research", task) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("pipeline", "requested_profile", "expected_profile"), [
    (HTN_PIPELINES[0], None, "auto"),
    ("react_base", None, "default"),
    (HTN_PIPELINES[0], "explicit-profile", "explicit-profile"),
])
async def test_subprocess_runner_preserves_version_and_profile_in_generated_task(
    tmp_path, monkeypatch, pipeline, requested_profile, expected_profile,
):
    from benchmarks.core.question import BenchmarkQuestion

    if pipeline == HTN_PIPELINES[0] and requested_profile is not None:
        module = import_module(f"workflows.{pipeline}.profile")
        profile_dir = tmp_path / "profiles"
        profile_dir.mkdir()
        (profile_dir / f"{requested_profile}.yaml").write_text("behavior: {strategy: research}\n")
        monkeypatch.setattr(module, "_PROFILES_DIR", profile_dir)
    question = BenchmarkQuestion("q1", "A scientific problem", "A rubric", "report")
    monkeypatch.setattr("benchmarks.core.registry.load_questions", lambda *args, **kwargs: [question])
    seen = []

    async def no_subprocess(qid, out_dir, benchmark, sem, mirror=False):
        task = out_dir / "tasks" / qid
        metadata = tomllib.loads((task / "task.toml").read_text())["metadata"]
        seen.append(metadata)
        assert metadata["pipeline_id"] == pipeline
        assert metadata["profile"] == expected_profile
        if expected_profile == "auto":
            assert resolve_judge_config(benchmark, task) == CONFIG
        return {"question_id": qid, "predicted_answer": "offline report", "is_correct": True,
                "judge_method": "offline-test"}

    monkeypatch.setattr(run_subprocess, "run_task_subprocess", no_subprocess)
    args = SimpleNamespace(
        benchmark="frontier_science_research", pipeline=pipeline, profile=requested_profile,
        limit=None, offset=0, answer_type=None, category=None, subset=None,
        no_shuffle=True, concurrency=1, judge_variant=None, resolved_benchmark_profile=SNAPSHOT,
    )
    result = await run_subprocess.run_eval(args, out_dir=tmp_path / "run", seed=0)
    assert len(seen) == 1
    assert result["pipeline"] == pipeline and result["profile"] == expected_profile


class _Client:
    def __init__(self, transport, calls, *, cancel=False):
        self.transport = transport
        self.calls = calls
        self.cancel = cancel
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def create(self, **kwargs):
        self.calls.append({"transport": self.transport, "request": kwargs})
        await asyncio.sleep(0)
        if self.cancel:
            raise asyncio.CancelledError()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="VERDICT: 8.5"))])


def _clients(monkeypatch, *, cancel=False):
    calls, clients = [], []

    def factory(**transport):
        client = _Client(transport, calls, cancel=cancel)
        clients.append(client)
        return client

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-canary")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.test/v1/chat/completions/")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-canary")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.test/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "unrelated-judge-canary")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://unrelated.test/v1")
    monkeypatch.setattr(frontier_science, "AsyncOpenAI", factory)
    return calls, clients


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline", HTN_PIPELINES)
async def test_main_runner_applies_fsr_profile_to_request_and_saves_safe_metadata(tmp_path, monkeypatch, pipeline):
    calls, clients = _clients(monkeypatch)
    task = _task(tmp_path, pipeline=pipeline)
    (task / "tests").mkdir()
    (task / "tests/evaluation.json").write_text(json.dumps({"benchmark_profile": SNAPSHOT}))

    class Agent:
        async def run(self, *_args):
            target = self.logs_dir / "agent"
            target.mkdir()
            (target / "final_answer.txt").write_text("Complete scientific report", encoding="utf-8")

    result = await run_one_task.run_one_question(
        Agent(),
        {"id": "q1", "question": "A scientific problem", "ground_truth": "A rubric", "answer_type": "report"},
        tmp_path / "trials" / "q1", task, "frontier_science_research",
    )

    assert result["judge_config"] == CONFIG
    assert result["rubric_score"] == 8.5
    assert result["is_correct"] is True
    [call] = calls
    assert call["request"]["model"] == "openai/gpt-5.4"
    assert call["request"]["reasoning_effort"] == "high"
    assert call["transport"] == {"api_key": "openrouter-test-canary", "base_url": "https://router.test/v1"}
    assert all(client.closed for client in clients)
    saved = (tmp_path / "trials" / "q1" / "result.json").read_text()
    assert "canary" not in saved and "router.test" not in saved


@pytest.mark.asyncio
async def test_concurrent_fsr_judges_keep_model_effort_and_provider_isolated(monkeypatch):
    calls, clients = _clients(monkeypatch)
    other = {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "low"}
    results = await asyncio.gather(*[
        score_answer("frontier_science_research", "question", "rubric", "report", judge_config=config)
        for config in (CONFIG, other)
    ])
    assert results == [("CORRECT", 8.5), ("CORRECT", 8.5)]
    assert {(call["request"]["model"], call["request"]["reasoning_effort"], call["transport"]["base_url"])
            for call in calls} == {
        ("openai/gpt-5.4", "high", "https://router.test/v1"),
        ("gpt-5.4", "low", "https://openai.test/v1"),
    }
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_profile_judge_client_closes_on_cancellation(monkeypatch):
    _, clients = _clients(monkeypatch, cancel=True)
    with pytest.raises(asyncio.CancelledError):
        await score_answer("frontier_science_research", "q", "rubric", "report", judge_config=CONFIG)
    assert len(clients) == 1 and clients[0].closed


@pytest.mark.asyncio
async def test_olympiad_does_not_inherit_fsr_profile_config(monkeypatch):
    captured = []

    async def call(kwargs, *, label, allow_effort_fallback=False):
        captured.append(kwargs)
        return "CORRECT"

    monkeypatch.setattr(frontier_science, "_judge_call_with_effort_fallback", call)
    await score_answer("frontier_science_olympiad", "q", "gold", "answer", judge_config=CONFIG)
    assert captured
    assert all(call["model"] == "openai/gpt-5.4" for call in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline", HTN_PIPELINES)
async def test_rejudge_reuses_saved_profile_instead_of_current_task_settings(tmp_path, monkeypatch, pipeline):
    task = _task(tmp_path, profile="not-the-saved-profile", pipeline=pipeline)
    (task / "tests").mkdir()
    (task / "tests" / "evaluation.json").write_text(json.dumps({
        "id": "q1", "benchmark": "frontier_science_research", "question": "Full scientific question",
        "ground_truth": "rubric", "answer_type": "report",
    }))
    path = tmp_path / "trials" / "q1" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"question_id": "q1", "predicted_answer": "report", "judge_config": CONFIG}))
    captured = []

    async def score(*_args, **kwargs):
        captured.append(kwargs["judge_config"])
        return 1, "llm", None, 9.0

    monkeypatch.setattr(rejudge, "_score_answer", score)
    assert await rejudge._rejudge_one(path, "frontier_science_research", {}, asyncio.Semaphore(1), True) == "ok"
    assert captured == [CONFIG]
    assert json.loads(path.read_text())["judge_config"] == CONFIG


def test_fsr_openai_uses_new_api_pair(monkeypatch):
    monkeypatch.setenv('NEW_API_API_KEY', 'new-test-key')
    monkeypatch.setenv('NEW_API_BASE_URL', 'https://new.example/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'old-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://old.example/v1')
    monkeypatch.setattr(frontier_science, 'AsyncOpenAI', lambda **kw: kw)
    assert frontier_science._profile_judge_client('openai') == {
        'api_key': 'new-test-key', 'base_url': 'https://new.example/v1'}
    monkeypatch.delenv('NEW_API_BASE_URL')
    with pytest.raises(ValueError, match='NEW_API_BASE_URL'):
        frontier_science._profile_judge_client('openai')


@pytest.mark.asyncio
async def test_rejudge_retries_null_score_and_then_skips_success(tmp_path, monkeypatch):
    task = _task(tmp_path)
    (task / 'tests').mkdir()
    (task / 'tests/evaluation.json').write_text(json.dumps({
        'id': 'q1', 'benchmark': 'frontier_science_research', 'question': 'question',
        'ground_truth': 'rubric', 'answer_type': 'report'}))
    path = tmp_path / 'trials/q1/result.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'question_id': 'q1', 'predicted_answer': 'keep this',
                               'rubric_score': None, 'is_correct': None,
                               'scoring_error': 'quota', 'judge_config': CONFIG}))

    async def score(*args, **kwargs):
        return 0, 'llm', None, 3.0

    monkeypatch.setattr(rejudge, '_score_answer', score)
    assert await rejudge._rejudge_one(path, 'frontier_science_research', {}, asyncio.Semaphore(1), False) == 'ok'
    assert json.loads(path.read_text())['predicted_answer'] == 'keep this'
    assert await rejudge._rejudge_one(path, 'frontier_science_research', {}, asyncio.Semaphore(1), False) == 'skip'


def _rejudge_snapshot(tmp_path, *, saved_config, pipeline=HTN_PIPELINES[0],
                      profile="not-the-saved-profile"):
    task = _task(tmp_path, pipeline=pipeline, profile=profile)
    (task / "tests").mkdir()
    snapshot = {
        "id": "q1", "benchmark": "frontier_science_research",
        "question": "Full scientific question. " * 30 + "FINAL PUBLIC REQUIREMENT",
        "ground_truth": "Saved complete rubric. " * 20 + "FINAL RUBRIC ITEM",
        "answer_type": "report",
    }
    if pipeline != "react_base":
        snapshot["benchmark_profile"] = SNAPSHOT
    (task / "tests" / "evaluation.json").write_text(json.dumps(snapshot))
    path = tmp_path / "trials" / "q1" / "result.json"
    path.parent.mkdir(parents=True)
    row = {
        "question_id": "q1", "question": "DISPLAY_PREVIEW_ONLY",
        "predicted_answer": "Complete original report with equations and derivations.",
        "duration_seconds": 1234.5, "rubric_score": None, "is_correct": None,
        "scoring_error": "previous provider failed",
    }
    if saved_config is not None:
        row["judge_config"] = saved_config
    path.write_text(json.dumps(row))
    return path, snapshot, row


@pytest.mark.asyncio
@pytest.mark.parametrize(("saved", "provider", "key", "endpoint"), [
    ({"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"},
     "openrouter", "openrouter-test-canary", "https://router.test/v1"),
    ({"provider": "openrouter", "model": "openai/gpt-5.4", "reasoning_effort": "low"},
     "openai", "new-test-canary", "https://new.test/v1"),
])
async def test_rejudge_provider_override_routes_saved_full_answer_without_changing_model_effort(
    tmp_path, monkeypatch, saved, provider, key, endpoint,
):
    calls, clients = _clients(monkeypatch)
    monkeypatch.setenv("NEW_API_API_KEY", "new-test-canary")
    monkeypatch.setenv("NEW_API_BASE_URL", "https://new.test/v1")
    path, snapshot, before = _rejudge_snapshot(tmp_path, saved_config=saved)
    assert await rejudge._rejudge_one(
        path, "frontier_science_research", {}, asyncio.Semaphore(1), False,
        judge_provider=provider,
    ) == "ok"
    [call] = calls
    assert call["transport"] == {"api_key": key, "base_url": endpoint}
    assert call["request"]["model"] == saved["model"]
    assert call["request"]["reasoning_effort"] == saved["reasoning_effort"]
    prompt = call["request"]["messages"][0]["content"]
    assert snapshot["question"] in prompt and snapshot["ground_truth"] in prompt
    assert before["predicted_answer"] in prompt and "DISPLAY_PREVIEW_ONLY" not in prompt
    result = json.loads(path.read_text())
    assert result["judge_config"] == {**saved, "provider": provider}
    assert result["predicted_answer"] == before["predicted_answer"]
    assert result["duration_seconds"] == before["duration_seconds"]
    assert result["rubric_score"] == 8.5 and result["scoring_error"] is None
    assert all(client.closed for client in clients)
    assert "canary" not in path.read_text() and endpoint not in path.read_text()
    graded_bytes = path.read_bytes()
    # Changing the requested provider must not rescore or rewrite a valid grade.
    other_provider = "openai" if provider == "openrouter" else "openrouter"
    assert await rejudge._rejudge_one(
        path, "frontier_science_research", {}, asyncio.Semaphore(1), False,
        judge_provider=other_provider,
    ) == "skip"
    assert len(calls) == 1 and path.read_bytes() == graded_bytes


@pytest.mark.asyncio
async def test_rejudge_provider_override_uses_task_profile_when_no_saved_config(tmp_path, monkeypatch):
    path, _, _ = _rejudge_snapshot(
        tmp_path, saved_config=None, profile="deepseek_v4_flash_fsr_report",
    )
    seen = []

    async def score(*args, **kwargs):
        seen.append(kwargs["judge_config"])
        return 0, "llm", None, 3.0

    monkeypatch.setattr(rejudge, "_score_answer", score)
    assert await rejudge._rejudge_one(
        path, "frontier_science_research", {}, asyncio.Semaphore(1), False,
        judge_provider="openai",
    ) == "ok"
    assert seen == [{**CONFIG, "provider": "openai"}]
    assert json.loads(path.read_text())["judge_config"] == seen[0]


@pytest.mark.asyncio
async def test_rejudge_provider_override_refuses_to_guess_missing_model_config(tmp_path, monkeypatch):
    path, _, before = _rejudge_snapshot(tmp_path, saved_config=None, pipeline="react_base")
    seen = []

    async def score(*args, **kwargs):
        seen.append(kwargs["judge_config"])
        return 1, "llm", None, 8.0

    monkeypatch.setattr(rejudge, "_score_answer", score)
    assert await rejudge._rejudge_one(
        path, "frontier_science_research", {}, asyncio.Semaphore(1), False,
        judge_provider="openrouter",
    ) == "fail"
    failed = json.loads(path.read_text())
    assert "refusing to guess model or reasoning effort" in failed["scoring_error"]
    assert failed["judge_config"] is None and failed["rubric_score"] is None
    assert failed["predicted_answer"] == before["predicted_answer"]
    assert failed["duration_seconds"] == before["duration_seconds"]
    assert seen == []
    # Omitting the new flag preserves the old unconfigured judge path.
    assert await rejudge._rejudge_one(
        path, "frontier_science_research", {}, asyncio.Semaphore(1), False,
    ) == "ok"
    assert seen == [None]


def test_rejudge_cli_forwards_provider_to_failed_fsr_trial(tmp_path, monkeypatch):
    calls, _ = _clients(monkeypatch)
    saved = {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"}
    path, _, _ = _rejudge_snapshot(tmp_path, saved_config=saved)
    (tmp_path / "config.json").write_text(json.dumps({"benchmark": "frontier_science_research"}))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(rejudge.sys, "argv", [
        "rejudge", str(tmp_path), "--judge-provider", "openrouter", "--concurrency", "1",
    ])
    assert rejudge.main() == 0
    assert len(calls) == 1
    assert json.loads(path.read_text())["judge_config"] == {**saved, "provider": "openrouter"}


@pytest.mark.parametrize("benchmark", ["frontier_science_olympiad", "browsecomp_200"])
def test_rejudge_cli_rejects_provider_for_non_fsr_before_collecting_trials(
    tmp_path, monkeypatch, capsys, benchmark,
):
    (tmp_path / "config.json").write_text(json.dumps({"benchmark": benchmark}))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(rejudge.sys, "argv", [
        "rejudge", str(tmp_path), "--judge-provider", "openrouter",
    ])
    monkeypatch.setattr(rejudge, "_collect_results", lambda *_: pytest.fail("must reject before reading trials"))
    with pytest.raises(SystemExit) as error:
        rejudge.main()
    assert error.value.code == 2
    assert "--judge-provider is only supported for frontier_science_research" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(("benchmark", "provider"), [
    ("frontier_science_olympiad", "openrouter"),
    ("frontier_science_research", "invalid-provider"),
])
async def test_direct_rejudge_rejects_invalid_provider_override_before_touching_result(
    tmp_path, benchmark, provider,
):
    with pytest.raises(ValueError, match="--judge-provider"):
        await rejudge._rejudge_one(
            tmp_path / "does-not-exist.json", benchmark, {}, asyncio.Semaphore(1), False,
            judge_provider=provider,
        )


@pytest.mark.asyncio
async def test_olympiad_gateway_alias_preserves_high_effort(monkeypatch):
    monkeypatch.setenv('JUDGE_BASE_URL', 'https://llm-hub.apodex.app/v1')
    captured = {}

    async def call(kwargs, **options):
        captured.update(kwargs=kwargs, options=options)
        return 'VERDICT: CORRECT'

    monkeypatch.setattr(frontier_science, '_judge_call_with_effort_fallback', call)
    assert await frontier_science._call_olympiad_verdict_once('q') == 'CORRECT'
    assert captured['kwargs']['model'] == 'gpt-5.4'
    assert captured['kwargs']['reasoning_effort'] == 'high'
    assert captured['options']['allow_effort_fallback'] is False
    monkeypatch.setenv('JUDGE_BASE_URL', 'https://openrouter.ai/api/v1')
    assert frontier_science._olympiad_judge_model() == 'openai/gpt-5.4'


def test_docker_benchmark_credentials_are_paired(monkeypatch):
    from scripts import benchmark_credentials as credentials
    monkeypatch.setattr(credentials, 'load_dotenv', lambda *a: None)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'https://gateway.example/v1')
    monkeypatch.setenv('JUDGE_API_KEY', 'old')
    monkeypatch.setenv('JUDGE_BASE_URL', 'https://old.example/v1')
    credentials.configure()
    import os
    assert os.environ['JUDGE_API_KEY'] == 'old'
    assert os.environ['JUDGE_BASE_URL'] == 'https://old.example/v1'


def test_openrouter_profile_routes_summary_without_changing_fsr_judge(monkeypatch):
    from scripts import benchmark_credentials as credentials
    import os
    monkeypatch.setattr(credentials, 'load_dotenv', lambda *a: None)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'router-test')
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'https://router.example/v1')
    monkeypatch.setenv('SUMMARY_LLM_API_KEY', 'old-summary')
    monkeypatch.setenv('SUMMARY_LLM_BASE_URL', 'https://old.example/v1')
    monkeypatch.setenv('JUDGE_API_KEY', 'judge-test')
    credentials.configure(profile='fsr_deepseek_v4_1_flash_openrouter')
    assert os.environ['SUMMARY_LLM_API_KEY'] == 'router-test'
    assert os.environ['SUMMARY_LLM_BASE_URL'] == 'https://router.example/v1'
    assert os.environ['JUDGE_API_KEY'] == 'judge-test'


@pytest.mark.parametrize("flag", ["--profile", "--meta-profile"])
def test_meta_cli_alias_preserves_requested_mapping(tmp_path, monkeypatch, flag):
    import sys
    seen = []
    async def run(args, **kwargs):
        seen.append(args.profile)
    monkeypatch.setattr(run_subprocess, "run_eval", run)
    monkeypatch.setattr(sys, "argv", ["run_subprocess", "--benchmark_profile", "browsecomp_200",
                        "--pipeline", HTN_PIPELINES[0], flag, "/configs/meta.yaml", "--out", str(tmp_path)])
    run_subprocess.main()
    assert seen == ["/configs/meta.yaml"]


@pytest.mark.asyncio
async def test_runner_rejects_missing_meta_before_loading_data_or_spawning(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Data/worker must not start before profile validation")
    monkeypatch.setattr("benchmarks.core.registry.load_questions", unexpected)
    monkeypatch.setattr(run_subprocess, "run_task_subprocess", unexpected)
    args = SimpleNamespace(benchmark="browsecomp_200", pipeline=HTN_PIPELINES[0],
                           profile=str(tmp_path / "missing-meta.yaml"))
    with pytest.raises(FileNotFoundError):
        await run_subprocess.run_eval(args, out_dir=tmp_path / "run", seed=0)
