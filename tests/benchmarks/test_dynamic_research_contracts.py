"""Offline dataset -> Harbor -> output -> judge contract tests for all five routes."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tomllib
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.core.contracts import extract_prediction
from benchmarks.core.harbor_task_generator import generate_task_dirs
from benchmarks.core.registry import get_config, load_questions
from benchmarks.runner import run_one_task, run_subprocess


@pytest.fixture
def harbor_agent_module(monkeypatch):
    """Only Harbor protocol objects are stubbed; the real AH adapter executes."""
    class BaseAgent:
        def __init__(self, *, logs_dir, **kwargs):
            self.logs_dir = logs_dir

    class AgentContext:
        def __init__(self):
            self.metadata = None

    for name, attrs in {
        "harbor.agents.base": {"BaseAgent": BaseAgent},
        "harbor.environments.base": {"BaseEnvironment": object},
        "harbor.models.agent.context": {"AgentContext": AgentContext},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    name = "benchmarks.harbor_agent.agent"
    old = sys.modules.pop(name, None)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)
    if old is not None:
        sys.modules[name] = old


def write_dataset(tmp_path, *, report=False):
    row = {
        "task_id": "task7", "idx": 7, "task_question": "Solver question with source restrictions",
        "ground_truth": ({"rubric": {"criterion": "hidden rubric"}, "blocked": {"secret": True}}
                         if report else "gold answer"),
        "judge_task": "Official shorter judge task", "description": "private evaluation context",
        "language": "en", "theme": "science", "category": "chemistry",
    }
    path = tmp_path / "subset.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(("dataset", "answer_type"), [
    ("browsecomp_200", "exactMatch"),
    ("frontier_science_olympiad", "single_answer"),
    ("superchem_text", "multipleChoice"),
    ("frontier_science_research", "report"),
    ("deepresearch_bench_ii", "report"),
])
def test_dataset_defaults_apply_before_filter(tmp_path, dataset, answer_type):
    path = write_dataset(tmp_path, report=dataset == "deepresearch_bench_ii")
    [q] = load_questions(dataset, subset=str(path), answer_type=answer_type)
    assert q.answer_type == answer_type
    assert q.id == ("7" if dataset == "deepresearch_bench_ii" else "task7")
    if dataset == "deepresearch_bench_ii":
        assert q.metadata["task_id"] == "task7"
        assert q.metadata["judge_task"] == "Official shorter judge task"
        assert json.loads(q.ground_truth)["rubric"] == {"criterion": "hidden rubric"}
    assert load_questions(dataset, subset=str(path), answer_type="nonexistent") == []
    assert get_config("browsecomp").default_answer_type == "exactMatch"


def test_report_is_verbatim_and_short_answer_uses_final_root_marker():
    formula = " + ".join(r"\\frac{x_{%d}}{y_{%d}}" % (i, i) for i in range(20))
    content = f"result: preliminary\n\n# Answer: {formula}\nResult: audit note\n"
    state = {"report": {"content": content}, "final_answer": "full fallback"}
    assert extract_prediction(state, "report") == content
    assert extract_prediction(state, "single_answer") == formula
    assert extract_prediction({"final_answer": "  react answer  "}, "exactMatch") == "react answer"
    assert extract_prediction({"final_content": content}, "report") == content
    mc = {"report": {"content": "The answer is A.\n# Answer: J\n"}}
    assert extract_prediction(mc, "multipleChoice") == "J"


def test_solver_metadata_excludes_evaluation_inputs(tmp_path, harbor_agent_module):
    private = {"judge_task": "hidden task", "rubric": {"gold": 1}, "ground_truth": "secret",
               "description": "private description", "judge_variant": "semantic"}
    public = {"profile": 'quoted "profile"', "idx": 7, "language": "en",
              "dynir_config": {"behavior": {"strategy": "auto"}}}
    tasks = tmp_path / "tasks"
    generate_task_dirs([{
        "id": "7", "question": "Visible prompt", "answer": "secret",
        "answer_type": "report", "metadata": {**public, **private},
        "file_path": "/tmp/input.csv", "benchmark": "deepresearch_bench_ii",
    }], tasks)
    task = tasks / "7"
    toml = (task / "task.toml").read_text()
    metadata = tomllib.loads(toml)["metadata"]
    assert not private.keys() & metadata.keys()
    assert metadata["profile"] == public["profile"]
    assert metadata["file_path"] == "/tmp/input.csv"
    assert "hidden" not in (task / "instruction.md").read_text()
    parsed = harbor_agent_module.parse_task_metadata(toml)
    assert parsed["profile"] == public["profile"]
    assert parsed["dynir_config"] == public["dynir_config"]
    assert not private.keys() & parsed.keys()
    evaluation = run_one_task.load_evaluation_question(task, "deepresearch_bench_ii", "7")
    assert evaluation["metadata"]["judge_task"] == "hidden task"
    assert evaluation["ground_truth"] == "secret"
    with pytest.raises(ValueError, match="identity mismatch"):
        run_one_task.load_evaluation_question(task, "deepresearch_bench_ii", "wrong")


@pytest.mark.asyncio
@pytest.mark.parametrize(("dataset", "content", "prediction", "score"), [
    ("browsecomp_200", "Derivation\n# Answer: gold answer\n", "gold answer", None),
    ("frontier_science_olympiad", "Derivation\n# Answer: gold answer\n", "gold answer", None),
    ("superchem_text", "Derivation\n# Answer: J\n", "J", None),
    ("frontier_science_research", "# Report\nFull scientific derivation\n",
     "# Report\nFull scientific derivation", 8.5),
    ("deepresearch_bench_ii", "# Report\nFull research report\n",
     "# Report\nFull research report", None),
])
async def test_five_routes_from_runner_to_scoring(
    tmp_path, monkeypatch, harbor_agent_module, dataset, content, prediction, score,
):
    from benchmarks import judges

    path = write_dataset(tmp_path, report=dataset == "deepresearch_bench_ii")
    variant = "semantic" if dataset == "browsecomp_200" else None
    calls = []

    async def judge(question, target, predicted, **_kwargs):
        calls.append((question, target, predicted))
        return ("CORRECT", score) if score is not None else "CORRECT"

    if variant:
        monkeypatch.setitem(judges.JUDGE_VARIANTS[dataset], variant, judge)
    elif score is not None:
        monkeypatch.setitem(judges.SCORE_REGISTRY, dataset, judge)
    elif dataset != "deepresearch_bench_ii":
        monkeypatch.setitem(judges.JUDGE_REGISTRY, dataset, judge)

    observed_solver_inputs = []

    async def fake_pipeline(instruction, *, meta, pipeline_id):
        observed_solver_inputs.append((instruction, meta))
        return {"report": {"content": content}, "final_answer": content}

    async def worker(qid, out_dir, benchmark, sem, mirror):
        task = out_dir / "tasks" / qid
        question = run_one_task.load_evaluation_question(task, benchmark, qid)
        agent = harbor_agent_module.AgentHarnessAgent(logs_dir=out_dir)
        agent._session = SimpleNamespace(run=fake_pipeline)
        return await run_one_task.run_one_question(
            agent, question, out_dir / "trials" / qid, task, benchmark,
        )

    monkeypatch.setattr(run_subprocess, "run_task_subprocess", worker)
    profile = ("report_research" if dataset == "frontier_science_research"
               else "local-fixture")
    if profile == "local-fixture" and dataset != "deepresearch_bench_ii":
        from workflows.dynamic_htn_informed_research import profile as profile_module
        directory = tmp_path / "profiles"
        directory.mkdir()
        (directory / "local-fixture.yaml").write_text("behavior: {strategy: research}\n")
        monkeypatch.setattr(profile_module, "_PROFILES_DIR", directory)
    args = argparse.Namespace(
        benchmark=dataset, pipeline="dynamic_htn_informed_research",
        profile=None if dataset == "deepresearch_bench_ii" else profile,
        limit=None, offset=0, answer_type=None, category=None, subset=str(path),
        no_shuffle=True, concurrency=1, judge_variant=variant,
    )
    out = tmp_path / "run"
    result = await run_subprocess.run_eval(args, out_dir=out, seed=42)
    [trial] = result["results"]
    assert trial["predicted_answer"] == prediction
    assert trial["rubric_score"] == score
    qid = trial["question_id"]
    assert (out / "trials" / qid / "agent" / "final_content.md").read_text() == content
    [(_, metadata)] = observed_solver_inputs
    assert metadata["profile"] == ("auto" if dataset == "deepresearch_bench_ii" else profile)
    assert metadata["benchmark"] == dataset
    assert not {"judge_task", "rubric", "ground_truth", "description"} & metadata.keys()
    if dataset == "deepresearch_bench_ii":
        assert not calls
        assert trial["reward"] is None and trial["is_correct"] is None
        assert trial["judge_method"] == "not_graded"
        assert trial["evaluation_metadata"]["judge_task"] == "Official shorter judge task"
        assert result["accuracy"] is None and result["graded"] == 0
        assert result["not_graded"] == 1
    else:
        assert calls == [("Solver question with source restrictions", "gold answer", prediction)]
        assert result["accuracy"] == 1.0 and result["graded"] == 1
        assert trial["judge_variant"] == (variant or "official")


@pytest.mark.asyncio
async def test_failed_judge_is_not_an_incorrect_answer(monkeypatch):
    monkeypatch.setattr(run_one_task, "score_answer", AsyncMock(return_value=("NOT_ATTEMPTED", None)))
    reward, method, error, _ = await run_one_task._score_answer(
        "answer", "gold", "question", "exactMatch", "1", "browsecomp_200",
    )
    assert reward is None and method == "scoring_failed" and error


@pytest.mark.asyncio
async def test_semantic_variant_and_model_settings_are_explicit(monkeypatch):
    from benchmarks import judges
    from benchmarks.judges import browsecomp

    captured = []

    async def fake_call(kwargs, *, label):
        captured.append(kwargs.copy())
        return "correct: yes"

    monkeypatch.setattr(browsecomp, "_judge_call_with_effort_fallback", fake_call)
    monkeypatch.setenv("BC200_JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("BC200_JUDGE_REASONING_EFFORT", "low")
    assert await judges.verify_answer("browsecomp_200", "q", "gold", "pred", variant="semantic") == "CORRECT"
    assert captured[0]["model"] == "gpt-5.4"
    assert captured[0]["reasoning_effort"] == "low"
    assert "SEMANTIC EQUIVALENCE" in captured[0]["messages"][0]["content"]
    await judges.verify_answer("browsecomp_200", "q", "gold", "pred")
    assert captured[1]["model"] == "gpt-5.4"
    assert "SEMANTIC EQUIVALENCE" not in captured[1]["messages"][0]["content"]
    with pytest.raises(KeyError):
        await judges.verify_answer("frontier_science_research", "q", "g", "p", variant="semantic")


@pytest.mark.asyncio
async def test_fsr_grades_complete_report_without_preprocessing(monkeypatch):
    from benchmarks.judges import frontier_science

    call = AsyncMock(return_value=8.5)
    monkeypatch.setattr(frontier_science, "_call_rubric_judge_once", call)
    report = "## References to prior work\nDerivation\n## References\nref list\n<!-- citation-map: {} -->"
    assert await frontier_science.score_frontier_science("problem", "rubric", report) == ("CORRECT", 8.5)
    [prompt] = call.await_args.args
    assert "References to prior work" in prompt and "Derivation" in prompt
    assert report in prompt


@pytest.mark.asyncio
async def test_superchem_judge_is_gpt54_without_changing_hle_model(monkeypatch):
    from types import SimpleNamespace
    from benchmarks import judges
    from benchmarks.judges import hle

    captured = []

    async def parse(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=SimpleNamespace(correct="yes"),
        ))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(hle, "_make_client", lambda: client)
    assert await judges.verify_answer("superchem_text", "q", "A", "A") == "CORRECT"
    assert captured[0]["model"] == "gpt-5.4"
    assert await judges.verify_answer("hle_text", "q", "gold", "answer") == "CORRECT"
    assert captured[1]["model"] == hle._HLE_JUDGE_MODEL
