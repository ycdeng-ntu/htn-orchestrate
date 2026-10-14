import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.prepare_prl import standardize, PROTOCOL
from benchmarks.core.contracts import is_graded, solver_metadata
from benchmarks.core.registry import load_questions
from benchmarks.judges import prl_bench


def raw_row():
    return dict(id="7", subfield="astro", title="Public title", introduction="Public motivation",
                tasks=[dict(id=2, prompt="Derive A"), dict(id=9, prompt="Calculate B")],
                rubrics=[dict(task_id=9, rubrics=["SECRET_B"]), dict(task_id=2, rubrics=["SECRET_A"])])


def test_conversion_maps_nonconsecutive_subtasks_without_leaking_rubrics():
    row = standardize([raw_row()])[0]
    assert "Derive A" in row["task_question"] and "Calculate B" in row["task_question"]
    assert "SECRET" not in row["task_question"]
    assert [c["criterion"] for c in row["ground_truth"]["criteria"]] == ["SECRET_A", "SECRET_B"]
    assert "ground_truth" not in solver_metadata(row)
    bad = raw_row(); bad["rubrics"].pop()
    with pytest.raises(ValueError, match="mapping"):
        standardize([bad])


def test_complete_release_and_profile():
    from collections import Counter
    from workflows.dynamic_htn_informed_research.nodes.dynamic_solve import prepare_profile_selection
    root = Path(__file__).resolve().parents[2]
    qs = load_questions("prl_bench", subset=str(root / "benchmarks/datasets/PRL-Bench/standardized_data.jsonl"))
    assert len(qs) == len({q.id for q in qs}) == 100
    assert Counter(q.metadata["subfield"] for q in qs) == dict(amo=20, astro=10, cmp=24, hep=16, qua=20, stat=10)
    assert sum(len(json.loads(q.ground_truth)["criteria"]) for q in qs) == 1898
    selected = prepare_profile_selection({"profile": "deepseek_v4_1_flash_prl_offline"})
    _, config = selected.select("report_research", "scientific_report")
    assert config["web"]["network_enabled"] is False
    assert config["agent"]["preserve_reasoning_fields"] is True


@pytest.mark.parametrize("bad", [
    [], [{"id": "wrong", "score": 1, "reason": "x"}],
    [{"id": "2:1", "score": True, "reason": "x"}],
    [{"id": "2:1", "score": 0.5, "reason": "x"}],
])
def test_judge_rejects_partial_or_invalid_results(bad):
    with pytest.raises(ValueError):
        prl_bench.validate(json.dumps({"criteria": bad}), [{"id": "2:1"}])


@pytest.mark.asyncio
async def test_scoring_is_continuous_auditable_and_uses_entire_report(monkeypatch):
    from benchmarks.runner.run_one_task import _score_answer
    target = standardize([raw_row()])[0]["ground_truth"]
    seen = []
    async def create(**kwargs):
        seen.append(kwargs)
        batch = json.loads(kwargs["messages"][1]["content"])
        assert batch["report"].endswith("TAIL_SENTINEL")
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({"criteria": [
            dict(id=c["id"], score=int(c["id"].startswith("2:")), reason="checked") for c in batch["criteria"]]})))])
    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        async def close(self): pass
    monkeypatch.setattr(prl_bench, "_profile_judge_client", lambda *_: Client())
    details = {}
    result = await _score_answer("x" * 180000 + "TAIL_SENTINEL", json.dumps(target), "question", "report", "7", "prl_bench", judge_details=details)
    assert result == (None, "rubric", None, 50.0)
    assert details["satisfied"] == 1 and len(details["criteria"]) == 2
    assert details["official_score_reproduction"] is False
    assert is_graded(dict(is_correct=None, judge_method="rubric", rubric_score=50))
    assert not is_graded(dict(benchmark="prl_bench", is_correct=False, judge_method="worker_failed"))


@pytest.mark.asyncio
async def test_offline_tools_hide_retrieval_and_disallow_host_dataset(tmp_path):
    import httpx
    from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig, ResearchToolRuntime
    from workflows.dynamic_htn_informed_research.tools.runtime import ResearchToolError
    config = ResearchToolConfig(network_enabled=False, read_roots=(tmp_path / "secret",))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("network request"))) as client:
        tools = ResearchToolRuntime("dynir_report_researcher", task_id="t", artifact_dir=tmp_path,
                                    config=config, http_client=client)
        try:
            assert {t.name for t in tools.tools} == {"run_python_code", "read_file", "submit_report"}
            assert tools.config.read_roots == ()
            with pytest.raises(ResearchToolError, match="disabled"):
                await tools._bind_tool("web_search")(query="paper")
        finally:
            await tools.aclose()


def test_opt_in_deepseek_reasoning_roundtrip():
    from agent_harness.infra.openai_client import _to_llm_response
    from agent_harness.core.runtime.loop.model_profile import DefaultMessageNormalizer, HistoryPolicy, ThinkingResult
    raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="tool", reasoning="opaque reasoning", reasoning_details=[{"type": "reasoning.text", "text": "opaque"}], tool_calls=[]), finish_reason="stop")])
    response = _to_llm_response(raw)
    normalizer = DefaultMessageNormalizer()
    tr = ThinkingResult(thinking="", visible_content="tool")
    baseline = normalizer.to_history(response, tr, HistoryPolicy())
    assert "reasoning_content" not in baseline and "reasoning_details" not in baseline
    fixed = normalizer.to_history(response, tr, HistoryPolicy(preserve_reasoning_fields=True))
    assert fixed["reasoning_content"] == "opaque reasoning"
    assert fixed["reasoning_details"] == raw.choices[0].message.reasoning_details
    assert fixed["reasoning_details"] is not raw.choices[0].message.reasoning_details


def test_offline_sandbox_disables_network_before_return_and_cleans_failed_bootstrap(monkeypatch):
    from e2b_code_interpreter import Sandbox
    from workflows.dynamic_htn_informed_research.tools.sandbox import e2b_factory
    events = []
    class FakeSandbox:
        def __init__(self, fail_install=False):
            self.fail_install = fail_install
            self.commands = self
            self._call_count = 0
        def update_network(self, options): events.append(options["allow_internet_access"])
        def run(self, command, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                return SimpleNamespace(exit_code=1)
            if self.fail_install and "pip install" in command:
                raise RuntimeError("install failed")
            return SimpleNamespace(exit_code=0)
        def kill(self): events.append("killed")
    fake = FakeSandbox()
    def create(**kwargs):
        assert kwargs["allow_internet_access"] is False
        return fake
    monkeypatch.setattr(Sandbox, "create", create)
    assert e2b_factory(api_key="fake", template="test", timeout=300, allow_internet_access=False) is fake
    assert events == [True, False]
    events.clear(); fake.fail_install = True; fake._call_count = 0
    with pytest.raises(RuntimeError, match="install failed"):
        e2b_factory(api_key="fake", template="test", timeout=300, allow_internet_access=False)
    assert events == [True, False, "killed"]
