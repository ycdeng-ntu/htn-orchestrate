"""Offline replay integration: adoption statistics must not invent a second audit PASS."""

from types import SimpleNamespace
import json

import pytest

from scripts import fsr_planning_replay as replay
from test_report_delivery_tree import Plan, PlanAuditor, plan_review_payload
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode


@pytest.mark.asyncio
async def test_replay_counts_mechanical_acceptance_separately_from_audit_pass(tmp_path, monkeypatch):
    cases = [
        {"case_id": mode, "question_id": mode, "mode": "fresh", "question": mode}
        for mode in ("clean", "revision", "fallback")
    ]
    monkeypatch.setattr(replay, "fresh_cases_from_tasks", lambda _: cases)
    monkeypatch.setattr(replay, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")

    class NoNetworkClient:
        def __init__(self, *args, **kwargs):
            async def forbidden(**kwargs):
                raise AssertionError("Offline test must not call the API")

            async def close():
                pass

            self._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(create=forbidden))),
                close=close,
            )

    class ScriptedPlanning:
        model = "offline-replay"

        def __init__(self):
            self.nodes = [HTNTaskNode(id="research", title="Derive", task="Derive and check the result")]
            self.planner = Plan(nodes=self.nodes)
            self.auditor = None
            self.revision_started = False
            self.calls = []

        async def chat(self, messages, tools=None):
            name = tools[0]["function"]["name"]
            self.calls.append(name)
            if name == "plan_report_research":
                task = next(c["question"] for c in cases if f"\n{c['question']}\n" in messages[1]["content"])
                self.auditor = PlanAuditor(responses=[
                    plan_review_payload(self.nodes, revise=task != "clean")])
                self.reviser = Plan("failure" if task == "fallback" else "decompose", nodes=self.nodes)
                return await self.planner.chat(messages, tools)
            if name == "revise_report_plan":
                self.revision_started = True
                return await self.reviser.chat(messages, tools)
            assert not self.revision_started, "No audit or synthesis calls after revision"
            return await self.auditor.chat(messages, tools)

    models = []

    def scripted(*args, **kwargs):
        model = ScriptedPlanning()
        models.append(model)
        return model

    monkeypatch.setattr(replay, "OpenAIClient", NoNetworkClient)
    monkeypatch.setattr(replay, "NativeHTNClient", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay, "HighEffortTimeoutFallbackLLMClient", scripted)
    out = tmp_path / "replay"
    await replay.main(SimpleNamespace(task_dir=str(tmp_path), dry_run=False, out=str(out),
                                      concurrency=3, timeout=60))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["accepted"] == 2
    assert summary["initial_audit_accepted"] == summary["revision_mechanically_accepted"] == 1
    assert summary["fallbacks"] == 1 and summary["errors"] == 0
    assert summary["accepted_research_node_distribution"] == {"1": 2}
    assert summary["api_attempts"] == summary["workers_executed"] == 0
    results = {r["case_id"]: r for r in summary["results"]}
    assert results["clean"]["audit_verdicts"] == ["PASS"]
    assert results["clean"]["accepted_by_audit"]
    assert results["revision"]["audit_verdicts"] == ["REVISE"]
    assert results["revision"]["accepted"] and not results["revision"]["accepted_by_audit"]
    assert results["revision"]["revision_mechanically_validated"]
    assert results["fallback"]["fallback"] and not results["fallback"]["accepted"]
    assert sorted(len(m.calls) for m in models) == [4, 6, 6]
