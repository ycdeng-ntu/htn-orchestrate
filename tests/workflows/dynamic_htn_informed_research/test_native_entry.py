"""Exercise the production entry, native envelopes, routing and report delivery offline."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness.core.llm import LLMResponse
from workflows.dynamic_htn_informed_research.diagnostics import scope_snapshot
from workflows.dynamic_htn_informed_research.htn import AgentResultRecord
from workflows.dynamic_htn_informed_research.nodes.dynamic_solve import dynamic_solve_node
from workflows.dynamic_htn_informed_research.profile import load_profile_selection, resolve_task_profile
from workflows.dynamic_htn_informed_research.runtime.context import override_runtime
from workflows.dynamic_htn_informed_research.task_strategies import get_active_task_strategy

SOURCE = "https://example.org/evidence"
SMALL_CONFIG = {
    "llm": {"high_effort_timeout_s": 0},
    "agent": {
        "max_depth": 1,
        "max_children": 2,
        "max_rounds": 3,
        "max_agents_per_round": 1,
        "max_iterations": 3,
        "subagent_max_turns": 3,
        "progressive_fan_in": False,
        "fan_in_grace_s": 0,
        "leaf_timeout_s": 10,
        "question_timeout_s": 30,
        "finalization_reserve_s": 5,
        "answer_format_timeout_s": 2,
        "max_rebuilds": 0,
        "recovery_timeout_s": 2,
        "min_rebuild_remaining_s": 2,
        "forced_finalization_high_timeout_s": 2,
    },
}
ROUTES = ["research", "science_olympiad", "science_mcq", "report_research"]


class ScriptedModel:
    model = "offline-fixture"

    def __init__(self, strategy):
        self.strategy = strategy
        self.calls, self.dispatched, self.unexpected = [], set(), []
        self.worker_roles = []
        self.finalizer_inputs = []

    @property
    def report(self):
        return "# Derivation\n\nUsing $x=6\\times7$, we obtain $x=42$.\n\nThe result is 42."

    def response(self, name, args):
        if name == "extract_evidence":
            args = {"challenges": [], **args}
        if name == "verify_report" and "premise_assessments" not in args:
            args = {**args, "premise_assessments": [{
                "check_id": "premise-001", "requirement_ids": [], "status": "satisfied",
                "reason": "The fixture satisfies the public-question premise check.",
                "corrective_instruction": "",
            }]}
        # These native id/type fields must be removed by the HTN adapter for
        # strict report parsers, and retained by the worker's native loop.
        return LLMResponse(
            tool_calls=[
                {
                    "id": f"call-{len(self.calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 5},
            model=self.model,
        )

    def final_review(self, instruction, verdict="pass", issue="Requested derivation omitted"):
        material, _ = json.JSONDecoder().raw_decode(instruction[instruction.index('{"question":'):])
        prior = material["previous_findings"]
        return {
            "verdict": verdict,
            "checks": [{"id": "result", "question_quote": material["question"],
                        "report_quote": material["draft"], "method": "Independent calculation",
                        "result": "6 * 7 = 42; checked requested derivation",
                        "status": "supported" if verdict == "pass" else "unresolved"}],
            "findings": [] if verdict == "pass" else [{
                "finding_id": prior[0]["finding_id"] if prior else "",
                "issue": issue, "action": "research" if verdict == "rebuild" else "revise",
                "affected_node_ids": [n["id"] for n in material["research"]],
                "check_ids": ["result"],
            }],
            "finding_reviews": [{"finding_id": f["finding_id"],
                                 "status": "resolved" if verdict == "pass" else "unresolved",
                                 "check_ids": ["result"], "reason": "Rechecked current derivation"}
                                for f in prior],
        }

    async def chat(self, messages, *, tools=None, **kwargs):
        await asyncio.sleep(0)  # Deliberately interleave concurrent tasks.
        names = {tool["function"]["name"] for tool in tools or []}
        scope = scope_snapshot()
        self.calls.append((names, scope))
        if "route_task" in names:
            args = {"strategy_name": self.strategy, "rationale": "Fixture route."}
            return self.response("route_task", args)
        if "compose_report" in names:
            self.finalizer_inputs.append(json.loads(messages[1]["content"]))
            return self.response("compose_report", {"action": "draft", "report": self.report,
                                                      "reason": "", "affected_node_ids": []})
        if "plan_report_research" in names:
            return self.response("plan_report_research", {"nodes": [{
                "id": "report-research", "title": "Derive", "task": "Derive and check six times seven",
                "depends_on": [],
            }], "rationale": "One bounded investigation"})
        for name in ("audit_report_fidelity", "audit_report_decomposition", "synthesize_report_plan_audit"):
            if name in names:
                payload = {"plan_review": {"verdict": "PASS", "issues": []}}
                if name == "synthesize_report_plan_audit":
                    payload["dispositions"] = []
                return self.response(name, payload)
        if "audit_report_plan" in names:
            plan = {"nodes": [{"id": node_id} for node_id in
                              re.findall(r"^ID: (.+)$", messages[1]["content"], re.M)]}
            return self.response("audit_report_plan", {
                "has_delivery": False,
                "nodes": [{"id": n["id"], "kind": "research", "reason": "Research calculation"}
                          for n in plan["nodes"]],
                "plan_review": {"verdict": "PASS", "issues": []},
            })
        if "execute_direct" in names:
            return self.response("execute_direct", {"rationale": "One bounded task."})
        if "dispatch" in names:
            leaf = scope.get("leaf_id", "root")
            if leaf not in self.dispatched:
                self.dispatched.add(leaf)
                role = get_active_task_strategy().workers.role_ids[0]
                return self.response(
                    "dispatch",
                    {
                        "role_id": role,
                        "task_summary": "Establish the result with evidence.",
                        "rationale": "Acquire evidence for the requested result.",
                    },
                )
            return self.response("terminate", {})
        if "derive_premise_checks" in names:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            return self.response("derive_premise_checks", {"checks": [{
                "question_quote": payload["question"], "constraint": "Respect the public question.",
                "independent_reasoning": "The requested arithmetic needs a direct derivation.",
                "test": "Independently recompute the arithmetic.",
            }]})
        if "derive_requirements" in names:
            return self.response(
                "derive_requirements",
                {
                    "requirements": [
                        {
                            "description": "Explain the result.",
                            "acceptance_criteria": ["State the result."],
                            "priority": 1,
                            "kind": "fact",
                        }
                    ]
                },
            )
        if "extract_evidence" in names:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            requirement = payload["candidate_requirements"][0]
            record = payload["worker_records"][0]
            return self.response(
                "extract_evidence",
                {
                    "cards": [
                        {
                            "record_id": record["record_id"],
                            "supports": [
                                {
                                    "requirement_id": requirement["id"],
                                    "acceptance_criterion_id": requirement["acceptance_criteria"][
                                        0
                                    ]["id"],
                                }
                            ],
                            "claim": "The result is 42.",
                            "source_url": None,
                            "source_title": None,
                            "support_summary": "The submitted evidence establishes the result as 42.",
                            "confidence": 0.9,
                        }
                    ]
                },
            )
        if "plan_report_outline" in names:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            self.finalizer_inputs.append(payload)
            return self.response(
                "plan_report_outline",
                {
                    "sections": [
                        {
                            "title": "Result",
                            "requirement_ids": [r["id"] for r in payload["requirements"]],
                        }
                    ]
                },
            )
        if "draft_report" in names:
            return self.response("draft_report", {"content": self.report})
        if "verify_report" in names:
            return self.response("verify_report", {"findings": []})
        if "finalize_root_answer" in names:
            return self.response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["result"],
                    "evidence_summary": "The evidence gives 42.",
                    "answer": self.report
                    if self.strategy == "report_research"
                    else "B"
                    if self.strategy == "science_mcq"
                    else "42",
                    "completion_outcome": "resolved",
                },
            )
        if "finalize_answer" in names:
            return self.response(
                "finalize_answer",
                {
                    "result_summary": "The result is 42.",
                    "completion_outcome": "resolved",
                },
            )
        if "submit_report" in names:
            role = scope.get("role_id", get_active_task_strategy().workers.role_ids[0])
            self.worker_roles.append(role)
            content = (json.dumps(self.final_review("\n".join(str(m.get("content", "")) for m in messages)))
                       if role == "dynir_report_final_verifier" else self.report)
            return self.response("submit_report", {"content": content, "confidence": 0.9})
        if not names:
            return LLMResponse(
                content="Establish the result and submit the supporting evidence.",
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            )
        self.unexpected.append(sorted(names))
        return LLMResponse(content="Unexpected fixture call.")

    async def dispatch(self, agent, instruction):
        self.worker_roles.append(agent.role_id)
        return AgentResultRecord(
            agent_id=agent.agent_id,
            role_id=agent.role_id,
            task_summary=agent.task_summary,
            result_summary=(json.dumps(self.final_review(instruction))
                            if agent.role_id == "dynir_report_final_verifier" else
                            f"The result is 42, from 6 times 7. [Fixture source]({SOURCE})"),
            execution_quality="submitted",
            submitted=True,
            observed_source_index=f'- [exact_observed] "Fixture source" — {SOURCE}',
        )


async def run_fixture(tmp_path, strategy, *, native_worker=False):
    model = ScriptedModel(strategy)
    metadata = {"profile": "auto", "dynir_config": SMALL_CONFIG, "_trial_dir": str(tmp_path)}
    seams = {"llm_factory": lambda *_: model}
    if not native_worker:
        seams["dispatch"] = model.dispatch
    with override_runtime(**seams):
        result = await dynamic_solve_node(
            {
                "question": "Establish the result of six times seven. A: 41; B: 42.",
                "metadata": metadata,
            },
            SimpleNamespace(task_id=f"fixture-{strategy}"),
        )
    assert not model.unexpected
    return result, model


@pytest.mark.asyncio
async def test_quota_during_finalization_preserves_worker_artifacts_and_stops_question(tmp_path):
    from agent_harness.core.llm_errors import LLMQuotaError

    class PaymentError(Exception):
        status_code = 402

    class QuotaModel(ScriptedModel):
        quota_calls = 0

        async def chat(self, messages, *, tools=None, **kwargs):
            if any(t["function"]["name"] == "compose_report" for t in tools or []):
                self.quota_calls += 1
                raise PaymentError("Synthetic payment error")
            return await super().chat(messages, tools=tools, **kwargs)

    model = QuotaModel("report_research")
    with override_runtime(llm_factory=lambda *_: model):
        with pytest.raises(LLMQuotaError):
            await dynamic_solve_node(
                {"question": "Compute six times seven and explain.",
                 "metadata": {"profile": "auto", "dynir_config": SMALL_CONFIG,
                              "_trial_dir": str(tmp_path)}},
                SimpleNamespace(task_id="quota-finalization"),
            )
    assert model.quota_calls == 1
    assert list(tmp_path.rglob("quota_error.json"))
    worker_results = list(tmp_path.rglob("workers/*/result.json"))
    assert worker_results and any("42" in path.read_text() for path in worker_results)
    assert any("42" in path.read_text() for path in tmp_path.rglob("refs/*.md"))
    assert list(tmp_path.rglob("usage.json"))
    assert list(tmp_path.rglob("tree_snapshots/*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("recheck", ["pass", "rebuild", "malformed"])
async def test_report_review_rebuilds_through_planner_with_full_material(tmp_path, recheck):
    from copy import deepcopy

    class RebuildModel(ScriptedModel):
        def __init__(self):
            super().__init__("report_research")
            self.verifications = 0
            self.planning_inputs = []
            self.full_memo = "Independent derivation with full evidence: " + "x" * 8000 + " END_EVIDENCE"

        async def chat(self, messages, *, tools=None, **kwargs):
            if any(t["function"]["name"] == "plan_report_research" for t in tools or []):
                self.planning_inputs.append(str(messages))
                # Each rebuilt tree gets new research, even with the same leaf ID.
                self.dispatched.clear()
            return await super().chat(messages, tools=tools, **kwargs)

        async def dispatch(self, agent, instruction):
            record = await super().dispatch(agent, instruction)
            assert not agent.agent_id.startswith(("report-check-", "report-gap-"))
            if agent.role_id == "dynir_report_final_verifier":
                self.verifications += 1
                verdict = "rebuild" if self.verifications == 1 else recheck
                if verdict == "malformed":
                    record.result_summary = "I think it is fine"  # Must never count as verified.
                else:
                    record.result_summary = json.dumps(self.final_review(
                        instruction, verdict, "Need an independent bound on the approximation error"))
                if self.verifications == 2:
                    material, _ = json.JSONDecoder().raw_decode(
                        instruction[instruction.index('{"question":'):])
                    assert material["previous_findings"][0]["finding_id"] == "e0-r0-f1"
                    assert material["previous_reviews"][0]["draft_id"] == "e0-r0"
                assert "END_EVIDENCE" in instruction
            else:
                record.result_summary = self.full_memo
            return record

    model = RebuildModel()
    config = deepcopy(SMALL_CONFIG)
    config["agent"]["max_rebuilds"] = 1
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {"question": "Derive the requested result and its approximation error.",
             "metadata": {"profile": "auto", "dynir_config": config,
                          "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id="report-rebuild"))
    assert model.verifications == (3 if recheck == "malformed" else 2)
    assert len(model.planning_inputs) == 2
    assert "END_EVIDENCE" in model.planning_inputs[1]
    assert "approximation error" in model.planning_inputs[1]
    assert "END_EVIDENCE" in model.finalizer_inputs[1]["previous_epoch"]
    tree = result["task_tree"]
    assert set(tree) == {"root", "report-research", "report-delivery"}
    assert tree["report-delivery"]["depends_on"] == ["report-research"]
    assert tree["root"]["result_summary"] == result["final_content"] == model.report
    assert tree["root"]["completion_outcome"] == (
        "resolved" if recheck == "pass" else "exhausted_unresolved")
    assert list(tmp_path.rglob("epoch-1-000-initial.json"))
    assert not list(tmp_path.rglob("report_repair/tree-appended.json"))
    assert not model.unexpected


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ROUTES)
async def test_four_routes_through_native_entry(tmp_path, strategy):
    result, model = await run_fixture(tmp_path, strategy)
    assert result["task_strategy"] == strategy
    assert "report_kind" not in result
    assert model.worker_roles, model.calls
    assert all(
        role.startswith(
            {
                "research": "dynir_",
                "science_olympiad": "dynir_science_",
                "science_mcq": "dynir_science_mcq_",
                "report_research": "dynir_report_",
            }[strategy]
        )
        for role in model.worker_roles
    )
    names = set().union(*(names for names, _ in model.calls))
    assert {"route_task", "dispatch"} <= names
    assert ("plan_report_research" if strategy == "report_research" else "execute_direct") in names
    if strategy == "report_research":
        assert result["delivery_mode"] == "report"
        assert result["final_content"] == model.report
        assert "compose_report" in names
        assert "finalize_root_answer" not in names
        assert not {"derive_requirements", "extract_evidence", "plan_report_outline",
                    "draft_report", "verify_report", "derive_premise_checks"} & names
        assert model.finalizer_inputs[0]["research"]
        assert "dynir_report_final_verifier" in model.worker_roles
    else:
        assert not names & {"audit_report_plan", "audit_report_fidelity",
                            "audit_report_decomposition", "synthesize_report_plan_audit"}
        assert "derive_premise_checks" not in names
        assert "# Answer:" in result["final_content"]
        assert result["final_answer"].endswith(
            "# Answer: " + ("B" if strategy == "science_mcq" else "42")
        )
    if strategy == "science_mcq":
        assert "instruction_generation" not in {scope.get("call_kind") for _, scope in model.calls}
    usage = result["usage_summary"]
    assert usage["unknown_usage_calls"] == 0
    assert usage["calls"] == len(model.calls)
    assert usage["prompt_tokens"] == 20 * len(model.calls)
    run_dir = Path(result["artifact_dir"])
    assert (run_dir / "report.md").read_text() == result["final_content"]
    assert json.loads((run_dir / "result.json").read_text())["task_strategy"] == strategy
    assert (run_dir / "events.jsonl").is_file()


@pytest.mark.asyncio
async def test_concurrent_routes_do_not_share_strategy_or_artifacts(tmp_path):
    previous = get_active_task_strategy()
    outcomes = await asyncio.gather(
        *(
            run_fixture(tmp_path / str(i), strategy)
            for i, strategy in enumerate(ROUTES)
        )
    )
    assert len({result["artifact_dir"] for result, _ in outcomes}) == len(ROUTES)
    assert [r["task_strategy"] for r, _ in outcomes] == ROUTES
    assert get_active_task_strategy() is previous


@pytest.mark.asyncio
async def test_entry_runs_native_worker_and_real_submit_tool(tmp_path):
    result, model = await run_fixture(tmp_path, "research", native_worker=True)
    assert result["final_answer"].endswith("# Answer: 42")
    assert any("submit_report" in names for names, _ in model.calls)
    records = list((Path(result["artifact_dir"]) / "workers").rglob("result.json"))
    assert records
    assert all(json.loads(path.read_text())["submitted"] for path in records)
    # Native worker reports and checkpoints must survive finalization.
    assert list(Path(result["artifact_dir"]).rglob("*.md"))
    assert model.worker_roles


def test_profile_override_is_task_local_and_smoke_stays_bounded(monkeypatch):
    monkeypatch.setenv("DYNAMIC_HTN_INFORMED_RESEARCH_PROFILE", "research_smoke")
    override = {"agent": {"max_depth": 0}, "llm": {"extra_body": {"reasoning_effort": "high"}}}
    _, first = resolve_task_profile({"dynir_config": override})
    _, second = resolve_task_profile()
    assert first["agent"]["max_depth"] == 0
    assert second["agent"]["max_depth"] == 1
    assert second["agent"]["max_rebuilds"] == 0
    assert second["llm"]["extra_body"]["reasoning_effort"] == "low"
    first["agent"]["max_depth"] = 9
    assert override["agent"]["max_depth"] == 0
    # Explicit selection must win over the compatibility environment setting.
    assert load_profile_selection({"profile": "auto"}).is_meta


@pytest.mark.asyncio
async def test_explicit_report_profile_uses_the_single_scientific_policy(tmp_path):
    model = ScriptedModel("report_research")
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {"question": "Write a scientific report.", "metadata": {
                "profile": "report_research", "_trial_dir": str(tmp_path),
                "dynir_config": {**SMALL_CONFIG, "behavior": {"strategy": "report_research"}},
            }}, SimpleNamespace(task_id="explicit-report"))
    assert result["delivery_mode"] == "report" and result["final_content"] == model.report
    assert not any("route_task" in names for names, _ in model.calls)
    assert "report_kind" not in result
    assert not model.unexpected


@pytest.mark.asyncio
async def test_invalid_budget_rejected_before_worker_or_control_calls(tmp_path):
    calls = []
    with override_runtime(llm_factory=lambda *args: calls.append(args)):
        with pytest.raises(ValueError, match="max_rounds must be positive"):
            await dynamic_solve_node(
                {
                    "question": "Calculate the result.",
                    "metadata": {
                        "profile": "research_smoke",
                        "_trial_dir": str(tmp_path),
                        "dynir_config": {"agent": {"max_rounds": 0}},
                    },
                },
                SimpleNamespace(task_id="invalid-budget"),
            )
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ROUTES)
async def test_explicit_yaml_survives_task_routing(tmp_path, monkeypatch, strategy):
    import yaml
    from workflows.dynamic_htn_informed_research import profile as profile_module

    config = {
        **SMALL_CONFIG,
        "behavior": {"strategy": "auto"},
        "llm": {"provider": "deepseek", "model": "deepseek-v4-flash", "high_effort_timeout_s": 0},
        "summary_llm": {"provider": "openai", "model": "qwen3.6-plus"},
    }
    # No built-in YAML exists here: routing must preserve the selected file.
    (tmp_path / "selected.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(profile_module, "_PROFILES_DIR", tmp_path)
    model = ScriptedModel(strategy)
    connections = []

    def factory(provider, model_name, sampling):
        connections.append((provider, model_name))
        return model

    with override_runtime(llm_factory=factory, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {"question": "Compute six times seven. A: 41; B: 42.",
             "metadata": {"profile": "selected", "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id="explicit-profile"),
        )
    assert result["task_strategy"] == strategy
    assert "report_kind" not in result
    assert connections[0] == ("deepseek", "deepseek-v4-flash")
    assert set(connections) == {("deepseek", "deepseek-v4-flash"), ("openai", "qwen3.6-plus")}
    effective = json.loads((Path(result["artifact_dir"]) / "effective_config.json").read_text())
    assert effective["profile"] == "selected"
    for section in ("llm", "summary_llm"):
        assert effective["configuration"][section] == config[section]
    assert not model.unexpected


def write_meta_fixture(tmp_path):
    """Distinct execution models, budgets and tools for all four routes."""
    import copy
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    routes, configs = {}, {}
    for index, strategy in enumerate(ROUTES):
        key = strategy
        config = copy.deepcopy(SMALL_CONFIG)
        config["llm"].update(provider="executor", model=f"model-{key}", temperature=0.2)
        config["summary_llm"] = {"provider": "extractor", "model": f"summary-{key}"}
        config["agent"].update(question_timeout_s=40 + index, max_total_workers=10 + index)
        config["web"] = {"extract_mode": f"extract-{key}"}
        config["reasoning_effort"] = {"coordinator": "low", "worker": "low"}
        config["behavior"] = {"strategy": strategy}
        configs[key] = config
        (tmp_path / f"{key}.yaml").write_text(yaml.safe_dump(config))
        routes[strategy] = f"./{key}.yaml"
    (tmp_path / "router.yaml").write_text(yaml.safe_dump({
        "llm": {"provider": "router", "model": "router-model", "temperature": 0.7,
                "extra_body": {"reasoning_effort": "low"}},
        "behavior": {"router_timeout_s": 3, "router_max_tokens": 128},
    }))
    path = tmp_path / "meta.yaml"
    path.write_text(yaml.safe_dump({"router": {"profile": "./router.yaml"}, "routes": routes}))
    return path, configs


async def run_meta_fixture(tmp_path, path, strategy, *, override=None, native=False, web_policy=None, tool_factory=None):
    from workflows.dynamic_htn_informed_research.runtime.context import current_runtime

    model = ScriptedModel(strategy)
    connections = []

    def factory(provider, name, sampling):
        import copy
        connections.append((provider, name, copy.deepcopy(sampling), copy.deepcopy(current_runtime().profile)))
        return model

    seams = {"llm_factory": factory}
    if tool_factory is not None:
        seams["tool_factory"] = tool_factory
    if not native:
        seams["dispatch"] = model.dispatch
    with override_runtime(**seams):
        result = await dynamic_solve_node(
            {"question": "Establish six times seven. A: 41; B: 42.",
             "metadata": {"profile": str(path), "dynir_config": override or {}, "_trial_dir": str(tmp_path),
                          "benchmark_web_policy": web_policy}},
            SimpleNamespace(task_id=f"meta-{strategy}"),
        )
    assert not model.unexpected
    effective = json.loads((Path(result["artifact_dir"]) / "effective_config.json").read_text())
    return result, connections, effective


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ROUTES)
async def test_meta_routes_bind_execution_profile_and_apply_overrides_after_routing(tmp_path, strategy):
    path, configs = write_meta_fixture(tmp_path / "configs")
    override = {"agent": {"max_depth": 0}, "llm": {"temperature": 0.4}}
    result, connections, effective = await run_meta_fixture(
        tmp_path, path, strategy, override=override,
    )
    key = strategy
    assert result["task_strategy"] == strategy and "report_kind" not in result
    assert connections[0][:2] == ("router", "router-model")
    assert connections[0][2]["temperature"] == 0.7  # Execution override never reaches router.
    assert connections[0][2]["max_tokens"] == 128
    assert {item[:2] for item in connections[1:]} == {
        ("executor", f"model-{key}"), ("extractor", f"summary-{key}"),
    }
    for provider, _, sampling, profile in connections[1:]:
        assert profile["agent"]["question_timeout_s"] == configs[key]["agent"]["question_timeout_s"]
        assert profile["web"] == configs[key]["web"]
        if provider == "executor":
            assert sampling["temperature"] == 0.4
    cfg = effective["configuration"]
    assert cfg["agent"]["max_depth"] == (1 if strategy == "report_research" else 0)
    assert cfg["agent"]["max_total_workers"] == configs[key]["agent"]["max_total_workers"]
    assert "judge" not in cfg and "evaluation" not in cfg
    assert cfg["behavior"]["strategy"] == strategy
    assert effective["profile"] == str(path.parent / f"{key}.yaml")
    assert effective["meta_profile"] == str(path)
    recorded = json.loads((Path(result["artifact_dir"]) / "profile_selection.json").read_text())
    assert set(recorded["routes"]) == set(ROUTES)
    assert len(recorded["file_sha256"]) == 6
    usage = json.loads((Path(result["artifact_dir"]) / "usage.json").read_text())
    assert usage[0]["purpose"] == "router"
    assert override == {"agent": {"max_depth": 0}, "llm": {"temperature": 0.4}}


@pytest.mark.asyncio
async def test_meta_concurrent_routes_keep_distinct_models_and_budgets(tmp_path):
    path, configs = write_meta_fixture(tmp_path / "configs")
    outcomes = await asyncio.gather(*(
        run_meta_fixture(tmp_path / str(i), path, strategy)
        for i, strategy in enumerate(ROUTES)
    ))
    for (strategy), (_, connections, effective) in zip(ROUTES, outcomes):
        key = strategy
        assert effective["configuration"]["agent"] == configs[key]["agent"]
        assert all(name in {"router-model", f"model-{key}", f"summary-{key}"}
                   for _, name, _, _ in connections)


@pytest.mark.asyncio
async def test_meta_runs_native_worker_under_selected_profile(tmp_path):
    path, _ = write_meta_fixture(tmp_path / "configs")
    result, _, effective = await run_meta_fixture(tmp_path, path, "science_mcq", native=True)
    assert result["final_answer"].endswith("# Answer: B")
    assert effective["configuration"]["llm"]["model"] == "model-science_mcq"
    assert list(Path(result["artifact_dir"]).glob("workers/*/result.json"))


@pytest.mark.asyncio
async def test_meta_rejects_bad_unselected_profile_before_any_model_call(tmp_path):
    import yaml
    path, configs = write_meta_fixture(tmp_path / "configs")
    broken = configs["science_mcq"]
    broken["agent"]["max_rounds"] = 0
    (path.parent / "science_mcq.yaml").write_text(yaml.safe_dump(broken))
    calls = []
    with override_runtime(llm_factory=lambda *args: calls.append(args)):
        with pytest.raises(ValueError, match="max_rounds must be positive"):
            await dynamic_solve_node(
                {"question": "Ordinary research", "metadata": {"profile": str(path)}},
                SimpleNamespace(task_id="preflight"),
            )
    assert calls == []


@pytest.mark.asyncio
async def test_meta_router_protocol_failure_never_falls_back_to_research(tmp_path):
    path, _ = write_meta_fixture(tmp_path / "configs")
    calls = []
    class BrokenRouter:
        model = "router-model"
        async def chat(self, messages, **kwargs):
            calls.append(kwargs)
            return LLMResponse(content="not a route")
    with override_runtime(llm_factory=lambda *_: BrokenRouter()):
        with pytest.raises(ValueError, match="no execution profile selected"):
            await dynamic_solve_node(
                {"question": "Classify", "metadata": {"profile": str(path), "_trial_dir": str(tmp_path)}},
                SimpleNamespace(task_id="broken-router"),
            )
    assert len(calls) == 3
    assert all(call["tools"][0]["function"]["name"] == "route_task" for call in calls)
    assert not list(tmp_path.rglob("workers"))


@pytest.mark.parametrize("fault", ["missing_route", "nested_report_routes", "missing_file", "nested_meta",
                                    "conflicting_strategy", "malformed_budget", "unknown_route"])
def test_meta_preflight_rejects_incomplete_or_inconsistent_configuration(tmp_path, fault):
    import yaml
    from workflows.dynamic_htn_informed_research.nodes.dynamic_solve import prepare_profile_selection

    path, configs = write_meta_fixture(tmp_path / "configs")
    data = yaml.safe_load(path.read_text())
    if fault == "missing_route":
        del data["routes"]["science_mcq"]
    elif fault == "nested_report_routes":
        data["routes"]["report_research"] = {"scientific_report": "report_research.yaml"}
    elif fault == "missing_file":
        data["routes"]["science_mcq"] = "missing.yaml"
    elif fault == "nested_meta":
        data["routes"]["science_mcq"] = "meta.yaml"
    elif fault == "unknown_route":
        data["routes"]["unknown"] = "research.yaml"
    else:
        if fault == "conflicting_strategy":
            configs["science_mcq"]["behavior"]["strategy"] = "research"
        else:
            configs["science_mcq"]["agent"]["max_rounds"] = "not-an-integer"
        (path.parent / "science_mcq.yaml").write_text(yaml.safe_dump(configs["science_mcq"]))
    path.write_text(yaml.safe_dump(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_profile_selection({"profile": str(path)})


def test_meta_relative_paths_preloaded_files_and_task_copies(tmp_path, monkeypatch):
    import yaml
    path, _ = write_meta_fixture(tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    selected = load_profile_selection({"profile": "configs/meta.yaml"})
    first_name, first = selected.select("science_mcq")
    assert first_name == str(path.parent / "science_mcq.yaml")
    first["agent"]["max_depth"] = 99
    (path.parent / "science_mcq.yaml").write_text(yaml.safe_dump({"broken": "after preload"}))
    _, second = selected.select("science_mcq")
    assert second["agent"]["max_depth"] == 1
    for strategy in ("unknown", "sourced_report", "scientific_report"):
        with pytest.raises(ValueError):
            selected.select(strategy)
    assert selected.select("report_research")[1]["behavior"]["strategy"] == "report_research"


def test_default_auto_restores_route_specific_profiles():
    selected = load_profile_selection({"profile": "auto"})
    _, research = selected.select("research")
    _, fsr = selected.select("report_research")
    _, mcq = selected.select("science_mcq")
    assert research["agent"]["max_total_workers"] == 0
    assert research["agent"]["question_timeout_s"] == 28800
    assert fsr["agent"]["question_timeout_s"] == 14400
    assert fsr["reasoning_effort"]["coordinator"] == "high"
    assert mcq["agent"]["question_timeout_s"] == 7200
    assert mcq["agent"]["subagent_max_turns"] == 45


@pytest.mark.asyncio
async def test_meta_router_sampling_is_owned_by_router_profile(tmp_path):
    import yaml
    path, _ = write_meta_fixture(tmp_path / "configs")
    router_file = path.parent / "router.yaml"
    data = yaml.safe_load(router_file.read_text())
    data["llm"].update(max_tokens=256, extra_body={"reasoning_effort": "medium"})
    del data["behavior"]["router_max_tokens"]
    router_file.write_text(yaml.safe_dump(data))
    _, connections, _ = await run_meta_fixture(tmp_path, path, "research")
    assert connections[0][2]["max_tokens"] == 256
    assert connections[0][2]["extra_body"]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ROUTES)
async def test_benchmark_web_policy_reaches_workers_after_every_route(tmp_path, strategy):
    from workflows.dynamic_htn_informed_research.tools import build_research_tools
    path, _ = write_meta_fixture(tmp_path / "profiles")
    seen = []
    policy = {"blocked_domains": ["evaluation-blocked.example"], "blocked_terms": ["leak-needle"]}

    def tools(*args, **kwargs):
        seen.append(kwargs["config"].blocked_needles)
        return build_research_tools(*args, **kwargs)

    result, _, effective = await run_meta_fixture(
        tmp_path, path, strategy, native=True, web_policy=policy, tool_factory=tools)
    assert seen and all(value == ("evaluation-blocked.example", "leak-needle") for value in seen)
    assert "blocked_domains" not in effective["configuration"].get("web", {})
    saved = json.loads((Path(result["artifact_dir"]) / "benchmark_web_policy.json").read_text())
    assert saved == {**policy, "network_enabled": True}


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ROUTES)
async def test_offline_policy_reaches_actual_workers_for_every_route(tmp_path, strategy):
    from workflows.dynamic_htn_informed_research.tools import ResearchToolRuntime
    path, _ = write_meta_fixture(tmp_path / 'profiles')
    runtimes = []
    def factory(*args, **kwargs):
        runtime = ResearchToolRuntime(*args, **kwargs)
        runtimes.append(runtime)
        return runtime
    result, _, effective = await run_meta_fixture(tmp_path, path, strategy, native=True,
        web_policy={'network_enabled': False}, tool_factory=factory)
    assert runtimes
    for runtime in runtimes:
        assert runtime.config.network_enabled is False
        assert not ({t.name for t in runtime.tools} & {'web_search', 'web_fetch', 'scholar_search'})
        assert runtime.config.read_roots and all(p.name == 'refs' for p in runtime.config.read_roots)
    assert 'network_enabled' not in effective['configuration'].get('web', {})
    assert result['task_strategy'] == strategy
    worker_results = list(Path(result['artifact_dir']).glob('workers/*/result.json'))
    assert worker_results
    assert all('External retrieval is disabled' in p.read_text() for p in worker_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["auto", "report_research"])
async def test_report_access_prompts_are_isolated_through_native_pipeline(tmp_path, profile):
    class CaptureModel(ScriptedModel):
        def __init__(self):
            super().__init__("report_research")
            self.messages_by_stage = {}

        async def chat(self, messages, *, tools=None, **kwargs):
            names = frozenset(t["function"]["name"] for t in tools or [])
            self.messages_by_stage.setdefault(names, []).append(messages)
            return await super().chat(messages, tools=tools, **kwargs)

    async def run(network_enabled):
        model = CaptureModel()
        with override_runtime(llm_factory=lambda *_: model):
            result = await dynamic_solve_node(
                {"question": "Derive six times seven and explain assumptions.", "metadata": {
                    "profile": profile, "dynir_config": SMALL_CONFIG,
                    "_trial_dir": str(tmp_path / str(network_enabled)),
                    "benchmark_web_policy": {"network_enabled": network_enabled},
                }}, SimpleNamespace(task_id=f"report-access-{network_enabled}"),
            )
        assert result["final_content"] == model.report
        assert not model.unexpected
        for names, batches in model.messages_by_stage.items():
            if "route_task" in names:
                continue
            for messages in batches:
                system = "\n".join(m["content"] for m in messages if m["role"] == "system")
                assert "[[WEB]]" not in system
                assert ("Web access is available" in system) == network_enabled, names
                assert ("Work offline" in system) != network_enabled, names
                for fragment in ("URLs", "verbatim supporting excerpts", "claims online"):
                    assert (fragment in system) == network_enabled, (names, fragment)
                if "submit_report" in names:
                    assert bool(names & {"web_search", "web_fetch", "scholar_search"}) == network_enabled
        stages = set().union(*model.messages_by_stage)
        assert {"plan_report_research", "audit_report_plan", "audit_report_fidelity",
                "audit_report_decomposition", "dispatch", "submit_report", "compose_report"} <= stages
        assert not {"derive_requirements", "extract_evidence", "plan_report_outline", "verify_report"} & stages
        assert frozenset() in model.messages_by_stage  # Assignment generation.
        assert model.finalizer_inputs[0]["policy"]["network_enabled"] == network_enabled
        return model

    previous = get_active_task_strategy()
    await asyncio.gather(run(False), run(True))
    assert get_active_task_strategy() is previous


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy,profile", [
    ("science_olympiad", "auto"),
    ("report_research", "auto"),
    ("science_mcq", "auto"),
    ("science_mcq", "science_mcq"),
])
async def test_access_policy_reaches_control_and_native_worker_prompts(tmp_path, strategy, profile):
    from benchmarks.core.profile import load_benchmark_profile

    class CaptureModel(ScriptedModel):
        def __init__(self):
            super().__init__(strategy)
            self.messages_by_stage = {}
            self.checker_dispatched = False

        async def chat(self, messages, *, tools=None, **kwargs):
            names = frozenset(t["function"]["name"] for t in tools or [])
            self.messages_by_stage.setdefault(names, []).append(messages)
            if (strategy in {"science_olympiad", "science_mcq"} and "dispatch" in names
                    and self.dispatched and not self.checker_dispatched):
                self.checker_dispatched = True
                return self.response("dispatch", {
                    "role_id": ("dynir_science_mcq_checker" if strategy == "science_mcq"
                                else "dynir_science_checker"),
                    "task_summary": "Independently recompute six times seven and check the candidate.",
                    "rationale": "Audit the completed solution.",
                })
            return await super().chat(messages, tools=tools, **kwargs)

    async def run(network_enabled):
        model = CaptureModel()
        benchmark = "superchem256" if strategy == "science_mcq" else "fso86"
        web_policy = {**load_benchmark_profile(benchmark).configuration["web_policy"],
                      "network_enabled": network_enabled}
        with override_runtime(llm_factory=lambda *_: model):
            result = await dynamic_solve_node(
                {"question": "Derive six times seven.", "metadata": {
                    "profile": profile, "dynir_config": SMALL_CONFIG,
                    "_trial_dir": str(tmp_path / str(network_enabled)),
                    "benchmark_web_policy": web_policy,
                }}, SimpleNamespace(task_id=f"{strategy}-access-{network_enabled}"),
            )
        assert result["task_strategy"] == strategy
        assert not model.unexpected
        worker_systems = []
        for names, batches in model.messages_by_stage.items():
            if "route_task" in names:
                continue
            for messages in batches:
                system = "\n".join(m["content"] for m in messages if m["role"] == "system")
                assert "[[WEB]]" not in system
                offline_hint = ("Worker capabilities for this task are offline" if strategy != "report_research"
                                else "Work offline")
                assert (offline_hint in system) != network_enabled, names
                if not network_enabled:
                    full = "\n".join(m.get("content") or "" for m in messages)
                    assert "External lookup is allowed" not in full
                    assert "Use external lookup" not in full
                    assert "Use external tools only" not in full
                    assert "or external lookup" not in full
                if "submit_report" in names:
                    worker_systems.append(system)
                    assert bool(names & {"web_search", "web_fetch", "scholar_search"}) == network_enabled
                    assert {"run_python_code", "read_file", "submit_report"} <= names
        assert worker_systems
        if strategy in {"science_olympiad", "science_mcq"}:
            assert any("Solve the assigned self-contained" in s for s in worker_systems)
            assert any("Audit a proposed scientific result" in s for s in worker_systems)
            assert "# Answer:" in result["final_content"]
        else:
            assert model.finalizer_inputs[0]["policy"]["network_enabled"] == network_enabled

    previous = get_active_task_strategy()
    await asyncio.gather(run(False), run(True))
    assert get_active_task_strategy() is previous


@pytest.mark.asyncio
async def test_full_research_memo_reaches_delivery_without_card_extraction(tmp_path):
    class GapModel(ScriptedModel):
        def __init__(self):
            super().__init__("report_research")
            self.gap_memo = "Detailed derivation step. " * 250 + "FINAL GAP RESULT: 42"
            self.gap_calls = 0

        async def chat(self, messages, *, tools=None, **kwargs):
            if any(t["function"]["name"] == "extract_evidence" for t in tools or []):
                return self.response("extract_evidence", {"cards": [], "challenges": []})
            return await super().chat(messages, tools=tools, **kwargs)

        async def dispatch(self, agent, instruction):
            result = await super().dispatch(agent, instruction)
            if agent.role_id == "dynir_report_researcher":
                self.gap_calls += 1
                result.result_summary = self.gap_memo
            return result

    model = GapModel()
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        await dynamic_solve_node(
            {"question": "Calculate six times seven with a derivation.",
             "metadata": {"profile": "auto", "dynir_config": SMALL_CONFIG,
                          "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id="report-gap-preservation"),
        )
    assert model.gap_calls == 1
    payload = model.finalizer_inputs[0]
    assert "evidence_cards" not in payload
    assert model.gap_memo == payload["research"][0]["worker_reports"][0]["result_summary"]


@pytest.mark.asyncio
async def test_unresolved_report_review_updates_root_even_without_research_repair(tmp_path):
    class RewriteModel(ScriptedModel):
        async def dispatch(self, agent, instruction):
            result = await super().dispatch(agent, instruction)
            if agent.role_id == "dynir_report_final_verifier":
                result.result_summary = json.dumps(self.final_review(instruction, "revise"))
            return result

    model = RewriteModel("report_research")
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {"question": "Calculate six times seven with a derivation.",
             "metadata": {"profile": "auto", "dynir_config": SMALL_CONFIG,
                          "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id="report-root-review"),
        )
    root = result["task_tree"]["root"]
    assert root["result_summary"] == result["final_content"]
    assert root["completion_outcome"] == "exhausted_unresolved"
    assert not any(node_id.startswith("report-check-") for node_id in result["task_tree"])


@pytest.mark.asyncio
async def test_native_report_verifier_submits_valid_review_and_root_passes_through(tmp_path):
    result, model = await run_fixture(tmp_path, "report_research", native_worker=True)
    root = result["task_tree"]["root"]
    assert root["completion_outcome"] == "resolved"
    assert root["result_summary"] == result["task_tree"]["report-delivery"]["result_summary"]
    assert "dynir_report_final_verifier" in model.worker_roles
    final_reviews = list(Path(result["artifact_dir"]).rglob("report_delivery/epoch-0/review-0.json"))
    assert len(final_reviews) == 1
    assert json.loads(final_reviews[0].read_text())["verdict"] == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("has_successor", [False, True])
async def test_report_plan_audit_in_native_workflow(tmp_path, has_successor):
    from copy import deepcopy

    class AuditedModel(ScriptedModel):
        def __init__(self):
            super().__init__("report_research")
            self.planning_inputs, self.revision_inputs, self.audits = [], [], []

        async def chat(self, messages, *, tools=None, **kwargs):
            names = {t["function"]["name"] for t in tools or []}
            if names & {"plan_report_research", "revise_report_plan"}:
                self.calls.append((names, scope_snapshot()))
                revising = "revise_report_plan" in names
                (self.revision_inputs if revising else self.planning_inputs).append(
                    messages[1]["content"])
                nodes = [{"id": "root-derive", "title": "Derive", "task": "Compute six times seven",
                          "depends_on": []}]
                if not revising:
                    nodes.append({"id": "root-draft", "title": "Write report",
                                  "task": "Write the complete final report", "depends_on": ["root-derive"]})
                    if has_successor:
                        nodes.append({"id": "root-later", "title": "Check later",
                                      "task": "Check the draft", "depends_on": ["root-draft"]})
                return self.response("revise_report_plan" if revising else "plan_report_research",
                                     {"nodes": nodes, "rationale": "Research calculation"})
            if "audit_report_plan" in names:
                self.calls.append((names, scope_snapshot()))
                material = messages[1]["content"]
                self.audits.append(material)
                nodes = [{"id": node_id} for node_id in re.findall(r"^ID: (.+)$", material, re.M)]
                return self.response("audit_report_plan", {
                    "has_delivery": any(n["id"] == "root-draft" for n in nodes),
                    "nodes": [{"id": n["id"], "kind": "report_delivery"
                               if n["id"] == "root-draft" else "research",
                               "reason": "Task classification"} for n in nodes],
                    "plan_review": {"verdict": "PASS", "issues": []},
                })
            return await super().chat(messages, tools=tools, **kwargs)

    model = AuditedModel()
    config = deepcopy(SMALL_CONFIG)
    config["agent"]["max_children"] = 4
    with override_runtime(llm_factory=lambda *_: model, dispatch=model.dispatch):
        result = await dynamic_solve_node({
            "question": "Establish the result of six times seven.",
            "metadata": {"profile": "auto", "dynir_config": config, "_trial_dir": str(tmp_path)},
        }, SimpleNamespace(task_id="plan-audit-integration"))
    assert not model.unexpected
    assert len(model.planning_inputs) == 1
    assert len(model.revision_inputs) == int(has_successor)
    assert len(model.audits) == 1
    if has_successor:
        revision = model.revision_inputs[0]
        assert revision.split("## Candidate plan")[0].rstrip() == model.planning_inputs[0]
        assert revision.split("## Audit feedback")[0].rstrip() == model.audits[0]
        assert "root-later" in revision.split("## Audit feedback")[1]
    tree = result["task_tree"]
    assert set(tree) == {"root", "root-derive", "report-delivery"}
    assert tree["report-delivery"]["depends_on"] == ["root-derive"]
    assert model.dispatched == {"root-derive"}
    assert result["final_content"] == model.report
    assert any(scope.get("report_stage") == "report_plan_audit" for _, scope in model.calls)
    if has_successor:
        assert any(scope.get("call_kind") == "planner_revision" for _, scope in model.calls)
    assert not any("audit_subtask_fidelity" in names or "revise_subtask" in names for names, _ in model.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['uncertain', 'malformed'])
async def test_review_protocol_failure_does_not_rebuild_research_tree(tmp_path, mode):
    from copy import deepcopy

    class Model(ScriptedModel):
        def __init__(self):
            super().__init__('report_research')
            self.plans = self.verifications = self.alignments = 0

        async def chat(self, messages, *, tools=None, **kwargs):
            names = {t['function']['name'] for t in tools or []}
            if 'plan_report_research' in names:
                self.plans += 1
            if 'align_review_quotes' in names:
                self.alignments += 1
                data = json.loads(messages[1]['content'])
                return self.response('align_review_quotes', {'matches': [
                    {'quote_id': q['quote_id'], 'status': 'uncertain', 'source_ids': [],
                     'reason': 'Cannot identify the same claim in the current draft'}
                    for q in data['citations']]})
            return await super().chat(messages, tools=tools, **kwargs)

        async def dispatch(self, agent, instruction):
            record = await super().dispatch(agent, instruction)
            if agent.role_id == 'dynir_report_final_verifier':
                self.verifications += 1
                p = self.final_review(instruction)
                p['checks'][0]['report_quote'] = 'Obsolete draft says 999'
                record.result_summary = json.dumps(p) if mode == 'uncertain' else 'bad json'
            return record

    model = Model()
    config = deepcopy(SMALL_CONFIG)
    config['agent']['max_rebuilds'] = 1
    samplings = []

    def factory(provider, name, sampling):
        samplings.append(sampling)
        return model

    with override_runtime(llm_factory=factory, dispatch=model.dispatch):
        result = await dynamic_solve_node(
            {'question': 'Derive the requested result.',
             'metadata': {'profile': 'auto', 'dynir_config': config, '_trial_dir': str(tmp_path)}},
            SimpleNamespace(task_id='review-failure'))
    assert model.plans == 1
    assert model.verifications == (1 if mode == 'uncertain' else 2)
    assert model.alignments == (1 if mode == 'uncertain' else 0)
    assert result['final_content'] == model.report
    assert result['task_tree']['root']['completion_outcome'] == 'exhausted_unresolved'
    assert not list(tmp_path.rglob('epoch-1-000-initial.json'))
    artifact = next(tmp_path.rglob('report_delivery/result.json'))
    assert json.loads(artifact.read_text())['status'] == 'verification_incomplete'
    assert any(s.get('max_tokens') == 8192 and s.get('extra_body', {}).get('reasoning_effort') == 'low'
               for s in samplings)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ROUTES)
async def test_configured_effort_reaches_report_delivery_and_workers(tmp_path, strategy):
    from copy import deepcopy

    model, calls = ScriptedModel(strategy), []
    config = deepcopy(SMALL_CONFIG)
    config["reasoning_effort"] = {"coordinator": "high", "worker": "high"}

    def factory(provider, model_name, sampling):
        effort = sampling.get("extra_body", {}).get("reasoning_effort")

        class BoundClient:
            model = model_name

            async def chat(self, messages, *, tools=None, **kwargs):
                names = {tool["function"]["name"] for tool in tools or []}
                calls.append((names, scope_snapshot(), effort))
                return await model.chat(messages, tools=tools, **kwargs)

        return BoundClient()

    # Use the real native workers too, so researcher and final-verifier effort
    # are observed at the same transport boundary as leaf coordination and report writing.
    with override_runtime(llm_factory=factory):
        result = await dynamic_solve_node(
            {"question": "Establish six times seven. A: 41; B: 42.",
             "metadata": {"profile": "auto", "dynir_config": config, "_trial_dir": str(tmp_path)}},
            SimpleNamespace(task_id=f"effort-{strategy}"),
        )
    assert not model.unexpected
    decisions = [effort for _, scope, effort in calls if scope.get("report_stage") == "report_decision"]
    writing = [effort for _, scope, effort in calls if scope.get("report_stage") == "delivery_draft"]
    workers = [(scope.get("role_id"), effort) for names, scope, effort in calls if "submit_report" in names]
    assert workers and all(effort == "high" for _, effort in workers)
    if strategy == "report_research":
        assert not decisions
        assert writing == ["high"]
        assert {role for role, _ in workers} == {"dynir_report_researcher", "dynir_report_final_verifier"}
        assert result["final_content"] == model.report
    else:
        assert not decisions and not writing
        ordinary_decisions = [effort for names, _, effort in calls if "dispatch" in names]
        assert ordinary_decisions and set(ordinary_decisions) == {"high"}
