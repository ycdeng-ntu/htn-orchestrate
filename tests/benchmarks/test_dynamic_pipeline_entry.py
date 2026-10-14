"""Exercise real benchmark bootstrap, workflow loading, scheduling and MiniDAG execution."""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness.core.runtime import registry
from agent_harness.core.runtime.dag.minidag import MiniDAGRunner
from agent_harness.core.types import TaskStatus
from agent_harness.scheduling.pipeline_registry import PipelineRegistry
from benchmarks.core.kernel_adapter import BenchmarkSession


def _native_fixture(pipeline="dynamic_htn_informed_research"):
    # Loading by path avoids collisions with third-party packages named tests.
    path = (Path(__file__).resolve().parents[1] / "workflows"
            / "dynamic_htn_informed_research" / "test_native_entry.py")
    if pipeline != "dynamic_htn_informed_research":
        path = Path(__file__).with_name("_dynamic_v1_fixture.py")
    spec = importlib.util.spec_from_file_location("_benchmark_native_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Bind the fixture to the selected package's context and result types.
    # Each workflow must use its own ContextVars, including diagnostics.
    package = f"workflows.{pipeline}"
    module.scope_snapshot = importlib.import_module(f"{package}.diagnostics").scope_snapshot
    module.get_active_task_strategy = importlib.import_module(
        f"{package}.task_strategies",
    ).get_active_task_strategy
    module.AgentResultRecord = importlib.import_module(f"{package}.htn").AgentResultRecord
    return module.ScriptedModel, module.SMALL_CONFIG


@pytest.mark.asyncio
@pytest.mark.parametrize("native_worker", [False, True])
@pytest.mark.parametrize("pipeline,strategy,report_kind,answer_type", [
    ("dynamic_htn_informed_research", "research", None, "exactMatch"),
    ("dynamic_htn_informed_research", "report_research", None, "report"),
])
async def test_benchmark_session_runs_registered_dynamic_pipeline(
    tmp_path, monkeypatch, strategy, report_kind, answer_type, pipeline, native_worker,
):
    from agent_harness.infra import config, llm_adapter

    scripted_model, small_config = _native_fixture(pipeline)
    model = scripted_model(strategy)
    # Keep the production bootstrap. Replace only provider configuration and
    # transport creation, so this test never reads .env or constructs an API client.
    monkeypatch.setattr(config, "get_config", lambda: SimpleNamespace())
    monkeypatch.setattr(llm_adapter, "create_llm", lambda _config: model)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_HARNESS_TASK_WALL_TIME_S", raising=False)

    module = importlib.import_module(
        f"workflows.{pipeline}.nodes.dynamic_solve",
    )
    original_node = module.dynamic_solve_node
    node_inputs, node_outputs = [], []

    async def observe_node(state, ctx):
        node_inputs.append((state.copy(), str(ctx.task_id)))
        output = await original_node(state, ctx)
        node_outputs.append(output.copy())
        return output

    monkeypatch.setattr(module, "dynamic_solve_node", observe_node)
    initial_registry = registry.snapshot()
    question = "Establish the result of six times seven. A: 41; B: 42."
    metadata = {
        "profile": "auto", "dynir_config": small_config,
        "answer_type": answer_type,
        "benchmark": ("deepresearch_bench_ii" if report_kind == "sourced_report"
                      else "frontier_science_research"),
        "_trial_dir": str(tmp_path / "trial"),
        "ground_truth": "never pass this to the solver",
        "judge_task": "private evaluator context", "rubric": "private rubric",
    }

    override_runtime = importlib.import_module(
        f"workflows.{pipeline}.runtime.context",
    ).override_runtime
    seams = {"llm_factory": lambda *_: model}
    if not native_worker:
        seams["dispatch"] = model.dispatch
    async with AsyncExitStack() as stack:
        if native_worker and report_kind == "sourced_report":
            import httpx

            # A cited report must actually observe its source through a tool.
            # Mock only the search HTTP transport; retain provenance validation.
            def search_response(request):
                assert request.url.path.endswith("/search")
                return httpx.Response(200, json={"organic": [{
                    "title": "Fixture source", "link": "https://example.org/evidence",
                    "snippet": "The result of six times seven is 42.",
                }]})

            http_client = await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.MockTransport(search_response)),
            )
            tools_module = importlib.import_module(f"workflows.{pipeline}.tools")

            def tool_factory(role_id, **kwargs):
                kwargs["config"].serper_api_key = "offline-fixture"
                kwargs["config"].serper_base_url = "https://search.test"
                return tools_module.build_research_tools(
                    role_id, http_client=http_client, **kwargs,
                )

            original_chat = model.chat

            async def search_before_submission(messages, *, tools=None, **kwargs):
                response = await original_chat(messages, tools=tools, **kwargs)
                names = {t["function"]["name"] for t in tools or []}
                if "web_search" in names and not any(m.get("role") == "tool" for m in messages):
                    return model.response("web_search", {"q": "six times seven", "num_results": 1})
                return response

            model.chat = search_before_submission
            seams["tool_factory"] = tool_factory
        stack.enter_context(override_runtime(**seams))
        async with BenchmarkSession() as session:
            assert registry.get(PipelineRegistry).has(pipeline)
            result = await session.run(
                question, pipeline_id=pipeline, meta=metadata,
                extra_input={"outside_context_policy": "must be filtered before node"},
            )
            graph = session.scheduler._compiled_cache[pipeline]
            assert isinstance(graph, MiniDAGRunner)
            task = await session.pm.get_task(result["task_id"])
            assert task.status == TaskStatus.COMPLETED

    assert registry.snapshot() == initial_registry
    assert len(node_inputs) == len(node_outputs) == 1
    node_input, task_id = node_inputs[0]
    assert node_input["original_question"] == question
    assert node_input["task_id"] == task_id == result["task_id"]
    assert node_input["metadata"]["profile"] == "auto"
    assert node_input["metadata"]["dynir_config"] == small_config
    assert node_input["metadata"]["answer_type"] == answer_type
    assert not {"ground_truth", "judge_task", "rubric"} & node_input["metadata"].keys()
    assert "outside_context_policy" not in node_input
    assert result["original_question"] == question
    # Compare every field emitted by the real node, not only final_answer:
    # report, tree, route, confidence, usage and artifact metadata must survive.
    assert node_outputs[0]
    for key, value in node_outputs[0].items():
        assert result[key] == value, f"Node output field was filtered or changed: {key}"
    assert result["task_strategy"] == strategy
    if pipeline == "dynamic_htn_informed_research":
        assert "report_kind" not in result
    else:
        assert result["report_kind"] == report_kind
    assert result["usage_summary"]["calls"] == len(model.calls)
    assert result["task_tree"]
    assert Path(result["artifact_dir"]).is_dir()
    if strategy == "report_research":
        assert result["delivery_mode"] == "report"
        assert result["report"]["content"] == result["final_content"] == model.report
    else:
        assert result["final_answer"].endswith("# Answer: 42")
    assert not model.unexpected
    prefix = "dynir_v1_" if pipeline.endswith("_v1") else "dynir_"
    assert model.worker_roles and all(role.startswith(prefix) for role in model.worker_roles)
