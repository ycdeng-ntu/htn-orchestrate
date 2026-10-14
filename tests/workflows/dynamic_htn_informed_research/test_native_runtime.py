"""Native boundary tests with no provider, search, or sandbox network traffic."""
from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse, StreamDelta
from agent_harness.core.tool import Tool
from workflows.dynamic_htn_informed_research.adapters import HighEffortTimeoutFallbackLLMClient
from workflows.dynamic_htn_informed_research.diagnostics import diagnostic_scope
from workflows.dynamic_htn_informed_research.htn import AgentTaskSummary
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse as HTNResponse
from workflows.dynamic_htn_informed_research.runtime.context import (
    TaskRuntime, current_runtime, override_runtime,
)
from workflows.dynamic_htn_informed_research.runtime.llm import (
    NativeHTNClient, ResearchToolCallParser, parse_partial_json,
)
from workflows.dynamic_htn_informed_research.runtime.worker import build_dispatch


def native(name, args=None, *, usage=None, content=""):
    return LLMResponse(content=content, usage=usage or {}, tool_calls=[{
        "id": "provider-id", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
    }])


class FakeLLM:
    model = "fake"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = 0

    async def chat(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)

    async def aclose(self):
        self.closed += 1


class NativeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    async def test_control_envelope_retains_unknown_companion_for_strict_validation(self):
        response = native("route_task", {"answer": "完整数据"})
        response.tool_calls.extend(native("unknown").tool_calls)
        llm = NativeHTNClient(FakeLLM(response))
        result = await llm.chat([], tools=[{"function": {"name": "route_task"}}])
        self.assertIsInstance(result, HTNResponse)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(all(set(call) == {"function"} for call in result.tool_calls))
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"answer": "完整数据"})

    async def test_argument_repair_never_silently_discards_report_or_code(self):
        parser = ResearchToolCallParser()
        for name, raw in (
            ("submit_report", '{"content":"a long report that was cut'),
            ("run_python_code", '{"code":"print(1)\n'),
            ("web_search", '["not an object"]'),
            ("web_fetch", '{"url":"https://example.test", "focus":oops}'),
        ):
            with self.subTest(name=name, raw=raw):
                response = native(name)
                response.tool_calls[0]["function"]["arguments"] = raw
                if name == "web_fetch":
                    # Deliberately supported short flat JSONish search values.
                    parsed = parser.parse(response, {name})
                    self.assertEqual(parsed[0]["args"]["focus"], "oops")
                else:
                    with self.assertRaises(ValueError):
                        parser.parse(response, {name})
                    self.assertEqual(response.tool_calls[0]["function"]["arguments"], raw)
        self.assertIsNone(parse_partial_json('{"answer":"keep", "confidence":oops}'))
        self.assertEqual(parse_partial_json('{"answer":"keep all characters'), {"answer": "keep all characters"})

    async def test_flat_search_repair_and_valid_report_keep_raw_data(self):
        parser = ResearchToolCallParser()
        response = native("web_search")
        response.tool_calls[0]["function"]["arguments"] = '{"q": marine bird census, "num_results": 10}'
        call = parser.parse(response, {"web_search"})[0]
        self.assertEqual(call["args"], {"q": "marine bird census", "num_results": 10})
        content = 'Unicode 汉字; quotes " and code \\n remain exact.'
        response = native("submit_report", {"content": content, "confidence": 0.4})
        self.assertEqual(parser.parse(response, {"submit_report"})[0]["args"]["content"], content)

    async def test_mixed_calls_keep_valid_arguments_in_either_order(self):
        content = 'Full report 汉字; quotes " and tail remain exact.'
        for name, raw in (("run_python_code", '{"code":"print(1)'),
                          ("submit_report", '{"content":"truncated'),
                          ("web_search", '["not an object"]'),
                          ("web_search", ["not an object"])):
            for bad_first in (True, False):
                with self.subTest(name=name, raw=raw, bad_first=bad_first):
                    valid = native("submit_report", {"content": content, "confidence": 0.7}).tool_calls[0]
                    valid["id"] = "valid"
                    bad = native(name).tool_calls[0]
                    bad["id"] = "bad"
                    bad["function"]["arguments"] = raw
                    response = LLMResponse(tool_calls=[bad, valid] if bad_first else [valid, bad])
                    original = copy.deepcopy(response)
                    parser = ResearchToolCallParser()
                    calls = parser.parse(response, {"submit_report", "run_python_code", "web_search"})
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]["id"], "valid")
                    self.assertEqual(calls[0]["args"], {"content": content, "confidence": 0.7})
                    self.assertEqual(parser.last_raw_response, original)

    async def test_unknown_call_cannot_rescue_malformed_known_call(self):
        response = native("submit_report")
        response.tool_calls[0]["function"]["arguments"] = '{"content":"truncated'
        response.tool_calls.extend(native("unknown").tool_calls)
        with self.assertRaises(ValueError):
            ResearchToolCallParser().parse(response, {"submit_report"})

    async def test_control_envelope_does_not_accept_valid_sibling_of_malformed_call(self):
        response = native("route_task", {"strategy_name": "research"})
        bad = native("run_python_code").tool_calls[0]
        bad["function"]["arguments"] = '{"code":"truncated'
        response.tool_calls.append(bad)
        result = await NativeHTNClient(FakeLLM(response)).chat(
            [], tools=[{"function": {"name": "route_task"}}],
        )
        self.assertEqual(result.tool_calls, [])

    async def test_task_isolation_unknown_usage_and_close(self):
        async def run(task_id):
            client = FakeLLM(LLMResponse(model=task_id, usage={"prompt_tokens": 10, "completion_tokens": 2}), LLMResponse())
            with override_runtime(llm_factory=lambda *_: client):
                async with TaskRuntime({}, task_id, self.root / task_id) as runtime:
                    await asyncio.sleep(0)
                    self.assertIs(current_runtime(), runtime)
                    wrapped = runtime.native_llm("fake", "fake", {})
                    self.assertIs(wrapped, runtime.native_llm("fake", "fake", {}))
                    await wrapped.chat([])
                    await wrapped.chat([])
                    totals = runtime.usage_summary()
                self.assertEqual(client.closed, 1)
                await runtime.aclose()
                self.assertEqual(client.closed, 1)
            return runtime, totals

        pairs = await asyncio.gather(run("a"), run("b"))
        for runtime, totals in pairs:
            self.assertEqual(totals, {"calls": 2, "unknown_usage_calls": 1, "prompt_tokens": 10, "completion_tokens": 2})
            self.assertEqual(runtime.usage_records[0]["model"], runtime.task_id)
        with self.assertRaises(RuntimeError):
            current_runtime()

    async def test_provider_credentials_summary_separation_and_redaction(self):
        made = []

        def constructor(model, **kwargs):
            made.append((model, kwargs))
            return FakeLLM()

        profile = {
            "llm": {"provider": "alpha", "api_key": "main-secret", "base_url": "https://main.test"},
            "summary_llm": {"provider": "alpha", "api_key": "summary-secret", "base_url": "https://summary.test"},
        }
        # Mock dotenv itself: these tests never open the workspace .env file.
        with patch("dotenv.dotenv_values", return_value={}), patch.dict(
            "os.environ", {"BETA_API_KEY": "beta-secret", "BETA_BASE_URL": "https://beta.test"}, clear=True,
        ), patch("agent_harness.infra.openai_client.OpenAIClient", side_effect=constructor):
            async with TaskRuntime(profile, "task", self.root) as runtime:
                runtime.native_llm("alpha", "same-model", {})
                runtime.native_llm("alpha", "same-model", {}, purpose="summary")
                runtime.native_llm("beta", "other-model", {})
                rendered = (self.root / "effective_config.json").read_text()
                self.assertNotIn("main-secret", rendered)
                self.assertNotIn("summary-secret", rendered)
        self.assertEqual([kw["api_key"] for _, kw in made], ["main-secret", "summary-secret", "beta-secret"])
        self.assertEqual([kw["base_url"] for _, kw in made], ["https://main.test", "https://summary.test", "https://beta.test"])

    async def test_close_failure_does_not_skip_remaining_clients(self):
        class Broken(FakeLLM):
            async def aclose(self):
                self.closed += 1
                raise RuntimeError("close failed")

        broken, healthy = Broken(), FakeLLM()
        with override_runtime(llm_factory=lambda provider, *_: broken if provider == "a" else healthy):
            async with TaskRuntime({}, "task", self.root) as runtime:
                runtime.native_llm("a", "fake", {})
                runtime.native_llm("b", "fake", {})
        self.assertEqual((broken.closed, healthy.closed), (1, 1))
        self.assertEqual(runtime.close_errors, ["RuntimeError"])

    async def test_summary_and_tool_credentials_resolve_from_task_local_dotenv(self):
        from workflows.dynamic_htn_informed_research.tools.summary import summary_candidates

        fake_env = {
            "OPENROUTER_API_KEY": "profile-provider-key",
            "SUMMARY_LLM_FALLBACK_API_KEY": "fallback-key",
            "SUMMARY_LLM_FALLBACK_BASE_URL": "https://fallback.test/v1",
            "SUMMARY_LLM_FALLBACK_MODEL": "fallback-model",
            "SERPER_API_KEY": "dotenv-search-key", "JINA_API_KEY": "dotenv-fetch-key",
            "E2B_API_KEY": "dotenv-sandbox-key",
        }
        profile = {"summary_llm": {"provider": "openrouter", "model": "selected-model"}}
        with patch("dotenv.dotenv_values", return_value=fake_env), patch.dict(
            "os.environ", {"SERPER_API_KEY": "process-search-key"}, clear=True,
        ):
            async with TaskRuntime(profile, "task", self.root) as runtime:
                resolved = runtime.resolved_summary_config()
                candidates = summary_candidates(resolved, environment=runtime.summary_environment())
                self.assertEqual([c["model"] for c in candidates], ["selected-model", "fallback-model"])
                self.assertEqual(candidates[0]["base_url"], "https://openrouter.ai/api/v1")
                self.assertEqual(candidates[0]["api_key"], "profile-provider-key")
                tools = runtime.resolved_tool_config()
                self.assertEqual(tools["serper_api_key"], "process-search-key")
                self.assertEqual(tools["jina_api_key"], "dotenv-fetch-key")
                self.assertEqual(tools["e2b_api_key"], "dotenv-sandbox-key")
                self.assertNotIn("SERPER_API_KEY", runtime.summary_environment())
            import os
            self.assertNotIn("JINA_API_KEY", os.environ)

    async def test_cancelled_native_request_is_recorded_as_unknown_once(self):
        class Waiting(FakeLLM):
            started = asyncio.Event()

            async def chat(self, messages, **kwargs):
                self.started.set()
                await asyncio.Event().wait()

        client = Waiting()
        with override_runtime(llm_factory=lambda *_: client):
            async with TaskRuntime({}, "task", self.root) as runtime:
                task = asyncio.create_task(runtime.native_llm("fake", "fake", {}).chat([]))
                await asyncio.wait_for(client.started.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(runtime.usage_summary()["calls"], 1)
                self.assertEqual(runtime.usage_summary()["unknown_usage_calls"], 1)
                self.assertEqual(runtime.usage_records[0]["status"], "CancelledError")

    async def test_sandbox_ttl_keeps_default_without_configuration(self):
        from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig

        runtime = TaskRuntime({}, "task", self.root)
        with patch.object(runtime, "_credential_source", return_value={}):
            resolved = runtime.resolved_tool_config()
        self.assertNotIn("e2b_timeout", resolved)
        self.assertEqual(ResearchToolConfig.from_profile({}, **resolved).e2b_timeout, 1800)

    async def test_sandbox_ttl_resolves_positive_integer_and_profile_precedence(self):
        from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig

        for profile, env, expected in (
            ({}, {"E2B_TIMEOUT": "7200"}, 7200),
            ({}, {"E2B_TIMEOUT": " 7500 "}, 7500),
            ({"web": {"e2b_timeout": 8000}}, {"E2B_TIMEOUT": "7200"}, 8000),
            ({"web": {"e2b_timeout": "9000"}}, {}, 9000),
        ):
            with self.subTest(profile=profile, env=env):
                runtime = TaskRuntime(profile, "task", self.root)
                with patch.object(runtime, "_credential_source", return_value=env):
                    config = ResearchToolConfig.from_profile(profile, **runtime.resolved_tool_config())
                self.assertIs(type(config.e2b_timeout), int)
                self.assertEqual(config.e2b_timeout, expected)

    async def test_sandbox_ttl_rejects_invalid_values_without_falling_back(self):
        invalid = (0, -1, True, False, 1.5, 7200.0, None, "", "  ", "0", "-1", "1.5", "1e3", "bad")
        for value in invalid:
            for use_profile in (True, False):
                with self.subTest(value=value, use_profile=use_profile):
                    profile = {"web": {"e2b_timeout": value}} if use_profile else {}
                    env = {"E2B_TIMEOUT": "7200" if use_profile else value}
                    runtime = TaskRuntime(profile, "task", self.root)
                    with patch.object(runtime, "_credential_source", return_value=env):
                        with self.assertRaisesRegex(ValueError, "positive integer"):
                            runtime.resolved_tool_config()

    async def test_stream_counts_once_and_closes_on_partial_consumption(self):
        class Streaming(FakeLLM):
            ended = False

            async def stream(self, messages, **kwargs):
                try:
                    yield StreamDelta(content="first")
                    yield StreamDelta(usage={"prompt_tokens": 5, "completion_tokens": 2})
                finally:
                    self.ended = True

        client = Streaming()
        with override_runtime(llm_factory=lambda *_: client):
            async with TaskRuntime({}, "task", self.root) as runtime:
                wrapped = runtime.native_llm("fake", "fake", {})
                stream = wrapped.stream([])
                await anext(stream)
                await stream.aclose()
                self.assertTrue(client.ended)
                self.assertEqual(runtime.usage_summary()["unknown_usage_calls"], 1)
                self.assertEqual(runtime.usage_records[0]["status"], "GeneratorExit")
                _ = [delta async for delta in wrapped.stream([])]
                self.assertEqual(runtime.usage_summary(), {"calls": 2, "unknown_usage_calls": 1, "prompt_tokens": 5, "completion_tokens": 2})

    async def test_high_timeout_uses_low_and_only_immediate_protocol_retry_stays_low(self):
        class EffortClient:
            model = "fake"

            def __init__(self, effort):
                self.reasoning_effort = effort
                self.calls = 0
                self.cancelled = False

            async def chat(self, messages, tools=None):
                self.calls += 1
                if self.reasoning_effort == "high" and self.calls == 1:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.cancelled = True
                return HTNResponse(content=self.reasoning_effort)

        high, low = EffortClient("high"), EffortClient("low")
        client = HighEffortTimeoutFallbackLLMClient(high, low, timeout_s=0.01)
        with diagnostic_scope(call_kind="planner"):
            self.assertEqual((await client.chat([])).content, "low")
        with diagnostic_scope(call_kind="planner_protocol_retry"):
            self.assertEqual((await client.chat([])).content, "low")
        with diagnostic_scope(call_kind="unrelated"):
            self.assertEqual((await client.chat([])).content, "high")
        self.assertTrue(high.cancelled)
        self.assertEqual((high.calls, low.calls), (2, 2))


class FakeTools:
    def __init__(self, *, bad_submit=False):
        self.report_latch = NS(submitted=False, report="", confidence=0.0)
        self.evidence_cards = []
        self.closed = False
        self.search_started = asyncio.Event()
        self.block_search = False

        async def search(**kwargs):
            self.search_started.set()
            if self.block_search:
                await asyncio.Event().wait()
            self.evidence_cards.append({"source": {"url": "https://example.test/source", "title": "Observed title"}})
            return "observed search result"

        async def submit(content, confidence):
            self.report_latch.submitted = True
            self.report_latch.report = content
            self.report_latch.confidence = confidence
            if bad_submit:
                raise RuntimeError("submit failed after a side effect")
            return "submitted"

        self.tools = [
            Tool("web_search", "search", {"type": "object", "properties": {}}, search),
            Tool("submit_report", "submit", {"type": "object", "properties": {}}, submit),
        ]

    async def aclose(self):
        self.closed = True


class WorkerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        dotenv = patch("dotenv.dotenv_values", return_value={})
        dotenv.start()
        self.addCleanup(dotenv.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.estimator = patch("agent_harness.core.runtime.loop.agent_loop.estimate_tokens", return_value=1)
        self.estimator.start()
        self.addCleanup(self.estimator.stop)
        self.agent = AgentTaskSummary("worker/id", "dynir_researcher", "find evidence")

    async def dispatch(self, llm, tools, *, max_turns=2, budget=0):
        with override_runtime(llm_factory=lambda *_: llm, tool_factory=lambda *_, **__: tools):
            async with TaskRuntime({"agent": {"worker_token_budget": budget, "max_llm_retries": 1}}, "task", self.root) as runtime:
                dispatch = build_dispatch("task", max_turns=max_turns, sampling={}, provider="fake", model="fake")
                return await dispatch(self.agent, "find evidence"), runtime

    async def test_explicit_submission_stops_and_links_only_observed_sources(self):
        report = "Finding based on https://example.test/source and unverified https://other.test/missing"
        llm = FakeLLM(native("web_search"), native("submit_report", {"content": report, "confidence": 0.6}))
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools)
        self.assertTrue(result.submitted)
        self.assertEqual(result.execution_quality, "submitted")
        self.assertEqual(result.stop_reason, "submitted_report")
        self.assertEqual(result.result_summary, report)
        self.assertIn('[exact_observed] "Observed title"', result.observed_source_index)
        self.assertIn("[unobserved] https://other.test/missing", result.observed_source_index)
        self.assertEqual([t["function"]["name"] for t in llm.requests[1][1]["tools"]], ["submit_report"])
        self.assertTrue(tools.closed)

    async def test_mixed_worker_calls_execute_search_then_submit_with_paired_history(self):
        response = native("web_search", {"q": "evidence"})
        response.tool_calls[0]["id"] = "good-search"
        bad = native("submit_report").tool_calls[0]
        bad["id"] = "bad-submit"
        bad["function"]["arguments"] = '{"content":"truncated report'
        response.tool_calls.insert(0, bad)
        report = "Complete report based on https://example.test/source"
        llm = FakeLLM(response, native("submit_report", {"content": report, "confidence": 0.6}))
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools)
        self.assertTrue(tools.search_started.is_set())
        self.assertTrue(result.submitted)
        self.assertEqual(result.result_summary, report)
        self.assertIn("exact_observed", result.observed_source_index)
        history = llm.requests[1][0]
        replies = {message["tool_call_id"]: message["content"] for message in history if message["role"] == "tool"}
        self.assertIn("observed search result", replies["good-search"])
        self.assertIn("not executed", replies["bad-submit"])
        original = next(message for message in history if message.get("tool_calls"))
        self.assertEqual(original["tool_calls"][0]["function"]["arguments"], bad["function"]["arguments"])
        self.assertTrue(tools.closed)

    async def test_failed_submission_cannot_latch_delivery_quality(self):
        llm = FakeLLM(native("submit_report", {"content": "enough readable report " * 4, "confidence": 0.6}))
        result, _ = await self.dispatch(llm, FakeTools(bad_submit=True), max_turns=1)
        self.assertFalse(result.submitted)
        self.assertNotEqual(result.execution_quality, "submitted")

    async def test_one_turn_worker_exposes_only_submit_and_rejects_unknown_text_companion(self):
        content = '<tool_call>{"tool":"submit_report","args":{"content":"report","confidence":0.5}}</tool_call><tool_call>{"tool":"unknown","args":{}}</tool_call>'
        llm = FakeLLM(LLMResponse(content=content))
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools, max_turns=1)
        self.assertEqual([t["function"]["name"] for t in llm.requests[0][1]["tools"]], ["submit_report"])
        self.assertFalse(result.submitted)
        self.assertFalse(tools.report_latch.submitted)
        self.assertEqual(result.stop_reason, "tool_policy_violation")
        self.assertEqual(result.execution_quality, "malformed")

    async def test_worker_budget_uses_normalized_usage(self):
        llm = FakeLLM(
            native("web_search", usage={"prompt_tokens": 1200, "completion_tokens": 300}),
            native("submit_report", {"content": "Report from the observed evidence", "confidence": 0.6},
                   usage={"prompt_tokens": 700, "completion_tokens": 100}),
        )
        result, runtime = await self.dispatch(llm, FakeTools(), budget=1000)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertFalse(result.submitted)
        self.assertEqual(runtime.usage_summary()["prompt_tokens"], 1200)
        self.assertEqual(runtime.usage_summary()["completion_tokens"], 300)
        self.assertEqual(runtime.usage_summary()["calls"], 1)

    async def test_malformed_first_response_keeps_original_arguments_in_artifact(self):
        response = native("submit_report")
        raw = '{"content":"keep this original truncated report'
        response.tool_calls[0]["function"]["arguments"] = raw
        llm = FakeLLM(response, response)
        result, _ = await self.dispatch(llm, FakeTools(), max_turns=1)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "malformed_tool_calls")
        self.assertEqual(result.execution_quality, "malformed")
        self.assertEqual(len(llm.requests), 2)
        artifact = next(self.root.glob("workers/*/result.json"))
        saved = json.loads(artifact.read_text())
        self.assertIn(raw, json.dumps(saved["messages"], ensure_ascii=False).replace('\\"', '"'))

    async def test_all_bad_calls_get_one_correction_with_usage_and_paired_history(self):
        bad = native("submit_report", usage={"prompt_tokens": 100, "completion_tokens": 10})
        raw = '{"content":"original unmodified report'
        bad.tool_calls[0]["function"]["arguments"] = raw
        report = 'Model-authored corrected report; Unicode 汉字 and quotes " remain exact.'
        llm = FakeLLM(bad, native("submit_report", {"content": report, "confidence": 0.7},
                                usage={"prompt_tokens": 200, "completion_tokens": 20}))
        tools = FakeTools()
        result, runtime = await self.dispatch(llm, tools, max_turns=1)
        self.assertTrue(result.submitted)
        self.assertEqual(result.result_summary, report)
        self.assertEqual(runtime.usage_summary()["prompt_tokens"], 300)
        self.assertEqual(runtime.usage_summary()["completion_tokens"], 30)
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual([t["function"]["name"] for t in llm.requests[1][1]["tools"]], ["submit_report"])
        history = llm.requests[1][0]
        original = next(m for m in history if m.get("tool_calls"))
        self.assertEqual(original["tool_calls"][0]["function"]["arguments"], raw)
        replies = [m for m in history if m["role"] == "tool"]
        self.assertEqual(len(replies), 1)
        self.assertIn("not executed", replies[0]["content"])
        self.assertTrue(any("one opportunity" in str(m.get("content")) for m in history))
        saved = json.loads(next(self.root.glob("workers/*/result.json")).read_text())
        self.assertEqual(saved["metadata"]["budget_tokens_used"], 330)
        self.assertEqual(saved["metadata"]["tool_call_correction"]["outcome"], "corrected")
        self.assertTrue(tools.closed)

    async def test_truncated_code_executes_only_model_authored_correction(self):
        bad = native("run_python_code")
        raw = '{"code":"print(unsafe incomplete'
        bad.tool_calls[0]["function"]["arguments"] = raw
        code = 'value = "原始值"\nprint(value)\n'
        llm = FakeLLM(bad, native("run_python_code", {"code": code}),
                      native("submit_report", {"content": "Report after valid calculation", "confidence": 0.7}))
        tools = FakeTools()
        executed = []

        async def run_code(code):
            executed.append(code)
            return "原始值"

        tools.tools.append(Tool("run_python_code", "code", {"type": "object", "properties": {}}, run_code))
        result, _ = await self.dispatch(llm, tools, max_turns=3)
        self.assertTrue(result.submitted)
        self.assertEqual(executed, [code])
        history = llm.requests[1][0]
        self.assertEqual(next(m for m in history if m.get("tool_calls"))["tool_calls"][0]["function"]["arguments"], raw)

    async def test_cancelling_parameter_correction_preserves_trace_and_closes_tools(self):
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        bad = native("submit_report", usage={"prompt_tokens": 123})
        bad.tool_calls[0]["function"]["arguments"] = '{"content":"incomplete'

        class WaitingLLM(FakeLLM):
            async def chat(self, messages, **kwargs):
                if not self.requests:
                    return await super().chat(messages, **kwargs)
                self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
                waiting.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        llm, tools = WaitingLLM(bad), FakeTools()
        task = asyncio.create_task(self.dispatch(llm, tools))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(cancelled.is_set() and tools.closed)
        self.assertEqual(llm.closed, 1)
        saved = json.loads(next(self.root.glob("workers/*/cancelled.json")).read_text())
        self.assertEqual(saved["metadata"]["budget_tokens_used"], 123)
        self.assertEqual(saved["metadata"]["tool_call_correction"]["attempts"], 1)
        self.assertFalse(saved["metadata"]["tool_call_correction"]["response_received"])
        self.assertTrue(any("not executed" in str(m.get("content")) for m in saved["messages"]))

    async def test_correction_of_penultimate_turn_keeps_research_then_arms_terminal_policy(self):
        bad = native("web_search")
        bad.tool_calls[0]["function"]["arguments"] = '["not an object"]'
        llm = FakeLLM(bad, native("web_search"), native("submit_report", {"content": "Complete evidence report", "confidence": 0.7}))
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools, max_turns=2)
        self.assertTrue(result.submitted)
        self.assertTrue(tools.search_started.is_set())
        self.assertEqual(len(llm.requests), 3)
        offered = [{t["function"]["name"] for t in request[1]["tools"]} for request in llm.requests]
        self.assertEqual(offered, [{"web_search", "submit_report"}, {"web_search", "submit_report"}, {"submit_report"}])
        self.assertNotIn("The next response is your final turn", json.dumps(llm.requests[1][0]))
        self.assertIn("The next response is your final turn", json.dumps(llm.requests[2][0]))

    async def test_plain_text_correction_does_not_get_another_nudge(self):
        bad = native("submit_report")
        bad.tool_calls[0]["function"]["arguments"] = '{"content":"truncated'
        llm = FakeLLM(bad, LLMResponse(content="This is ordinary report text."))
        result, _ = await self.dispatch(llm, FakeTools(), max_turns=3)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "malformed_tool_calls")
        self.assertEqual(len(llm.requests), 2)

    async def test_later_malformed_response_does_not_receive_a_second_correction(self):
        bad = native("web_search")
        bad.tool_calls[0]["function"]["arguments"] = '[]'
        llm = FakeLLM(bad, native("web_search"), bad)
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools, max_turns=4)
        self.assertEqual(result.stop_reason, "malformed_tool_calls")
        self.assertEqual(len(llm.requests), 3)
        self.assertTrue(tools.search_started.is_set())
        saved = json.loads(next(self.root.glob("workers/*/result.json")).read_text())
        self.assertEqual(saved["metadata"]["tool_call_correction"]["attempts"], 1)
        self.assertEqual(saved["metadata"]["tool_call_correction"]["outcome"], "failed")
        self.assertEqual(len(saved["evidence_cards"]), 1)

    async def test_exhausted_budget_prevents_parameter_correction(self):
        bad = native("submit_report", usage={"prompt_tokens": 1000})
        bad.tool_calls[0]["function"]["arguments"] = '{"content":"truncated'
        llm = FakeLLM(bad)
        result, runtime = await self.dispatch(llm, FakeTools(), max_turns=2, budget=1000)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(runtime.usage_summary()["prompt_tokens"], 1000)

    async def test_final_turn_correction_cannot_reopen_research(self):
        bad = native("submit_report")
        bad.tool_calls[0]["function"]["arguments"] = '{"content":"truncated'
        llm = FakeLLM(native("web_search"), bad, native("web_search"))
        tools = FakeTools()
        result, _ = await self.dispatch(llm, tools, max_turns=2)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "tool_policy_violation")
        self.assertEqual(len(tools.evidence_cards), 1)
        self.assertEqual([t["function"]["name"] for t in llm.requests[2][1]["tools"]], ["submit_report"])

    async def test_cancelled_worker_closes_tools_and_preserves_current_trace(self):
        tools = FakeTools()
        tools.block_search = True
        llm = FakeLLM(native("web_search"))
        with override_runtime(llm_factory=lambda *_: llm, tool_factory=lambda *_, **__: tools):
            async with TaskRuntime({}, "task", self.root) as runtime:
                dispatch = build_dispatch("task", max_turns=2, sampling={}, provider="fake", model="fake")
                task = asyncio.create_task(dispatch(self.agent, "find evidence"))
                await asyncio.wait_for(tools.search_started.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(tools.closed)
                artifact = next(self.root.glob("workers/*/cancelled.json"))
                saved = json.loads(artifact.read_text())
                self.assertTrue(saved["messages"])
                self.assertFalse(saved["submitted"])
                self.assertEqual(runtime.usage_summary()["calls"], 1)


if __name__ == "__main__":
    unittest.main()


class RouterEndpointIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_model_and_sampling_do_not_reuse_router_endpoint_for_execution(self):
        made = []
        def constructor(model, **kwargs):
            made.append(kwargs)
            return FakeLLM()
        router = {"llm": {"provider": "openai", "model": "same", "api_key": "router-key",
                          "base_url": "https://router.test/v1", "headers": {"X-Route": "router"}}}
        execution = {"llm": {"provider": "openai", "model": "same", "api_key": "worker-key",
                             "base_url": "https://worker.test/v1", "headers": {"X-Route": "worker"}}}
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.dotenv_values", return_value={}), patch.dict(
            "os.environ", {}, clear=True,
        ), patch("agent_harness.infra.openai_client.OpenAIClient", side_effect=constructor):
            async with TaskRuntime(router, "isolation", Path(directory), router_profile=router) as runtime:
                routing = runtime.native_llm("openai", "same", {}, purpose="router")
                runtime.profile = execution
                worker = runtime.native_llm("openai", "same", {})
                self.assertIsNot(routing, worker)
                self.assertIs(routing, runtime.native_llm("openai", "same", {}, purpose="router"))
                self.assertIs(worker, runtime.native_llm("openai", "same", {}))
        self.assertEqual([c["api_key"] for c in made], ["router-key", "worker-key"])
        self.assertEqual([c["base_url"] for c in made], ["https://router.test/v1", "https://worker.test/v1"])
        self.assertEqual([c["default_headers"] for c in made], [{"X-Route": "router"}, {"X-Route": "worker"}])
