"""Budget-bound worker delivery through the native loop, without network access."""
from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse
from agent_harness.core.tool import Tool
from workflows.dynamic_htn_informed_research.htn import AgentTaskSummary
from workflows.dynamic_htn_informed_research.runtime.context import TaskRuntime, override_runtime
from workflows.dynamic_htn_informed_research.runtime.worker import build_dispatch


EVIDENCE = "Observed evidence from https://example.test/budget-source"
REPORT = "Supported finding from https://example.test/budget-source; remaining gaps are explicit."


def response(name=None, args=None, *, prompt=0, completion=0, content=""):
    calls = []
    if name:
        calls = [{
            "id": "provider-call", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }]
    return LLMResponse(
        content=content, tool_calls=calls,
        usage={"prompt_tokens": prompt, "completion_tokens": completion},
    )


def submission(**kwargs):
    return response("submit_report", {"content": REPORT, "confidence": 0.7}, **kwargs)


class ScriptedLLM:
    model = "fake-budget-model"

    def __init__(self, *responses, wait_after_script=False):
        self.responses = list(responses)
        self.requests = []
        self.wait_after_script = wait_after_script
        self.waiting = asyncio.Event()
        self.cancelled = False
        self.closed = False

    async def chat(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        index = len(self.requests) - 1
        if index < len(self.responses):
            return copy.deepcopy(self.responses[index])
        if self.wait_after_script:
            self.waiting.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        raise AssertionError("Worker requested an unexpected additional model response")

    async def aclose(self):
        self.closed = True


class EvidenceTools:
    def __init__(self, *, fail_submission=False):
        self.report_latch = SimpleNamespace(submitted=False, report="", confidence=0.0)
        self.evidence_cards = []
        self.calls = []
        self.closed = False

        async def search(**kwargs):
            self.calls.append("web_search")
            self.evidence_cards.append({
                "source": {"url": "https://example.test/budget-source", "title": "Budget evidence"},
            })
            return EVIDENCE

        async def submit(content, confidence):
            self.calls.append("submit_report")
            self.report_latch.submitted = True
            self.report_latch.report = content
            self.report_latch.confidence = confidence
            if fail_submission:
                raise RuntimeError("Submission failed after mutating the report latch")
            return "Report accepted"

        self.tools = [
            Tool("web_search", "Read evidence", {"type": "object", "properties": {}}, search),
            Tool("submit_report", "Deliver report", {
                "type": "object", "properties": {
                    "content": {"type": "string"}, "confidence": {"type": "number"},
                }, "required": ["content", "confidence"],
            }, submit),
        ]

    async def aclose(self):
        self.closed = True


class WorkerTokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Neither credentials nor real provider/sandbox clients are needed.
        self.dotenv = patch("dotenv.dotenv_values", return_value={})
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.estimator = patch(
            "agent_harness.core.runtime.loop.agent_loop.estimate_tokens", return_value=1,
        )
        self.estimator.start()
        self.addCleanup(self.estimator.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = AgentTaskSummary("budget-worker", "dynir_researcher", "find evidence")

    async def dispatch(self, llm, tools, *, max_turns=70, budget=None, job_timeout_s=None):
        profile = {"agent": {
            "max_llm_retries": 1, "tool_timeout_s": 5,
        }}
        if budget is not None:
            profile["agent"]["worker_token_budget"] = budget
        with override_runtime(llm_factory=lambda *_: llm, tool_factory=lambda *_, **__: tools):
            async with TaskRuntime(profile, "budget-task", self.root) as runtime:
                dispatch = build_dispatch(
                    "budget-task", max_turns=max_turns, sampling={}, provider="fake", model="fake",
                    job_timeout_s=job_timeout_s,
                )
                result = await dispatch(self.agent, "Research, then deliver a supported report")
                return result, runtime

    def artifact(self, name="result.json"):
        return json.loads(next(self.root.glob(f"workers/*/{name}")).read_text())

    def assert_tools_offered(self, llm, request, expected):
        tools = llm.requests[request][1].get("tools") or []
        self.assertEqual({tool["function"]["name"] for tool in tools}, set(expected))

    def assert_evidence_in_request(self, llm, request):
        messages = llm.requests[request][0]
        self.assertTrue(any(
            message.get("role") == "tool" and EVIDENCE in str(message.get("content"))
            for message in messages
        ))

    async def test_explicit_budget_warns_at_192k_and_keeps_research_available(self):
        llm = ScriptedLLM(
            response("web_search", prompt=160000),
            response("web_search", prompt=31999),
            response("web_search", completion=1),
            response("web_search", prompt=1000),
            submission(prompt=1000),
        )
        tools = EvidenceTools()
        result, runtime = await self.dispatch(llm, tools, budget=240000)
        self.assertTrue(result.submitted)
        self.assertEqual(result.stop_reason, "submitted_report")
        self.assertEqual(tools.calls, ["web_search"] * 4 + ["submit_report"])
        for request in range(5):
            self.assert_tools_offered(llm, request, {"web_search", "submit_report"})
        for request in (1, 2):
            self.assertNotIn("Warning: 80%", json.dumps(llm.requests[request][0]))
        warning_messages = [m for m in llm.requests[4][0] if "Warning: 80%" in str(m.get("content"))]
        self.assertEqual(len(warning_messages), 1)
        self.assertIn("192,000 / 240,000", warning_messages[0]["content"])
        self.assert_evidence_in_request(llm, 4)
        metadata = self.artifact()["metadata"]
        self.assertEqual(metadata["budget_tokens_limit"], 240000)
        self.assertEqual(metadata["budget_tokens_used"], 194000)
        self.assertNotIn("budget_finalization", metadata)
        self.assertEqual(runtime.usage_summary()["calls"], 5)
        self.assertTrue(tools.closed)

    async def test_explicit_budget_stops_at_240k_without_a_submission_grace(self):
        llm = ScriptedLLM(
            response("web_search", prompt=239999),
            response("web_search", completion=1),
            submission(),  # Must never be requested.
        )
        tools = EvidenceTools()
        result, _ = await self.dispatch(llm, tools, budget=240000)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(tools.calls, ["web_search", "web_search"])
        self.assert_tools_offered(llm, 1, {"web_search", "submit_report"})
        saved = self.artifact()
        self.assertEqual(saved["metadata"]["budget_tokens_used"], 240000)
        self.assertEqual(len(saved["evidence_cards"]), 2)
        self.assertTrue(tools.closed)

    async def test_response_overshoot_finishes_its_tools_without_another_model_call(self):
        llm = ScriptedLLM(response("web_search", prompt=1200, completion=300), submission())
        tools = EvidenceTools()
        result, runtime = await self.dispatch(llm, tools, budget=1000)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(tools.calls, ["web_search"])
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(runtime.usage_summary()["calls"], 1)
        self.assertEqual(self.artifact()["metadata"]["budget_tokens_used"], 1500)
        self.assertEqual(len(self.artifact()["evidence_cards"]), 1)

    async def test_no_tool_response_at_budget_stops_without_nudge(self):
        llm = ScriptedLLM(response(content="Partial findings", prompt=240000), submission())
        tools = EvidenceTools()
        result, _ = await self.dispatch(llm, tools, budget=240000)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(tools.calls, [])
        self.assertIn("Partial findings", json.dumps(self.artifact()["messages"]))

    async def test_successful_submission_at_or_above_budget_keeps_delivery_outcome(self):
        for max_turns, tokens in ((70, 240000), (70, 250000), (1, 250000)):
            with self.subTest(max_turns=max_turns, tokens=tokens):
                llm = ScriptedLLM(submission(prompt=tokens))
                tools = EvidenceTools()
                result, _ = await self.dispatch(llm, tools, max_turns=max_turns, budget=240000)
                self.assertTrue(result.submitted)
                self.assertEqual(result.stop_reason, "submitted_report")
                self.assertEqual(result.result_summary, REPORT)
                self.assertEqual(len(llm.requests), 1)
                if max_turns == 1:
                    self.assert_tools_offered(llm, 0, {"submit_report"})

    async def test_failed_submission_at_budget_does_not_count_as_delivery(self):
        llm = ScriptedLLM(submission(prompt=240000), submission())
        result, _ = await self.dispatch(llm, EvidenceTools(fail_submission=True), budget=240000)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(len(llm.requests), 1)

    async def test_default_unlimited_budget_reaches_turn_70_and_reserves_submission(self):
        llm = ScriptedLLM(*[response("web_search", prompt=50000) for _ in range(69)], submission(prompt=100))
        tools = EvidenceTools()
        result, _ = await self.dispatch(llm, tools)
        self.assertTrue(result.submitted)
        self.assertEqual(len(llm.requests), 70)
        self.assert_tools_offered(llm, 68, {"web_search", "submit_report"})
        self.assert_tools_offered(llm, 69, {"submit_report"})
        self.assertEqual(tools.calls, ["web_search"] * 69 + ["submit_report"])
        metadata = self.artifact()["metadata"]
        self.assertEqual(metadata["budget_tokens_limit"], 0)
        self.assertEqual(metadata["budget_tokens_used"], 3450100)
        self.assertFalse(any("token budget used" in str(m.get("content", ""))
                             for messages, _ in llm.requests for m in messages))

    async def test_zero_budget_keeps_token_accounting_without_stopping(self):
        llm = ScriptedLLM(response("web_search", prompt=300000), submission(prompt=100))
        result, _ = await self.dispatch(llm, EvidenceTools(), budget=0)
        self.assertTrue(result.submitted)
        self.assertEqual(self.artifact()["metadata"]["budget_tokens_limit"], 0)
        self.assertEqual(self.artifact()["metadata"]["budget_tokens_used"], 300100)
        self.assert_tools_offered(llm, 1, {"web_search", "submit_report"})

    async def test_each_dispatch_starts_with_its_own_budget(self):
        llm = ScriptedLLM(
            response("web_search", prompt=800), submission(prompt=10),
            response("web_search", prompt=800), submission(prompt=10),
        )
        made = []

        def tool_factory(*_, **__):
            tools = EvidenceTools()
            made.append(tools)
            return tools

        with override_runtime(llm_factory=lambda *_: llm, tool_factory=tool_factory):
            async with TaskRuntime({"agent": {"worker_token_budget": 1000}}, "budget-task", self.root):
                dispatch = build_dispatch("budget-task", max_turns=70, sampling={}, provider="fake", model="fake")
                first = await dispatch(self.agent, "First worker")
                second = await dispatch(self.agent, "Second worker")
        self.assertTrue(first.submitted and second.submitted)
        self.assertEqual(len(llm.requests), 4)
        self.assert_tools_offered(llm, 1, {"web_search", "submit_report"})
        self.assert_tools_offered(llm, 3, {"web_search", "submit_report"})
        self.assertTrue(all(tools.closed for tools in made))
        saved = [json.loads(path.read_text()) for path in self.root.glob("workers/*/result.json")]
        self.assertEqual([item["metadata"]["budget_tokens_used"] for item in saved], [810, 810])

    async def test_cancellation_after_warning_preserves_evidence_and_usage(self):
        llm = ScriptedLLM(response("web_search", prompt=192000), wait_after_script=True)
        tools = EvidenceTools()
        task = asyncio.create_task(self.dispatch(llm, tools, budget=240000))
        try:
            await asyncio.wait_for(llm.waiting.wait(), timeout=3)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(llm.cancelled and llm.closed and tools.closed)
        self.assert_tools_offered(llm, 1, {"web_search", "submit_report"})
        saved = self.artifact("cancelled.json")
        self.assertFalse(saved["submitted"])
        self.assertEqual(saved["metadata"]["budget_tokens_used"], 192000)
        self.assertNotIn("budget_finalization", saved["metadata"])
        self.assertTrue(any(EVIDENCE in str(m.get("content")) for m in saved["messages"]))

    async def test_worker_deadline_still_bounds_research_with_cap_disabled(self):
        llm = ScriptedLLM(response("web_search", prompt=400000), wait_after_script=True)
        tools = EvidenceTools()
        result, _ = await self.dispatch(llm, tools, job_timeout_s=0.2)
        self.assertFalse(result.submitted)
        self.assertEqual(result.stop_reason, "timeout")
        self.assertTrue(llm.cancelled and llm.closed and tools.closed)
        self.assertEqual(self.artifact()["metadata"]["budget_tokens_used"], 400000)
        self.assertEqual(len(self.artifact()["evidence_cards"]), 1)
