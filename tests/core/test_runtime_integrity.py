"""Offline regressions for usage, streaming and tool protocol boundaries."""
from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from agent_harness.components.observers.budget_observer import BudgetObserver
from agent_harness.components.observers.context_size_guard import ContextSizeGuard
from agent_harness.core.llm import LLMResponse, StreamDelta
from agent_harness.core.loop_types import BaseObserver, Intervention, LoopConfig, LoopPolicy
from agent_harness.core.runtime.loop.agent_loop import (
    PER_TURN_TOOL_ALLOWLIST_KEY,
    run_agent_loop,
)
from agent_harness.core.runtime.loop.llm_client import _stream_llm_response, extract_usage
from agent_harness.core.runtime.loop.model_profile import ModelProfile
from agent_harness.core.tool import Tool
from agent_harness.infra.openai_client import OpenAIClient, _to_llm_response


def native(*names: str, ids: list[str] | None = None) -> LLMResponse:
    return LLMResponse(tool_calls=[
        {"id": ids[i] if ids is not None else f"native_{i}", "type": "function",
         "function": {"name": name, "arguments": "{}"}}
        for i, name in enumerate(names)
    ])


class SequenceLLM:
    model = "offline"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return copy.deepcopy(next(self.responses))


class ArmAllowlist(BaseObserver):
    critical = True

    def __init__(self, names):
        self.names = names
        self.tool_hooks = []

    async def on_turn_end(self, ctx):
        if ctx.turn == 1:
            ctx.metadata[PER_TURN_TOOL_ALLOWLIST_KEY] = self.names

    async def on_tool_call(self, ctx, call):
        self.tool_hooks.append(call["name"])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Token estimation is unrelated to these cases and must not download
        # a tiktoken encoding when the test environment has no local cache.
        self.estimator = patch(
            "agent_harness.core.runtime.loop.agent_loop.estimate_tokens", return_value=1,
        )
        self.estimator.start()
        self.addCleanup(self.estimator.stop)
        self.executed = []
        self.tools = []
        for name in ("probe", "allowed", "forbidden", "terminal"):
            async def run(_name=name):
                self.executed.append(_name)
                return "ok"
            self.tools.append(Tool(name, name, {"type": "object", "properties": {}}, run))

    async def run_loop(self, llm, observers=None, **options):
        profile = options.pop("model_profile", None)
        config = LoopConfig(
            max_turns=options.pop("max_turns", 4), max_llm_retries=1,
            compact_after_turns=1000, context_token_limit=10**9,
            loop_policy=LoopPolicy(no_tool_behavior="stop"), **options,
        )
        return await run_agent_loop(
            system_prompt="offline", user_message="task", llm=llm,
            tools=self.tools, config=config, observers=observers,
            model_profile=profile,
        )

    def assert_paired(self, messages):
        pending = set()
        seen = set()
        for message in messages:
            if message["role"] == "tool":
                self.assertIn(message["tool_call_id"], pending)
                pending.remove(message["tool_call_id"])
            else:
                self.assertFalse(pending, "unanswered calls before the next message")
                if message["role"] == "assistant":
                    for call in message.get("tool_calls", []):
                        self.assertTrue(call["id"])
                        self.assertNotIn(call["id"], seen)
                        seen.add(call["id"])
                        pending.add(call["id"])
        self.assertFalse(pending)

    async def test_unknown_companion_and_cap_receive_responses(self):
        llm = SequenceLLM(native("allowed", "missing", "forbidden"), LLMResponse("done"))
        result = await self.run_loop(llm, max_tool_calls_per_turn=1)
        self.assertEqual(self.executed, ["allowed"])
        self.assertEqual(result.tool_calls_count, 1)
        self.assert_paired(result.messages)
        self.assert_paired(llm.requests[1][0])
        replies = [m["content"] for m in result.messages if m["role"] == "tool"]
        self.assertTrue(any("unknown tool" in reply for reply in replies))
        self.assertTrue(any("not dispatched" in reply for reply in replies))

    async def test_missing_duplicate_and_reused_native_ids_are_paired(self):
        llm = SequenceLLM(
            native("allowed", "probe", ids=["", ""]),
            native("allowed", "probe", ids=["reused", "reused"]),
            native("allowed", ids=["reused"]), LLMResponse("done"),
        )
        result = await self.run_loop(llm)
        self.assertEqual(len(self.executed), 5)
        self.assert_paired(result.messages)
        for messages, _ in llm.requests:
            self.assert_paired(messages)

    async def test_text_calls_are_recorded_and_ids_unique_across_turns(self):
        text = '<tool_call>{"tool":"allowed","args":{}}</tool_call>'
        llm = SequenceLLM(LLMResponse(text), LLMResponse(text), LLMResponse("done"))
        result = await self.run_loop(llm)
        self.assertEqual(self.executed, ["allowed", "allowed"])
        self.assert_paired(result.messages)
        assistant = [m for m in result.messages if m.get("tool_calls")]
        self.assertEqual(json.loads(assistant[0]["tool_calls"][0]["function"]["arguments"]), {})

    async def test_calls_recovered_from_thinking_are_recorded(self):
        llm = SequenceLLM(LLMResponse(
            '<think><tool_call><function=allowed></function></tool_call></think>'
        ), LLMResponse("done"))
        result = await self.run_loop(
            llm, model_profile=ModelProfile("offline", "openai", thinking_format="tag"),
        )
        self.assertEqual(self.executed, ["allowed"])
        self.assert_paired(result.messages)

    async def test_rollback_removes_entire_rejected_turn(self):
        class Rollback(BaseObserver):
            critical = True
            fired = False

            async def on_llm_response(self, ctx):
                if not self.fired:
                    self.fired = True
                    return Intervention(pop_last_message=True, continue_to_next_turn=True)

        llm = SequenceLLM(native("allowed", "missing", "probe"), LLMResponse("done"))
        result = await self.run_loop(llm, [Rollback()], max_tool_calls_per_turn=1)
        self.assertEqual(self.executed, [])
        self.assertEqual([m["role"] for m in llm.requests[1][0]], ["system", "user"])
        self.assertEqual(result.turns_used, 1)
        self.assert_paired(result.messages)

    async def test_observer_skip_and_early_stop_keep_history_complete(self):
        class Skip(BaseObserver):
            critical = True

            async def on_llm_response(self, ctx):
                return Intervention(skip_tool_execution=True)

        class Stop(BaseObserver):
            critical = True

            async def on_llm_response(self, ctx):
                return Intervention(stop_reason="observer_stop")

        for observer in (Skip(), Stop()):
            with self.subTest(observer=type(observer).__name__):
                llm = SequenceLLM(native("allowed", "missing"), LLMResponse("done"))
                result = await self.run_loop(llm, [observer])
                self.assert_paired(result.messages)
                self.assertEqual(self.executed, [])

    async def test_allowlist_filters_schema_and_is_consumed_once(self):
        arm = ArmAllowlist(["allowed", "does_not_exist"])
        llm = SequenceLLM(native("probe"), native("allowed"), native("forbidden"), LLMResponse("done"))
        result = await self.run_loop(llm, [arm])
        schemas = [schema["function"]["name"] for schema in llm.requests[1][1]["tools"]]
        self.assertEqual(schemas, ["allowed"])
        self.assertEqual(len(llm.requests[2][1]["tools"]), 4)
        self.assertEqual(self.executed, ["probe", "allowed", "forbidden"])
        self.assertNotIn(PER_TURN_TOOL_ALLOWLIST_KEY, result.metadata)
        self.assert_paired(result.messages)

    async def test_allowlist_rejects_entire_mixed_turn_before_cap_or_hooks(self):
        for forbidden in ("forbidden", "unknown"):
            with self.subTest(forbidden=forbidden):
                self.executed.clear()
                arm = ArmAllowlist("terminal")
                llm = SequenceLLM(native("probe"), native("terminal", forbidden))
                result = await self.run_loop(llm, [arm], max_tool_calls_per_turn=1)
                self.assertEqual(result.stopped_by, "tool_policy_violation")
                self.assertEqual(self.executed, ["probe"])
                self.assertEqual(arm.tool_hooks, ["probe"])
                self.assertTrue(result.metadata["tool_policy_violation"]["mixed_with_allowed"])
                self.assert_paired(result.messages)

    async def test_unknown_text_companion_is_not_silently_dropped_by_policy(self):
        for forbidden in ("forbidden", "unknown"):
            for suffix in (
                f'<tool_call>{{"tool":"{forbidden}","args":{{}}}}</tool_call>',
                f'<tool_call><function={forbidden}></function></tool_call>',
            ):
                with self.subTest(forbidden=forbidden, suffix=suffix):
                    self.executed.clear()
                    arm = ArmAllowlist(["terminal"])
                    llm = SequenceLLM(native("probe"), LLMResponse(
                        '<tool_call>{"tool":"terminal","args":{}}</tool_call>' + suffix,
                    ))
                    result = await self.run_loop(llm, [arm])
                    self.assertEqual(result.stopped_by, "tool_policy_violation")
                    self.assertEqual(self.executed, ["probe"])
                    self.assertEqual(arm.tool_hooks, ["probe"])
                    self.assertIn(forbidden, result.metadata["tool_policy_violation"]["disallowed_tools"])
                    self.assert_paired(result.messages)

    async def test_empty_allowlist_hides_schemas_and_rejects_calls(self):
        llm = SequenceLLM(native("probe"), native("allowed"))
        result = await self.run_loop(llm, [ArmAllowlist([])])
        self.assertIsNone(llm.requests[1][1]["tools"])
        self.assertEqual(result.stopped_by, "tool_policy_violation")
        self.assertEqual(self.executed, ["probe"])
        self.assert_paired(result.messages)


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalized_usage_triggers_budget_and_context_guard(self):
        usage = extract_usage(LLMResponse(usage={"prompt_tokens": 1200, "completion_tokens": 300}))
        ctx = NS(usage=usage, metadata={}, turn=1)
        budget = BudgetObserver(max_tokens=1000)
        guard = ContextSizeGuard(max_input_tokens=1000)
        await budget.on_llm_response(ctx)
        self.assertEqual(budget.tokens_used, 1500)
        self.assertEqual((await budget.on_turn_end(ctx)).stop_reason, "budget_exhausted")
        self.assertIsNotNone(await guard.on_llm_response(ctx))
        await budget.on_loop_start(LoopConfig())
        await guard.on_loop_start(LoopConfig())
        self.assertEqual(budget.tokens_used, 0)
        self.assertIsNotNone(await guard.on_llm_response(ctx))

    async def test_usage_aliases_and_missing_counts_do_not_double_count(self):
        for usage in (
            {"input_tokens": 1200, "output_tokens": 300},
            {"prompt_tokens": 1200, "completion_tokens": 300, "input_tokens": 1200, "output_tokens": 300},
        ):
            with self.subTest(usage=usage):
                budget = BudgetObserver()
                await budget.on_llm_response(NS(usage=usage, metadata={}))
                self.assertEqual(budget.tokens_used, 1500)
                await budget.on_llm_response(NS(usage=None, metadata={}))
                self.assertEqual(budget.tokens_used, 1500)

    async def test_stream_preserves_usage_only_chunk_finish_reason_and_model(self):
        usage = NS(prompt_tokens=1200, completion_tokens=300, total_tokens=1500,
                   prompt_tokens_details=NS(cached_tokens=100))
        chunks = [
            NS(choices=[NS(delta=NS(content="answer"), finish_reason=None)], model="served", usage=None),
            NS(choices=[NS(delta=NS(), finish_reason="length")], model="served", usage=None),
            NS(choices=[], model="served", usage=usage),
        ]
        requests = []

        async def create(**kwargs):
            requests.append(kwargs)

            async def iterate():
                for chunk in chunks:
                    yield chunk
            return iterate()

        # Bypass SDK construction entirely: no credentials and no transport.
        client = object.__new__(OpenAIClient)
        client.model = "configured"
        client.default_temperature = client.default_max_tokens = client.default_timeout = None
        client.extra_body = {}
        client._client = NS(chat=NS(completions=NS(create=create)))
        deltas = []

        async def on_delta(*args):
            deltas.append(args)

        response = await _stream_llm_response(client, [], None, None, 5, on_delta)
        self.assertTrue(requests[0]["stream_options"]["include_usage"])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.model, "served")
        self.assertEqual(response.usage, {"prompt_tokens": 1200, "completion_tokens": 300,
                                          "total_tokens": 1500, "cached_tokens": 100})
        self.assertEqual(len(deltas), 1)
        complete = _to_llm_response(NS(
            choices=[NS(message=NS(content="answer"), finish_reason="length")],
            usage=usage, model="served", id="response-id",
        ))
        self.assertEqual(response.usage, complete.usage)
        self.assertEqual(response.finish_reason, complete.finish_reason)

    async def test_stream_uses_last_usage_snapshot_without_summing(self):
        class Client:
            async def stream(self, messages, **kwargs):
                yield StreamDelta(usage={"prompt_tokens": 20, "completion_tokens": 1})
                yield StreamDelta(content="ok", model="actual")
                yield StreamDelta(usage={"prompt_tokens": 20, "completion_tokens": 4}, finish_reason="stop")
                yield StreamDelta()

        async def on_delta(*args):
            pass

        response = await _stream_llm_response(Client(), [], None, None, 5, on_delta)
        self.assertEqual(response.usage, {"prompt_tokens": 20, "completion_tokens": 4})
        self.assertEqual(response.model, "actual")
        self.assertEqual(response.finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
