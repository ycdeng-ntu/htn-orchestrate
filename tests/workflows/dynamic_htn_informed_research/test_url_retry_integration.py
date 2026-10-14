"""Question-wide URL retries and partial fetch batches, using only fake HTTP."""
from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from agent_harness.core.llm import LLMResponse
from workflows.dynamic_htn_informed_research.htn import AgentTaskSummary
from workflows.dynamic_htn_informed_research.runtime.context import TaskRuntime, override_runtime
from workflows.dynamic_htn_informed_research.runtime.worker import build_dispatch
from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig, ResearchToolError, build_research_tools
from workflows.dynamic_htn_informed_research.tools.scrape_cache import _ScrapeCache


URL = "https://paper.example/article"
BODY = "# Research evidence\nThe independent measured value is forty-two."


def config(cache, **overrides):
    values = {
        "serper_api_key": "", "serper_base_url": "https://search.example",
        "jina_api_key": "fake-key", "jina_base_url": "https://jina.example",
        "e2b_api_key": "", "e2b_template": "unused",
        "ncbi_api_key": "", "unpaywall_email": "",
        "summary_use_environment": False, "direct_fetch_enabled": False,
        "retry_delay": 0, "scrape_cache": cache,
    }
    values.update(overrides)
    return ResearchToolConfig(**values)


def fetch(runtime, url=URL, **kwargs):
    tool = next(tool for tool in runtime.tools if tool.name == "web_fetch")
    return tool.ainvoke({"url": url, **kwargs})


async def worker(stack, tmp_path, http, cache, **overrides):
    return await stack.enter_async_context(build_research_tools(
        "dynir_researcher", task_id="question", artifact_dir=tmp_path,
        config=config(cache, **overrides), http_client=http,
    ))


@pytest.mark.asyncio
async def test_two_workers_share_direct_403_after_empty_jina(tmp_path):
    requests = []

    def provider(request):
        requests.append(request.method)
        return httpx.Response(200, json={"data": {"content": ""}}) if request.method == "POST" else httpx.Response(403)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        task = TaskRuntime({}, "question", tmp_path)
        first = await worker(stack, tmp_path, http, task.scrape_cache, direct_fetch_enabled=True)
        second = await worker(stack, tmp_path, http, task.scrape_cache, direct_fetch_enabled=True)
        with pytest.raises(ResearchToolError, match="403"):
            await fetch(first)
        with pytest.raises(ResearchToolError, match="403") as skipped:
            await fetch(second)
        assert "429" not in str(skipped.value)
        assert requests == ["POST", "GET"]
        assert task.scrape_cache.check(URL).fail_count == 1
        assert first._negative_cache is second._negative_cache is task.scrape_cache
        assert not first.evidence_cards and not second.evidence_cards


@pytest.mark.asyncio
async def test_expired_cooldowns_retain_failures_and_fourth_stops_question(tmp_path):
    now, requests = [1000.0], []
    cache = _ScrapeCache(clock=lambda: now[0])

    def provider(request):
        requests.append(request)
        return httpx.Response(503)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        for count, cooldown in enumerate((60, 120, 240), 1):
            with pytest.raises(ResearchToolError, match="503"):
                await fetch(tools)
            assert len(requests) == count
            assert cache.check(URL).fail_count == count
            now[0] += cooldown - 0.1
            with pytest.raises(ResearchToolError):
                await fetch(tools)
            assert len(requests) == count
            now[0] += 0.1
        with pytest.raises(ResearchToolError):
            await fetch(tools)
        assert len(requests) == 4
        assert cache.check(URL).fail_count == 4
        now[0] += 1_000_000
        later_worker = await worker(stack, tmp_path, http, cache)
        with pytest.raises(ResearchToolError):
            await fetch(later_worker)
        assert len(requests) == 4


@pytest.mark.asyncio
async def test_later_success_clears_failures_and_next_failure_restarts_backoff(tmp_path):
    now, calls = [1000.0], []
    cache = _ScrapeCache(clock=lambda: now[0])

    def provider(request):
        calls.append(request)
        if len(calls) == 2:
            return httpx.Response(200, json={"data": {"content": BODY}})
        return httpx.Response(503)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        with pytest.raises(ResearchToolError):
            await fetch(tools)
        now[0] += 60
        assert await fetch(tools) == BODY
        assert cache.size() == 0
        with pytest.raises(ResearchToolError):
            await fetch(tools)
        assert cache.check(URL).fail_count == 1
        now[0] += 60
        assert cache.check(URL) is None
        assert len(calls) == 3


@pytest.mark.parametrize("failure", [429, 503, "ReadTimeout"])
@pytest.mark.asyncio
async def test_jina_transient_failure_is_one_attempt_and_preserves_reason(tmp_path, failure):
    calls = []
    cache = _ScrapeCache()

    def provider(request):
        calls.append(request)
        if failure == "ReadTimeout":
            raise httpx.ReadTimeout("fake transport timeout", request=request)
        return httpx.Response(failure)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        with pytest.raises(ResearchToolError) as initial:
            await fetch(tools)
        with pytest.raises(ResearchToolError) as skipped:
            await fetch(tools)
        assert len(calls) == 1
        reason = "timeout" if failure == "ReadTimeout" else str(failure)
        assert reason in str(initial.value)
        assert reason in str(skipped.value)
        if failure != 429:
            assert "429" not in str(initial.value) + str(skipped.value)
        assert cache.check(URL).fail_count == 1
        attempt = next(item for item in tools.provenance if item["tool_name"] == "web_fetch_attempt" and item["outcome"] == "failed")
        saved = json.loads(Path(attempt["path"]).read_text())
        backend = saved["backend_attempts"][0]
        if failure == "ReadTimeout":
            assert backend["error"] == "ReadTimeout"
            assert backend["reason"] == "timeout"
            assert backend["status"] is None
        else:
            assert backend["status"] == failure


@pytest.mark.parametrize("status", [403, 422])
@pytest.mark.asyncio
async def test_browser_fallback_success_does_not_create_url_failure(tmp_path, status):
    engines = []
    cache = _ScrapeCache()

    def provider(request):
        engines.append(request.headers.get("X-Engine", "default"))
        if engines[-1] == "browser":
            return httpx.Response(200, json={"data": {"content": BODY}})
        return httpx.Response(status)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        assert await fetch(tools) == BODY
        assert engines == ["default", "browser"]
        assert cache.size() == 0
        assert len(tools.evidence_cards) == 1


@pytest.mark.asyncio
async def test_direct_fallback_success_does_not_create_url_failure(tmp_path):
    requests = []
    cache = _ScrapeCache()
    html = "<html><body><article><h1>Research evidence</h1><p>" + "The independent measurement is forty-two. " * 20 + "</p></article></body></html>"

    def provider(request):
        requests.append(request.method)
        return httpx.Response(503) if request.method == "POST" else httpx.Response(200, text=html)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache, direct_fetch_enabled=True)
        assert "independent measurement" in await fetch(tools)
        assert requests == ["POST", "GET"]
        assert cache.size() == 0
        assert len(tools.evidence_cards) == 1


@pytest.mark.asyncio
async def test_oa_fallback_success_preserves_alternative_and_does_not_ban_requested_url(tmp_path):
    cache, requests = _ScrapeCache(), []
    original = "https://paper.example/doi/10.1234/example"
    alternative = "https://repository.example/article.pdf"

    def provider(request):
        requests.append(request)
        if request.url.host == "api.unpaywall.org":
            return httpx.Response(200, json={"best_oa_location": {"url_for_pdf": alternative}})
        url = json.loads(request.content)["url"]
        return httpx.Response(503) if url == original else httpx.Response(200, json={"data": {"content": BODY}})

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache, unpaywall_email="research@example.org")
        assert await fetch(tools, original) == BODY
        assert len(requests) == 3
        assert cache.size() == 0
        assert tools.evidence_cards[0]["source"]["url"] == original
        assert tools.evidence_cards[0]["actual_url"] == alternative


@pytest.mark.asyncio
async def test_short_valid_oa_replaces_long_blocked_preprint(tmp_path, monkeypatch):
    cache, requested = _ScrapeCache(), []
    original = "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1"
    alternative = "https://repository.example/paper"

    def provider(request):
        url = json.loads(request.content)["url"]
        requested.append(url)
        body = BODY if url == alternative else "# Access denied\n" + "x" * 600
        return httpx.Response(200, json={"data": {"content": body}})

    async def oa_url(_):
        return alternative

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        monkeypatch.setattr(tools.academic, "oa_url", oa_url)
        assert await fetch(tools, original) == BODY
        assert requested == [original + ".full.pdf", alternative]
        assert tools.evidence_cards[0]["actual_url"] == alternative
        assert cache.size() == 0


@pytest.mark.parametrize("original", [
    "https://pubs.acs.org/doi/10.1234/example",
    "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1",
])
@pytest.mark.asyncio
async def test_short_valid_alternative_survives_long_blocked_original(tmp_path, monkeypatch, original):
    cache = _ScrapeCache()
    alternative = original + ".full.pdf" if "biorxiv" in original else "https://repository.example/paper"

    def provider(request):
        url = json.loads(request.content)["url"]
        body = "# Access denied\n" + "x" * 600 if url == original else BODY
        return httpx.Response(200, json={"data": {"content": body}})

    async def oa_url(_):
        return alternative

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        monkeypatch.setattr(tools.academic, "oa_url", oa_url)
        assert await fetch(tools, original) == BODY
        assert tools.evidence_cards[0]["actual_url"] == alternative
        assert cache.size() == 0


@pytest.mark.asyncio
async def test_summary_failure_keeps_retrieved_content_and_does_not_ban_url(tmp_path):
    cache = _ScrapeCache()

    async def broken_summary(*_):
        raise RuntimeError("fake summary failure")

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": {"content": BODY}}),
        )))
        tools = await worker(stack, tmp_path, http, cache)
        tools.summary_client = broken_summary
        result = await fetch(tools, info_to_extract="Find the measured value")
        assert BODY in result
        assert cache.size() == 0
        assert tools.evidence_cards[0]["source"]["url"] == URL


@pytest.mark.asyncio
async def test_same_url_concurrent_failure_makes_one_actual_attempt(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    cache = _ScrapeCache()

    async def provider(request):
        calls.append(request)
        started.set()
        await release.wait()
        return httpx.Response(503)

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        runtimes = [await worker(stack, tmp_path, http, cache) for _ in range(3)]
        tasks = [asyncio.create_task(fetch(runtime)) for runtime in runtimes]
        try:
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
            assert all(isinstance(result, ResearchToolError) for result in results)
            assert len(calls) == 1
            assert cache.check(URL).fail_count == 1
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("cancel_owner", [False, True])
@pytest.mark.asyncio
async def test_cancelling_url_owner_or_waiter_does_not_poison_lock(tmp_path, cancel_owner):
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    cache = _ScrapeCache()

    async def provider(request):
        calls.append(request)
        if len(calls) == 1:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return httpx.Response(200, json={"data": {"content": BODY}})

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        first = await worker(stack, tmp_path, http, cache)
        second = await worker(stack, tmp_path, http, cache)
        owner = asyncio.create_task(fetch(first))
        waiter = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            waiter = asyncio.create_task(fetch(second))
            await asyncio.sleep(0)
            victim, survivor = (owner, waiter) if cancel_owner else (waiter, owner)
            victim.cancel()
            with pytest.raises(asyncio.CancelledError):
                await victim
            if not cancel_owner:
                assert not cancelled.is_set()
                release.set()
            assert await asyncio.wait_for(survivor, 2) == BODY
            assert cancelled.is_set() is cancel_owner
            assert cache.size() == 0
            assert await asyncio.wait_for(fetch(second), 2) == BODY
        finally:
            for task in (owner, waiter):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in (owner, waiter) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_partial_batch_keeps_slow_success_and_its_evidence(tmp_path):
    slow_started, bad_finished = asyncio.Event(), asyncio.Event()
    requests = []
    cache = _ScrapeCache()

    async def provider(request):
        url = json.loads(request.content)["url"]
        requests.append(url)
        if url.endswith("/bad"):
            await slow_started.wait()
            bad_finished.set()
            return httpx.Response(503)
        slow_started.set()
        await bad_finished.wait()
        await asyncio.sleep(0)
        return httpx.Response(200, json={"data": {"content": BODY}})

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        urls = ["https://paper.example/bad", "https://paper.example/good"]
        result = await asyncio.wait_for(fetch(tools, urls), 2)
        assert "Error:" in result and "Info:" in result
        assert all(url in result for url in urls)
        assert BODY in result
        assert len(requests) == 2
        assert [card["source"]["url"] for card in tools.evidence_cards] == [urls[1]]
        saved = json.loads(Path(tools.evidence_cards[0]["raw_path"]).read_text())
        assert saved["content"] == BODY
        assert cache.check(urls[0]).fail_count == 1
        assert cache.check(urls[1]) is None


@pytest.mark.asyncio
async def test_all_failed_batch_is_tool_error_and_records_each_url(tmp_path):
    cache = _ScrapeCache()
    urls = ["https://paper.example/first", "https://paper.example/second"]
    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503))))
        tools = await worker(stack, tmp_path, http, cache)
        with pytest.raises(ResearchToolError) as error:
            await fetch(tools, urls)
        assert all(url in str(error.value) for url in urls)
        assert all(cache.check(url).fail_count == 1 for url in urls)
        assert not tools.evidence_cards


@pytest.mark.asyncio
async def test_external_batch_cancellation_cleans_all_http_requests(tmp_path):
    all_started = asyncio.Event()
    started, cancelled = set(), set()
    cache = _ScrapeCache()

    async def provider(request):
        url = json.loads(request.content)["url"]
        started.add(url)
        if len(started) == 2:
            all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(url)
            raise

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(provider)))
        tools = await worker(stack, tmp_path, http, cache)
        batch = asyncio.create_task(fetch(tools, ["https://paper.example/a", "https://paper.example/b"]))
        try:
            await asyncio.wait_for(all_started.wait(), 2)
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await batch
            assert cancelled == started
            assert cache.size() == 0
            assert not tools.evidence_cards
        finally:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_workers_share_question_cache_and_other_questions_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda *_: {})
    monkeypatch.setattr(TaskRuntime, "_credential_source", lambda self: {})
    requests, observed_caches = [], []

    def provider(request):
        requests.append(request)
        return httpx.Response(503)

    class Model:
        model = "offline-url-retry"

        async def chat(self, messages, **kwargs):
            seen_fetch = any(message.get("role") == "tool" for message in messages)
            name = "submit_report" if seen_fetch else "web_fetch"
            arguments = {"content": "The requested source was unavailable; no factual claim is established.", "confidence": 0.0} if seen_fetch else {"url": URL}
            return LLMResponse(tool_calls=[{
                "id": f"call-{len(messages)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }], usage={"prompt_tokens": 10, "completion_tokens": 5})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        def factory(role, **kwargs):
            observed_caches.append(kwargs["config"].scrape_cache)
            kwargs["config"] = replace(config(kwargs["config"].scrape_cache), summary_llm={})
            return build_research_tools(role, **kwargs, http_client=http)

        with override_runtime(llm_factory=lambda *_: Model(), tool_factory=factory):
            caches = []
            for question in ("first", "second"):
                async with TaskRuntime({"agent": {"max_llm_retries": 1}}, question, tmp_path / question) as runtime:
                    caches.append(runtime.scrape_cache)
                    dispatch = build_dispatch(question, max_turns=3, sampling={}, provider="fake", model="fake", job_timeout_s=5)
                    for index in range(2):
                        await dispatch(AgentTaskSummary(f"worker-{index}", "dynir_researcher", "Check source"), "Fetch the source, then submit findings.")
        assert len(requests) == 2
        assert observed_caches == [caches[0], caches[0], caches[1], caches[1]]
        assert caches[0] is not caches[1]
        for question in ("first", "second"):
            state = json.loads((tmp_path / question / "fetch_retry_state.json").read_text())
            assert state[URL]["fail_count"] == 1
            assert state[URL]["status"] == 503
