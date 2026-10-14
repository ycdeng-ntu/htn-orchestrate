from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_harness.core.llm import LLMResponse
from agent_harness.core.llm_errors import LLMQuotaError
from agent_harness.core.runtime.loop.llm_client import call_llm
from workflows.dynamic_htn_informed_research.runtime.context import TaskRuntime
from workflows.dynamic_htn_informed_research.runtime.llm import MeteredLLM, NativeHTNClient
from workflows.dynamic_htn_informed_research.tools.summary import SummaryExtractor


class APIError(Exception):
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body
        super().__init__("sensitive-provider-body-must-not-escape")


class Client:
    model = "test-model"

    def __init__(self, error):
        self.error, self.calls = error, 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return LLMResponse(content="Recovered")

    async def stream(self, *args, **kwargs):
        self.calls += 1
        raise self.error
        yield  # pragma: no cover - async generator protocol


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status,body", [
    (402, None), (429, {"error": {"code": "insufficient_quota"}}),
    (429, {"type": "insufficient_quota"}),
])
async def test_quota_raises_once_without_sleep_or_losing_type(monkeypatch, stream, status, body):
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    client = Client(APIError(status, body))
    with pytest.raises(LLMQuotaError) as caught:
        await call_llm(client, [], 1, 4, 1, on_delta=AsyncMock() if stream else None)
    assert caught.value.status_code == status
    assert "sensitive-provider" not in str(caught.value)
    assert client.calls == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [APIError(429, {"error": {"code": "rate_limit_exceeded"}}),
                                   APIError(503), TimeoutError()])
async def test_transient_errors_still_retry(monkeypatch, error):
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    client = Client(error)
    result = await call_llm(client, [], 1, 4, 1)
    assert result.content == "Recovered"
    assert client.calls == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_htn_control_retries_do_not_swallow_quota():
    client = Client(APIError(402))
    with pytest.raises(LLMQuotaError):
        await NativeHTNClient(client).chat([])
    assert client.calls == 1


@pytest.mark.asyncio
async def test_question_watcher_stops_swallowed_error_and_settles_children(tmp_path):
    runtime = TaskRuntime({}, "quota-task", tmp_path)
    bad = Client(APIError(402))
    wrapped = MeteredLLM(bad, runtime, "test", "test-model")
    cleaned = asyncio.Event()

    async def work():
        try:
            try:
                await wrapped.chat([])
            except Exception:
                pass  # A strategy's generic recovery handler must not hide failure.
            with pytest.raises(LLMQuotaError):
                await wrapped.chat([])
            await asyncio.Event().wait()
        finally:
            (tmp_path / "preserved-report.md").write_text("Existing supported work")
            cleaned.set()

    with runtime:
        with pytest.raises(LLMQuotaError):
            await runtime.run_until_quota(work())
    assert bad.calls == 1 and cleaned.is_set()
    assert len(runtime.usage_records) == 1  # Suppressed requests are not billed requests.
    assert (tmp_path / "preserved-report.md").read_text() == "Existing supported work"
    assert json.loads((tmp_path / "quota_error.json").read_text())["error_code"] == "llm_quota_exhausted"
    other = TaskRuntime({}, "independent-task", tmp_path / "other")
    good = Client(APIError(503))
    good.calls = 1
    assert (await MeteredLLM(good, other, "test", "test-model").chat([])).content == "Recovered"


@pytest.mark.asyncio
async def test_streaming_quota_notifies_question_watcher(tmp_path):
    runtime = TaskRuntime({}, "stream", tmp_path)
    client = MeteredLLM(Client(APIError(402)), runtime, "test", "test-model")
    with runtime:
        with pytest.raises(LLMQuotaError):
            async for _ in client.stream([]):
                pass
    assert runtime.quota_failure is not None
    assert runtime.usage_records[0]["status"] == "LLMQuotaError"


@pytest.mark.asyncio
async def test_summary_model_quota_is_reported_without_retry(tmp_path):
    runtime = TaskRuntime({}, "summary", tmp_path)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    with runtime:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            summary = SummaryExtractor(http, {"model": "test", "api_key": "fake",
                                              "base_url": "https://test.invalid/v1"},
                                       use_environment=False, usage_sink=runtime.record_external_usage)
            with pytest.raises(LLMQuotaError):
                await summary("Public evidence", "Summarize")
    assert len(requests) == 1
    assert runtime.quota_failure.status_code == 429


@pytest.mark.asyncio
async def test_webpage_paywall_does_not_trip_model_quota_watch(tmp_path):
    runtime = TaskRuntime({}, "web", tmp_path)

    async def work():
        try:
            httpx.Response(402, request=httpx.Request("GET", "https://test.invalid/article")).raise_for_status()
        except httpx.HTTPStatusError:
            return "Use another source"

    assert await runtime.run_until_quota(work()) == "Use another source"
    assert runtime.quota_failure is None
