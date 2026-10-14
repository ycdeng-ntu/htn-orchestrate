from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError

import pytest

from workflows.dynamic_htn_informed_research.tools.scrape_cache import (
    MAX_ATTEMPTS,
    RETRY_COOLDOWNS,
    _ScrapeCache,
    format_skip_message,
)

URL = "https://example.org/research"


def test_cooldown_boundaries_preserve_consecutive_history() -> None:
    cache = _ScrapeCache()
    now = 1000.0
    for attempt, delay in enumerate(RETRY_COOLDOWNS, start=1):
        assert cache.check(URL, now=now) is None
        entry = cache.record_failure(URL, 503, now=now, reason="http_error")
        assert entry is not None
        assert entry.fail_count == attempt
        assert entry.ban_until == now + delay
        assert cache.check(URL, now=now + delay - 0.001) is entry
        assert cache.check(URL, now=now + delay) is None
        assert cache.size() == 1
        now += delay

    final = cache.record_failure(URL, 403, now=now, reason="http_error")
    assert final is not None and final.exhausted
    assert final.fail_count == MAX_ATTEMPTS == 4
    assert final.ban_until == 0
    assert cache.check(URL, now=now + 10**9) is final
    assert cache.record_failure(URL, 503, now=now + 10**9) is final


def test_blocked_checks_do_not_extend_cooldown_or_consume_attempts() -> None:
    cache = _ScrapeCache()
    entry = cache.record_failure(URL, 403, now=100.0)
    before = cache.snapshot()
    for now in (100.0, 110.0, 120.0, 159.999):
        assert cache.check(URL, now=now) is entry
    assert cache.snapshot() == before
    assert cache.check(URL, now=160.0) is None


def test_success_resets_failure_history_including_exhaustion() -> None:
    cache = _ScrapeCache()
    now = 0.0
    for delay in (*RETRY_COOLDOWNS, 1):
        cache.record_failure(URL, 500, now=now)
        now += delay
    assert cache.check(URL, now=now).exhausted
    cache.record_success(URL)
    assert cache.size() == 0
    assert cache.check(URL, now=now) is None
    assert cache.record_failure(URL, 404, now=now).fail_count == 1


@pytest.mark.parametrize(
    ("status", "reason"),
    [(403, "http_error"), (422, "empty_content"), (429, "http_error"),
     (503, "http_error"), (None, "read_timeout"), (None, "connection_error")],
)
def test_real_failure_status_and_reason_are_preserved(status, reason) -> None:
    cache = _ScrapeCache()
    entry = cache.record_failure(URL, status, now=100, reason=reason, backend="reader")
    assert entry.status == status
    assert entry.reason == reason
    assert entry.backend == "reader"
    message = format_skip_message(URL, entry, now=100)
    assert reason in message
    assert "reader" in message
    if status != 429:
        assert "429" not in message
    if status is None:
        assert "HTTP" not in message


def test_changed_failure_type_does_not_reset_attempt_count() -> None:
    cache = _ScrapeCache()
    cache.record_failure(URL, 503, now=0, reason="http_error", backend="reader")
    entry = cache.record_failure(URL, None, now=60, reason="read_timeout", backend="direct")
    assert entry.fail_count == 2
    assert entry.status is None
    assert entry.reason == "read_timeout"
    assert entry.backend == "direct"
    assert entry.ban_until == 180


def test_snapshot_is_json_safe_immutable_and_keeps_expired_entries() -> None:
    cache = _ScrapeCache()
    entry = cache.record_failure(URL, 503, now=0)
    with pytest.raises(FrozenInstanceError):
        entry.fail_count = 50
    assert cache.check(URL, now=60) is None
    snap = cache.snapshot()
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap
    snap[URL]["fail_count"] = 50
    assert cache.snapshot()[URL]["fail_count"] == 1
    for now in (60, 180, 420):
        cache.record_failure(URL, 503, now=now)
    assert json.loads(json.dumps(cache.snapshot(), allow_nan=False))[URL]["exhausted"]


def test_empty_url_is_ignored_and_clear_drops_history() -> None:
    cache = _ScrapeCache()
    assert cache.record_failure("", 503) is None
    assert cache.check("") is None
    cache.record_failure(URL, 503)
    cache.clear()
    assert cache.snapshot() == {}
    assert cache.size() == 0


def test_injected_clock_and_skip_message_boundaries() -> None:
    now = [100.0]
    cache = _ScrapeCache(clock=lambda: now[0])
    entry = cache.record_failure(URL, None, reason="read_timeout")
    assert entry.ban_until == 160.0
    now[0] = 159.1
    assert cache.check(URL) is entry
    assert "1 seconds" in format_skip_message(URL, entry, now=now[0])
    now[0] = 160.0
    assert cache.check(URL) is None
    for t in (160, 280, 520):
        entry = cache.record_failure(URL, 503, now=t)
    message = format_skip_message(URL, entry, now=10**9)
    assert "all 3 retries failed" in message
    assert "do not fetch this URL again during this question" in message


@pytest.mark.asyncio
async def test_concurrent_failed_fetches_admit_one_attempt_per_cooldown() -> None:
    now = [0.0]
    cache = _ScrapeCache(clock=lambda: now[0])
    actual_fetches = 0

    async def fetch() -> str:
        nonlocal actual_fetches
        async with cache.guard(URL):
            if cache.check(URL) is not None:
                return "skipped"
            actual_fetches += 1
            await asyncio.sleep(0)
            cache.record_failure(URL, 503, reason="http_error")
            return "failed"

    for attempt, when in enumerate((0.0, 60.0, 180.0, 420.0), start=1):
        now[0] = when
        outcomes = await asyncio.gather(*(fetch() for _ in range(33)))
        assert outcomes.count("failed") == 1
        assert outcomes.count("skipped") == 32
        assert actual_fetches == attempt
        assert cache._guards == {}
    now[0] = 10**9
    assert await asyncio.gather(*(fetch() for _ in range(33))) == ["skipped"] * 33
    assert actual_fetches == 4
    assert cache._guards == {}


@pytest.mark.asyncio
async def test_success_is_not_cached_and_unrelated_urls_do_not_wait() -> None:
    cache = _ScrapeCache()
    first_entered = asyncio.Event()
    release = asyncio.Event()
    other_entered = asyncio.Event()
    same_entered = asyncio.Event()

    async def hold() -> None:
        async with cache.guard(URL):
            first_entered.set()
            await release.wait()
            cache.record_success(URL)

    async def same_url() -> None:
        async with cache.guard(URL):
            assert cache.check(URL) is None
            same_entered.set()

    async def other_url() -> None:
        async with cache.guard("https://example.net/independent"):
            other_entered.set()

    holder = asyncio.create_task(hold())
    await first_entered.wait()
    same = asyncio.create_task(same_url())
    other = asyncio.create_task(other_url())
    await asyncio.wait_for(other_entered.wait(), 1)
    assert not same_entered.is_set()
    release.set()
    await asyncio.gather(holder, same, other)
    assert same_entered.is_set()
    assert cache._guards == {}


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_existing_lock_until_holder_finishes() -> None:
    cache = _ScrapeCache()
    held = asyncio.Event()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def hold() -> None:
        async with cache.guard(URL):
            held.set()
            await release.wait()

    async def wait() -> None:
        async with cache.guard(URL):
            entered.set()

    holder = asyncio.create_task(hold())
    await held.wait()
    waiter = asyncio.create_task(wait())
    await asyncio.sleep(0)
    original_lock = cache._guards[URL].lock
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cache._guards[URL].users == 1
    replacement = asyncio.create_task(wait())
    await asyncio.sleep(0)
    assert cache._guards[URL].lock is original_lock
    assert not entered.is_set()
    release.set()
    await asyncio.gather(holder, replacement)
    assert entered.is_set()
    assert cache._guards == {}


@pytest.mark.asyncio
async def test_cancelled_holder_releases_waiter_and_cleans_registry() -> None:
    cache = _ScrapeCache()
    held = asyncio.Event()
    entered = asyncio.Event()

    async def hold() -> None:
        async with cache.guard(URL):
            held.set()
            await asyncio.Event().wait()

    async def wait() -> None:
        async with cache.guard(URL):
            entered.set()

    holder = asyncio.create_task(hold())
    await held.wait()
    waiter = asyncio.create_task(wait())
    await asyncio.sleep(0)
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder
    await asyncio.wait_for(waiter, 1)
    assert entered.is_set()
    assert cache._guards == {}


@pytest.mark.asyncio
async def test_clear_does_not_replace_a_live_guard() -> None:
    cache = _ScrapeCache()
    async with cache.guard(URL):
        cache.record_failure(URL, 503)
        guard = cache._guards[URL]
        cache.clear()
        assert cache._guards[URL] is guard
        assert cache.size() == 0
    assert cache._guards == {}
