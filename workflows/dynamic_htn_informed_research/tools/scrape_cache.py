"""Question-owned URL failure history, cooldowns, and fetch serialization.

A failed fetch gets at most three later retries, after 60, 120, and 240
seconds respectively. The fourth consecutive failure stops further fetches
for this question. Cooldown expiry preserves the failure count; success
clears it. Callers return a skip result during cooldown instead of sleeping.

All workers for a question share one instance. A per-URL async guard keeps
concurrent workers from starting the same failed fetch before its outcome
can be recorded. No successful response or background task is retained.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

RETRY_COOLDOWNS = (60.0, 120.0, 240.0)
MAX_ATTEMPTS = len(RETRY_COOLDOWNS) + 1


@dataclass(frozen=True)
class _Entry:
    status: int | None
    fail_count: int
    ban_until: float
    last_failed_at: float
    reason: str = "fetch_failed"
    backend: str = ""

    @property
    def exhausted(self) -> bool:
        return self.fail_count >= MAX_ATTEMPTS


@dataclass
class _GuardState:
    lock: asyncio.Lock
    users: int = 0


class _ScrapeCache:
    """Shared failure state; async guards run on the question's event loop."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._data: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._guards: dict[str, _GuardState] = {}
        self._clock = clock or time.time

    @asynccontextmanager
    async def guard(self, url: str) -> AsyncIterator[None]:
        """Serialize a URL's check/fetch/record transaction without leaking locks.

        Register waiters before awaiting acquisition. Both cancelled waiters
        and cancelled holders release their registration in the same finally
        block, so another waiter cannot accidentally acquire a replacement
        lock while the original lock is still in use.
        """
        with self._lock:
            state = self._guards.get(url)
            if state is None:
                state = _GuardState(lock=asyncio.Lock())
                self._guards[url] = state
            state.users += 1
        try:
            async with state.lock:
                yield
        finally:
            with self._lock:
                state.users -= 1
                if state.users == 0 and self._guards.get(url) is state:
                    del self._guards[url]

    def check(self, url: str, now: float | None = None) -> _Entry | None:
        """Return a cooldown/exhaustion entry without changing its history."""
        if not url:
            return None
        t = self._clock() if now is None else now
        with self._lock:
            entry = self._data.get(url)
            if entry is not None and (entry.exhausted or t < entry.ban_until):
                return entry
            return None

    def record_failure(
        self,
        url: str,
        status: int | None = None,
        now: float | None = None,
        *,
        reason: str = "fetch_failed",
        backend: str = "",
    ) -> _Entry | None:
        """Record one actual failed fetch, preserving its real failure category.

        Call only for a fetch that was admitted after checking under guard().
        Rejected cache checks are not attempts and must not call this method.
        """
        if not url:
            return None
        t = self._clock() if now is None else now
        with self._lock:
            previous = self._data.get(url)
            if previous is not None and previous.exhausted:
                return previous
            fail_count = 1 if previous is None else previous.fail_count + 1
            entry = _Entry(
                status=status,
                fail_count=fail_count,
                ban_until=t + RETRY_COOLDOWNS[fail_count - 1]
                if fail_count < MAX_ATTEMPTS
                else 0.0,
                last_failed_at=t,
                reason=reason,
                backend=backend,
            )
            self._data[url] = entry
            return entry

    def record_success(self, url: str) -> None:
        """Clear consecutive failures after a successful fetch."""
        with self._lock:
            self._data.pop(url, None)

    def clear(self) -> None:
        """Clear failure history without disturbing active URL guards."""
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return JSON-safe failure history, including expired cooldowns."""
        with self._lock:
            return {
                url: {**asdict(entry), "exhausted": entry.exhausted}
                for url, entry in self._data.items()
            }


def format_skip_message(url: str, entry: _Entry, now: float | None = None) -> str:
    """Explain a skipped fetch without mislabelling transport errors as HTTP 429."""
    failure = entry.reason
    if entry.status is not None:
        failure += f" (HTTP {entry.status})"
    if entry.backend:
        failure += f" via {entry.backend}"
    prefix = (
        f"URL skipped: {url}. Last fetch failed: {failure}. "
        f"Consecutive failed attempts: {entry.fail_count}/{MAX_ATTEMPTS}. "
    )
    if entry.exhausted:
        return (
            prefix
            + "The initial attempt and all 3 retries failed; do not fetch this URL "
            "again during this question. Use another source or search query."
        )
    t = time.time() if now is None else now
    remaining = max(0, math.ceil(entry.ban_until - t))
    return (
        prefix
        + f"Retry is allowed in {remaining} seconds. Continue with other sources "
        "or search queries in the meantime; do not wait by repeatedly fetching this URL."
    )
