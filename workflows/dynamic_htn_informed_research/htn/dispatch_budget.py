"""Question-wide cumulative worker admission, shared across leaves and epochs."""

from __future__ import annotations


class QuestionDispatchBudget:
    """Count admitted workers without refunding failures or cancellations.

    One instance belongs to one question and is reused by every leaf and rebuild.
    ``try_acquire`` is synchronous: callers on the question's event loop must
    create the worker immediately after admission, without an intervening await.
    This is an event-loop primitive, not a thread/process synchronization lock.
    A limit of zero explicitly disables the cap while retaining usage accounting.
    """

    __slots__ = ("_limit", "_used")

    def __init__(self, limit: int = 0) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("Question worker dispatch limit must be an integer.")
        if limit < 0:
            raise ValueError("Question worker dispatch limit must be non-negative.")
        self._limit = limit
        self._used = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int | None:
        return max(0, self._limit - self._used) if self._limit else None

    @property
    def exhausted(self) -> bool:
        return self._limit > 0 and self._used >= self._limit

    def try_acquire(self) -> bool:
        """Reserve one actual dispatch, atomically within a single event loop."""
        if self.exhausted:
            return False
        self._used += 1
        return True

    def snapshot(self) -> dict[str, int | bool | None]:
        return {
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }
