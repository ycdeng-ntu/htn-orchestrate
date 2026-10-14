"""Fail-open diagnostics for Dynamic HTN Informed.

This module is deliberately observational: it never participates in a planning,
dispatch, retry, or result-selection decision.  Every write is best-effort so a
diagnostic filesystem failure cannot fail a benchmark question.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agent_harness.core.loop_types import BaseObserver

logger = logging.getLogger(__name__)

_scope: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "dynamic_htn_informed_research_diagnostic_scope",
    default={},
)


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]", "_", str(value or "unknown"))
    return text[:180] or "unknown"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def scope_snapshot() -> dict[str, Any]:
    """Return a copy of the diagnostic correlation fields in this async task."""
    return dict(_scope.get())


@contextmanager
def diagnostic_scope(**fields: Any) -> Iterator[None]:
    """Temporarily add correlation fields; ContextVars propagate to child tasks."""
    merged = scope_snapshot()
    merged.update({k: v for k, v in fields.items() if v is not None})
    token = _scope.set(merged)
    try:
        yield
    finally:
        _scope.reset(token)


class DiagnosticRecorder:
    """Append structured events and full diagnostic-only artifacts."""

    def __init__(self, root: Path, **base_fields: Any) -> None:
        self.root = Path(root)
        self._events = self.root / "events.jsonl"
        self._base = {k: v for k, v in base_fields.items() if v is not None}
        self._origin = time.monotonic()

    def emit(self, event: str, **fields: Any) -> None:
        try:
            now = time.monotonic()
            row = {
                "event": event,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": round(now, 6),
                "question_elapsed_ms": round((now - self._origin) * 1000, 3),
                **self._base,
                **scope_snapshot(),
                **fields,
            }
            self.root.mkdir(parents=True, exist_ok=True)
            with self._events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — diagnostics must be fail-open
            logger.debug("dynamic_htn_informed_research diagnostic event failed: %s", exc)

    def write_text(self, relative_path: str | Path, content: Any) -> str | None:
        try:
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                content if isinstance(content, str) else str(content),
                encoding="utf-8",
            )
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dynamic_htn_informed_research diagnostic artifact failed: %s", exc)
            return None

    def write_json(self, relative_path: str | Path, value: Any) -> str | None:
        try:
            return self.write_text(
                relative_path,
                json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("dynamic_htn_informed_research diagnostic JSON failed: %s", exc)
            return None

    def artifact_path(self, category: str, name: Any, suffix: str) -> Path:
        return Path(category) / f"{_safe_name(name)}{suffix}"


def message_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages or []:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(str(content))
    return total


def serialize_messages(messages: list[Any]) -> list[Any]:
    rows: list[Any] = []
    for message in messages or []:
        if isinstance(message, dict):
            rows.append(_jsonable(message))
            continue
        if hasattr(message, "model_dump"):
            try:
                rows.append(_jsonable(message.model_dump()))
                continue
            except Exception:  # noqa: BLE001
                pass
        rows.append(
            {
                "type": type(message).__name__,
                "content": _jsonable(getattr(message, "content", "")),
                "tool_calls": _jsonable(getattr(message, "tool_calls", None)),
                "tool_call_id": getattr(message, "tool_call_id", None),
                "name": getattr(message, "name", None),
                "additional_kwargs": _jsonable(
                    getattr(message, "additional_kwargs", None)
                ),
                "response_metadata": _jsonable(
                    getattr(message, "response_metadata", None)
                ),
            }
        )
    return rows


class AgentLoopDiagnosticObserver(BaseObserver):
    """Per-turn timing and full-transcript capture for one AgentBus job."""

    critical = True

    def __init__(
        self,
        recorder: DiagnosticRecorder,
        job_id: str,
        role_id: str,
        reasoning_effort: str | None = None,
    ) -> None:
        self._recorder = recorder
        self._job_id = job_id
        self._role_id = role_id
        self._reasoning_effort = reasoning_effort
        self._scope = scope_snapshot()
        self._started = time.monotonic()
        self._last = self._started

    def _emit(self, event: str, **fields: Any) -> None:
        now = time.monotonic()
        merged = {
            **self._scope,
            "job_id": self._job_id,
            "role_id": self._role_id,
            "reasoning_effort": self._reasoning_effort,
            "job_elapsed_ms": round((now - self._started) * 1000, 3),
            "since_job_event_ms": round((now - self._last) * 1000, 3),
            **fields,
        }
        self._recorder.emit(event, **merged)
        self._last = now

    async def on_loop_start(self, config: Any) -> None:
        self._started = time.monotonic()
        self._last = self._started
        self._emit(
            "agent_loop_start",
            max_turns=getattr(config, "max_turns", None),
            llm_timeout_s=getattr(config, "llm_timeout", None),
            tool_timeout_s=getattr(config, "tool_timeout", None),
            context_token_limit=getattr(config, "context_token_limit", None),
            compact_after_turns=getattr(config, "compact_after_turns", None),
        )

    async def on_llm_response(self, ctx: Any) -> None:
        usage = getattr(ctx, "usage", None) or {}
        tool_calls = getattr(ctx, "tool_calls", None) or []
        metadata = getattr(ctx, "metadata", None) or {}
        self._emit(
            "agent_llm_response",
            turn=getattr(ctx, "turn", None),
            message_count=len(getattr(ctx, "messages", None) or []),
            context_chars=message_chars(getattr(ctx, "messages", None) or []),
            visible_chars=len(getattr(ctx, "ai_text", "") or ""),
            thinking_chars=len(getattr(ctx, "thinking", "") or ""),
            leaked_reasoning_chars=len(getattr(ctx, "leaked_reasoning", "") or ""),
            tool_call_count=len(tool_calls),
            tool_names=[tc.get("name") for tc in tool_calls if isinstance(tc, dict)],
            input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
            output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
            tool_policy=metadata.get("tool_policy")
            if isinstance(metadata, dict)
            else None,
            tool_policy_violation=metadata.get("tool_policy_violation")
            if isinstance(metadata, dict)
            else None,
        )

    async def on_tool_call(self, ctx: Any, tool_call: dict) -> None:
        args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
        self._emit(
            "agent_tool_call",
            turn=getattr(ctx, "turn", None),
            tool_call_id=tool_call.get("id") if isinstance(tool_call, dict) else None,
            tool_name=tool_call.get("name") if isinstance(tool_call, dict) else None,
            args_chars=len(str(args)),
        )

    async def on_tool_result(self, ctx: Any, result: Any) -> None:
        raw = getattr(result, "result", "")
        self._emit(
            "agent_tool_result",
            turn=getattr(ctx, "turn", None),
            tool_call_id=getattr(result, "tool_call_id", None),
            tool_name=getattr(result, "name", None),
            duration_ms=getattr(result, "duration_ms", None),
            is_error=bool(getattr(result, "is_error", False)),
            result_chars=len(raw if isinstance(raw, str) else str(raw)),
        )

    async def on_turn_end(self, ctx: Any) -> None:
        self._emit("agent_turn_end", turn=getattr(ctx, "turn", None))

    async def on_loop_end(self, result: Any) -> None:
        messages = list(getattr(result, "messages", None) or [])
        transcript = self._recorder.write_json(
            self._recorder.artifact_path(
                "agent_traces", self._job_id, ".messages.json"
            ),
            serialize_messages(messages),
        )
        metadata = getattr(result, "metadata", None) or {}
        self._emit(
            "agent_loop_end",
            stopped_by=getattr(result, "stopped_by", None),
            turns_used=getattr(result, "turns_used", None),
            tool_calls_count=getattr(result, "tool_calls_count", None),
            message_count=len(messages),
            transcript_chars=message_chars(messages),
            final_chars=len(getattr(result, "final_content", "") or ""),
            evidence_count=len(metadata.get("evidence_cards", []) or [])
            if isinstance(metadata, dict)
            else None,
            assertion_count=len(metadata.get("assertion_cards", []) or [])
            if isinstance(metadata, dict)
            else None,
            source_count=len(metadata.get("source_urls", []) or [])
            if isinstance(metadata, dict)
            else None,
            tool_policy=metadata.get("tool_policy")
            if isinstance(metadata, dict)
            else None,
            tool_policy_violation=metadata.get("tool_policy_violation")
            if isinstance(metadata, dict)
            else None,
            transcript_path=transcript,
        )


__all__ = [
    "AgentLoopDiagnosticObserver",
    "DiagnosticRecorder",
    "diagnostic_scope",
    "message_chars",
    "scope_snapshot",
    "serialize_messages",
]
