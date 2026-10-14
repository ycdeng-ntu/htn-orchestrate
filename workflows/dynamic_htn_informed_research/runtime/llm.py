"""Native transport and the strict, function-only HTN response envelope."""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import uuid
from copy import deepcopy
from typing import Any

from agent_harness.core.llm import LLMResponse as NativeResponse
from agent_harness.core.llm_errors import quota_error
from agent_harness.core.runtime.loop.tool_call_parser import MultiFormatToolCallParser

from ..diagnostics import scope_snapshot
from ..htn.llm_types import LLMResponse
from ..markup import strip_leaked_tool_markup


# Report validators need the untouched native arguments, including malformed JSON.
# Context-local opt-in leaves worker and other strategy parsing unchanged.
_RAW_TOOL_ENVELOPE = contextvars.ContextVar("dynir_raw_tool_envelope", default=False)


@contextlib.contextmanager
def raw_tool_envelope():
    token = _RAW_TOOL_ENVELOPE.set(True)
    try:
        yield
    finally:
        _RAW_TOOL_ENVELOPE.reset(token)


class MeteredLLM:
    def __init__(
        self, client, runtime, provider: str, model: str, *, purpose: str = "main",
        reasoning_effort: str | None = None,
    ):
        self.client, self.runtime = client, runtime
        self.provider, self.model = provider, model
        self.purpose = purpose
        self.reasoning_effort = reasoning_effort

    async def chat(self, messages, **kwargs):
        self.runtime.raise_if_quota_failed()
        try:
            response = await self.client.chat(messages, **kwargs)
        except BaseException as exc:
            terminal = quota_error(exc)
            self.runtime.record_usage(
                self.provider, self.model, None, status=type(terminal or exc).__name__,
                purpose=self.purpose, reasoning_effort=self.reasoning_effort,
            )
            if terminal is not None:
                self.runtime.fail_quota(terminal, self.provider, self.model, self.purpose)
                raise terminal from None
            raise
        self.runtime.record_usage(
            self.provider, self.model, response, purpose=self.purpose,
            reasoning_effort=self.reasoning_effort,
        )
        return response

    async def stream(self, messages, **kwargs):
        self.runtime.raise_if_quota_failed()
        usage, model = {}, self.model
        status = "completed"
        source = self.client.stream(messages, **kwargs)
        try:
            async for delta in source:
                if getattr(delta, "usage", None):
                    usage = delta.usage
                model = getattr(delta, "model", "") or model
                yield delta
        except BaseException as exc:
            terminal = quota_error(exc)
            status = type(terminal or exc).__name__
            if terminal is not None:
                self.runtime.fail_quota(terminal, self.provider, self.model, self.purpose)
                raise terminal from None
            raise
        finally:
            self.runtime.record_usage(
                self.provider, model, NativeResponse(usage=usage), status=status,
                purpose=self.purpose, reasoning_effort=self.reasoning_effort,
            )
            close = getattr(source, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()


def parse_partial_json(text: str) -> Any:
    """Close incomplete JSON without discarding any argument text.

    Unfinished values or escapes that cannot be closed losslessly are rejected.
    In particular, never repeatedly trim a broken tail until an earlier subset
    of a report happens to become valid JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stack, in_string, escaped = [], False, False
    out = []
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char == "\n":
                char = "\\n"
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
        out.append(char)
    if in_string:
        if escaped:
            return None
        out.append('"')
    try:
        return json.loads("".join(out) + "".join(reversed(stack)))
    except json.JSONDecodeError:
        return None


class ResearchToolCallParser(MultiFormatToolCallParser):
    """Repair raw native arguments before the generic parser can discard them."""

    def __init__(self, *, allow_partial_calls: bool = True):
        self.last_raw_response: NativeResponse | None = None
        self.allow_partial_calls = allow_partial_calls

    def parse_text(self, text: str, tool_names: set[str] | None) -> list[dict]:
        return self._parse_json_tool_calls(text, tool_names) or super().parse_text(text, tool_names)

    @staticmethod
    def _repair_call(call: dict) -> dict:
        from ..adapters import _parse_flat_jsonish_object

        function = call.get("function", {})
        name = function.get("name", "")
        raw = function.get("arguments", "")
        if not isinstance(raw, str):
            if not isinstance(raw, dict):
                raise ValueError(f"Arguments for tool {name!r} must be an object")
            return call
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            pass
        else:
            if not isinstance(value, dict):
                raise ValueError(f"Arguments for tool {name!r} must be an object")
            return call
        args = None
        # Code and reports must not acquire invented syntax or lost tails.
        if name in {"web_search", "web_fetch", "scholar_search", "read_file"}:
            args = _parse_flat_jsonish_object(raw)
        if args is None and name not in {"submit_report", "run_python_code"}:
            args = parse_partial_json(raw)
        if not isinstance(args, dict) or not args:
            raise ValueError(f"Malformed arguments for tool {name!r}")
        return {**call, "function": {**function, "arguments": json.dumps(args)}}

    def parse(self, response: NativeResponse, tool_names: set[str] | None) -> list[dict]:
        self.last_raw_response = deepcopy(response)
        repaired = []
        errors = []
        for call in response.tool_calls or []:
            try:
                repaired.append(self._repair_call(call))
            except ValueError as exc:
                errors.append(exc)
        # Worker calls are independent. Keep usable siblings without repairing
        # authored code/reports. The loop retains raw calls in history and adds
        # a non-execution reply for each omitted call. Control envelopes remain
        # atomic, and an entirely unusable worker response still fails.
        if errors and (not self.allow_partial_calls or not any(
            tool_names is None or call["function"].get("name") in tool_names
            for call in repaired
        )):
            raise errors[0]
        response.tool_calls = repaired
        return super().parse(response, tool_names)


class NativeHTNClient:
    """Keep HTN control retries separate from the worker's native loop retries."""

    def __init__(self, client, *, diagnostics=None, reasoning_effort=None, attempts: int = 4):
        if attempts < 1:
            raise ValueError("Control LLM attempts must be positive")
        self.native = client
        self.model = client.model
        self.reasoning_effort = reasoning_effort
        self.diagnostics = diagnostics
        self.attempts = attempts

    async def chat(self, messages, tools=None):
        parser = ResearchToolCallParser(allow_partial_calls=False)
        diag = self.diagnostics
        call_id = uuid.uuid4().hex[:16]
        if diag is not None:
            diag.write_json(diag.artifact_path("llm_calls", call_id, ".request.json"), {
                "scope": scope_snapshot(), "messages": messages, "tools": tools,
                "reasoning_effort": self.reasoning_effort,
            })
        response = None
        last_error = None
        for attempt in range(self.attempts):
            try:
                response = await self.native.chat(messages, tools=tools)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                terminal = quota_error(exc)
                if terminal is not None:
                    raise terminal from None
                if diag is not None:
                    diag.emit("llm_attempt_error", llm_call_id=call_id, attempt=attempt + 1, error_type=type(exc).__name__)
                if attempt + 1 < self.attempts:
                    await asyncio.sleep(1.5 * (attempt + 1))
        if response is None:
            if _RAW_TOOL_ENVELOPE.get() and last_error is not None:
                raise last_error
            return LLMResponse(content="", tool_calls=[])
        if diag is not None:
            diag.write_json(diag.artifact_path("llm_calls", call_id, ".native.json"), vars(response))
        if _RAW_TOOL_ENVELOPE.get():
            # Strip only native envelope metadata; never repair arguments, drop
            # companion calls, or recover a tool call from reasoning text.
            result = LLMResponse(
                content=response.content,
                finish_reason=response.finish_reason,
                tool_calls=[{"function": deepcopy(call.get("function"))}
                            for call in response.tool_calls or []],
            )
            if diag is not None:
                diag.write_json(diag.artifact_path("llm_calls", call_id, ".response.json"), vars(result))
            return result
        names = {tool["function"]["name"] for tool in tools or []}
        calls = []
        if tools:
            try:
                # Native unknown companion calls must survive to the strict HTN
                # parser, which rejects malformed envelopes as a whole.
                parsed = parser.parse(response, None if response.tool_calls else names)
            except ValueError:
                parsed = []
            if not parsed:
                parsed = parser.parse_text(getattr(response, "reasoning_content", "") or "", names)
            calls = [{"function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}} for c in parsed]
        content = response.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        result = LLMResponse(content=strip_leaked_tool_markup(content or ""), tool_calls=calls,
                             finish_reason=response.finish_reason)
        if diag is not None:
            diag.write_json(diag.artifact_path("llm_calls", call_id, ".response.json"), vars(result))
        return result


def build_worker_llm(
    provider: str, model: str, sampling: dict[str, Any], *, diagnostics=None,
    purpose: str = "main",
):
    from .context import current_runtime

    effort = (sampling.get("extra_body") or {}).get("reasoning_effort")
    return NativeHTNClient(
        current_runtime().native_llm(provider, model, sampling, purpose=purpose),
        diagnostics=diagnostics, reasoning_effort=effort,
    )
