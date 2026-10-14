"""One worker-local correction for an entirely unusable native tool response."""
from __future__ import annotations

from agent_harness.core.loop_types import BaseObserver, Intervention
from agent_harness.core.runtime.loop.agent_loop import PER_TURN_TOOL_ALLOWLIST_KEY

from ..terminal_turn import InformedTerminalTurnObserver
from .llm import ResearchToolCallParser


class WorkerToolCallRecovery(ResearchToolCallParser, BaseObserver):
    """Defer parse failure to an observer so failed responses retain usage/history.

    A protocol retry repeats the same logical turn and its tool allowlist.
    Delegate terminal-turn arming only after parsing succeeds, so a malformed
    penultimate response cannot prematurely restrict its own correction.
    The HTN control client continues to use its strict parser separately.
    """

    critical = True

    def __init__(self):
        super().__init__()
        self._parse_error = None
        self._used = False
        self._pending = False
        self._terminal = InformedTerminalTurnObserver()

    def parse(self, response, tool_names):
        self._parse_error = None
        try:
            return super().parse(response, tool_names)
        except ValueError as exc:
            self._parse_error = str(exc)
            return []

    def parse_text(self, text, tool_names):
        # Do not replace a rejected authored report/code call with a different
        # tool call recovered from thinking or ordinary response text.
        if self._parse_error is not None:
            return []
        return super().parse_text(text, tool_names)

    async def on_llm_response(self, ctx):
        if self._pending:
            self._pending = False
            state = ctx.metadata["tool_call_correction"]
            state["response_received"] = True
            state["outcome"] = "corrected" if ctx.tool_calls and self._parse_error is None else "failed"
            if state["outcome"] == "failed":
                state["error"] = self._parse_error or "correction contained no usable tool call"
                return Intervention(stop_reason="malformed_tool_calls")

        if self._parse_error is None:
            return await self._terminal.on_llm_response(ctx)
        state = ctx.metadata.setdefault("tool_call_correction", {})
        state.update(error=self._parse_error, turn=ctx.turn)
        if self._used:
            state["outcome"] = "failed"
            return Intervention(stop_reason="malformed_tool_calls")
        # Budget and policy observers still own their stop reasons. A rejected
        # response must not consume a correction opportunity after either stop.
        limit = ctx.metadata.get("budget_tokens_limit", 0)
        if limit > 0 and ctx.metadata.get("budget_tokens_used", 0) >= limit:
            state["outcome"] = "skipped_budget"
            return None
        violation = ctx.metadata.get("tool_policy_violation") or {}
        if violation.get("turn") == ctx.turn:
            state["outcome"] = "skipped_tool_policy"
            return None
        self._used = self._pending = True
        state.update(attempts=1, response_received=False, outcome="requested")
        policy = ctx.metadata.get("tool_policy") or {}
        if policy.get("turn") == ctx.turn:
            ctx.metadata[PER_TURN_TOOL_ALLOWLIST_KEY] = list(policy["allowed_tools"])
        return Intervention(
            continue_to_next_turn=True,
            inject_messages=[
                "Tool-call protocol correction (one opportunity): none of your native tool calls "
                "could be executed. Reissue the intended call using a complete JSON object with "
                "valid argument types. The original calls and their unmodified arguments remain "
                "in the preceding history; no rejected call was executed. Preserve the intended "
                "code or full report content yourself; do not substitute ordinary chat text for "
                "a tool call. Follow the tools currently offered. "
                f"Parser error: {self._parse_error}"
            ],
        )
