"""Final-turn submission policy for Dynamic HTN Informed workers."""

from __future__ import annotations

from agent_harness.core.loop_types import BaseObserver, Intervention, TurnContext
from agent_harness.core.runtime.loop.agent_loop import PER_TURN_TOOL_ALLOWLIST_KEY


class InformedTerminalTurnObserver(BaseObserver):
    """Expose only ``submit_report`` on the worker's final model turn.

    The observer is workflow-owned; the shared loop merely provides the
    opt-in, one-shot filtering mechanism. Earlier turns retain the full
    research tool set.
    """

    critical = True

    def __init__(self, terminal_tool: str = "submit_report") -> None:
        self._terminal_tool = terminal_tool
        self._fired = False

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if self._fired or ctx.turn != ctx.max_turns - 1:
            return None
        final_answer = ctx.metadata.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            return None

        self._fired = True
        ctx.metadata[PER_TURN_TOOL_ALLOWLIST_KEY] = [self._terminal_tool]
        return Intervention(
            inject_messages=[
                "The next response is your final turn. Only "
                f"`{self._terminal_tool}` is available. Do not call search, "
                "fetch, code, or file tools. Submit the best evidence-based "
                "report you have now, even if some clues remain unresolved; "
                "state those gaps explicitly. Call "
                f"`{self._terminal_tool}` now with a numeric confidence "
                "between 0 and 1 that reflects the strength and completeness "
                "of your evidence."
            ]
        )


__all__ = ["InformedTerminalTurnObserver"]
