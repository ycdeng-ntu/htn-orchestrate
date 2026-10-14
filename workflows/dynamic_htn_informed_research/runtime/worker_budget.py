"""Record worker token usage; enforce a cap only when explicitly configured."""
from __future__ import annotations

from agent_harness.components.observers.budget_observer import BudgetObserver
from agent_harness.core.loop_types import Intervention, TurnContext


DEFAULT_WORKER_TOKEN_BUDGET = 0  # Unlimited; continue recording token usage.


class InformedWorkerBudgetObserver(BudgetObserver):
    """Keep research tools available after the warning; never add a grace turn.

    Usage is reported after a response, so that response can exceed the limit.
    Finish its tools, then stop before requesting another model response.
    """

    def __init__(self, max_tokens: int = DEFAULT_WORKER_TOKEN_BUDGET) -> None:
        super().__init__(max_tokens=max_tokens)

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        await super().on_llm_response(ctx)
        # The core's no-tool path bypasses on_turn_end. It must not nudge the
        # model into another response once the cumulative budget is exhausted.
        if not ctx.tool_calls and self.max_tokens > 0 and self.tokens_used >= self.max_tokens:
            return await super().on_turn_end(ctx)
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        # A successfully accepted report keeps its normal submission outcome,
        # including when its response crosses the token limit.
        if "final_answer" in ctx.metadata:
            return None
        return await super().on_turn_end(ctx)
