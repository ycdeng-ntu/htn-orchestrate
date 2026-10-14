"""Result-informed HTN workflow with task-local strategy selection.

Registration exposes every worker role without activating profiles or changing
process configuration. The dynamic node selects the strategy for each question.
"""

from __future__ import annotations

from agent_harness.core.runtime.registries.workflows import WorkflowContext
from workflows.dynamic_htn_informed_research.task_strategies import get_all_task_strategies


def _registered_agent_definitions() -> list:
    definitions = {}
    for strategy in get_all_task_strategies():
        for definition in strategy.workers.agent_definitions:
            definitions[definition.role_id] = definition
    return list(definitions.values())


def register(ctx: WorkflowContext) -> None:
    from workflows.dynamic_htn_informed_research.spec import (
        DYNAMIC_HTN_INFORMED_RESEARCH_SPEC,
    )

    ctx.register_agents(_registered_agent_definitions())
    ctx.register_pipeline(DYNAMIC_HTN_INFORMED_RESEARCH_SPEC)
