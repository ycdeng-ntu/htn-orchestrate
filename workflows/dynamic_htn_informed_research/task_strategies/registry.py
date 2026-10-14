"""Explicit task-strategy selection with a fail-closed registry."""

from __future__ import annotations

from contextvars import ContextVar

from .contracts import TaskStrategy


_ACTIVE_TASK_STRATEGY: ContextVar[TaskStrategy | None] = ContextVar(
    "dynamic_htn_informed_research_active_task_strategy",
    default=None,
)

AUTO_TASK_STRATEGY_NAME = "auto"


def normalize_task_strategy_name(name: str | None) -> str:
    return (name or "research").strip().lower().replace("-", "_")


def is_auto_task_strategy(name: str | None) -> bool:
    return normalize_task_strategy_name(name) == AUTO_TASK_STRATEGY_NAME


def get_task_strategy(name: str | None) -> TaskStrategy:
    normalized = normalize_task_strategy_name(name)
    if normalized in {"research", "browsecomp_research"}:
        from .research import RESEARCH_STRATEGY

        return RESEARCH_STRATEGY
    if normalized in {
        "report_research",
        "report",
        "long_report",
        "deep_research_report",
    }:
        from .report_research import REPORT_RESEARCH_STRATEGY

        return REPORT_RESEARCH_STRATEGY
    if normalized in {"science_olympiad", "fso", "frontier_science_olympiad"}:
        from .science_olympiad import SCIENCE_OLYMPIAD_STRATEGY

        return SCIENCE_OLYMPIAD_STRATEGY
    if normalized in {"science_mcq", "mcq"}:
        from .science_mcq import SCIENCE_MCQ_STRATEGY

        return SCIENCE_MCQ_STRATEGY
    raise ValueError(
        f"unknown dynamic_htn_informed_research task strategy {name!r}; "
        "available strategies: research, report_research, science_olympiad, science_mcq"
    )


def get_all_task_strategies() -> tuple[TaskStrategy, ...]:
    """Return every concrete strategy whose worker roles the runtime must register."""
    return (
        get_task_strategy("research"),
        get_task_strategy("report_research"),
        get_task_strategy("science_olympiad"),
        get_task_strategy("science_mcq"),
    )


def set_active_task_strategy(strategy: TaskStrategy) -> None:
    """Select the task-local strategy for helper APIs with legacy signatures."""
    _ACTIVE_TASK_STRATEGY.set(strategy)


def get_active_task_strategy() -> TaskStrategy:
    """Return the configured strategy, defaulting to legacy research behavior."""
    return _ACTIVE_TASK_STRATEGY.get() or get_task_strategy("research")
