"""HTN Planner scaffolding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree

logger = logging.getLogger(__name__)


def _log_text(value: str, limit: int = 1000) -> str:
    """Keep model-authored rationales useful without producing unbounded logs."""
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[:limit] + "...[truncated]"


@dataclass(frozen=True)
class DecompositionBudget:
    """Structural budget exposed to decomposition strategies."""

    current_depth: int
    max_depth: int
    remaining_depth: int
    child_depth: int
    child_remaining_depth: int
    max_children: int

    @classmethod
    def for_node(
        cls,
        *,
        current_depth: int,
        max_depth: int,
        max_children: int,
    ) -> "DecompositionBudget":
        """Build a budget for decomposing a node at ``current_depth``."""
        child_depth = current_depth + 1
        return cls(
            current_depth=current_depth,
            max_depth=max_depth,
            remaining_depth=max(max_depth - current_depth, 0),
            child_depth=child_depth,
            child_remaining_depth=max(max_depth - child_depth, 0),
            max_children=max_children,
        )


@dataclass(frozen=True)
class DecompositionProfile:
    """Plug-and-play decomposition rules for a task family."""

    name: str = "default"
    prompt_template: str = ""
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecompositionDecision:
    """One explicit result from a decomposition strategy.

    ``execute_direct`` is a deliberate planning decision after inspecting the
    available evidence. It is distinct from a malformed/failed strategy call,
    even though both ultimately make the current node executable.
    """

    action: Literal["decompose", "execute_direct"]
    children: tuple[HTNTaskNode, ...] = ()
    rationale: str = ""


class DecompositionStrategy(Protocol):
    """Strategy that decomposes one ready HTN leaf."""

    profile: DecompositionProfile

    async def decompose(
        self,
        tree: HTNTree,
        node: HTNTaskNode,
        budget: DecompositionBudget,
    ) -> DecompositionDecision:
        """Return an explicit decompose-or-execute decision for ``node``."""
        ...


@dataclass
class HTNPlanner:
    """Incremental HTN decomposer with depth/width budget enforcement."""

    strategy: DecompositionStrategy
    max_depth: int = 3
    max_children: int = 6

    async def decompose(self, tree: HTNTree, node_id: str) -> list[HTNTaskNode]:
        """Decompose a ready leaf and attach the produced children to the tree.

        Returns the newly attached child nodes, or an empty list when the node
        has been converted to an executable leaf (max depth reached or strategy
        returned no children). The caller should re-query ``ready_for_execution``
        to pick up the converted node.
        """
        node = tree.get_node(node_id)
        if not tree.is_ready(node_id):
            raise ValueError(f"HTN node '{node_id}' is not ready for planning")
        if not node.needs_decomposition:
            raise ValueError(f"HTN node '{node_id}' does not need decomposition")

        budget = self.budget_for(tree, node_id)
        if budget.current_depth >= budget.max_depth:
            logger.info(
                "dynamic_htn_informed_research planning_decision node=%s action=execute_direct "
                "source=depth_limit depth=%d max_depth=%d",
                node_id,
                budget.current_depth,
                budget.max_depth,
            )
            tree.mark_executable(node_id)
            return []

        try:
            decision = await self.strategy.decompose(tree, node, budget)
        except Exception as exc:
            logger.warning(
                "dynamic_htn_informed_research planning_decision node=%s action=execute_direct "
                "source=strategy_error error_type=%s",
                node_id,
                type(exc).__name__,
            )
            tree.mark_executable(node_id)
            return []

        if decision.action == "execute_direct":
            logger.info(
                "dynamic_htn_informed_research planning_decision node=%s action=execute_direct "
                "source=strategy rationale=%s",
                node_id,
                _log_text(decision.rationale) or "(none)",
            )
            tree.mark_executable(node_id)
            return []

        children = list(decision.children)
        if not children:
            logger.warning(
                "dynamic_htn_informed_research planning_decision node=%s action=execute_direct "
                "source=empty_decomposition",
                node_id,
            )
            tree.mark_executable(node_id)
            return []

        children = self._apply_budget(children, budget)
        attached = tree.decompose(node_id, children)
        logger.info(
            "dynamic_htn_informed_research planning_decision node=%s action=decompose "
            "child_count=%d children=%s rationale=%s",
            node_id,
            len(attached),
            [
                {
                    "id": child.id,
                    "depends_on": child.depends_on,
                    "needs_decomposition": child.needs_decomposition,
                }
                for child in attached
            ],
            _log_text(decision.rationale) or "(none)",
        )
        return attached

    def ready_for_planning(self, tree: HTNTree) -> list[HTNTaskNode]:
        """Return ready leaves that should be routed to the Planner."""
        return tree.ready_for_planning()

    def ready_for_execution(self, tree: HTNTree) -> list[HTNTaskNode]:
        """Return ready leaves executable by the lower-level Orchestrator."""
        return tree.ready_for_execution()

    def budget_for(self, tree: HTNTree, node_id: str) -> DecompositionBudget:
        """Return the structural decomposition budget for a node."""
        return DecompositionBudget.for_node(
            current_depth=tree.depth_of(node_id),
            max_depth=self.max_depth,
            max_children=self.max_children,
        )

    def _apply_budget(
        self,
        children: list[HTNTaskNode],
        budget: DecompositionBudget,
    ) -> list[HTNTaskNode]:
        if len(children) > budget.max_children:
            logger.warning(
                "HTNPlanner: strategy returned %d children, capping at %d.",
                len(children),
                budget.max_children,
            )
            children = children[: budget.max_children]
        if budget.child_remaining_depth == 0:
            return [
                child.model_copy(update={"needs_decomposition": False}, deep=True)
                for child in children
            ]
        return children
