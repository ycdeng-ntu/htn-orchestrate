"""HTN task-tree node model.

``HTNTaskNode`` is the shared representation for every node in an HTN task
tree: the root task, internal decomposed tasks, and executable leaf subtasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class HTNTaskStatus(str, Enum):
    """Runtime status for a node in an HTN task tree."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HTNCompletionOutcome(str, Enum):
    """Semantic completion state produced by node synthesis."""

    RESOLVED = "resolved"
    EXHAUSTED_UNRESOLVED = "exhausted_unresolved"


@dataclass(frozen=True)
class HTNNodeCompletion:
    """Structured semantic result returned by a node synthesizer."""

    result_summary: str
    completion_outcome: HTNCompletionOutcome
    failure_reason: str | None = None


class HTNTaskNode(BaseModel):
    """Unified node for root tasks, internal subtasks, and leaf subtasks.

    ``depends_on`` contains sibling node IDs that must complete before this
    node is ready. ``children`` contains child node IDs; an empty list means the
    node is currently a leaf and may be executed by the lower-level
    orchestrator once dependencies are satisfied.

    ``result_summary`` is set after completion. For a leaf, it summarizes the
    orchestrator result. For an internal node, it summarizes completed child
    results. ``ref`` is intended for leaf nodes only and points to the full
    result artifact, such as a file path.
    """

    id: str = Field(..., description="Unique node ID within one HTN tree.")
    parent_id: str | None = Field(
        default=None,
        description="Parent node ID. None only for the root node.",
    )
    title: str = Field(default="", description="Short human-readable label.")
    task: str = Field(default="", description="Task instruction or objective.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Sibling node IDs that must complete before this node.",
    )
    execution_kind: Literal["standard", "report_delivery"] = "standard"
    needs_decomposition: bool = Field(
        default=False,
        description="Whether this leaf should be routed back to the Planner when ready.",
    )
    status: HTNTaskStatus = HTNTaskStatus.PENDING
    children: list[str] = Field(
        default_factory=list,
        description="Child node IDs. Empty means the node is currently a leaf.",
    )
    result_summary: str | None = Field(
        default=None,
        description="Summary of the completed node result.",
    )
    completion_outcome: HTNCompletionOutcome | None = Field(
        default=None,
        description=(
            "Whether node synthesis resolved the task or exhausted the current approach. "
            "Partial evidence remains in result_summary rather than a third outcome."
        ),
    )
    failure_reason: str | None = Field(
        default=None,
        description="Execution or synthesis failure; not a finding about the task.",
    )
    ref: str | None = Field(
        default=None,
        description="Leaf-only reference to the full result artifact.",
    )

    @property
    def is_leaf(self) -> bool:
        """Return True when the node has no children."""
        return not self.children

    @property
    def is_internal(self) -> bool:
        """Return True when the node has at least one child."""
        return bool(self.children)
