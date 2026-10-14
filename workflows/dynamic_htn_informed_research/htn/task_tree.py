"""HTN task-tree runtime container."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNNodeCompletion,
    HTNTaskNode,
    HTNTaskStatus,
)

logger = logging.getLogger(__name__)

HTNSummarizer: TypeAlias = Callable[
    [HTNTaskNode, HTNTaskNode, list[HTNTaskNode]],
    str | HTNNodeCompletion | Awaitable[str | HTNNodeCompletion],
]


@dataclass
class HTNTree:
    """Runtime container for one HTN task tree.

    The tree owns structural operations over ``HTNTaskNode`` objects:
    validation, readiness checks, leaf completion, and upward AND-completion
    propagation. A summarizer may be provided to synthesize internal-node
    summaries when all children complete. An absent summarizer uses deterministic
    aggregation. A failed summarizer preserves child evidence on the failed node.
    """

    root_id: str
    nodes: dict[str, HTNTaskNode]
    summarizer: HTNSummarizer | None = None
    epoch_id: int = 0
    restart_context: str = ""
    restart_answer_context: str = ""
    _summarizing_nodes: set[str] = field(default_factory=set, init=False, repr=False)

    def get_node(self, node_id: str) -> HTNTaskNode:
        """Return a node by ID."""
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown HTN node '{node_id}'") from exc

    def root(self) -> HTNTaskNode:
        """Return the root node."""
        return self.get_node(self.root_id)

    def children_of(self, node_id: str) -> list[HTNTaskNode]:
        """Return child nodes in the order stored by the parent."""
        node = self.get_node(node_id)
        return [self.get_node(child_id) for child_id in node.children]

    def siblings_of(self, node_id: str) -> list[HTNTaskNode]:
        """Return sibling nodes in parent-child order, including the node."""
        node = self.get_node(node_id)
        if node.parent_id is None:
            return [node]
        return self.children_of(node.parent_id)

    def is_ready(self, node_id: str) -> bool:
        """Return True when a PENDING leaf's dependencies are all satisfied.

        Readiness = "on the frontier": a pending leaf whose ``depends_on`` are
        all completed. The planning-vs-execution split is made downstream by
        ``ready_for_planning`` / ``ready_for_execution`` on ``needs_decomposition``
        — it is NOT excluded here (the upstream scaffolding excluded
        ``needs_decomposition`` nodes, which made ``ready_for_planning`` — and
        therefore ``HTNPlanner.decompose`` — unreachable, so the tree could never
        grow past the root).
        """
        node = self.get_node(node_id)
        if not node.is_leaf or node.status is not HTNTaskStatus.PENDING:
            return False
        return all(
            self.get_node(dep_id).status is HTNTaskStatus.COMPLETED
            for dep_id in node.depends_on
        )

    def ready_leaves(self) -> list[HTNTaskNode]:
        """Return structurally ready leaf nodes.

        Ready leaves form the current HTN frontier. Leaves with
        ``needs_decomposition=True`` should be routed to the Planner; the rest
        may be routed to the lower-level Orchestrator.
        """
        return [node for node in self.nodes.values() if self.is_ready(node.id)]

    def ready_for_planning(self) -> list[HTNTaskNode]:
        """Return ready leaves that should be decomposed further."""
        return [node for node in self.ready_leaves() if node.needs_decomposition]

    def ready_for_execution(self) -> list[HTNTaskNode]:
        """Return ready leaves that may be executed by the Orchestrator."""
        return [node for node in self.ready_leaves() if not node.needs_decomposition]

    def depth_of(self, node_id: str) -> int:
        """Return a node's depth, with the root at depth 0."""
        depth = 0
        seen: set[str] = set()
        node = self.get_node(node_id)
        while node.parent_id is not None:
            if node.id in seen:
                raise ValueError(f"Parent cycle detected while computing depth for '{node_id}'")
            seen.add(node.id)
            node = self.get_node(node.parent_id)
            depth += 1
        return depth

    def mark_running(self, node_id: str) -> None:
        """Mark a ready leaf node as running."""
        node = self.get_node(node_id)
        if not self.is_ready(node_id):
            raise ValueError(f"HTN node '{node_id}' is not ready to run")
        if node.needs_decomposition:
            raise ValueError(f"HTN node '{node_id}' still needs decomposition")
        node.status = HTNTaskStatus.RUNNING

    async def mark_completed(
        self,
        node_id: str,
        result_summary: str,
        ref: str | None = None,
        completion_outcome: HTNCompletionOutcome = HTNCompletionOutcome.RESOLVED,
        *,
        interrupt_on_semantic_exhaustion: bool = True,
    ) -> HTNTaskNode | None:
        """Mark a leaf complete and return an exhausted or failed ancestor, if any.

        ``interrupt_on_semantic_exhaustion`` controls propagation only.  Epoch 0
        stops at the first exhausted root/internal node so the caller can
        rebuild.  A final epoch keeps the same semantic outcome but propagates
        through every ready ancestor before returning the first exhausted node.
        Execution failures always stop propagation and return the failed node;
        they do not change the completion state of its successful children.
        """
        node = self.get_node(node_id)
        if not node.is_leaf:
            raise ValueError(f"HTN node '{node_id}' is internal and cannot be completed directly")
        if node.status is not HTNTaskStatus.RUNNING:
            raise ValueError(f"HTN node '{node_id}' is not running and cannot be completed")

        node.result_summary = result_summary
        node.ref = ref
        node.completion_outcome = completion_outcome
        node.status = HTNTaskStatus.COMPLETED
        if (
            completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            and node.id == self.root_id
        ):
            return node
        return await self._propagate_completion(
            node,
            interrupt_on_semantic_exhaustion=interrupt_on_semantic_exhaustion,
        )

    def mark_failed(self, node_id: str) -> None:
        """Mark a node as failed without propagating failure.

        Failure recovery/backtracking is owned by the caller. This method only
        records the local node failure and rejects completed nodes.
        """
        node = self.get_node(node_id)
        if node.status is HTNTaskStatus.COMPLETED:
            raise ValueError(f"HTN node '{node_id}' is completed and cannot be failed")
        node.status = HTNTaskStatus.FAILED

    def mark_executable(self, node_id: str) -> None:
        """Mark a pending leaf as executable instead of planner-routed."""
        node = self.get_node(node_id)
        if not node.is_leaf:
            raise ValueError(f"HTN node '{node_id}' is internal and cannot be executable")
        if node.status is not HTNTaskStatus.PENDING:
            raise ValueError(f"HTN node '{node_id}' is not pending and cannot be executable")
        if not node.needs_decomposition:
            raise ValueError(f"HTN node '{node_id}' does not need decomposition")
        node.needs_decomposition = False

    def decompose(self, node_id: str, children: list[HTNTaskNode]) -> list[HTNTaskNode]:
        """Replace a pending leaf node with child subtasks.

        The tree stores copies of the supplied child nodes so failed validation
        or later caller-side mutation cannot corrupt accepted tree state.
        """
        node = self.get_node(node_id)
        if not node.is_leaf:
            raise ValueError(f"HTN node '{node_id}' is already decomposed")
        if node.status is not HTNTaskStatus.PENDING:
            raise ValueError(f"HTN node '{node_id}' is not pending and cannot be decomposed")
        if not node.needs_decomposition:
            raise ValueError(f"HTN node '{node_id}' does not need decomposition")
        if not children:
            raise ValueError("Decomposition must produce at least one child")

        child_ids = [child.id for child in children]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError(f"Decomposition for node '{node_id}' has duplicate child IDs")
        existing = [child_id for child_id in child_ids if child_id in self.nodes]
        if existing:
            raise ValueError(f"Decomposition child IDs already exist: {', '.join(existing)}")

        child_copies = [
            child.model_copy(update={"parent_id": node_id}, deep=True)
            for child in children
        ]
        candidate_parent = node.model_copy(
            update={"children": child_ids, "needs_decomposition": False},
            deep=True,
        )
        candidate_nodes = dict(self.nodes)
        candidate_nodes[node_id] = candidate_parent
        for child in child_copies:
            candidate_nodes[child.id] = child

        candidate = HTNTree(
            root_id=self.root_id,
            nodes=candidate_nodes,
            summarizer=self.summarizer,
        )
        errors = candidate.validate()
        if errors:
            raise ValueError(f"Invalid HTN decomposition: {'; '.join(errors)}")

        for child in child_copies:
            self.nodes[child.id] = child
        node.children = child_ids
        node.needs_decomposition = False
        return child_copies

    def append_report_checks(self, children: list[HTNTaskNode]) -> list[HTNTaskNode]:
        """Append executable report checks to a quiescent root, preserving old branches.

        These are supplementary investigations, not replacements for existing
        nodes. Only the root aggregate is invalidated. The report finalizer owns
        its subsequent synthesis; ordinary completion propagation must not run
        while the report checks execute. Validation precedes every mutation.
        """
        if self._summarizing_nodes or any(
            node.status is HTNTaskStatus.RUNNING for node in self.nodes.values()
        ):
            raise ValueError("Report checks require a quiescent tree")
        ids = [child.id for child in children]
        if not ids or len(ids) != len(set(ids)) or any(i in self.nodes for i in ids):
            raise ValueError("Report checks require fresh, unique node IDs")
        if any(
            not child.is_leaf or child.needs_decomposition
            or child.status is not HTNTaskStatus.PENDING or child.depends_on
            for child in children
        ):
            raise ValueError("Report checks must be independent pending executable leaves")
        copies = [child.model_copy(update={"parent_id": self.root_id}, deep=True)
                  for child in children]
        root = self.root()
        updated = root.model_copy(update={
            "children": [*root.children, *ids], "needs_decomposition": False,
            "status": HTNTaskStatus.PENDING, "completion_outcome": None,
            "result_summary": None, "failure_reason": None, "ref": None,
        }, deep=True)
        candidate_nodes = {**self.nodes, self.root_id: updated,
                           **{child.id: child for child in copies}}
        errors = HTNTree(self.root_id, candidate_nodes).validate()
        if errors:
            raise ValueError(f"Invalid report checks: {'; '.join(errors)}")
        # Preserve root object identity for the question driver's live reference.
        for name in ("children", "needs_decomposition", "status", "completion_outcome",
                     "result_summary", "failure_reason", "ref"):
            setattr(root, name, getattr(updated, name))
        self.nodes.update({child.id: child for child in copies})
        return copies

    def validate(self) -> list[str]:
        """Return structural validation errors. Empty means the tree is valid."""
        errors: list[str] = []

        if self.root_id not in self.nodes:
            errors.append(f"Root node '{self.root_id}' is missing")
            return errors

        root = self.nodes[self.root_id]
        if root.parent_id is not None:
            errors.append("Root node parent_id must be None")

        for node in self.nodes.values():
            errors.extend(self._validate_node_links(node))
            if node.is_internal and node.needs_decomposition:
                errors.append(f"Internal node '{node.id}' cannot need decomposition")
            if node.needs_decomposition and node.status is not HTNTaskStatus.PENDING:
                errors.append(
                    f"Node '{node.id}' has needs_decomposition=True but status is {node.status.value}"
                )

        errors.extend(self._validate_reachability())
        errors.extend(self._validate_child_cycles())
        errors.extend(self._validate_dependency_cycles())

        return errors

    async def _propagate_completion(
        self,
        node: HTNTaskNode,
        *,
        interrupt_on_semantic_exhaustion: bool = True,
    ) -> HTNTaskNode | None:
        completed = node
        parent_id = completed.parent_id
        first_exhausted: HTNTaskNode | None = None
        while parent_id is not None:
            parent = self.get_node(parent_id)
            if completed.id not in parent.children:
                raise RuntimeError(
                    f"HTN tree is inconsistent: node '{completed.id}' points to parent "
                    f"'{parent_id}', but parent does not list it as a child"
                )
            if parent.status in (HTNTaskStatus.COMPLETED, HTNTaskStatus.FAILED):
                return None
            if parent.id in self._summarizing_nodes:
                logger.debug("HTN node '%s' summarization already in progress", parent.id)
                return None

            children = self.children_of(parent_id)
            if not children:
                raise RuntimeError(
                    f"HTN tree is inconsistent: parent '{parent_id}' has no children "
                    f"while propagating completion from '{completed.id}'"
                )
            if any(child.status is not HTNTaskStatus.COMPLETED for child in children):
                return None

            self._summarizing_nodes.add(parent.id)
            try:
                completion = await self._summarize_internal(parent, children)
            finally:
                self._summarizing_nodes.discard(parent.id)
            parent.result_summary = completion.result_summary
            parent.completion_outcome = completion.completion_outcome
            parent.failure_reason = completion.failure_reason
            if completion.failure_reason:
                parent.completion_outcome = HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
                parent.status = HTNTaskStatus.FAILED
                return parent
            parent.status = HTNTaskStatus.COMPLETED
            if (
                parent.completion_outcome
                is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            ):
                if first_exhausted is None:
                    first_exhausted = parent
                if interrupt_on_semantic_exhaustion:
                    return first_exhausted
            completed = parent
            parent_id = parent.parent_id
        return first_exhausted

    async def _summarize_internal(
        self,
        parent: HTNTaskNode,
        children: list[HTNTaskNode],
    ) -> HTNNodeCompletion:
        if self.summarizer is None:
            return self._fallback_completion(children)

        try:
            summary = self.summarizer(self.root(), parent, children)
            if inspect.isawaitable(summary):
                summary = await summary
            if isinstance(summary, HTNNodeCompletion):
                return summary
            return HTNNodeCompletion(
                result_summary=str(summary),
                completion_outcome=self._fallback_outcome(children),
            )
        except Exception as exc:
            logger.exception("HTN summarizer failed for node '%s'; preserving evidence", parent.id)
            return HTNNodeCompletion(
                result_summary=self._fallback_summary(children),
                completion_outcome=HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
                failure_reason=f"Tree synthesis failed: {type(exc).__name__}: {exc}",
            )

    def _fallback_completion(self, children: list[HTNTaskNode]) -> HTNNodeCompletion:
        return HTNNodeCompletion(
            result_summary=self._fallback_summary(children),
            completion_outcome=self._fallback_outcome(children),
        )

    def _fallback_outcome(
        self,
        children: list[HTNTaskNode],
    ) -> HTNCompletionOutcome:
        # HTN decomposition is conjunctive. If synthesis is unavailable, one
        # exhausted child means the parent task was not fully resolved.
        if any(
            child.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            for child in children
        ):
            return HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
        return HTNCompletionOutcome.RESOLVED

    def _fallback_summary(self, children: list[HTNTaskNode]) -> str:
        sections: list[str] = []
        for child in children:
            title = child.title or child.id
            summary = child.result_summary if child.result_summary is not None else "(no summary)"
            sections.append(f"## {title}\n{summary}")
        return "\n\n".join(sections)

    def _validate_node_links(self, node: HTNTaskNode) -> list[str]:
        errors: list[str] = []

        if len(node.children) != len(set(node.children)):
            errors.append(f"Node '{node.id}' has duplicate child IDs")

        for child_id in node.children:
            child = self.nodes.get(child_id)
            if child is None:
                errors.append(f"Node '{node.id}' references missing child '{child_id}'")
                continue
            if child.parent_id != node.id:
                errors.append(
                    f"Child '{child_id}' parent_id is {child.parent_id!r}, expected '{node.id}'"
                )

        if node.id != self.root_id:
            if node.parent_id is None:
                errors.append(f"Non-root node '{node.id}' must have a parent_id")
            elif node.parent_id not in self.nodes:
                errors.append(f"Node '{node.id}' references missing parent '{node.parent_id}'")
            elif node.id not in self.nodes[node.parent_id].children:
                errors.append(f"Node '{node.id}' is not listed in parent '{node.parent_id}' children")

        sibling_ids = self._sibling_ids(node)
        for dep_id in node.depends_on:
            if dep_id == node.id:
                errors.append(f"Node '{node.id}' cannot depend on itself")
            elif dep_id not in self.nodes:
                errors.append(f"Node '{node.id}' depends on missing node '{dep_id}'")
            elif dep_id not in sibling_ids:
                errors.append(f"Node '{node.id}' depends on non-sibling node '{dep_id}'")

        return errors

    def _sibling_ids(self, node: HTNTaskNode) -> set[str]:
        if node.parent_id is None or node.parent_id not in self.nodes:
            return set()
        return set(self.nodes[node.parent_id].children)

    def _validate_child_cycles(self) -> list[str]:
        errors: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                errors.append(f"Child cycle detected at node '{node_id}'")
                return
            visiting.add(node_id)
            node = self.nodes.get(node_id)
            if node is not None:
                for child_id in node.children:
                    if child_id in self.nodes:
                        visit(child_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)
        return errors

    def _validate_reachability(self) -> list[str]:
        reachable: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in reachable:
                return
            node = self.nodes.get(node_id)
            if node is None:
                return
            reachable.add(node_id)
            for child_id in node.children:
                visit(child_id)

        visit(self.root_id)
        return [
            f"Node '{node_id}' is unreachable from root '{self.root_id}'"
            for node_id in self.nodes
            if node_id not in reachable
        ]

    def _validate_dependency_cycles(self) -> list[str]:
        errors: list[str] = []
        for parent in self.nodes.values():
            child_ids = set(parent.children)
            if not child_ids:
                continue

            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(node_id: str) -> None:
                if node_id in visited:
                    return
                if node_id in visiting:
                    errors.append(f"Sibling dependency cycle detected under node '{parent.id}'")
                    return
                visiting.add(node_id)
                node = self.nodes.get(node_id)
                if node is None:
                    visiting.remove(node_id)
                    return
                for dep_id in node.depends_on:
                    if dep_id in child_ids:
                        visit(dep_id)
                visiting.remove(node_id)
                visited.add(node_id)

            for child_id in parent.children:
                if child_id in self.nodes:
                    visit(child_id)

        return errors
