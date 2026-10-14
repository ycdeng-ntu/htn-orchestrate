"""Run one disposable informed-HTN planning epoch."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder, diagnostic_scope
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    OrchestratorResult,
    ExecutionBudget,
    Orchestrator,
    OrchestratorRequest,
)
from workflows.dynamic_htn_informed_research.htn.planner import HTNPlanner
from workflows.dynamic_htn_informed_research.htn.strategies import ResearchDecompositionStrategy
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies.contracts import TaskStrategy


class EpochContext(Protocol):
    """The sole host service used by the otherwise portable epoch driver."""

    async def emit_event(self, event: str, payload: dict[str, Any]) -> None: ...

if TYPE_CHECKING:
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import (
        ReportRuntime,
    )

logger = logging.getLogger(__name__)


class LeafExecutor(Protocol):
    """Shared lifecycle for a research team or a report delivery session."""

    _results: list[AgentResultRecord]

    @property
    def results(self) -> tuple[AgentResultRecord, ...]: ...

    async def run(self) -> OrchestratorResult: ...

    def partial_result(self, *, reason: str = "") -> OrchestratorResult: ...


@dataclass(frozen=True)
class LeafTimeoutIncident(Exception):
    leaf_id: str
    leaf_task: str
    report_count: int
    partial_chars: int
    ref: str | None = None


@dataclass(frozen=True)
class SemanticExhaustionIncident(Exception):
    """A root/internal node established that the current tree cannot resolve its task."""

    node_id: str
    node_task: str
    result_summary: str
    ref: str | None = None


@dataclass
class EpochRunResult:
    tree: HTNTree
    orchestrators: dict[str, LeafExecutor]
    timeout_incidents: list[LeafTimeoutIncident]
    exhaustion_incidents: list[SemanticExhaustionIncident] = field(default_factory=list)
    nonterminal_exhaustion_incidents: list[SemanticExhaustionIncident] = field(
        default_factory=list
    )
    question_deadline_exhausted: bool = False
    dispatch_budget_exhausted: bool = False
    execution_failure_incidents: list[ExecutionFailureIncident] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionFailureIncident:
    """An execution/protocol failure, distinct from a research conclusion."""

    node_id: str
    node_task: str
    error_type: str
    error: str
    report_count: int
    partial_chars: int
    ref: str | None = None


async def run_epoch(
    *,
    question: str,
    epoch_id: int,
    restart_context: str,
    restart_answer_context: str = "",
    research_deadline: float,
    max_iterations: int,
    max_depth: int,
    max_children: int,
    max_precursor_chars: int,
    max_single_summary_chars: int,
    max_rounds: int,
    max_agents_per_round: int,
    progressive_fan_in: bool,
    fan_in_grace_s: int,
    leaf_timeout_s: int,
    available_agents: list[AgentDescriptor],
    llm: Any,
    dispatch: Any,
    summarizer: Any,
    output_dir: Path,
    diagnostics: DiagnosticRecorder,
    ctx: EpochContext,
    task_strategy: TaskStrategy | None = None,
    interrupt_on_semantic_exhaustion: bool = True,
    planner_llm: Any | None = None,
    planner_audit_llm: Any | None = None,
    planner_revision_llm: Any | None = None,
    coordinator_llm: Any | None = None,
    instruction_llm: Any | None = None,
    leaf_synthesis_llm: Any | None = None,
    root_answer_context_provider: Any | None = None,
    report_runtime: ReportRuntime | None = None,
    blocked_source_directive: str = "",
    dispatch_budget: QuestionDispatchBudget | None = None,
    report_research_deadline: float | None = None,
    report_network_enabled: bool = True,
    report_alignment_llm: Any | None = None,
) -> EpochRunResult:
    """Execute one tree until completion, leaf timeout, or question deadline."""
    if task_strategy is None:
        from workflows.dynamic_htn_informed_research.task_strategies import (
            get_active_task_strategy,
        )

        task_strategy = get_active_task_strategy()
    report_tree = task_strategy.delivery.output_mode == "report"
    if report_tree:
        from ..task_strategies.report_research.topology import (
            ReportPlanningStrategy, delivery_root_summary,
        )
        summarizer = delivery_root_summary
        max_depth = 1  # ROOT at depth zero; research and delivery are siblings.
        max_children = max(2, max_children)
    root = HTNTaskNode(
        id="root",
        parent_id=None,
        title="Question",
        task=question,
        needs_decomposition=True,
    )
    tree = HTNTree(
        root_id="root",
        nodes={"root": root},
        summarizer=summarizer,
        epoch_id=epoch_id,
        restart_context=restart_context,
        restart_answer_context=restart_answer_context,
    )
    planner = HTNPlanner(
        strategy=ResearchDecompositionStrategy(
            llm=planner_llm or llm,
            task_strategy=task_strategy.planning,
            audit_llm=planner_audit_llm,
            revision_llm=planner_revision_llm,
            max_precursor_chars=max_precursor_chars,
            max_single_summary_chars=max_single_summary_chars,
            diagnostics=diagnostics,
        ),
        max_depth=max_depth,
        max_children=max_children,
    )
    if report_tree:
        planner.strategy = ReportPlanningStrategy(
            planner_llm=planner_llm or llm,
            audit_llm=planner_audit_llm or planner_llm or llm,
            revision_llm=planner_revision_llm or planner_llm or llm,
            planner_system=task_strategy.planning.decomposition_system,
            network_enabled=report_network_enabled,
            deadline=min(research_deadline, report_research_deadline)
            if report_research_deadline is not None else research_deadline,
            diagnostics=diagnostics,
        )
    active_orchestrators: dict[str, LeafExecutor] = {}
    nonterminal_exhaustion_incidents: list[SemanticExhaustionIncident] = []
    execution_failure_incidents: list[ExecutionFailureIncident] = []
    epoch_output_dir = output_dir / f"epoch-{epoch_id}"

    def dispatch_exhausted() -> bool:
        return dispatch_budget is not None and dispatch_budget.exhausted

    def epoch_result(**kwargs: Any) -> EpochRunResult:
        return EpochRunResult(
            dispatch_budget_exhausted=dispatch_exhausted(),
            execution_failure_incidents=list(execution_failure_incidents),
            **kwargs,
        )

    def snapshot_tree(label: str) -> None:
        diagnostics.write_json(
            diagnostics.artifact_path(
                "tree_snapshots",
                f"epoch-{epoch_id}-{label}",
                ".json",
            ),
            {nid: node.model_dump() for nid, node in tree.nodes.items()},
        )

    def fail_running_leaf(leaf: HTNTaskNode, ref: str | None = None) -> None:
        if leaf.status is HTNTaskStatus.RUNNING:
            leaf.status = HTNTaskStatus.FAILED
            leaf.ref = ref

    def record_semantic_exhaustion(
        trigger: HTNTaskNode,
        *,
        completed_leaf_id: str,
    ) -> SemanticExhaustionIncident | None:
        """Record an exhausted internal/root node and return it only if terminal."""
        incident = SemanticExhaustionIncident(
            node_id=trigger.id,
            node_task=trigger.task,
            result_summary=trigger.result_summary or "",
            ref=trigger.ref,
        )
        if interrupt_on_semantic_exhaustion and not dispatch_exhausted():
            diagnostics.emit(
                "semantic_exhaustion_checkpointed",
                trigger_node_id=incident.node_id,
                trigger_is_root=incident.node_id == tree.root_id,
                output_chars=len(incident.result_summary),
                ref=incident.ref,
            )
            snapshot_tree(f"semantic-exhaustion-{incident.node_id}")
            return incident

        nonterminal_exhaustion_incidents.append(incident)
        diagnostics.emit(
            "semantic_exhaustion_nonterminal",
            trigger_node_id=incident.node_id,
            trigger_is_root=incident.node_id == tree.root_id,
            trigger_depth=tree.depth_of(incident.node_id),
            active_leaf_ids=[
                node.id
                for node in tree.nodes.values()
                if node.status is HTNTaskStatus.RUNNING
                and node.id != completed_leaf_id
            ],
            output_chars=len(incident.result_summary),
            ref=incident.ref,
        )
        snapshot_tree(f"semantic-exhaustion-nonterminal-{incident.node_id}")
        return None

    async def solve_leaf(leaf: HTNTaskNode) -> None:
        report_delivery_context = (
            report_runtime.delivery_context_for(leaf)
            if report_runtime is not None
            else ""
        )
        delivery_context = "\n\n".join(
            part.strip()
            for part in (blocked_source_directive, report_delivery_context)
            if part.strip()
        )
        req = OrchestratorRequest(
            tree=tree,
            leaf=leaf,
            available_agents=(
                [agent for agent in available_agents if agent.role_id != "dynir_report_final_verifier"]
                if report_tree else available_agents
            ),
            execution_budget=ExecutionBudget(
                max_rounds=max_rounds,
                max_agents_per_round=max_agents_per_round,
                progressive_fan_in=progressive_fan_in,
                fan_in_grace_s=max(0, fan_in_grace_s),
            ),
            output_dir=epoch_output_dir,
            llm=llm,
            dispatch=dispatch,
            dispatch_budget=dispatch_budget,
            task_strategy=task_strategy,
            diagnostics=diagnostics,
            coordinator_llm=coordinator_llm,
            instruction_llm=instruction_llm,
            synthesis_llm=leaf_synthesis_llm,
            root_answer_context_provider=root_answer_context_provider,
            delivery_context=delivery_context,
        )
        leaf_deadline = research_deadline
        is_delivery = report_tree and leaf.execution_kind == "report_delivery"
        if report_tree and not is_delivery and report_research_deadline is not None:
            leaf_deadline = min(leaf_deadline, report_research_deadline)
        leaf_budget = max(0.001 if report_tree else 1.0,
                          min(leaf_timeout_s, leaf_deadline - time.monotonic()))
        if is_delivery:
            from ..task_strategies.report_research.delivery import ReportDeliveryOrchestrator
            orchestrator = ReportDeliveryOrchestrator(
                req, research_orchestrators=active_orchestrators,
                network_enabled=report_network_enabled,
                alignment_llm=report_alignment_llm,
                deadline=min(leaf_deadline, time.monotonic() + leaf_budget),
            )
        else:
            orchestrator = Orchestrator(req)
        active_orchestrators[leaf.id] = orchestrator
        orchestrator_task = asyncio.create_task(
            orchestrator.run(),
            name=f"dynamic_htn_informed_research:epoch-{epoch_id}:{leaf.id}:orchestrator",
        )
        leaf_started = time.monotonic()
        diagnostics.emit(
            "leaf_start",
            leaf_id=leaf.id,
            leaf_budget_s=leaf_budget,
            leaf_task_chars=len(leaf.task or ""),
            depends_on=list(leaf.depends_on),
        )
        leaf_cancelled = False

        def checkpoint_execution_failure(
            error_type: str, error: str, *, partial: Any | None = None,
            node: HTNTaskNode | None = None,
        ) -> None:
            failed_node = node if node is not None else leaf
            reason = "Leaf execution failed; the task remains unresolved."
            if dispatch_exhausted():
                reason += " The question worker dispatch budget was exhausted."
            if partial is None:
                try:
                    partial = orchestrator.partial_result(reason=reason)
                except Exception as checkpoint_error:
                    # Failure to build a checkpoint must not turn the original
                    # error into success or discard already returned reports.
                    logger.warning("Could not checkpoint failed leaf %s: %s", leaf.id, checkpoint_error)
                    partial = None
            failed_node.result_summary = (
                partial.result_summary if partial is not None else "\n\n".join(
                    f"Agent report [{record.agent_id}] ({record.execution_quality}):\n{record.result_summary}"
                    for record in orchestrator._results
                )
            ) or ""
            failed_node.ref = partial.ref if partial is not None else None
            failed_node.status = HTNTaskStatus.FAILED
            failed_node.completion_outcome = HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            failed_node.failure_reason = f"{error_type}: {error}"
            subtree_ids = {failed_node.id}
            remaining_nodes = list(failed_node.children)
            while remaining_nodes:
                child_id = remaining_nodes.pop()
                subtree_ids.add(child_id)
                remaining_nodes.extend(tree.get_node(child_id).children)
            report_count = sum(
                len(owner._results) for node_id, owner in active_orchestrators.items()
                if node_id in subtree_ids
            )
            incident = ExecutionFailureIncident(
                node_id=failed_node.id, node_task=failed_node.task,
                error_type=error_type, error=error,
                report_count=report_count,
                partial_chars=len(failed_node.result_summary), ref=failed_node.ref,
            )
            execution_failure_incidents.append(incident)
            diagnostics.emit(
                "node_execution_failure_checkpointed", node_id=failed_node.id,
                completed_leaf_id=leaf.id if failed_node is not leaf else None,
                error_type=error_type, error=error,
                recovered_report_count=incident.report_count,
                output_chars=incident.partial_chars, ref=incident.ref,
            )
            if failed_node is leaf:
                diagnostics.emit(
                    "leaf_end", leaf_id=leaf.id,
                    duration_ms=round((time.monotonic() - leaf_started) * 1000, 3),
                    status="error", error_type=error_type, error=error,
                    report_count=incident.report_count, output_chars=incident.partial_chars,
                    ref=incident.ref, completion_outcome=leaf.completion_outcome.value,
                )
            snapshot_tree(f"node-error-{failed_node.id}")

        try:
            done, _ = await asyncio.wait({orchestrator_task}, timeout=leaf_budget)
            if not done:
                orchestrator_task.cancel()
                await asyncio.gather(orchestrator_task, return_exceptions=True)
                partial = orchestrator.partial_result()
                fail_running_leaf(leaf, partial.ref)
                if report_tree:
                    leaf.result_summary = partial.result_summary
                    leaf.completion_outcome = HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
                incident = LeafTimeoutIncident(
                    leaf_id=leaf.id,
                    leaf_task=leaf.task,
                    report_count=len(orchestrator._results),
                    partial_chars=len(partial.result_summary or ""),
                    ref=partial.ref,
                )
                logger.warning(
                    "dynamic_solve: epoch %d leaf %s hit %.0fs timeout; terminating epoch",
                    epoch_id,
                    leaf.id,
                    leaf_budget,
                )
                diagnostics.emit(
                    "leaf_timeout_checkpointed",
                    leaf_id=leaf.id,
                    output_chars=incident.partial_chars,
                    recovered_report_count=incident.report_count,
                    ref=incident.ref,
                )
                diagnostics.emit(
                    "leaf_end",
                    leaf_id=leaf.id,
                    duration_ms=round((time.monotonic() - leaf_started) * 1000, 3),
                    status="timeout",
                    report_count=incident.report_count,
                )
                snapshot_tree(f"leaf-timeout-{leaf.id}")
                raise incident

            # Awaiting a completed task deliberately propagates an internal
            # TimeoutError into the ordinary leaf-error path below.  Only the
            # explicit wall-clock expiry above is classified as a leaf timeout.
            result = await orchestrator_task
            failure_reason = getattr(result, "failure_reason", None)
            if failure_reason:
                checkpoint_execution_failure("SynthesisError", failure_reason, partial=result)
                return
            completion_outcome = getattr(
                result,
                "completion_outcome",
                HTNCompletionOutcome.RESOLVED,
            )
            with diagnostic_scope(leaf_id=leaf.id, call_kind="tree_completion_synthesis"):
                exhaustion_trigger = await tree.mark_completed(
                    leaf.id,
                    result.result_summary or "",
                    result.ref,
                    completion_outcome,
                    interrupt_on_semantic_exhaustion=(
                        interrupt_on_semantic_exhaustion and not dispatch_exhausted()
                    ),
                )
            diagnostics.emit(
                "leaf_end",
                leaf_id=leaf.id,
                duration_ms=round((time.monotonic() - leaf_started) * 1000, 3),
                status="completed",
                output_chars=len(result.result_summary or ""),
                report_count=len(orchestrator._results),
                ref=result.ref,
                completion_outcome=completion_outcome.value,
            )
            snapshot_tree(f"leaf-completed-{leaf.id}")
            if exhaustion_trigger is not None:
                if exhaustion_trigger.failure_reason:
                    checkpoint_execution_failure(
                        "TreeSynthesisError", exhaustion_trigger.failure_reason,
                        partial=exhaustion_trigger, node=exhaustion_trigger,
                    )
                    return
                terminal_incident = record_semantic_exhaustion(
                    exhaustion_trigger,
                    completed_leaf_id=leaf.id,
                )
                if terminal_incident is not None:
                    raise terminal_incident
        except LeafTimeoutIncident:
            raise
        except SemanticExhaustionIncident:
            raise
        except asyncio.CancelledError:
            leaf_cancelled = True
            if not orchestrator_task.done():
                orchestrator_task.cancel()
                await asyncio.gather(orchestrator_task, return_exceptions=True)
            partial = orchestrator.partial_result()
            fail_running_leaf(leaf, partial.ref)
            if report_tree:
                leaf.result_summary = partial.result_summary
                leaf.completion_outcome = HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            diagnostics.emit(
                "leaf_end",
                leaf_id=leaf.id,
                duration_ms=round((time.monotonic() - leaf_started) * 1000, 3),
                status="cancelled_by_epoch_quiesce",
                output_chars=len(partial.result_summary or ""),
                report_count=len(orchestrator._results),
                ref=partial.ref,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - checkpoint failure for bounded recovery
            logger.warning("dynamic_solve: leaf %s failed: %s", leaf.id, exc)
            checkpoint_execution_failure(type(exc).__name__, str(exc))
        finally:
            if report_runtime is not None and not leaf_cancelled:
                try:
                    await report_runtime.observe_leaf(
                        leaf,
                        orchestrator.results,
                        leaf_synthesis_llm or llm,
                        deadline=research_deadline,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - observation is fail-open
                    logger.warning(
                        "dynamic_solve: report observation failed for leaf %s: %s",
                        leaf.id,
                        exc,
                    )
                    diagnostics.emit(
                        "report_leaf_observation_failed",
                        leaf_id=leaf.id,
                        report_count=len(orchestrator.results),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

    diagnostics.emit(
        "epoch_start",
        restart_context_chars=len(restart_context),
        restart_answer_context_chars=len(restart_answer_context),
        remaining_research_s=max(0.0, research_deadline - time.monotonic()),
    )
    snapshot_tree("000-initial")

    for iteration in range(max_iterations):
        if execution_failure_incidents:
            # The previous admitted wave has settled under its original time
            # limit. Never plan/run dependents of a failed execution; let the
            # question driver recover from the preserved evidence instead.
            diagnostics.emit("epoch_end", status="execution_failure")
            return epoch_result(
                tree=tree, orchestrators=active_orchestrators, timeout_incidents=[],
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
            )
        if tree.root().status is HTNTaskStatus.COMPLETED:
            break
        if dispatch_exhausted():
            # Waves settle before this check. Preserve their evidence and do
            # not plan dependent leaves that cannot receive any worker slots.
            diagnostics.emit(
                "epoch_end", status="question_dispatch_budget_exhausted",
                dispatch_budget=dispatch_budget.snapshot(),
            )
            snapshot_tree(f"iter-{iteration}-dispatch-budget-exhausted")
            return epoch_result(
                tree=tree, orchestrators=active_orchestrators, timeout_incidents=[],
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
            )
        if time.monotonic() > research_deadline:
            diagnostics.emit("epoch_end", status="question_deadline_before_wave")
            return epoch_result(
                tree=tree,
                orchestrators=active_orchestrators,
                timeout_incidents=[],
                exhaustion_incidents=[],
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
                question_deadline_exhausted=True,
            )

        while True:
            planning = tree.ready_for_planning()
            if not planning:
                break
            for node in planning:
                planning_started = time.monotonic()
                diagnostics.emit(
                    "planning_start",
                    iteration=iteration,
                    node_id=node.id,
                    node_depth=tree.depth_of(node.id),
                    depends_on=list(node.depends_on),
                )
                try:
                    with diagnostic_scope(
                        iteration=iteration,
                        node_id=node.id,
                        call_kind="planner_decompose",
                    ):
                        children = await planner.decompose(tree, node.id)
                        if report_tree and (not children or sum(
                            child.execution_kind == "report_delivery" for child in children
                        ) != 1):
                            raise RuntimeError("Report planning must retain the mandatory delivery leaf")
                    diagnostics.emit(
                        "planning_end",
                        iteration=iteration,
                        node_id=node.id,
                        duration_ms=round((time.monotonic() - planning_started) * 1000, 3),
                        status="completed",
                        child_count=len(children),
                        child_ids=[child.id for child in children],
                        needs_decomposition=tree.get_node(node.id).needs_decomposition,
                    )
                    snapshot_tree(f"iter-{iteration}-planned-{node.id}")
                except Exception as exc:  # noqa: BLE001 - existing planner fallback
                    logger.warning("dynamic_solve: decompose %s failed: %s", node.id, exc)
                    if report_tree:
                        # Do not silently bypass the mandatory delivery leaf.
                        raise
                    try:
                        tree.mark_executable(node.id)
                    except Exception:  # noqa: BLE001
                        pass
                    diagnostics.emit(
                        "planning_end",
                        iteration=iteration,
                        node_id=node.id,
                        duration_ms=round((time.monotonic() - planning_started) * 1000, 3),
                        status="error_marked_executable",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    snapshot_tree(f"iter-{iteration}-planning-error-{node.id}")

        execs = tree.ready_for_execution()
        if not execs:
            if not tree.ready_for_planning():
                break
            continue
        if report_runtime is not None:
            await report_runtime.refresh_ownership(tree, planner_llm or llm)
        for leaf in execs:
            tree.mark_running(leaf.id)
        wave_started = time.monotonic()
        diagnostics.emit(
            "wave_start",
            iteration=iteration,
            leaf_ids=[leaf.id for leaf in execs],
            remaining_research_s=max(0.0, research_deadline - time.monotonic()),
        )
        await ctx.emit_event(
            "dynamic_htn_informed_research_wave",
            {
                "epoch_id": epoch_id,
                "iteration": iteration,
                "leaves": [leaf.id for leaf in execs],
            },
        )

        remaining = max(1.0, research_deadline - time.monotonic())
        tasks = [
            asyncio.create_task(
                solve_leaf(leaf),
                name=f"dynamic_htn_informed_research:epoch-{epoch_id}:{leaf.id}",
            )
            for leaf in execs
        ]
        wave_deadline = time.monotonic() + remaining
        try:
            pending = set(tasks)
            while pending:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending),
                        timeout=max(0.0, wave_deadline - time.monotonic()),
                    )
                    break
                except SemanticExhaustionIncident:
                    # Report observation runs in solve_leaf's finally block.
                    # Another leaf can exhaust the shared quota during that
                    # await, after this incident was classified as terminal.
                    if not dispatch_exhausted():
                        raise
                    for leaf, task in zip(execs, tasks, strict=True):
                        if task not in pending or not task.done():
                            continue
                        try:
                            task.result()
                        except SemanticExhaustionIncident as incident:
                            record_semantic_exhaustion(
                                tree.get_node(incident.node_id),
                                completed_leaf_id=leaf.id,
                            )
                    pending = {task for task in pending if not task.done()}
            diagnostics.emit(
                "wave_end",
                iteration=iteration,
                duration_ms=round((time.monotonic() - wave_started) * 1000, 3),
                status="completed",
                leaf_ids=[leaf.id for leaf in execs],
            )
        except LeafTimeoutIncident as first:
            diagnostics.emit(
                "epoch_quiesce_start",
                trigger_leaf_id=first.leaf_id,
                active_leaf_ids=[leaf.id for leaf in execs if leaf.status is HTNTaskStatus.RUNNING],
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            settled = await asyncio.gather(*tasks, return_exceptions=True)
            incidents = [item for item in settled if isinstance(item, LeafTimeoutIncident)]
            if not any(item.leaf_id == first.leaf_id for item in incidents):
                incidents.insert(0, first)
            for leaf in execs:
                fail_running_leaf(leaf)
            diagnostics.emit(
                "wave_end",
                iteration=iteration,
                duration_ms=round((time.monotonic() - wave_started) * 1000, 3),
                status="epoch_timeout",
                leaf_ids=[leaf.id for leaf in execs],
                timeout_leaf_ids=[item.leaf_id for item in incidents],
            )
            diagnostics.emit(
                "epoch_quiesce_end",
                timeout_leaf_ids=[item.leaf_id for item in incidents],
                checkpointed_report_count=sum(
                    len(orchestrator._results) for orchestrator in active_orchestrators.values()
                ),
            )
            snapshot_tree(f"iter-{iteration}-epoch-timeout")
            diagnostics.emit("epoch_end", status="leaf_timeout")
            return epoch_result(
                tree=tree,
                orchestrators=active_orchestrators,
                timeout_incidents=incidents,
                exhaustion_incidents=[],
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
            )
        except SemanticExhaustionIncident as first:
            diagnostics.emit(
                "epoch_quiesce_start",
                trigger="semantic_exhaustion",
                trigger_node_id=first.node_id,
                active_leaf_ids=[
                    leaf.id for leaf in execs if leaf.status is HTNTaskStatus.RUNNING
                ],
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            settled = await asyncio.gather(*tasks, return_exceptions=True)
            incidents = [
                item
                for item in settled
                if isinstance(item, SemanticExhaustionIncident)
            ]
            if not any(item.node_id == first.node_id for item in incidents):
                incidents.insert(0, first)
            for leaf in execs:
                fail_running_leaf(leaf)
            diagnostics.emit(
                "wave_end",
                iteration=iteration,
                duration_ms=round((time.monotonic() - wave_started) * 1000, 3),
                status="semantic_exhaustion",
                leaf_ids=[leaf.id for leaf in execs],
                exhaustion_node_ids=[item.node_id for item in incidents],
            )
            diagnostics.emit(
                "epoch_quiesce_end",
                trigger="semantic_exhaustion",
                exhaustion_node_ids=[item.node_id for item in incidents],
                checkpointed_report_count=sum(
                    len(orchestrator._results)
                    for orchestrator in active_orchestrators.values()
                ),
            )
            snapshot_tree(f"iter-{iteration}-semantic-exhaustion")
            diagnostics.emit("epoch_end", status="semantic_exhaustion")
            return epoch_result(
                tree=tree,
                orchestrators=active_orchestrators,
                timeout_incidents=[],
                exhaustion_incidents=incidents,
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
            )
        except (TimeoutError, asyncio.TimeoutError):
            diagnostics.emit(
                "epoch_quiesce_start",
                trigger="question_deadline",
                active_leaf_ids=[leaf.id for leaf in execs if leaf.status is HTNTaskStatus.RUNNING],
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for leaf in execs:
                fail_running_leaf(leaf)
            diagnostics.emit(
                "wave_end",
                iteration=iteration,
                duration_ms=round((time.monotonic() - wave_started) * 1000, 3),
                status="question_timeout",
                leaf_ids=[leaf.id for leaf in execs],
            )
            diagnostics.emit("epoch_quiesce_end", trigger="question_deadline")
            snapshot_tree(f"iter-{iteration}-question-timeout")
            diagnostics.emit("epoch_end", status="question_deadline")
            return epoch_result(
                tree=tree,
                orchestrators=active_orchestrators,
                timeout_incidents=[],
                exhaustion_incidents=[],
                nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
                question_deadline_exhausted=True,
            )

    diagnostics.emit(
        "epoch_end",
        status=("execution_failure" if execution_failure_incidents else
                "completed" if tree.root().status is HTNTaskStatus.COMPLETED else "frontier_exhausted"),
        node_count=len(tree.nodes),
        root_status=tree.root().status.value,
    )
    return epoch_result(
        tree=tree,
        orchestrators=active_orchestrators,
        timeout_incidents=[],
        exhaustion_incidents=[],
        nonterminal_exhaustion_incidents=nonterminal_exhaustion_incidents,
    )
