from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.epoch import run_epoch
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    AgentTaskSummary,
    ExecutionBudget,
    Orchestrator,
    OrchestratorRequest,
    OrchestratorResult,
    _render_c_compact,
    build_stable_context,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy
from workflows.dynamic_htn_informed_research.task_strategies.report_research.evidence import (
    EvidenceCard,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import (
    Requirement,
    RequirementSet,
    RequirementStatus,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import (
    ReportRuntime,
)


class _Ctx:
    task_id = "report-epoch-integration"

    def service(self, _kind):  # noqa: ANN001
        return object()

    async def emit_event(self, _name: str, _payload: dict[str, Any]) -> None:
        return None


class _RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.writes: list[tuple[Any, Any]] = []

    def emit(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def write_json(self, path: Any, value: Any) -> str:
        self.writes.append((path, value))
        return str(path)

    def write_text(self, path: Any, value: Any) -> str:
        self.writes.append((path, value))
        return str(path)

    def artifact_path(self, category: str, name: Any, suffix: str) -> Path:
        return Path(category) / f"{name}{suffix}"


class _UnusedLLM:
    model = "unused"

    async def chat(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("this test does not cross an LLM boundary")


def _record(
    agent_id: str,
    summary: str = "A sufficiently detailed submitted memo.",
    *,
    observed_source_index: str | None = None,
) -> AgentResultRecord:
    return AgentResultRecord(
        agent_id=agent_id,
        role_id="dynir_report_researcher",
        task_summary="Research the assigned requirement.",
        result_summary=summary,
        execution_quality="submitted",
        submitted=True,
        observed_source_index=observed_source_index,
    )


def _epoch_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "question": "Produce the requested report.",
        "epoch_id": 0,
        "restart_context": "",
        "research_deadline": time.monotonic() + 60,
        "max_iterations": 2,
        "max_depth": 2,
        "max_children": 3,
        "max_precursor_chars": 4_000,
        "max_single_summary_chars": 2_000,
        "max_rounds": 2,
        "max_agents_per_round": 1,
        "progressive_fan_in": False,
        "fan_in_grace_s": 1,
        "leaf_timeout_s": 30,
        "available_agents": [AgentDescriptor("dynir_report_researcher")],
        "llm": _UnusedLLM(),
        "dispatch": object(),
        "summarizer": None,
        "output_dir": tmp_path,
        "diagnostics": _RecordingDiagnostics(),
        "ctx": _Ctx(),
        "task_strategy": get_task_strategy("research"),
    }
    values.update(overrides)
    return values


def _install_direct_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    class _Planner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def decompose(self, tree: HTNTree, node_id: str) -> list[HTNTaskNode]:
            tree.mark_executable(node_id)
            return []

    monkeypatch.setattr(epoch_module, "HTNPlanner", _Planner)


def _install_success_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[Any],
    *,
    events: list[str] | None = None,
) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    class _Orchestrator:
        def __init__(self, request) -> None:  # noqa: ANN001
            requests.append(request)
            self._results = [_record(f"worker-{request.leaf.id}")]
            if events is not None:
                events.append("orchestrator_created")

        @property
        def results(self) -> tuple[AgentResultRecord, ...]:
            return tuple(self._results)

        async def run(self) -> OrchestratorResult:
            return OrchestratorResult("Resolved leaf result.\n\n# Answer: resolved")

        def partial_result(self) -> OrchestratorResult:
            return OrchestratorResult("Partial leaf result.")

    monkeypatch.setattr(epoch_module, "Orchestrator", _Orchestrator)


def _runtime(*requirements: Requirement, diagnostics: DiagnosticRecorder | None = None) -> ReportRuntime:
    return ReportRuntime(
        question="Compare public facts in a sourced report.",
        policy=SCIENTIFIC_REPORT_POLICY,
        requirements=RequirementSet("scientific_report", list(requirements)),
        _diagnostics=diagnostics,
    )


def _requirement(
    requirement_id: str,
    *,
    priority: int,
    description: str | None = None,
    status: RequirementStatus = RequirementStatus.PENDING,
) -> Requirement:
    return Requirement(
        id=requirement_id,
        description=description or f"Research public fact {requirement_id}.",
        acceptance_criteria=(f"Cite authoritative support for {requirement_id}.",),
        priority=priority,
        kind="fact",
        status=status,
    )


def _tool_response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            }
        ]
    )


def test_report_delivery_context_is_rendered_only_when_nonempty() -> None:
    root = HTNTaskNode(id="root", title="Report", task="Write a sourced report.")
    tree = HTNTree(root_id="root", nodes={"root": root})

    short_render = _render_c_compact(build_stable_context(tree, root))
    report_render = _render_c_compact(
        build_stable_context(
            tree,
            root,
            delivery_context="[req-003] Include the audited revenue and year.",
        )
    )

    assert "REPORT DELIVERY CONTRACT" not in short_render
    assert "REPORT DELIVERY CONTRACT" in report_render
    assert "BEGIN REPORT DELIVERY CONTRACT" in report_render
    assert "END REPORT DELIVERY CONTRACT" in report_render
    assert "[req-003] Include the audited revenue and year." in report_render


@pytest.mark.asyncio
async def test_run_epoch_default_none_preserves_short_answer_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    _install_direct_planner(monkeypatch)
    _install_success_orchestrator(monkeypatch, requests)

    result = await run_epoch(
        **_epoch_kwargs(tmp_path),
        report_runtime=None,
    )

    assert result.tree.root().status is HTNTaskStatus.COMPLETED
    assert len(requests) == 1
    assert requests[0].delivery_context == ""


@pytest.mark.asyncio
async def test_epoch_injects_question_forbidden_directive_into_ordinary_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    directive = "FORBIDDEN SOURCE: avoid the named work and every mirror."
    _install_direct_planner(monkeypatch)
    _install_success_orchestrator(monkeypatch, requests)

    result = await run_epoch(
        **_epoch_kwargs(tmp_path),
        blocked_source_directive=directive,
    )

    assert result.tree.root().status is HTNTaskStatus.COMPLETED
    assert requests[0].delivery_context == directive


@pytest.mark.asyncio
async def test_epoch_refreshes_ownership_after_decomposition_before_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    events: list[str] = []
    requests: list[Any] = []
    leaf_llm = _UnusedLLM()

    class _Planner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def decompose(self, tree: HTNTree, node_id: str) -> list[HTNTaskNode]:
            events.append("decomposed")
            return tree.decompose(
                node_id,
                [
                    HTNTaskNode(
                        id="leaf-revenue",
                        title="Revenue",
                        task="Find audited revenue and year.",
                    )
                ],
            )

    class _RuntimeProbe:
        async def refresh_ownership(self, tree: HTNTree, llm: Any) -> None:
            assert "leaf-revenue" in tree.nodes
            assert llm is planner_llm
            events.append("ownership_refreshed")

        def delivery_context_for(self, leaf: HTNTaskNode) -> str:
            assert leaf.id == "leaf-revenue"
            events.append("delivery_context_built")
            return "[req-001] Include audited revenue and year."

        async def observe_leaf(
            self,
            leaf: HTNTaskNode,
            records: tuple[AgentResultRecord, ...],
            llm: Any,
            **_kwargs: Any,
        ) -> None:
            assert leaf.id == "leaf-revenue"
            assert [record.agent_id for record in records] == ["worker-leaf-revenue"]
            assert llm is leaf_llm
            events.append("leaf_observed")

    planner_llm = _UnusedLLM()
    monkeypatch.setattr(epoch_module, "HTNPlanner", _Planner)
    _install_success_orchestrator(monkeypatch, requests, events=events)

    result = await run_epoch(
        **_epoch_kwargs(
            tmp_path,
            task_strategy=get_task_strategy("research"),  # Legacy optional observer seam.
            planner_llm=planner_llm,
            leaf_synthesis_llm=leaf_llm,
        ),
        report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
    )

    assert result.tree.root().status is HTNTaskStatus.COMPLETED
    assert events == [
        "decomposed",
        "ownership_refreshed",
        "delivery_context_built",
        "orchestrator_created",
        "leaf_observed",
    ]
    assert requests[0].delivery_context == "[req-001] Include audited revenue and year."


@pytest.mark.asyncio
async def test_epoch_observes_checkpointed_records_on_leaf_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    _install_direct_planner(monkeypatch)
    observations: list[tuple[str, tuple[str, ...], Any]] = []

    class _RuntimeProbe:
        async def refresh_ownership(self, _tree: HTNTree, _llm: Any) -> None:
            return None

        def delivery_context_for(self, _leaf: HTNTaskNode) -> str:
            return "timeout delivery contract"

        async def observe_leaf(
            self,
            leaf: HTNTaskNode,
            records: tuple[AgentResultRecord, ...],
            llm: Any,
            **_kwargs: Any,
        ) -> None:
            observations.append((leaf.id, tuple(record.agent_id for record in records), llm))

    class _TimeoutOrchestrator:
        def __init__(self, request) -> None:  # noqa: ANN001
            self._results = [_record("worker-before-timeout")]

        @property
        def results(self) -> tuple[AgentResultRecord, ...]:
            return tuple(self._results)

        async def run(self) -> OrchestratorResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def partial_result(self) -> OrchestratorResult:
            return OrchestratorResult("Checkpointed partial result.")

    real_wait = asyncio.wait

    async def _expire_orchestrator(awaitables, *, timeout=None):  # noqa: ANN001
        tasks = set(awaitables)
        if any(task.get_name().endswith(":root:orchestrator") for task in tasks):
            return set(), tasks
        return await real_wait(tasks, timeout=timeout)

    leaf_llm = _UnusedLLM()
    monkeypatch.setattr(epoch_module, "Orchestrator", _TimeoutOrchestrator)
    monkeypatch.setattr(epoch_module.asyncio, "wait", _expire_orchestrator)

    result = await run_epoch(
        **_epoch_kwargs(tmp_path, leaf_synthesis_llm=leaf_llm),
        report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
    )

    assert [incident.leaf_id for incident in result.timeout_incidents] == ["root"]
    assert observations == [("root", ("worker-before-timeout",), leaf_llm)]


class _EmptyEvidenceLLM:
    model = "empty-evidence"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
        self.calls.append(messages)
        return _tool_response("extract_evidence", {"cards": [], "challenges": []})


@pytest.mark.asyncio
async def test_gap_recovery_drains_eligible_gaps_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )

    diagnostics = DiagnosticRecorder(tmp_path)
    policy = replace(
        SCIENTIFIC_REPORT_POLICY,
        max_recovery_requirements=4,
        max_recovery_agents=3,
    )
    runtime = ReportRuntime(
        question="Compare public facts in a sourced report.",
        policy=policy,
        requirements=RequirementSet(
            "scientific_report",
            [
                _requirement("req-005", priority=2),
                _requirement("req-004", priority=4),
                _requirement("req-003", priority=1),
                _requirement("req-002", priority=1),
                _requirement("req-001", priority=1, status=RequirementStatus.GROUNDED),
            ],
        ),
    )
    runtime.evidence_cards["ev-known"] = EvidenceCard(
        id="ev-known",
        leaf_id="leaf-prior",
        requirement_ids=("req-002",),
        claim="A prior memo found a provisional public value.",
        source_url="https://example.test/public",
        source_title="Public source",
        support_summary="The observed source supports the provisional value.",
        confidence=0.7,
    )

    active = 0
    max_active = 0
    dispatched: list[tuple[AgentTaskSummary, str]] = []

    async def dispatch(agent: AgentTaskSummary, instruction: str) -> AgentResultRecord:
        nonlocal active, max_active
        dispatched.append((agent, instruction))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return _record(agent.agent_id, f"Submitted evidence memo for {agent.agent_id} with support.")

    evidence_llm = _EmptyEvidenceLLM()
    first = await run_gap_recovery_wave(
        runtime=runtime,
        dispatch=dispatch,
        research_role_id="dynir_report_researcher",
        evidence_llm=evidence_llm,
        question_deadline=time.monotonic() + 30,
        reserve_s=5,
        max_live_agents=2,
        diagnostics=diagnostics,
        blocked_source_directive=(
            "FORBIDDEN SOURCE: do not view or cite the named Alpha study."
        ),
    )
    second = await run_gap_recovery_wave(
        runtime=runtime,
        dispatch=dispatch,
        research_role_id="dynir_report_researcher",
        evidence_llm=evidence_llm,
        question_deadline=time.monotonic() + 30,
        reserve_s=5,
        max_live_agents=2,
        diagnostics=diagnostics,
    )

    assert [record.agent_id for record in first] == [
        "report-gap-req-002",
        "report-gap-req-003",
        "report-gap-req-005",
    ]
    assert second == ()
    assert max_active == 2
    assert [agent.agent_id for agent, _ in dispatched] == [
        "report-gap-req-002",
        "report-gap-req-003",
        "report-gap-req-005",
    ]
    assert runtime.recovery_wave_used is True
    assert len(evidence_llm.calls) == 3
    extraction_prompts = [messages[-1]["content"] for messages in evidence_llm.calls]
    assert "req-002" in extraction_prompts[0] and "req-003" not in extraction_prompts[0]
    assert "req-003" in extraction_prompts[1] and "req-002" not in extraction_prompts[1]

    assignments = {agent.agent_id: instruction for agent, instruction in dispatched}
    assert runtime.question in assignments["report-gap-req-002"]
    assert "ev-known" in assignments["report-gap-req-002"]
    assert policy.requirement_prompt in assignments["report-gap-req-002"]
    assert "answer key" in assignments["report-gap-req-002"].lower()
    assert "private rubric" in assignments["report-gap-req-002"].lower()
    assert "do not view or cite the named alpha study" in assignments[
        "report-gap-req-002"
    ].lower()
    assert "req-004" not in "\n".join(assignments.values())
    assert "req-005" in assignments["report-gap-req-005"]


@pytest.mark.asyncio
async def test_gap_recovery_preserves_reserve_when_deadline_is_insufficient() -> None:
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )

    runtime = _runtime(_requirement("req-001", priority=1))
    diagnostics = _RecordingDiagnostics()
    dispatch_calls = 0

    async def dispatch(_agent: AgentTaskSummary, _instruction: str) -> AgentResultRecord:
        nonlocal dispatch_calls
        dispatch_calls += 1
        return _record("unexpected")

    result = await run_gap_recovery_wave(
        runtime=runtime,
        dispatch=dispatch,
        research_role_id="dynir_report_researcher",
        evidence_llm=_EmptyEvidenceLLM(),
        question_deadline=time.monotonic() + 2,
        reserve_s=5,
        max_live_agents=1,
        diagnostics=diagnostics,  # type: ignore[arg-type]
    )

    assert result == ()
    assert dispatch_calls == 0
    assert any(
        event == "report_gap_recovery_skipped"
        and fields["reason"] == "insufficient_time_to_preserve_reserve"
        for event, fields in diagnostics.events
    )


@pytest.mark.asyncio
async def test_gap_recovery_caller_cancellation_quiesces_children_before_cutoff() -> None:
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )

    policy = replace(
        SCIENTIFIC_REPORT_POLICY,
        max_recovery_requirements=2,
        max_recovery_agents=2,
    )
    runtime = ReportRuntime(
        question="Recover two public report gaps.",
        policy=policy,
        requirements=RequirementSet(
            "scientific_report",
            [
                _requirement("req-001", priority=1),
                _requirement("req-002", priority=1),
            ],
        ),
    )
    started = [asyncio.Event(), asyncio.Event()]
    cancelled = [asyncio.Event(), asyncio.Event()]
    cleanup_finished = [asyncio.Event(), asyncio.Event()]
    continued_after_dispatch = [asyncio.Event(), asyncio.Event()]

    async def dispatch(agent: AgentTaskSummary, _instruction: str) -> AgentResultRecord:
        index = int(agent.agent_id.rsplit("-", 1)[1]) - 1
        started[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled[index].set()
            try:
                # Simulate an AgentBus abort that would exceed the report reserve
                # unless the absolute recovery cutoff bounds settlement.
                await asyncio.sleep(10)
            finally:
                cleanup_finished[index].set()
            raise
        continued_after_dispatch[index].set()
        return _record(agent.agent_id)

    loop = asyncio.get_running_loop()
    question_deadline = loop.time() + 1.2
    reserve_s = 0.1
    wave = asyncio.create_task(
        run_gap_recovery_wave(
            runtime=runtime,
            dispatch=dispatch,
            research_role_id="dynir_report_researcher",
            evidence_llm=_EmptyEvidenceLLM(),
            question_deadline=question_deadline,
            reserve_s=reserve_s,
            max_live_agents=2,
            diagnostics=_RecordingDiagnostics(),  # type: ignore[arg-type]
        )
    )
    await asyncio.gather(*(event.wait() for event in started))
    cancellation_started = loop.time()
    wave.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await wave
        elapsed = loop.time() - cancellation_started
        assert all(event.is_set() for event in cancelled)
        assert all(event.is_set() for event in cleanup_finished)
        assert elapsed < 1.35
        assert loop.time() <= question_deadline - reserve_s + 0.15
        assert not any(event.is_set() for event in continued_after_dispatch)
        assert not any(
            task.get_name().startswith("report-gap-recovery:")
            and not task.done()
            for task in asyncio.all_tasks()
        )
        await asyncio.sleep(0)
        assert not any(event.is_set() for event in continued_after_dispatch)
    finally:
        leftovers = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("report-gap-recovery:")
            and not task.done()
        ]
        for task in leftovers:
            task.cancel()
        await asyncio.sleep(0)
        for task in leftovers:
            if not task.done():
                task.cancel()
        if leftovers:
            await asyncio.gather(*leftovers, return_exceptions=True)


@pytest.mark.asyncio
async def test_gap_recovery_preserves_first_cancellation_during_child_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )

    runtime = ReportRuntime(
        question="Recover one public report gap.",
        policy=replace(
            SCIENTIFIC_REPORT_POLICY,
            max_recovery_requirements=1,
            max_recovery_agents=1,
        ),
        requirements=RequirementSet(
            "scientific_report",
            [_requirement("req-001", priority=1)],
        ),
    )
    timeline: list[str] = []

    class _OrderedDiagnostics(_RecordingDiagnostics):
        def emit(self, event: str, **fields: Any) -> None:
            timeline.append(event)
            super().emit(event, **fields)

    diagnostics = _OrderedDiagnostics()
    started = asyncio.Event()
    child_cleanup_started = asyncio.Event()
    release_survivor = asyncio.Event()
    wake_survivor = asyncio.Event()
    cancellation_count = 0
    late_observations: list[str] = []

    async def observe_records(**_kwargs: Any) -> None:
        late_observations.append("observed")

    monkeypatch.setattr(runtime, "observe_records", observe_records)

    async def dispatch(_agent: AgentTaskSummary, _instruction: str) -> AgentResultRecord:
        nonlocal cancellation_count
        started.set()
        while not release_survivor.is_set():
            try:
                await wake_survivor.wait()
            except asyncio.CancelledError:
                # Model a non-cooperative AgentBus child that suppresses both the
                # initial cancellation and the cutoff cancellation.
                cancellation_count += 1
                if cancellation_count == 1:
                    child_cleanup_started.set()
        return _record("report-gap-req-001")

    loop = asyncio.get_running_loop()
    recovery_cutoff = loop.time() + 1.05
    wave = asyncio.create_task(
        run_gap_recovery_wave(
            runtime=runtime,
            dispatch=dispatch,
            research_role_id="dynir_report_researcher",
            evidence_llm=_EmptyEvidenceLLM(),
            question_deadline=recovery_cutoff,
            reserve_s=0,
            max_live_agents=1,
            diagnostics=diagnostics,  # type: ignore[arg-type]
        )
    )
    await started.wait()
    wave.cancel("caller requested stop")
    await child_cleanup_started.wait()
    wave.cancel("later cancellation must not replace the first")
    survivor: asyncio.Task[Any] | None = None
    try:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await wave
        timeline.append("caller_observed_cancelled")

        terminal_events = [
            fields
            for event, fields in diagnostics.events
            if event == "report_gap_recovery_end"
        ]
        assert terminal_events == [
            {
                "status": "cancelled",
                "requirement_ids": ["req-001"],
                "result_count": 0,
                "timed_out_count": 1,
                "unsettled_count": 1,
            }
        ]
        assert timeline.index("report_gap_recovery_end") < timeline.index(
            "caller_observed_cancelled"
        )
        assert isinstance(cancelled.value, asyncio.CancelledError)
        assert cancelled.value.args == ("caller requested stop",)
        assert wave.cancelled()
        assert wave.cancelling() == 2
        assert cancellation_count >= 2
        assert loop.time() >= recovery_cutoff - 0.1
        assert loop.time() <= recovery_cutoff + 0.15
        assert not any(
            task.get_name() == "report-gap-recovery:cleanup" and not task.done()
            for task in asyncio.all_tasks()
        )
    finally:
        survivors = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name() == "report-gap-recovery:req-001"
            and not task.done()
        ]
        if survivors:
            survivor = survivors[0]
            release_survivor.set()
            wake_survivor.set()
            await asyncio.wait({survivor}, timeout=0.25)
        if survivor is not None and not survivor.done():
            survivor.cancel()
        if survivor is not None:
            await asyncio.gather(survivor, return_exceptions=True)

    assert late_observations == []


@pytest.mark.asyncio
async def test_successful_epoch_survives_report_observation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    diagnostics = _RecordingDiagnostics()
    _install_direct_planner(monkeypatch)
    _install_success_orchestrator(monkeypatch, requests)

    class _RuntimeProbe:
        async def refresh_ownership(self, _tree: HTNTree, _llm: Any) -> None:
            return None

        def delivery_context_for(self, _leaf: HTNTaskNode) -> str:
            return "report contract"

        async def observe_leaf(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("evidence extraction failed")

    result = await run_epoch(
        **_epoch_kwargs(tmp_path, diagnostics=diagnostics),
        report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
    )

    assert result.tree.root().status is HTNTaskStatus.COMPLETED
    assert any(
        event == "report_leaf_observation_failed"
        and fields["leaf_id"] == "root"
        and fields["error_type"] == "RuntimeError"
        for event, fields in diagnostics.events
    )


@pytest.mark.asyncio
async def test_leaf_timeout_survives_report_observation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    diagnostics = _RecordingDiagnostics()
    _install_direct_planner(monkeypatch)

    class _RuntimeProbe:
        async def refresh_ownership(self, _tree: HTNTree, _llm: Any) -> None:
            return None

        def delivery_context_for(self, _leaf: HTNTaskNode) -> str:
            return "report contract"

        async def observe_leaf(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("evidence extraction failed")

    class _TimeoutOrchestrator:
        def __init__(self, _request: Any) -> None:
            self._results = [_record("worker-before-timeout")]

        @property
        def results(self) -> tuple[AgentResultRecord, ...]:
            return tuple(self._results)

        async def run(self) -> OrchestratorResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def partial_result(self) -> OrchestratorResult:
            return OrchestratorResult("Checkpointed partial result.")

    real_wait = asyncio.wait

    async def _expire_orchestrator(awaitables, *, timeout=None):  # noqa: ANN001
        tasks = set(awaitables)
        if any(task.get_name().endswith(":root:orchestrator") for task in tasks):
            return set(), tasks
        return await real_wait(tasks, timeout=timeout)

    monkeypatch.setattr(epoch_module, "Orchestrator", _TimeoutOrchestrator)
    monkeypatch.setattr(epoch_module.asyncio, "wait", _expire_orchestrator)

    result = await run_epoch(
        **_epoch_kwargs(tmp_path, diagnostics=diagnostics),
        report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
    )

    assert [incident.leaf_id for incident in result.timeout_incidents] == ["root"]
    assert any(
        event == "report_leaf_observation_failed"
        and fields["leaf_id"] == "root"
        and fields["error_type"] == "RuntimeError"
        for event, fields in diagnostics.events
    )


@pytest.mark.asyncio
async def test_report_observation_does_not_swallow_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    _install_direct_planner(monkeypatch)
    _install_success_orchestrator(monkeypatch, requests)

    class _RuntimeProbe:
        async def refresh_ownership(self, _tree: HTNTree, _llm: Any) -> None:
            return None

        def delivery_context_for(self, _leaf: HTNTaskNode) -> str:
            return "report contract"

        async def observe_leaf(self, *_args: Any, **_kwargs: Any) -> None:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_epoch(
            **_epoch_kwargs(tmp_path),
            report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_cancelling_live_leaf_skips_post_cancel_report_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from workflows.dynamic_htn_informed_research.htn import epoch as epoch_module

    _install_direct_planner(monkeypatch)
    orchestrator_started = asyncio.Event()
    observation_called = False

    class _BlockingOrchestrator:
        def __init__(self, _request: Any) -> None:
            self._results = [_record("worker-before-cancel")]

        @property
        def results(self) -> tuple[AgentResultRecord, ...]:
            return tuple(self._results)

        async def run(self) -> OrchestratorResult:
            orchestrator_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def partial_result(self) -> OrchestratorResult:
            return OrchestratorResult("Checkpointed partial result.")

    class _RuntimeProbe:
        async def refresh_ownership(self, _tree: HTNTree, _llm: Any) -> None:
            return None

        def delivery_context_for(self, _leaf: HTNTaskNode) -> str:
            return "report contract"

        async def observe_leaf(self, *_args: Any, **_kwargs: Any) -> None:
            nonlocal observation_called
            observation_called = True

    monkeypatch.setattr(epoch_module, "Orchestrator", _BlockingOrchestrator)
    epoch_task = asyncio.create_task(
        run_epoch(
            **_epoch_kwargs(tmp_path),
            report_runtime=_RuntimeProbe(),  # type: ignore[arg-type]
        )
    )
    await orchestrator_started.wait()
    epoch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await epoch_task

    assert observation_called is False


@pytest.mark.asyncio
async def test_root_answer_context_reconstruction_preserves_delivery_contract(
    tmp_path: Path,
) -> None:
    delivery_contract = "[req-001] Preserve this exact report delivery contract."
    answer_context = "Confirmed recovery fact for root answer synthesis."
    root = HTNTaskNode(
        id="root",
        title="Report",
        task="Write the complete sourced report.",
    )
    tree = HTNTree(root_id="root", nodes={"root": root})

    class _SynthesisLLM:
        model = "root-synthesis"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            rendered = messages[-1]["content"]
            assert delivery_contract in rendered
            assert answer_context in rendered
            return _tool_response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["complete report"],
                    "evidence_summary": "Evidence supports the report.",
                    "answer": "# Complete report\n\nGrounded content.",
                    "completion_outcome": "resolved",
                },
            )

    async def root_answer_context_provider() -> str:
        return answer_context

    llm = _SynthesisLLM()
    orchestrator = Orchestrator(
        OrchestratorRequest(
            tree=tree,
            leaf=root,
            available_agents=[AgentDescriptor("dynir_report_researcher")],
            execution_budget=ExecutionBudget(max_rounds=1, max_agents_per_round=1),
            output_dir=tmp_path,
            llm=llm,
            dispatch=lambda *_args: asyncio.sleep(0),  # type: ignore[arg-type]
            task_strategy=get_task_strategy("report_research"),
            synthesis_llm=llm,
            root_answer_context_provider=root_answer_context_provider,
            delivery_context=delivery_contract,
        )
    )
    context = build_stable_context(
        tree,
        root,
        delivery_context=delivery_contract,
    )

    result = await orchestrator._synthesize_result(context)

    assert result.result_summary == "# Complete report\n\nGrounded content."



@pytest.mark.asyncio
async def test_gap_recovery_refills_free_slot_prioritizes_disputes_and_caps_total(tmp_path):
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )
    requirements = [_requirement(f'req-{i:03d}', priority=1) for i in range(1, 7)]
    disputed = _requirement('req-007', priority=5)
    disputed.reopen('The verifier disproved the model assumption.')
    requirements.append(disputed)
    runtime = ReportRuntime(
        question='Resolve the scientific requirements.',
        policy=replace(SCIENTIFIC_REPORT_POLICY, max_recovery_requirements=4, max_recovery_agents=2),
        requirements=RequirementSet('scientific_report', requirements),
    )
    third_started = asyncio.Event()
    active = 0
    max_active = 0
    dispatched = []

    async def dispatch(agent, instruction):
        nonlocal active, max_active
        dispatched.append(agent.agent_id)
        active += 1
        max_active = max(max_active, active)
        try:
            if agent.agent_id == 'report-gap-req-007':
                assert 'disproved the model assumption' in instruction
                # Requires the free slot to be reused while this task remains
                # active: a batch barrier would deadlock this fixture.
                await asyncio.wait_for(third_started.wait(), timeout=1)
            if len(dispatched) == 3:
                third_started.set()
            await asyncio.sleep(0)
            return _record(agent.agent_id)
        finally:
            active -= 1

    records = await run_gap_recovery_wave(
        runtime=runtime, dispatch=dispatch, research_role_id='dynir_report_researcher',
        evidence_llm=_EmptyEvidenceLLM(), question_deadline=time.monotonic() + 5,
        reserve_s=1, max_live_agents=2, diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert dispatched == ['report-gap-req-007', 'report-gap-req-001',
                          'report-gap-req-002', 'report-gap-req-003']
    assert [record.agent_id for record in records] == dispatched
    assert max_active == 2
    # Empty evidence leaves gaps unresolved; that must not trigger an endless retry.
    assert len({record.agent_id for record in records}) == 4


@pytest.mark.asyncio
async def test_gap_recovery_deadline_cancels_active_worker_without_starting_queue(tmp_path):
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import (
        run_gap_recovery_wave,
    )
    runtime = _runtime(*[_requirement(f'req-{i:03d}', priority=1) for i in range(3)])
    dispatched, cancelled = [], []

    async def dispatch(agent, _instruction):
        dispatched.append(agent.agent_id)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(agent.agent_id)

    records = await run_gap_recovery_wave(
        runtime=runtime, dispatch=dispatch, research_role_id='dynir_report_researcher',
        evidence_llm=_EmptyEvidenceLLM(), question_deadline=time.monotonic() + 1.25,
        reserve_s=.15, max_live_agents=1, diagnostics=DiagnosticRecorder(tmp_path),
    )
    assert records == ()
    assert dispatched == cancelled == ['report-gap-req-000']
