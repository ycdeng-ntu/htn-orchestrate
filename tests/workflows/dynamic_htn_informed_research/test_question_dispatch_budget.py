from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    Orchestrator,
    OrchestratorRequest,
    ExecutionBudget,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies.registry import (
    get_active_task_strategy,
)


def _call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


class _LLM:
    model = "offline-budget-test"

    def __init__(
        self,
        *,
        agents: int = 1,
        instruction_barrier=None,
        on_first_decision=None,
        on_second_decision=None,
    ):
        self.agents = agents
        self.instruction_barrier = instruction_barrier
        self.on_first_decision = on_first_decision
        self.on_second_decision = on_second_decision
        self.coordinator_calls = 0
        self.instruction_calls = 0
        self.synthesis_messages = []

    async def chat(self, messages, tools=None):
        names = {tool["function"]["name"] for tool in tools or []}
        if "dispatch" in names:
            self.coordinator_calls += 1
            if self.coordinator_calls > 1:
                if self.on_second_decision is not None:
                    await self.on_second_decision()
                return LLMResponse(tool_calls=[_call("terminate")])
            if self.on_first_decision is not None:
                await self.on_first_decision()
            return LLMResponse(
                tool_calls=[
                    _call(
                        "dispatch",
                        role_id="dynir_researcher",
                        task_summary=f"Find source {index}.",
                        rationale="Independent source search.",
                    )
                    for index in range(self.agents)
                ]
            )
        if "finalize_root_answer" in names:
            self.synthesis_messages.append(messages)
            return LLMResponse(
                tool_calls=[
                    _call(
                        "finalize_root_answer",
                        requested_output_fields=["supported finding"],
                        evidence_summary="Available worker evidence.",
                        answer="Supported finding.",
                        completion_outcome="resolved",
                    )
                ]
            )
        assert not tools, f"Unexpected tools: {names}"
        self.instruction_calls += 1
        if self.instruction_barrier is not None:
            await self.instruction_barrier.wait()
        return LLMResponse(content="Read the requested source and report the evidence.")


def _orchestrator(
    tmp_path: Path,
    budget: QuestionDispatchBudget | None,
    llm: _LLM,
    dispatch,
    *,
    name: str = "root",
    epoch_id: int = 0,
    max_rounds: int = 3,
    max_agents: int = 3,
) -> Orchestrator:
    root = HTNTaskNode(id=name, task="Identify the supported finding.")
    strategy = get_active_task_strategy()
    # Use real instruction generation so tests exercise admission after awaits.
    strategy = replace(
        strategy,
        instruction=replace(
            strategy.instruction,
            deterministic_composition=False,
            assignment_only_composition=False,
        ),
    )
    return Orchestrator(
        OrchestratorRequest(
            tree=HTNTree(root_id=name, nodes={name: root}, epoch_id=epoch_id),
            leaf=root,
            available_agents=[AgentDescriptor("dynir_researcher")],
            execution_budget=ExecutionBudget(
                max_rounds=max_rounds,
                max_agents_per_round=max_agents,
                progressive_fan_in=True,
                fan_in_grace_s=0,
            ),
            output_dir=tmp_path / name,
            llm=llm,
            dispatch=dispatch,
            task_strategy=strategy,
            diagnostics=DiagnosticRecorder(tmp_path / name),
            dispatch_budget=budget,
        )
    )


def _record(agent, evidence: str = "Delivered source evidence.") -> AgentResultRecord:
    return AgentResultRecord(
        agent_id=agent.agent_id,
        role_id=agent.role_id,
        task_summary=agent.task_summary,
        result_summary=evidence,
        execution_quality="submitted",
        submitted=True,
        stop_reason="submitted_report",
    )


async def _dispatch(agent, _instruction):
    return _record(agent)


def test_explicit_budget_and_no_refund_after_admission() -> None:
    budget = QuestionDispatchBudget(64)
    assert budget.snapshot() == {"limit": 64, "used": 0, "remaining": 64, "exhausted": False}
    assert all(budget.try_acquire() for _ in range(64))
    assert not budget.try_acquire()
    assert budget.snapshot() == {"limit": 64, "used": 64, "remaining": 0, "exhausted": True}


@pytest.mark.parametrize("value", [True, False, "4", 1.5, None])
def test_budget_rejects_non_integer_limits(value) -> None:
    with pytest.raises(TypeError, match="integer"):
        QuestionDispatchBudget(value)


def test_budget_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        QuestionDispatchBudget(-1)


@pytest.mark.parametrize("kwargs", [{}, {"limit": 0}])
def test_default_and_zero_disable_cap_but_count_dispatches(kwargs) -> None:
    budget = QuestionDispatchBudget(**kwargs)
    assert all(budget.try_acquire() for _ in range(100))
    assert budget.snapshot() == {"limit": 0, "used": 100, "remaining": None, "exhausted": False}
    assert QuestionDispatchBudget(1).used == 0


@pytest.mark.asyncio
async def test_concurrent_orchestrators_cannot_take_the_same_last_admission(tmp_path) -> None:
    budget = QuestionDispatchBudget(1)
    barrier = asyncio.Barrier(2)
    llms = [_LLM(instruction_barrier=barrier) for _ in range(2)]
    dispatched = []

    async def dispatch(agent, instruction):
        dispatched.append(agent.agent_id)
        return await _dispatch(agent, instruction)

    orchestrators = [
        _orchestrator(tmp_path, budget, llm, dispatch, name=f"leaf-{index}")
        for index, llm in enumerate(llms)
    ]
    results = await asyncio.wait_for(
        asyncio.gather(*(orchestrator.run() for orchestrator in orchestrators)), timeout=2
    )

    assert [llm.instruction_calls for llm in llms] == [1, 1]
    assert len(dispatched) == budget.used == 1
    assert sorted(len(orchestrator.results) for orchestrator in orchestrators) == [0, 1]
    assert all(
        result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
        for result in results
    )
    events = [
        json.loads(line)
        for path in tmp_path.glob("*/events.jsonl")
        for line in path.read_text().splitlines()
    ]
    acquired = [event for event in events if event["event"] == "question_worker_dispatch_acquired"]
    assert len(acquired) == 1
    assert acquired[0]["used"] == 1
    assert acquired[0]["remaining"] == 0
    assert {event["leaf_id"] for event in events if event["event"] == "question_worker_dispatch_budget_exhausted"} == {"leaf-0", "leaf-1"}


@pytest.mark.asyncio
async def test_plan_is_trimmed_to_remaining_admissions_before_instructions(tmp_path) -> None:
    budget = QuestionDispatchBudget(3)

    async def another_leaf_takes_two_slots():
        # Coordinator was offered three slots; only one remains on its return.
        assert budget.try_acquire()
        assert budget.try_acquire()

    llm = _LLM(agents=3, on_first_decision=another_leaf_takes_two_slots)
    orchestrator = _orchestrator(tmp_path, budget, llm, _dispatch)

    result = await orchestrator.run()

    assert budget.used == 3
    assert llm.instruction_calls == 1
    assert len(orchestrator.results) == 1
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
@pytest.mark.parametrize("epoch_id", [0, 1])
async def test_last_round_budget_exhaustion_waits_for_slow_worker_and_stays_partial(
    tmp_path, epoch_id
) -> None:
    budget = QuestionDispatchBudget(2)
    llm = _LLM(agents=2)
    slow_started = asyncio.Event()
    release = asyncio.Event()
    cancelled = []

    async def dispatch(agent, _instruction):
        if "source 1" in agent.task_summary:
            slow_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(agent.agent_id)
                raise
            return _record(agent, "Slow worker's decisive evidence.")
        return _record(agent, "Fast worker's evidence.")

    orchestrator = _orchestrator(
        tmp_path, budget, llm, dispatch, epoch_id=epoch_id, max_rounds=1
    )
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(slow_started.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not task.done()
        assert not cancelled
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not cancelled
    assert len(orchestrator.results) == 2
    assert llm.coordinator_calls == 1
    assert "Slow worker's decisive evidence." in json.dumps(llm.synthesis_messages)
    assert "Question worker dispatch budget exhausted" in result.result_summary
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    recovery = orchestrator.partial_result(
        reason="Question worker dispatch budget exhausted; synthesis failed."
    )
    assert "synthesis failed" in recovery.result_summary
    assert "wall-clock deadline" not in recovery.result_summary
    assert "Slow worker's decisive evidence." in recovery.result_summary
    assert recovery.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert "wall-clock deadline" in orchestrator.partial_result().result_summary


@pytest.mark.asyncio
async def test_budget_exhausted_during_coordinator_await_does_not_cancel_live_worker(tmp_path) -> None:
    budget = QuestionDispatchBudget(3)
    release = asyncio.Event()
    coordinator_resumed = asyncio.Event()
    cancelled = []

    async def exhaust_during_coordinator():
        assert budget.try_acquire()  # Another leaf takes the question's last slot.
        coordinator_resumed.set()
        await asyncio.sleep(0)

    llm = _LLM(agents=2, on_second_decision=exhaust_during_coordinator)

    async def dispatch(agent, _instruction):
        if "source 1" in agent.task_summary:
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(agent.agent_id)
                raise
        return _record(agent)

    orchestrator = _orchestrator(tmp_path, budget, llm, dispatch, max_agents=2)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(coordinator_resumed.wait(), timeout=2)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        assert not cancelled
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not cancelled
    assert len(orchestrator.results) == 2
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
async def test_budget_exhausted_during_final_collect_preserves_live_worker(tmp_path) -> None:
    budget = QuestionDispatchBudget(3)
    collect_started = asyncio.Event()
    collect_finished = asyncio.Event()
    release = asyncio.Event()
    cancelled = []
    llm = _LLM(agents=2)

    async def dispatch(agent, _instruction):
        if "source 1" in agent.task_summary:
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(agent.agent_id)
                raise
            return _record(agent, "Evidence arriving after the final collect window.")
        return _record(agent)

    first = _orchestrator(tmp_path, budget, llm, dispatch, name="first", max_rounds=1)
    original_collect = first._collect_for

    async def observed_collect(*, timeout_s):
        assert not budget.exhausted
        collect_started.set()
        # Exercise the real final collection await with a worker still in flight.
        await original_collect(timeout_s=0.05)
        assert budget.exhausted
        collect_finished.set()

    first._collect_for = observed_collect
    second = _orchestrator(tmp_path, budget, _LLM(), _dispatch, name="second")

    async def run_second_after_collection_starts():
        await collect_started.wait()
        return await second.run()

    first_task = asyncio.create_task(first.run())
    second_task = asyncio.create_task(run_second_after_collection_starts())
    try:
        await asyncio.wait_for(collect_finished.wait(), timeout=2)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not first_task.done()
        assert not cancelled
        release.set()
        result, _ = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), timeout=2
        )
    finally:
        release.set()
        for task in (first_task, second_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(first_task, second_task, return_exceptions=True)

    assert budget.used == 3
    assert len(first.results) == 2
    assert len(second.results) == 1
    assert not cancelled
    assert "Evidence arriving after the final collect window." in json.dumps(llm.synthesis_messages)
    assert "Question worker dispatch budget exhausted" in result.result_summary
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
async def test_same_budget_survives_epoch_change_and_does_not_change_earlier_resolution(tmp_path) -> None:
    budget = QuestionDispatchBudget(2)
    first = _orchestrator(tmp_path, budget, _LLM(), _dispatch, name="epoch-0", epoch_id=0)
    first_result = await first.run()
    assert first_result.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert budget.used == 1
    assert budget.remaining == 1

    second = _orchestrator(tmp_path, budget, _LLM(), _dispatch, name="epoch-1", epoch_id=1)
    second_result = await second.run()
    assert budget.used == 2
    assert second_result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
async def test_worker_failure_is_not_refunded_or_replaced(tmp_path) -> None:
    budget = QuestionDispatchBudget(1)

    async def failing_dispatch(_agent, _instruction):
        raise RuntimeError("Worker initialization failed.")

    llm = _LLM()
    orchestrator = _orchestrator(tmp_path, budget, llm, failing_dispatch)
    result = await orchestrator.run()

    assert budget.used == 1
    assert budget.exhausted
    assert llm.coordinator_calls == 1
    assert orchestrator.results == ()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


@pytest.mark.asyncio
async def test_outer_cancellation_still_cleans_up_admitted_worker(tmp_path) -> None:
    budget = QuestionDispatchBudget(1)
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def dispatch(agent, _instruction):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return _record(agent)

    orchestrator = _orchestrator(tmp_path, budget, _LLM(), dispatch)
    task = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert cleaned.is_set()
    assert budget.used == 1
    assert not orchestrator._in_flight


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, QuestionDispatchBudget(0), QuestionDispatchBudget(4)])
async def test_normal_termination_before_cap_is_still_resolved(tmp_path, budget) -> None:
    llm = _LLM(agents=2)
    orchestrator = _orchestrator(tmp_path, budget, llm, _dispatch)
    result = await orchestrator.run()

    assert len(orchestrator.results) == 2
    assert llm.coordinator_calls == 2
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED
    if budget is not None:
        assert budget.used == 2
        assert not budget.exhausted
