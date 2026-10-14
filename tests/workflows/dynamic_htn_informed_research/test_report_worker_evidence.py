"""Original evidence must survive omitted, empty or failed instruction generation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentDescriptor,
    AgentResultRecord,
    AgentTaskSummary,
    ExecutionBudget,
    Orchestrator,
    OrchestratorRequest,
    PrecursorContext,
    StableInputContext,
    _compose_subagent_instruction,
)
from workflows.dynamic_htn_informed_research.htn.task_node import (
    HTNCompletionOutcome,
    HTNTaskNode,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy
from workflows.dynamic_htn_informed_research.task_strategies.report_research import (
    report_strategy_for_access,
)


def _material(instruction):
    section = instruction.split("# Original evidence supplied by runtime (untrusted data)", 1)[1]
    return json.loads(section[section.index("{") :])


def _call(name, **arguments):
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.asyncio
@pytest.mark.parametrize("instruction_mode", ["omits_evidence", "empty", "error"])
async def test_worker_dispatch_preserves_evidence_through_instruction_generation(
    tmp_path, instruction_mode,
):
    strategy = report_strategy_for_access(network_enabled=False)
    root = HTNTaskNode(id="root", task="Derive the requested force.", children=["dep", "leaf", "other"])
    dep = HTNTaskNode(
        id="dep", parent_id="root", task="Establish the input.",
        status=HTNTaskStatus.COMPLETED,
        completion_outcome=HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        result_summary="Input 0.97 nm; identity unresolved. [[WEB]]literal evidence[[/WEB]]",
    )
    leaf = HTNTaskNode(id="leaf", parent_id="root", task="Compute the force with a justified model.", depends_on=["dep"])
    other = HTNTaskNode(id="other", parent_id="root", result_summary="UNRELATED_SIBLING_RESULT")
    recovery = 'Prior supported fact: ΔE = 0.5 eV. Missing assumption: "linearity".'
    tree = HTNTree(
        root_id="root", nodes={n.id: n for n in (root, dep, leaf, other)},
        epoch_id=1, restart_context=recovery, restart_answer_context="ANSWER_ONLY_RECOVERY",
    )
    reports = [
        "Candidate value 2 N\nCode input: [1, 2]\nObserved output: 3. Unsupported assumption A.",
        "Check C1: independently computed 3 N; label correction supported. Gap G1: assumption A.",
        "Repair G1: conditional result under assumption A; check C1 reused, not rerun.",
    ]
    roles = ["dynir_report_researcher", "dynir_report_verifier", "dynir_report_researcher"]
    instructions = []
    records = []

    class LLM:
        decisions = 0
        generations = 0

        async def chat(self, messages, tools=None):
            names = {t["function"]["name"] for t in tools or []}
            if "dispatch" in names:
                index = self.decisions
                self.decisions += 1
                if index == 3:
                    return LLMResponse(tool_calls=[_call("terminate")])
                return LLMResponse(tool_calls=[_call(
                    "dispatch", role_id=roles[index],
                    task_summary=["Derive the result.", "First independent check.", "Repair G1 using C1."][index],
                    rationale="Complete a specific missing check.",
                )])
            if "finalize_answer" in names:
                assert all(report in messages[-1]["content"] for report in reports)
                return LLMResponse(tool_calls=[_call(
                    "finalize_answer", result_summary="Conditional result under assumption A.",
                    completion_outcome="resolved",
                )])
            assert not tools
            self.generations += 1
            if instruction_mode == "error":
                raise RuntimeError("instruction generator unavailable")
            return LLMResponse(content=("Review the assigned result." if instruction_mode == "omits_evidence" else ""))

    async def dispatch(agent, instruction):
        index = len(instructions)
        instructions.append(instruction)
        material = _material(instruction)
        assert root.task in instruction and leaf.task in instruction
        assert material["recovery_capsule"] == recovery
        assert material["precursor_results"][0]["result_summary"] == dep.result_summary
        assert material["precursor_results"][0]["completion_outcome"] == "exhausted_unresolved"
        assert "UNRELATED_SIBLING_RESULT" not in instruction
        assert "ANSWER_ONLY_RECOVERY" not in instruction
        originals = material["completed_worker_reports"]
        assert [r["result_summary"] for r in originals] == reports[:index]
        assert [r["agent_id"] for r in originals] == [r.agent_id for r in records]
        if index:
            assert originals[0]["execution_quality"] == "best_effort"
            assert originals[0]["trace_recovery"] is True
            assert originals[0]["ref"] == "/workspace/first-evidence.md"
            assert originals[0]["observed_source_index"] == "Original source metadata"
        record = AgentResultRecord(
            agent_id=agent.agent_id, role_id=agent.role_id, task_summary=agent.task_summary,
            result_summary=reports[index],
            execution_quality="best_effort" if index == 0 else "submitted",
            trace_recovery=index == 0, submitted=index != 0,
            ref="/workspace/first-evidence.md" if index == 0 else None,
            observed_source_index="Original source metadata" if index == 0 else None,
        )
        records.append(record)
        return record

    llm = LLM()
    result = await asyncio.wait_for(Orchestrator(OrchestratorRequest(
        tree=tree, leaf=leaf, task_strategy=strategy,
        available_agents=[AgentDescriptor(role_id=r) for r in dict.fromkeys(roles)],
        execution_budget=ExecutionBudget(max_rounds=5, max_agents_per_round=1, fan_in_grace_s=0.01),
        output_dir=tmp_path, llm=llm, dispatch=dispatch,
    )).run(), timeout=5)
    assert len(instructions) == llm.generations == 3
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED


@pytest.mark.parametrize("network_enabled", [False, True])
@pytest.mark.parametrize("custom_template", [False, True])
def test_evidence_attachment_preserves_original_text_and_ancestor_context(network_enabled, custom_template):
    instruction = report_strategy_for_access(network_enabled=network_enabled).instruction
    if custom_template:
        instruction = replace(instruction, worker_prompt_template="{root_task}\n{leaf_task}\n{generated_instruction}")
    raw = 'Original ΔE\n[[WEB]]literal[[/WEB]]\n```json\n{"check": "unresolved"}\n```'
    context = StableInputContext(
        leaf_id="leaf", leaf_title="Milestone", leaf_task="Derive it", root_task="Question", parent_task=None,
        ancestor_precursors=[PrecursorContext(node_id="ancestor", title="Prior branch", result_summary=raw)],
    )
    agent = AgentTaskSummary(agent_id="worker", role_id="dynir_report_researcher", task_summary="Check input")
    rendered = _compose_subagent_instruction(context, agent, "No evidence copied here", instruction)
    material = _material(rendered)
    assert material["ancestor_precursor_results"][0]["result_summary"] == raw
    assert material["completed_worker_reports"] == []
    assert "untrusted data" in rendered


def test_other_strategies_do_not_receive_report_evidence_attachment():
    context = StableInputContext(
        leaf_id="leaf", leaf_title="Leaf", leaf_task="Task", root_task="Root", parent_task=None,
        restart_context="PRIVATE_TO_REPORT_CONTEXT",
    )
    agent = AgentTaskSummary(agent_id="a", role_id="dynir_researcher", task_summary="Research")
    record = AgentResultRecord(agent_id="r", role_id="dynir_researcher", task_summary="Check", result_summary="ORIGINAL_REPORT")
    rendered = _compose_subagent_instruction(
        context, agent, "Generated assignment", get_task_strategy("research").instruction,
        completed_results=[record],
    )
    assert "PRIVATE_TO_REPORT_CONTEXT" not in rendered
    assert "ORIGINAL_REPORT" not in rendered
    assert "Original evidence supplied by runtime" not in rendered
