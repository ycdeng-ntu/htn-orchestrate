from __future__ import annotations

from pathlib import Path

from workflows.dynamic_htn_informed_research.task_strategies.registry import (
    get_task_strategy,
)


MCQ_PACKAGE = Path(__file__).parents[3] / "workflows" / "dynamic_htn_informed_research" / "task_strategies" / "science_mcq"


def test_science_mcq_has_no_fso_source_dependency() -> None:
    source = "\n".join(path.read_text() for path in sorted(MCQ_PACKAGE.glob("*.py")))

    assert "science_olympiad" not in source
    assert "SCIENCE_OLYMPIAD" not in source
    assert "frontierscience" not in source.lower()


def test_science_mcq_owns_legacy_direct_semantics() -> None:
    strategy = get_task_strategy("science_mcq")
    science = get_task_strategy("science_olympiad")

    assert strategy.name == "science_mcq"
    for section in (
        "planning",
        "coordination",
        "instruction",
        "workers",
        "recovery",
        "synthesis",
    ):
        assert getattr(strategy, section) is not getattr(science, section)
    assert strategy.planning.decompose_tool is not science.planning.decompose_tool
    assert strategy.planning.execute_direct_tool is not science.planning.execute_direct_tool
    assert strategy.planning.audit_subtask_tool is not science.planning.audit_subtask_tool
    assert strategy.planning.revise_subtask_tool is not science.planning.revise_subtask_tool
    assert strategy.recovery.finalize_gate_tool is not science.recovery.finalize_gate_tool
    assert strategy.recovery.supported_facts_tool is not science.recovery.supported_facts_tool
    assert strategy.recovery.answer_annotation_tool is not science.recovery.answer_annotation_tool
    assert strategy.synthesis.forced_audit_tool is not science.synthesis.forced_audit_tool
    assert strategy.planning.minimum_decomposition_children == 1
    assert not strategy.planning.include_sibling_task_semantics_in_audit
    assert not strategy.planning.retry_overlong_subtask_revision
    assert not strategy.coordination.precursor_evidence_unlocks_verifier
    assert not strategy.synthesis.include_child_semantics

    assert "Decomposition children are jointly required deliverables" in (
        strategy.planning.decomposition_system
    )
    assert "First classify the current leaf" not in strategy.coordination.strategic_decision_system
    assert "Allocated route:" not in strategy.coordination.strategic_decision_system
    assert "Reconciliation:" not in strategy.coordination.strategic_decision_system
    assert "route portfolio" not in strategy.synthesis.synthesis_system.lower()
    assert "route portfolio" not in strategy.synthesis.tree_root_system.lower()


def test_science_mcq_keeps_its_mcq_contract() -> None:
    strategy = get_task_strategy("science_mcq")

    assert strategy.coordination.primary_role_id == "dynir_science_mcq_solver"
    assert strategy.coordination.verifier_role_id == "dynir_science_mcq_checker"
    assert strategy.instruction.assignment_only_composition
    assert strategy.synthesis.root_answer_pattern == r"(?:[Oo]ption\s+)?\(?([A-Z])\)?"
    assert "Multiple-choice root contract" in strategy.synthesis.root_direct_system
    assert tuple(agent.role_id for agent in strategy.workers.agent_definitions) == (
        "dynir_science_mcq_solver",
        "dynir_science_mcq_checker",
    )


def test_offline_mcq_rendering_does_not_mutate_online_strategy() -> None:
    from workflows.dynamic_htn_informed_research.task_strategies.science_mcq import (
        mcq_strategy_for_access,
    )

    online = get_task_strategy("science_mcq")
    offline = mcq_strategy_for_access(network_enabled=False)
    assert "Use external tools only" not in offline.workers.agent_definitions[0].system_prompt
    assert "External lookup is allowed" not in offline.coordination.strategic_decision_system
    assert "or external lookup" not in offline.instruction.solving_protocol
    for definition in offline.workers.agent_definitions:
        assert "Do not request external retrieval" in definition.system_prompt
    offline.workers.agent_definitions[0].allowed_tools.clear()
    offline.planning.decompose_tool["function"]["description"] = "changed"
    assert online == mcq_strategy_for_access(network_enabled=True)
    assert online.workers.agent_definitions[0].allowed_tools
    assert online.planning.decompose_tool["function"]["description"] != "changed"
    assert "Use external tools only" in online.workers.agent_definitions[0].system_prompt
    assert "External lookup is allowed" in online.coordination.strategic_decision_system
    assert "or external lookup" in online.instruction.solving_protocol
