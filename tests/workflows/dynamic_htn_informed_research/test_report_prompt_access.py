"""Capability rendering and premise verification contracts across report stages."""
from dataclasses import replace

import pytest

from workflows.dynamic_htn_informed_research.task_strategies.report_research import (
    REPORT_RESEARCH_STRATEGY, report_strategy_for_access,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research import finalize
from workflows.dynamic_htn_informed_research.task_strategies.report_research.gap_recovery import _render_assignment
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import SCIENTIFIC_REPORT_POLICY
from workflows.dynamic_htn_informed_research.task_strategies.report_research.prompt_access import render_web_sections
from workflows.dynamic_htn_informed_research.task_strategies.report_research.requirements import Requirement, RequirementSet
from workflows.dynamic_htn_informed_research.task_strategies.report_research.runtime import ReportRuntime


@pytest.mark.parametrize("network_enabled", [False, True])
def test_recovery_and_verifier_prompts_preserve_premise_review(network_enabled):
    strategy = report_strategy_for_access(network_enabled=network_enabled)
    verifier = strategy.workers.agent_definitions[1].system_prompt
    assert "Separate\nexplicit question facts from upstream hypotheses" in verifier
    assert "Reopen each load-bearing identity" in verifier
    assert "premise verdict" in verifier
    assert "weights or model" in verifier
    assert "not claim a blind review" in verifier
    assert "reopen load-bearing identities" in strategy.coordination.strategic_decision_system
    assert "Never freeze an upstream answer" in strategy.instruction.generation_system
    assert "reopen\nupstream premises" in strategy.instruction.scope_boundary
    target = Requirement(id="R1", description="Derive the result.",
                         acceptance_criteria=("Justify the model.",), priority=1, kind="fact")
    policy = replace(SCIENTIFIC_REPORT_POLICY, network_enabled=network_enabled)
    # User data that resembles template syntax must be preserved literally.
    question = "Explain [[WEB]]literal text[[/WEB]] and https://provided.example/equation."
    runtime = ReportRuntime(question=question, policy=policy,
                            requirements=RequirementSet(policy.key, [target]))
    recovery = _render_assignment(runtime, target)
    assert question in recovery
    assert "Reopen any upstream premise" in recovery
    final_system = finalize._messages(finalize._VERIFY_SYSTEM,
                                      {"policy": {"network_enabled": network_enabled}})[0]["content"]
    assert "upstream identities" in final_system
    assert "Use research repair" in final_system
    for text in (recovery, verifier, final_system,
                 strategy.recovery.malformed_trace_system,
                 strategy.recovery.supported_facts_system,
                 strategy.synthesis.synthesis_system):
        assert ("URLs" in text) == network_enabled
        assert ("verbatim" in text) == network_enabled
        assert ("Work offline" in text) != network_enabled


def test_prompt_instances_do_not_mutate_shared_worker_definitions_or_tools():
    first = report_strategy_for_access(network_enabled=False)
    second = report_strategy_for_access(network_enabled=True)
    first.workers.agent_definitions[0].allowed_tools.clear()
    first.planning.decompose_tool["function"]["description"] = "changed"
    assert second.workers.agent_definitions[0].allowed_tools
    assert REPORT_RESEARCH_STRATEGY.workers.agent_definitions[0].allowed_tools
    assert second.planning.decompose_tool["function"]["description"] != "changed"
    assert "Work offline" not in REPORT_RESEARCH_STRATEGY.workers.agent_definitions[0].system_prompt


@pytest.mark.parametrize("template", ["[[WEB]]oops", "oops[[/WEB]]", "[[OFFLINE]]oops"])
def test_broken_authored_templates_fail_before_dispatch(template):
    with pytest.raises(ValueError, match="Malformed"):
        render_web_sections(template, network_enabled=False)
