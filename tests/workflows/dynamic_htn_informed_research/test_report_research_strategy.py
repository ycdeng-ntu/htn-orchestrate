from __future__ import annotations

import builtins
import importlib
import json
import sys

import pytest

from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    _parse_root_synthesis_response,
    _synthesize,
    build_stable_context,
)
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy
from workflows.dynamic_htn_informed_research.task_strategies.research import (
    RESEARCH_STRATEGY,
)


def test_report_strategy_owns_distinct_prompt_constants_and_roles() -> None:
    strategy = get_task_strategy("report_research")

    assert strategy.name == "report_research"
    assert strategy.delivery.output_mode == "report"
    assert strategy.coordination.primary_role_id == "dynir_report_researcher"
    assert strategy.coordination.verifier_role_id == "dynir_report_verifier"
    assert set(strategy.workers.role_ids) == {
        "dynir_report_researcher",
        "dynir_report_verifier",
        "dynir_report_final_verifier",
    }
    assert strategy.planning.decomposition_system != RESEARCH_STRATEGY.planning.decomposition_system
    assert strategy.synthesis.synthesis_system != RESEARCH_STRATEGY.synthesis.synthesis_system
    assert {
        definition.role_id for definition in strategy.workers.agent_definitions
    } == set(strategy.workers.role_ids)


def test_report_prompts_require_evidence_rich_coverage_memos() -> None:
    strategy = get_task_strategy("report_research")
    prompt_bundle = "\n".join(
        (
            strategy.planning.decomposition_system,
            strategy.coordination.strategic_decision_system,
            strategy.instruction.generation_system,
            *(definition.system_prompt for definition in strategy.workers.agent_definitions),
            strategy.recovery.finalize_gate_system,
            strategy.synthesis.synthesis_system,
            strategy.synthesis.tree_root_system,
        )
    ).lower()

    assert "coverage" in prompt_bundle
    assert "evidence memo" in prompt_bundle
    assert "source url" in prompt_bundle
    assert "enumerat" in prompt_bundle
    assert "table cell" in prompt_bundle
    assert "sub-question" in prompt_bundle
    assert "equation" in prompt_bundle
    assert "named method" in prompt_bundle
    assert "assumption" in prompt_bundle
    assert "scientific specific" in prompt_bundle


def test_report_prompts_forbid_private_benchmark_material_and_fabricated_citations() -> None:
    strategy = get_task_strategy("report_research")
    prompt_bundle = "\n".join(
        definition.system_prompt for definition in strategy.workers.agent_definitions
    ).lower()

    assert "benchmark leakage" in prompt_bundle
    assert "answer-key searches" in prompt_bundle
    assert "rubric access" in prompt_bundle
    assert "fabricated citations" in prompt_bundle


def test_report_workers_expose_only_controlled_research_and_submission_tools() -> None:
    strategy = get_task_strategy("report_research")

    for definition in strategy.workers.agent_definitions:
        assert set(definition.allowed_tools) == {
            "web_search",
            "web_fetch",
            "scholar_search",
            "run_python_code",
            "read_file",
            "submit_report",
        }


def test_report_package_has_no_runtime_adaptive_htn_imports(monkeypatch) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        imported.append(name)
        if name.startswith("workflows.adaptive_htn"):
            raise AssertionError(f"forbidden runtime import: {name}")
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name.startswith(
            "workflows.dynamic_htn_informed_research.task_strategies.report_research"
        ):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module(
        "workflows.dynamic_htn_informed_research.task_strategies.report_research"
    )

    assert module.REPORT_RESEARCH_STRATEGY.name == "report_research"
    assert not any(name.startswith("workflows.adaptive_htn") for name in imported)


def test_existing_research_contract_remains_short_answer() -> None:
    assert get_task_strategy("research").delivery.output_mode == "short_answer"


def test_report_root_parser_preserves_multiline_report_content() -> None:
    strategy = get_task_strategy("report_research")
    report = """# Findings

Primary evidence supports the result [1].

| Field | Value |
| --- | --- |
| Coverage | Complete |

## References

1. https://example.test/source
""".strip()
    response = LLMResponse(
        content="",
        tool_calls=[
            {
                "function": {
                    "name": "finalize_root_answer",
                    "arguments": json.dumps(
                        {
                            "requested_output_fields": ["complete report"],
                            "evidence_summary": "Internal synthesis note.",
                            "answer": report,
                            "completion_outcome": "resolved",
                        }
                    ),
                }
            }
        ],
    )

    parsed = _parse_root_synthesis_response(
        response,
        fallback="fallback",
        strategy=strategy.synthesis,
    )

    assert parsed.result_summary == report


@pytest.mark.asyncio
async def test_report_root_synthesis_accepts_multiline_report_without_retry(
    tmp_path,
) -> None:
    root = HTNTaskNode(id="root", task="Write the complete sourced report.")
    tree = HTNTree(root_id="root", nodes={"root": root})
    report = "# Report\n\nComplete section one.\n\n## Sources\n\n- https://example.test"
    response = LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "finalize_root_answer",
                    "arguments": json.dumps(
                        {
                            "requested_output_fields": ["complete report"],
                            "evidence_summary": "Internal synthesis note.",
                            "answer": report,
                            "completion_outcome": "resolved",
                        }
                    ),
                }
            }
        ]
    )

    class _OneResponseLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            self.calls += 1
            return response

    llm = _OneResponseLLM()
    result = await _synthesize(
        build_stable_context(tree, root),
        [],
        tree,
        llm,
        tmp_path,
        strategy=get_task_strategy("report_research").synthesis,
    )

    assert llm.calls == 1
    assert result.result_summary == report
