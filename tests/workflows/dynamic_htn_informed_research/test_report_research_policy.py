from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy
from workflows.dynamic_htn_informed_research.task_strategies.contracts import (
    DeliveryStrategy,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
)


def test_report_policies_are_immutable_delivery_contracts() -> None:
    assert SCIENTIFIC_REPORT_POLICY.citations_required is False
    assert SCIENTIFIC_REPORT_POLICY.enumerative_coverage is False
    assert SCIENTIFIC_REPORT_POLICY.scientific_delivery is True

    with pytest.raises(FrozenInstanceError):
        SCIENTIFIC_REPORT_POLICY.citations_required = True


def test_fsr_policy_carries_specificity_delivery_contract() -> None:
    # Keep part-level granularity and scientific specificity for all report tasks.
    assert SCIENTIFIC_REPORT_POLICY.max_requirements == 24
    assert "one requirement per part" in SCIENTIFIC_REPORT_POLICY.requirement_prompt

    for gate in ("GENERIC", "UNCOMMITTED", "DROPPED FROM FINDINGS"):
        assert gate in SCIENTIFIC_REPORT_POLICY.verification_prompt
    assert "wrong species" in SCIENTIFIC_REPORT_POLICY.verification_prompt

    directive = SCIENTIFIC_REPORT_POLICY.delivery_directive
    assert "Develop every requested part fully" in directive
    assert "Depth over breadth" in directive
    assert "Carry every specific result" in directive
    # Specificity must not force certainty when the problem is underdetermined.
    assert "flag the uncertainty" in directive
    assert "Do not force a unique answer" in directive
    assert "conditional result" in directive
    assert "unsupported parts as uncovered" in directive
    assert "commit to the best-supported" not in directive
    assert "line by line" in directive
    assert len(directive) <= 2_000



def test_existing_strategy_delivery_defaults_to_short_answer() -> None:
    delivery = get_task_strategy("research").delivery

    assert delivery == DeliveryStrategy()
    assert delivery.output_mode == "short_answer"
    assert not hasattr(delivery, "report_policy_resolver")
