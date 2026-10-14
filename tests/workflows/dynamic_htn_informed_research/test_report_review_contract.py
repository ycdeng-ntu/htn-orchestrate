"""Review provenance and finding continuity using small, domain-neutral reports."""

from copy import deepcopy
import json

import pytest

from workflows.dynamic_htn_informed_research.task_strategies.report_research.review import (
    ReviewState,
    parse_review,
)

QUESTION = "Calculate total cost for two items at 3 each. Explain the cost assumptions."
DRAFT = "Two items at 3 each cost 6. Table total: 6. Both items have the same unit price."


def accepted_review():
    return {
        "verdict": "pass",
        "checks": [
            {
                "id": "cost",
                "question_quote": "Calculate total cost for two items at 3 each.",
                "report_quote": "Two items at 3 each cost 6. Table total: 6.",
                "method": "Multiply quantity by unit price and compare formula and table",
                "result": "2 * 3 = 6, matching the expression and table total",
                "status": "supported",
            },
            {
                "id": "assumptions",
                "question_quote": "Explain the cost assumptions.",
                "report_quote": "Both items have the same unit price.",
                "method": "Compare the assumption with the stated per-item price",
                "result": "Uniform price agrees with 3 each in the question",
                "status": "supported",
            },
        ],
        "findings": [],
        "finding_reviews": [],
    }


def parse(payload, *, prior=(), draft=DRAFT, question=QUESTION):
    return parse_review(
        json.dumps(payload),
        question=question,
        draft=draft,
        previous_findings=list(prior),
        research_node_ids={"derive"},
    )


def tracked_finding():
    return {
        "finding_id": "e0-r0-f1",
        "issue": "The earlier table said 3 instead of 6",
        "action": "revise",
        "affected_node_ids": ["derive"],
        "check_ids": ["cost"],
    }


def closure(status="resolved"):
    return {
        "finding_id": "e0-r0-f1",
        "status": status,
        "check_ids": ["cost"],
        "reason": "Recomputed 2 * 3 = 6; the current formula and table both give 6",
    }


def test_complete_public_requirement_checks_pass():
    assert parse(accepted_review())["verdict"] == "pass"


@pytest.mark.parametrize(
    "defect",
    [
        "legacy_string_check",
        "invented_requirement",
        "stale_formula",
        "no_report_passage",
        "missing_method",
        "blank_result",
        "duplicate_check_id",
        "unknown_check_status",
        "unsuccessful_check_pass",
        "missing_assessment",
        "duplicate_assessment",
        "invented_assessment",
        "unsupported_closure",
        "closure_without_evidence",
        "unknown_evidence_id",
        "unresolved_finding_omitted",
    ],
)
def test_incomplete_or_unanchored_reviews_cannot_pass(defect):
    payload = accepted_review()
    prior = [tracked_finding()]
    payload["finding_reviews"] = [closure()]
    if defect == "legacy_string_check":
        payload["checks"] = ["Everything looks good"]
    elif defect == "invented_requirement":
        payload["checks"][0]["question_quote"] = "Private rubric expects this answer"
    elif defect == "stale_formula":
        payload["checks"][0]["report_quote"] = "Two items at 3 each cost 3."
    elif defect == "no_report_passage":
        payload["checks"][0]["report_quote"] = ""
    elif defect == "missing_method":
        del payload["checks"][0]["method"]
    elif defect == "blank_result":
        payload["checks"][0]["result"] = " "
    elif defect == "duplicate_check_id":
        payload["checks"][1]["id"] = "cost"
    elif defect == "unknown_check_status":
        payload["checks"][0]["status"] = "probably"
    elif defect == "unsuccessful_check_pass":
        payload["checks"][0]["status"] = "contradicted"
    elif defect == "missing_assessment":
        payload["finding_reviews"] = []
    elif defect == "duplicate_assessment":
        payload["finding_reviews"] *= 2
    elif defect == "invented_assessment":
        payload["finding_reviews"][0]["finding_id"] = "invented"
    elif defect == "unsupported_closure":
        payload["finding_reviews"][0]["reason"] = " "
    elif defect == "closure_without_evidence":
        payload["finding_reviews"][0]["check_ids"] = []
    elif defect == "unknown_evidence_id":
        payload["finding_reviews"][0]["check_ids"] = ["prior-review-cost-check"]
    elif defect == "unresolved_finding_omitted":
        payload["finding_reviews"][0]["status"] = "unresolved"
    with pytest.raises(ValueError):
        parse(payload, prior=prior)


@pytest.mark.parametrize("status", ["resolved", "refuted"])
def test_supported_fix_or_refutation_can_close_old_finding(status):
    payload = accepted_review()
    payload["finding_reviews"] = [closure(status)]
    if status == "refuted":
        payload["finding_reviews"][0]["reason"] = (
            "Old finding treated the per-item price as the total; 2 * 3 proves the total is 6"
        )
    result = parse(payload, prior=[tracked_finding()])
    assert result["verdict"] == "pass"
    assert result["finding_reviews"][0]["status"] == status


def failing_review(*, existing=False):
    payload = accepted_review()
    payload["verdict"] = "revise"
    payload["checks"][0].update(
        report_quote="", status="unresolved", result="No total is stated in the current draft"
    )
    payload["findings"] = [
        {
            "finding_id": "e0-r0-f1" if existing else "",
            "issue": "Missing cost result",
            "action": "revise",
            "affected_node_ids": ["derive"],
            "check_ids": ["cost"],
        }
    ]
    if existing:
        payload["finding_reviews"] = [closure("unresolved")]
    return payload


def test_missing_deliverable_is_a_finding_not_a_fake_quote():
    assert parse(failing_review())["verdict"] == "revise"


@pytest.mark.parametrize(
    "defect",
    [
        "unlinked_check",
        "unknown_check",
        "unknown_node",
        "invented_finding_id",
        "closed_but_still_open",
    ],
)
def test_findings_must_link_current_checks_and_known_identifiers(defect):
    payload = failing_review(existing=True)
    if defect == "unlinked_check":
        payload["findings"] = []
    elif defect == "unknown_check":
        payload["findings"][0]["check_ids"] = ["unknown"]
    elif defect == "unknown_node":
        payload["findings"][0]["affected_node_ids"] = ["old-tree-node"]
    elif defect == "invented_finding_id":
        payload["findings"][0]["finding_id"] = "renamed-old-issue"
    elif defect == "closed_but_still_open":
        payload["finding_reviews"][0] = closure("resolved")
    with pytest.raises(ValueError):
        parse(payload, prior=[tracked_finding()])


def test_research_finding_cannot_be_hidden_behind_revise():
    payload = failing_review()
    payload["findings"][0]["action"] = "research"
    assert parse(payload)["verdict"] == "rebuild"


def test_finding_ids_and_closed_state_survive_rebuild_and_can_reopen():
    state = ReviewState()
    first = state.accept(parse(failing_review()), epoch=0, revision=0)
    assert first["findings"][0]["finding_id"] == "e0-r0-f1"
    second = accepted_review()
    second["finding_reviews"] = [closure()]
    state.accept(parse(second, prior=state.snapshot()["findings"]), epoch=0, revision=1)
    assert state.snapshot()["findings"][0]["last_assessment"]["status"] == "resolved"
    # A previously fixed defect reappears after Planner rebuild; the ID remains stable.
    restored = ReviewState(json.dumps({"material": [{"review_state": state.snapshot()}]}))
    third = restored.accept(
        parse(failing_review(existing=True), prior=restored.snapshot()["findings"]),
        epoch=1,
        revision=0,
    )
    assert third["findings"][0]["finding_id"] == "e0-r0-f1"
    assert restored.snapshot()["findings"][0]["last_assessment"]["status"] == "unresolved"
    assert len(restored.reviews) == 3
    assert state.snapshot()["findings"][0]["last_assessment"]["status"] == "resolved"
    # Reading a snapshot must not let a caller mutate the saved review history.
    snapshot = restored.snapshot()
    snapshot["reviews"].clear()
    assert len(restored.reviews) == 3


def test_closed_findings_cannot_disappear_from_later_reviews():
    prior = tracked_finding()
    prior["last_assessment"] = closure("resolved")
    with pytest.raises(ValueError, match="Every prior finding"):
        parse(accepted_review(), prior=[prior])


def test_quote_matching_allows_layout_but_not_changed_numbers_or_signs():
    payload = accepted_review()
    payload["checks"][0]["report_quote"] = "Two items at 3 each cost 6.\nTable total: 6."
    assert parse(payload)["verdict"] == "pass"
    payload["checks"][0]["report_quote"] = "Two items at 3 each cost -6. Table total: 6."
    with pytest.raises(ValueError, match="CURRENT draft"):
        parse(payload)


def test_known_math_serialization_is_not_a_false_missing_quote():
    payload = accepted_review()
    payload["checks"] = [deepcopy(payload["checks"][0])]
    payload["checks"][0]["question_quote"] = r"\(\alpha+\beta\)"
    question = r"Compute \\(\\alpha+\\beta\\)."
    assert parse(payload, question=question)["verdict"] == "pass"
    payload["checks"][0]["question_quote"] = r"\(\alpha-\beta\)"
    with pytest.raises(ValueError, match="public question"):
        parse(payload, question=question)


def test_epoch_that_never_reaches_delivery_preserves_prior_findings():
    from types import SimpleNamespace

    from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
    from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.topology import (
        rebuild_context,
    )

    state = ReviewState()
    state.accept(parse(failing_review()), epoch=0, revision=0)
    root = HTNTaskNode(id="root", task=QUESTION)
    tree = HTNTree(
        root_id="root",
        nodes={"root": root},
        epoch_id=1,
        restart_context=json.dumps({"review_state": state.snapshot()}),
    )
    restored = ReviewState(rebuild_context(SimpleNamespace(tree=tree, orchestrators={})))
    assert restored.snapshot() == state.snapshot()
