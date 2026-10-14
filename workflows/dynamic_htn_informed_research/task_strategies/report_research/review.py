"""Draft-anchored final checks and explicit reassessment of prior findings.

These checks enforce provenance and review completeness, not scientific truth.
Only public question/draft text and solver reviews enter this contract.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .premise_review import _source_quote


def _object(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} requires exactly {', '.join(sorted(keys))}")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a {'possibly empty' if empty else 'nonempty'} string")
    return value


def _refs(value: Any, allowed: set[str], label: str, *, empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not empty and not value)
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{label} requires a list of IDs")
    if len(value) != len(set(value)) or not set(value) <= allowed:
        raise ValueError(f"{label} contains duplicate or unknown IDs")
    return value


def _quoted(quote: str, source: str) -> bool:
    # Reuse the existing narrowly scoped serialization fix; never equate formulas.
    try:
        _source_quote(" ".join(quote.split()), " ".join(source.split()))
        return True
    except ValueError:
        return False


def parse_review(
    text: str,
    *,
    question: str,
    draft: str,
    previous_findings: list[dict],
    research_node_ids: set[str],
    validate_quotes: bool = True,
    aligned_quotes: set[tuple[str, str]] | None = None,
) -> dict:
    """Require concrete current-draft checks and close/reopen every tracked finding."""
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("```"):
        value = value[8:-3].strip()
    review = _object(
        json.loads(value), {"verdict", "checks", "findings", "finding_reviews"}, "Final review"
    )
    if review["verdict"] not in ("pass", "revise", "rebuild"):
        raise ValueError("Unknown review verdict")
    if not isinstance(review["checks"], list) or not review["checks"]:
        raise ValueError("Final review must describe actual checks")
    checks = {}
    for check in review["checks"]:
        _object(
            check, {"id", "question_quote", "report_quote", "method", "result", "status"}, "Check"
        )
        for field in ("id", "question_quote", "method", "result"):
            _text(check[field], f"Check {field}")
        _text(check["report_quote"], "Check report_quote", empty=True)
        if check["id"] in checks:
            raise ValueError("Duplicate check ID")
        if check["status"] not in ("supported", "contradicted", "unresolved"):
            raise ValueError("Unknown check status")
        if (validate_quotes and (check["id"], "question_quote") not in (aligned_quotes or set())
                and not _quoted(check["question_quote"], question)):
            raise ValueError("Check question_quote is not in the public question")
        if check["report_quote"].strip():
            if (validate_quotes and (check["id"], "report_quote") not in (aligned_quotes or set())
                    and not _quoted(check["report_quote"], draft)):
                raise ValueError("Check report_quote is not in the CURRENT draft")
        elif check["status"] != "unresolved":
            raise ValueError("A missing report passage must be an unresolved check")
        checks[check["id"]] = check

    previous_ids = {item["finding_id"] for item in previous_findings}
    if not isinstance(review["findings"], list):
        raise ValueError("Review findings must be a list")
    linked_checks: set[str] = set()
    repeated_ids: set[str] = set()
    for finding in review["findings"]:
        _object(
            finding, {"finding_id", "issue", "action", "affected_node_ids", "check_ids"}, "Finding"
        )
        _text(finding["finding_id"], "Finding ID", empty=True)
        _text(finding["issue"], "Finding issue")
        fid = finding["finding_id"]
        if fid:
            if fid not in previous_ids or fid in repeated_ids:
                raise ValueError("Repeated finding must reference a unique existing ID")
            repeated_ids.add(fid)
        if finding["action"] not in ("revise", "research"):
            raise ValueError("Unknown finding action")
        _refs(finding["affected_node_ids"], research_node_ids, "Affected node IDs", empty=True)
        refs = _refs(finding["check_ids"], set(checks), "Finding check_ids")
        if all(checks[c]["status"] == "supported" for c in refs):
            raise ValueError("An unresolved finding must link to an unsuccessful check")
        linked_checks.update(refs)
    unsuccessful = {cid for cid, c in checks.items() if c["status"] != "supported"}
    if not unsuccessful <= linked_checks:
        raise ValueError("Every unsuccessful check needs an actionable finding")

    if not isinstance(review["finding_reviews"], list):
        raise ValueError("finding_reviews must be a list")
    assessed: set[str] = set()
    for assessment in review["finding_reviews"]:
        _object(assessment, {"finding_id", "status", "check_ids", "reason"}, "Finding review")
        _text(assessment["finding_id"], "Reviewed finding ID")
        _text(assessment["reason"], "Finding review reason")
        fid = assessment["finding_id"]
        if fid not in previous_ids or fid in assessed:
            raise ValueError("Finding review has an unknown or duplicate ID")
        assessed.add(fid)
        if assessment["status"] not in ("resolved", "refuted", "unresolved"):
            raise ValueError("Unknown finding review status")
        refs = _refs(assessment["check_ids"], set(checks), "Finding review check_ids")
        if assessment["status"] == "unresolved":
            if fid not in repeated_ids:
                raise ValueError("An unresolved prior finding must remain in findings with its ID")
            finding = next(f for f in review["findings"] if f["finding_id"] == fid)
            if not set(refs) & set(finding["check_ids"]) & unsuccessful:
                raise ValueError("Unresolved finding review must link its unsuccessful check")
        elif fid in repeated_ids or any(checks[c]["status"] != "supported" for c in refs):
            raise ValueError(
                "Resolved/refuted findings need supported checks and no open duplicate"
            )
    if assessed != previous_ids:
        raise ValueError("Every prior finding must be explicitly reassessed, including closed ones")
    if review["verdict"] == "pass" and (review["findings"] or unsuccessful):
        raise ValueError("Unresolved findings/checks cannot pass")
    if review["verdict"] != "pass" and not review["findings"]:
        raise ValueError("Non-pass verdict must identify its defects")
    if any(f["action"] == "research" for f in review["findings"]):
        review["verdict"] = "rebuild"
    return review


class ReviewState:
    """Stable finding IDs plus accepted reviews, including across Planner rebuilds."""

    def __init__(self, restart_context: str = "") -> None:
        self.findings: dict[str, dict] = {}
        self.reviews: list[dict] = []
        try:
            prior = json.loads(restart_context)
        except (ValueError, TypeError):
            return  # Legacy/non-report restart context has no structured review history.
        if not isinstance(prior, dict):
            return
        if prior.get("review_state") is not None:
            states = [prior["review_state"]]
        else:
            states = [
                item["review_state"]
                for item in prior.get("material", [])
                if isinstance(item, dict) and item.get("review_state") is not None
            ]
        for state in states:
            _object(state, {"findings", "reviews"}, "Saved review state")
            if not isinstance(state["findings"], list) or not isinstance(state["reviews"], list):
                raise ValueError("Invalid saved review state")
            for finding in state["findings"]:
                fid = _text(finding.get("finding_id"), "Saved finding ID")
                if fid in self.findings:
                    raise ValueError("Duplicate saved finding ID")
                self.findings[fid] = deepcopy(finding)
            self.reviews.extend(deepcopy(state["reviews"]))

    def snapshot(self) -> dict:
        return {
            "findings": deepcopy(list(self.findings.values())),
            "reviews": deepcopy(self.reviews),
        }

    def accept(self, review: dict, *, epoch: int, revision: int) -> dict:
        review = deepcopy(review)
        for assessment in review["finding_reviews"]:
            self.findings[assessment["finding_id"]]["last_assessment"] = deepcopy(assessment)
        for index, finding in enumerate(review["findings"], 1):
            if not finding["finding_id"]:
                finding["finding_id"] = f"e{epoch}-r{revision}-f{index}"
            fid = finding["finding_id"]
            self.findings[fid] = {
                **deepcopy(finding),
                "last_assessment": {"status": "unresolved", "reason": finding["issue"]},
            }
        self.reviews.append({"draft_id": f"e{epoch}-r{revision}", "review": deepcopy(review)})
        return review
