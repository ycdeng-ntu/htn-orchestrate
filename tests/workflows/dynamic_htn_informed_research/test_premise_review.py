"""Question quote provenance and narrowly scoped serialization compatibility."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from workflows.dynamic_htn_informed_research.task_strategies.report_research.premise_review import (
    parse_premise_checks,
)


def _payload(quote):
    return {"checks": [{
        "question_quote": quote,
        "constraint": "Use the stated expression.",
        "independent_reasoning": "Changing its sign changes the equation.",
        "test": "Compare the expression and its sign with the question.",
    }]}


def test_recorded_q10_quotes_recover_original_source_without_changing_checks():
    fixture = json.loads((Path(__file__).parent / "fixtures/q10_premise_quotes.json").read_text())
    payload, question = fixture["payload"], fixture["question"]
    original = deepcopy(payload)
    assert sum(c["question_quote"] in question for c in payload["checks"]) == 3
    checks = parse_premise_checks(payload, question)
    assert len(checks) == 11
    assert payload == original
    for check, raw in zip(checks, payload["checks"], strict=True):
        assert check.question_quote in question
        for field in ("constraint", "independent_reasoning", "test"):
            assert getattr(check, field) == raw[field]


@pytest.mark.parametrize(("question", "quote", "expected"), [
    (r"Use \\(b\_\\omega^\\dagger + a\\).", r"b_\omega^\dagger + a", r"b\_\\omega^\\dagger + a"),
    (r"Use \\(a\\) and \\(b\\).", r"\(a\) and \(b\)", r"\\(a\\) and \\(b\\)"),
    (r"Use \\(x+\\omega\\).", r"\omega", r"\omega"),
    (r"Use \\(x\_y\\).", "_y", "_y"),
    (r"Use \(x\\ y\).", r"x\\ y", r"x\\ y"),
    (r"Use \\(\\omega+b\_\\nu\\).", r"\omega+b_\nu", r"\\omega+b\_\\nu"),
    ("假设：α ≥ 0，且能量守恒。", "α ≥ 0，且能量守恒", "α ≥ 0，且能量守恒"),
])
def test_accepted_quote_is_an_actual_source_excerpt(question, quote, expected):
    check, = parse_premise_checks(_payload(quote), question)
    assert check.question_quote == expected
    assert expected in question


@pytest.mark.parametrize(("question", "quote"), [
    (r"Use \\(b\_\\omega^\\dagger + a\\).", r"b_\omega^\dagger - a"),
    (r"Use \\(b\_\\omega^\\dagger + a\\).", r"b_\nu^\dagger + a"),
    (r"Use \\(b\_\\omega^\\dagger + a\\).", r"b_\omega + a"),
    (r"Use \\(2a\\).", r"\(3a\)"),
    (r"Use \(x\\ y\).", r"\(x\ y\)"),
    (r"Use \\(x\\[4pt]y\\).", r"x\[4pt]y"),
    (r"Use \(x\_y\).", r"\(x_y\)"),
    (r"literal x\_y outside math", "literal x_y outside math"),
    (r"literal \\omega outside math", r"literal \omega outside math"),
    (r"Use \\(x+\omega\\).", r"\(x+\omega\)"),
    (r"Use \\(x+\\\omega\\).", r"\(x+\omega\)"),
    ("The field does not vanish.", "The field does vanish."),
    ("Assume x > 0.", "Assume x ≥ 0."),
    (r"Use \\(x\\).", "Invented excerpt"),
])
def test_formula_changes_and_unrecognized_escapes_remain_rejected(question, quote):
    with pytest.raises(ValueError, match="quote is absent"):
        parse_premise_checks(_payload(quote), question)


def test_quote_feedback_locates_each_mismatch_without_accepting_paraphrases():
    question = "The sample remains cold. Cell A contains 4 units."
    payload = _payload("The sample stayed cold.")
    payload["checks"].extend(_payload("cell A contains 4 units.")["checks"])
    with pytest.raises(ValueError) as exc:
        parse_premise_checks(payload, question)
    message = str(exc.value)
    assert "checks[0].question_quote" in message
    assert "checks[1].question_quote" in message
    assert "Cell A contains 4 units." in message
    assert "not an accepted replacement" in message
