"""Planning presentation must preserve evidence and distinguish audit provenance."""
from copy import deepcopy

import pytest

from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_rendering import (
    render_planning_input,
)


@pytest.mark.parametrize("rebuilding", [False, True])
def test_markdown_roles_preserve_long_multiline_inputs_and_audit_findings(rebuilding):
    question = 'Derive ΔE.\n\n## Given data\n\n| x | y |\n| 1 | 2 |\n\n```python\nx = "a\\nb"\n```'
    recovery = 'Supported check C1:\n' + ('Measured α = 2.\n' * 1800) + 'Remaining gap: coupling sign.'
    context = {"task": question}
    if rebuilding:
        context["recovery_capsule"] = recovery
    constraints = {"max_research_nodes": 4, "runtime_owns_final_delivery": True}
    task = 'Establish the result:\n' + ('Use $E = \\hbar\\omega$ under assumption A.\n' * 800) + 'Last required check: units.'
    plan = {
        "rationale": 'Compare two models.\nRetain the given {"label": "A"} literally.',
        "nodes": [
            {"id": "M1", "title": "Model α", "task": task, "depends_on": []},
            {"id": "M2", "title": "Derived observable", "task": "Compute B.\nCheck its sign.", "depends_on": ["M1"]},
        ],
    }
    audit = {
        "has_delivery": False,
        "nodes": [
            {"id": "M1", "kind": "research", "reason": "Establishes α.\nCheck C1 remains provisional."},
            {"id": "M2", "kind": "research", "reason": "Derives the measured response."},
        ],
        "plan_review": {"verdict": "REVISE", "issues": [
            {"node_ids": ["M1", "M2"], "issue": "Boundary splits one calculation.\nThe sign check needs the same model.",
             "repair": "Merge the coupled result.\nKeep the independent check inside it."},
            {"node_ids": [], "issue": "The requested uncertainty is missing.", "repair": "Include uncertainty propagation."},
        ]},
        "model_plan_review": {"verdict": "PASS", "issues": []},
        "runtime_findings": ["Delivery nodes have successors: [\"M2\"]", "Candidate node IDs are not unique"],
    }
    before = deepcopy((context, constraints, plan, audit))
    initial = render_planning_input(planner_input=context, constraints=constraints)
    auditing = render_planning_input(planner_input=context, constraints=constraints, candidate_plan=plan)
    revising = render_planning_input(planner_input=context, constraints=constraints, candidate_plan=plan, audit_feedback=audit)

    assert auditing.startswith(initial + "\n\n")
    assert revising.startswith(auditing + "\n\n")
    assert question in initial
    assert (recovery in initial) == rebuilding
    for text in (auditing, revising):
        assert task in text  # No length clipping, escaped newlines or paraphrasing.
        assert plan["rationale"] in text
        assert text.index("ID: M1") < text.index("ID: M2")
        assert "Depends on: M1" in text
    for finding in audit["nodes"]:
        assert finding["reason"] in revising
    for issue in audit["plan_review"]["issues"]:
        assert issue["issue"] in revising and issue["repair"] in revising
    assert "Affected nodes: M1, M2" in revising
    assert "None (missing coverage)" in revising
    assert "Plan verdict: REVISE" in revising
    prior = revising.split("### Auditor verdict before runtime checks", 1)[1]
    assert "Plan verdict: PASS" in prior
    assert all(finding in revising for finding in audit["runtime_findings"])
    assert (context, constraints, plan, audit) == before


def test_rendering_keeps_duplicate_ids_and_unknown_dependencies_visible_to_auditor():
    plan = {"rationale": "Candidate needing structural review", "nodes": [
        {"id": "same", "title": "First", "task": "Original task A", "depends_on": []},
        {"id": "same", "title": "Second", "task": "Original task B", "depends_on": ["missing"]},
    ]}
    before = deepcopy(plan)
    text = render_planning_input(
        planner_input={"task": "Public task"},
        constraints={"max_research_nodes": 4, "runtime_owns_final_delivery": True},
        candidate_plan=plan,
    )
    assert text.count("ID: same") == 2
    assert "Original task A" in text and "Original task B" in text
    assert "Depends on: missing" in text
    assert plan == before


def _parse_markdown(text):
    return pytest.importorskip("markdown_it").MarkdownIt("commonmark").parse(text)


def test_source_headings_and_code_fences_cannot_escape_into_document_structure():
    # Longer closing fences, setext headings and nested code are all source data.
    source = '# Source heading\n\nSetext heading\n===\n\n```python\nprint("α")\n````````\n\n~~~\n## Still source\n~~~'
    plan = {"rationale": source, "nodes": [
        {"id": "M1", "title": source, "task": source, "depends_on": []},
    ]}
    audit = {
        "has_delivery": False,
        "nodes": [{"id": "M1", "kind": "research", "reason": source}],
        "plan_review": {"verdict": "REVISE", "issues": [
            {"node_ids": ["M1"], "issue": source, "repair": source},
        ]},
        "runtime_findings": [source],
    }
    text = render_planning_input(
        planner_input={"task": source, "recovery_capsule": source},
        constraints={"max_research_nodes": 4, "runtime_owns_final_delivery": True},
        candidate_plan=plan, audit_feedback=audit,
    )
    tokens = _parse_markdown(text)
    headings = [(t.tag, tokens[i + 1].content) for i, t in enumerate(tokens)
                if t.type == "heading_open"]
    assert headings == [
        ("h1", "Planning input"), ("h2", "Original task"),
        ("h2", "Recovery capsule (background evidence)"), ("h2", "Constraints"),
        ("h2", "Candidate plan"), ("h3", "Node 1"), ("h2", "Audit feedback"),
        ("h3", "Issue 1"), ("h3", "Node classifications"),
        ("h4", "Classification 1"), ("h3", "Runtime structural findings"),
        ("h4", "Finding 1"),
    ]
    blocks = [t.content for t in tokens if t.type == "fence"]
    assert blocks == [source + "\n"] * 9


def test_structured_recovery_keeps_every_evidence_string_and_review_field():
    import json
    from collections import Counter

    evidence = '# Evidence\n\n```python\na = 1\n````\n' + ('Check ΔE and units.\n' * 2000)
    capsule = {
        "previous_epoch": 2,
        "review_state": {
            "findings": [{"finding_id": "e2-r0-f1", "issue": "Missing coupling sign.\nCheck C1.",
                          "last_assessment": {"status": "unresolved", "reason": "Need another derivation."}}],
            "reviews": [{"draft_id": "e2-r0", "review": {"verdict": "rebuild", "findings": []}}],
        },
        "instruction": "Reuse checked derivations.\nClose the sign gap.",
        "material": [
            {"id": "M1", "task": "# Derive E\nUnder assumption A.", "status": "completed",
             "outcome": "resolved", "memo": evidence, "failure": None,
             "worker_reports": [{"role": "researcher", "quality": "valid", "content": evidence},
                                {"role": "verifier", "quality": "valid", "content": '{"literal": "x\\ny"}'}],
             "rebuild_feedback": "Keep the model; check the sign."},
            {"id": "M2", "task": "Compute response", "status": "pending", "outcome": None,
             "memo": None, "failure": "Worker failed.\nRetry needed.", "worker_reports": [],
             "rebuild_feedback": ""},
        ],
        "future_metadata": {"confidence": 0.75, "checked": False, "extra": "Do not discard this."},
    }
    context = {"task": "# Task\nDerive and check the result.",
               "recovery_capsule": json.dumps(capsule, ensure_ascii=False)}
    before = deepcopy(context)
    rendered = render_planning_input(
        planner_input=context,
        constraints={"max_research_nodes": 4, "runtime_owns_final_delivery": True},
    )
    tokens = _parse_markdown(rendered)

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    # Compare parsed text, not escaped/indented Markdown source. This also detects
    # loss of duplicate reports, blank text, internal fences and trailing newlines.
    assert Counter(t.content for t in tokens if t.type == "fence") == Counter(
        text + "\n" for text in [context["task"], *strings(capsule)]
    )
    headings = [tokens[i + 1].content for i, t in enumerate(tokens) if t.type == "heading_open"]
    assert headings == ["Planning input", "Original task", "Recovery capsule (background evidence)",
                        "Previous epoch", "Review state", "Recovery guidance", "Recovered nodes",
                        "Recovered node 1", "Recovered node 2", "Additional recovery field", "Constraints"]
    assert "0.75" in rendered and "false" in rendered and "null" in rendered
    assert context == before  # Structured diagnostics and stored restart_context stay intact.


@pytest.mark.parametrize("capsule", [
    "# Legacy recovery\n\n```text\nPrior evidence\n```",
    '{"previous_epoch": 0, "material": [broken JSON',
    '{"unrelated": "JSON should stay literal"}',
    '["legacy", "data"]',
    "",
])
def test_unstructured_or_unrecognized_recovery_is_preserved_literally(capsule):
    rendered = render_planning_input(
        planner_input={"task": "Question", "recovery_capsule": capsule},
        constraints={"max_research_nodes": 4, "runtime_owns_final_delivery": True},
    )
    assert [t.content for t in _parse_markdown(rendered) if t.type == "fence"] == [
        "Question\n", capsule + "\n",
    ]
