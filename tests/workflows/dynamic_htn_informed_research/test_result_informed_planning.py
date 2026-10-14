from __future__ import annotations

import json

import pytest

from agent_harness.core.runtime.registries.agents import AgentRegistry
from agent_harness.core.runtime.registries.workflows import WorkflowContext
from agent_harness.scheduling.pipeline_registry import PipelineRegistry
from workflows.dynamic_htn_informed_research.htn import (
    DecompositionDecision,
    HTNPlanner,
    HTNTaskNode,
    HTNTaskStatus,
    HTNTree,
    LLMResponse,
)
from workflows.dynamic_htn_informed_research.htn.planner import DecompositionBudget
from workflows.dynamic_htn_informed_research.htn.strategies import (
    ResearchDecompositionStrategy,
    _AUDIT_SUBTASK_TOOL,
    _DECOMPOSE_TOOL,
    _DECOMPOSITION_REPAIR_SYSTEM,
    _PLANNING_PROTOCOL_RETRY_USER,
    _RESEARCH_DECOMPOSITION_SYSTEM,
    _REVISE_SUBTASK_TOOL,
    _SUBTASK_FIDELITY_AUDITOR_SYSTEM,
    _SUBTASK_FIDELITY_REVISER_SYSTEM,
    _complete_planning_action,
    _novel_numeric_literals,
    _render_precursor_evidence,
    _structural_decomposition_issues,
    _structural_repair_preservation_issues,
)
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy


def test_react_and_informed_workflows_register_without_collisions() -> None:
    from workflows.react_base import register as register_original
    from workflows.dynamic_htn_informed_research import register as register_informed

    pipelines = PipelineRegistry()
    agents = AgentRegistry()
    ctx = WorkflowContext(pipelines, agents)

    register_original(ctx)
    register_informed(ctx)

    assert pipelines.has("react_base")
    assert pipelines.has("dynamic_htn_informed_research")
    assert agents.has("react_solver")
    assert agents.has("dynir_researcher")
    assert agents.has("dynir_verifier")
    for strategy in (
        "research", "report_research", "science_olympiad", "science_mcq",
    ):
        for definition in get_task_strategy(strategy).workers.agent_definitions:
            assert agents.has(definition.role_id)
            assert definition.metadata["enable_skills"] is False


def test_planner_preserves_literal_constraints_and_keeps_hypotheses_optional() -> None:
    prompt = _RESEARCH_DECOMPOSITION_SYSTEM
    assert "has two" in prompt
    assert "exactly two" in prompt
    assert "does NOT imply that the person is Muslim" in prompt
    assert "accused" in prompt
    assert "official" in prompt
    assert "optional" in prompt

    assert "constraint-delta check" in _RESEARCH_DECOMPOSITION_SYSTEM
    assert "Source authority affects evidence quality" in _RESEARCH_DECOMPOSITION_SYSTEM
    normalized = " ".join(_RESEARCH_DECOMPOSITION_SYSTEM.lower().split())
    assert "exact" in normalized and "phrase" in normalized
    assert "veto" in normalized
    assert "narrowed interpretation" in normalized
    assert "unresolved" in normalized

    assert "centers on a chemical element" in _RESEARCH_DECOMPOSITION_SYSTEM
    assert "implies contextual recency" in " ".join(
        _RESEARCH_DECOMPOSITION_SYSTEM.split()
    )
    assert "fixed exact date window as a rejection criterion" in " ".join(
        _RESEARCH_DECOMPOSITION_SYSTEM.split()
    )


def test_subtask_audit_owns_boundedness_without_closing_candidate_discovery() -> None:
    normalized = " ".join(_SUBTASK_FIDELITY_AUDITOR_SYSTEM.split())
    assert "Operational boundedness" in normalized
    assert "do not infer an unstated cap" in normalized
    assert "A candidate-agnostic route that actually instructs search" in normalized
    assert 'merely saying "return a ranked list" supplies no bound' in normalized
    assert "is not itself a search instruction" in normalized
    assert "at least two" in normalized
    assert "does not by itself say \"exactly two\"" in normalized
    assert '"contains an element" does NOT entail "centers on a chemical element"' in normalized
    assert "unsupported type restriction" in normalized
    assert "world knowledge" in normalized
    assert "no veto authority" in normalized
    assert "highest authority" in normalized
    assert "album title includes X" in normalized
    assert '"90+5"' in normalized
    assert "Level 100 robustness" in normalized
    assert "asymmetric uncertainty" in normalized

    audit_schema = _AUDIT_SUBTASK_TOOL["function"]["parameters"]["properties"]
    assert audit_schema["authoritative_context_quote"]["minLength"] == 1
    assert audit_schema["subtask_quote"]["minLength"] == 1

    reviser = " ".join(_SUBTASK_FIDELITY_REVISER_SYSTEM.split())
    assert "operationally_unbounded" in reviser
    assert "Do not delete a candidate-agnostic discovery route" in reviser
    assert 'say "no new candidates"' in reviser
    assert "must never exceed 1000 characters" in reviser
    assert "aim for 900 characters or fewer" in reviser

    revision_task_schema = _REVISE_SUBTASK_TOOL["function"]["parameters"][
        "properties"
    ]["task"]
    assert revision_task_schema["maxLength"] == 1000
    assert "900 characters or fewer" in revision_task_schema["description"]


def test_planner_keeps_structural_tasks_compact_and_defers_worker_tactics() -> None:
    task_schema = _DECOMPOSE_TOOL["function"]["parameters"]["properties"]["task"]
    rationale_schema = _DECOMPOSE_TOOL["function"]["parameters"]["properties"][
        "rationale"
    ]

    assert task_schema["maxLength"] == 1000
    assert rationale_schema["maxLength"] == 800
    assert "instruction generation expands those later" in task_schema["description"]

    for prompt in (
        _RESEARCH_DECOMPOSITION_SYSTEM,
        _PLANNING_PROTOCOL_RETRY_USER,
    ):
        normalized = " ".join(prompt.lower().split())
        assert "600 characters or fewer" in normalized
        assert "query lists" in normalized
        assert "report templates" in normalized

    assert "compact structural plan" in _RESEARCH_DECOMPOSITION_SYSTEM
    assert "compact structural objective" in " ".join(
        _PLANNING_PROTOCOL_RETRY_USER.split()
    )


def test_whole_plan_repair_is_strictly_structural() -> None:
    normalized = " ".join(_DECOMPOSITION_REPAIR_SYSTEM.split())
    assert "structural graph repairer" in normalized
    assert "Structural Defects list is exhaustive" in normalized
    assert "copy its `title`, `task`, and `needs_decomposition` values exactly" in normalized
    assert "remove only the minimum number" in normalized
    assert "Do not alter objectives" in normalized
    assert "separate per-subtask audit owns both" in normalized
    assert "finite stopping condition" not in normalized
    assert "candidate-agnostic route" not in normalized


def test_numeric_delta_is_attention_only_and_respects_precursor_numbers() -> None:
    hints = _novel_numeric_literals(
        "Search 2006-2014 (exclusive); return at most 5 candidates from the 2011 lead.",
        root_task="The debut was between 2005 and 2015 (exclusive).",
        parent_task="Find the director.",
        precursor_evidence="A completed source mentions a 2011 lead.",
        restart_context="",
    )

    assert hints == ["5", "2006", "2014"]


def _tool_response(name: str, **arguments: object) -> LLMResponse:
    if name == "decompose" and isinstance(arguments.get("subtasks"), list):
        rationale = arguments.get("rationale", "Test decomposition branch.")
        return LLMResponse(
            tool_calls=[
                {
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            {
                                **subtask,
                                "rationale": rationale,
                            }
                        ),
                    }
                }
                for subtask in arguments["subtasks"]
            ]
        )
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            }
        ]
    )


def _fragmented_decompose_response() -> LLMResponse:
    """Return flat calls that are incomplete because each lacks rationale."""
    return LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "decompose",
                    "arguments": json.dumps(
                        {
                            "id": f"route-{index}",
                            "title": f"Route {index}",
                            "task": f"Investigate route {index}.",
                            "depends_on": [],
                            "needs_decomposition": False,
                        }
                    ),
                }
            }
            for index in range(1, 4)
        ]
    )


def test_planner_merges_flat_decompose_calls_into_one_atomic_plan() -> None:
    response = _tool_response(
        "decompose",
        rationale="Run discovery before reconciliation.",
        subtasks=[
            {
                "id": "discover",
                "title": "Discover candidates",
                "task": "Return a bounded candidate ledger.",
                "depends_on": [],
                "needs_decomposition": False,
            },
            {
                "id": "reconcile",
                "title": "Reconcile candidates",
                "task": "Check candidates against every original constraint.",
                "depends_on": ["discover"],
                "needs_decomposition": True,
            },
        ],
    )

    parsed, issue = _complete_planning_action(response)

    assert issue == ""
    assert parsed is not None
    assert parsed[0] == "decompose"
    assert [subtask["id"] for subtask in parsed[1]["subtasks"]] == [
        "discover",
        "reconcile",
    ]


def test_planner_rejects_legacy_nested_array_and_mixed_actions() -> None:
    legacy = LLMResponse(
        tool_calls=[
            {
                "function": {
                    "name": "decompose",
                    "arguments": json.dumps(
                        {
                            "subtasks": [],
                            "rationale": "Legacy envelope.",
                        }
                    ),
                }
            }
        ]
    )
    parsed, issue = _complete_planning_action(legacy)
    assert parsed is None
    assert "legacy subtasks array" in issue

    flat = _tool_response(
        "decompose",
        id="discover",
        title="Discover candidates",
        task="Return a bounded candidate ledger.",
        depends_on=[],
        needs_decomposition=False,
        rationale="Discovery branch.",
    )
    direct = _tool_response("execute_direct", rationale="Keep one leaf.")
    mixed = LLMResponse(tool_calls=[*flat.tool_calls, *direct.tool_calls])
    parsed, issue = _complete_planning_action(mixed)
    assert parsed is None
    assert "execute_direct must be the only" in issue


def test_planner_structural_validation_rejects_invalid_dependency_graph() -> None:
    subtasks = [
        {
            "id": "route-a",
            "task": "Investigate route A.",
            "depends_on": ["route-b"],
        },
        {
            "id": "route-b",
            "task": "Investigate route B.",
            "depends_on": ["route-a", "missing"],
        },
    ]
    issues = _structural_decomposition_issues(
        subtasks,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert any("unknown sibling IDs: missing" in issue for issue in issues)
    assert "Subtask dependency graph contains a cycle." in issues


def test_structural_repair_may_change_edges_but_not_subtask_semantics() -> None:
    original = [
        {
            "id": "route",
            "title": "Discover candidates",
            "task": "Preserve this exact open discovery route.",
            "depends_on": ["missing"],
            "needs_decomposition": False,
        },
        {
            "id": "route",
            "title": "Reconcile evidence",
            "task": "Preserve this exact reconciliation objective.",
            "depends_on": ["route"],
            "needs_decomposition": True,
        },
    ]
    repaired = [
        {**original[0], "id": "discover", "depends_on": []},
        {**original[1], "id": "reconcile", "depends_on": ["discover"]},
    ]
    budget = DecompositionBudget.for_node(
        current_depth=0,
        max_depth=2,
        max_children=4,
    )

    assert _structural_repair_preservation_issues(original, repaired, budget) == []

    rewritten = [
        repaired[0],
        {
            **repaired[1],
            "task": "A strategically improved but unauthorized objective.",
        },
    ]
    issues = _structural_repair_preservation_issues(original, rewritten, budget)
    assert any("changed/reordered title, task" in issue for issue in issues)


def test_child_limit_repair_removes_only_the_minimum_without_rewriting() -> None:
    original = [
        {
            "id": f"route-{index}",
            "title": f"Route {index}",
            "task": f"Investigate route {index}.",
            "depends_on": [],
            "needs_decomposition": False,
        }
        for index in range(3)
    ]
    budget = DecompositionBudget.for_node(
        current_depth=0,
        max_depth=2,
        max_children=2,
    )

    assert _structural_repair_preservation_issues(
        original,
        original[:2],
        budget,
    ) == []
    over_pruned = _structural_repair_preservation_issues(
        original,
        original[:1],
        budget,
    )
    assert any("retain exactly 2 subtasks" in issue for issue in over_pruned)


class _RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


@pytest.mark.asyncio
async def test_planner_strictly_normalizes_complete_stringified_arrays() -> None:
    class _LLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            self.calls += 1
            return _tool_response(
                "decompose",
                id="route-a",
                title="Investigate route A",
                task="Return a bounded candidate ledger from route A.",
                depends_on=json.dumps([]),
                needs_decomposition=False,
                rationale="One bounded route is sufficient.",
            )

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _LLM()
    diagnostics = _RecordingDiagnostics()
    decision = await ResearchDecompositionStrategy(
        llm=llm,
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    assert decision.children[0].depends_on == []
    assert llm.calls == 1
    attempt = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_protocol_attempt"
    )
    assert attempt["normalization_modes"] == ["depends_on_json_string"]
    assert attempt["normalization_recovered"] is True
    outcome = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_protocol_outcome"
    )
    assert outcome["outcome"] == "initial_normalized"


@pytest.mark.asyncio
async def test_planner_protocol_retry_recovers_fragmented_decomposition() -> None:
    class _LLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return _fragmented_decompose_response()
            assert len(messages) == 3
            assert "one homogeneous `decompose` tool call per subtask" in messages[-1][
                "content"
            ]
            assert [tool["function"]["name"] for tool in tools] == ["decompose"]
            return _tool_response(
                "decompose",
                rationale="Two independent routes should run in parallel.",
                subtasks=[
                    {
                        "id": "route-a",
                        "title": "Investigate route A",
                        "task": "Return a bounded candidate ledger from route A.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "route-b",
                        "title": "Investigate route B",
                        "task": "Return a bounded candidate ledger from route B.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                ],
            )

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _LLM()
    decision = await ResearchDecompositionStrategy(llm=llm).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    assert [child.id for child in decision.children] == ["root-route-a", "root-route-b"]
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_planner_second_fresh_protocol_retry_recovers() -> None:
    class _LLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            if self.calls <= 2:
                return _fragmented_decompose_response()
            assert len(messages) == 3
            assert "one homogeneous `decompose` tool call per subtask" in messages[-1][
                "content"
            ]
            assert [tool["function"]["name"] for tool in tools] == ["decompose"]
            return _tool_response(
                "decompose",
                rationale="Two independent routes should run in parallel.",
                subtasks=[
                    {
                        "id": "route-a",
                        "title": "Investigate route A",
                        "task": "Return a bounded candidate ledger from route A.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "route-b",
                        "title": "Investigate route B",
                        "task": "Return a bounded candidate ledger from route B.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                ],
            )

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _LLM()
    diagnostics = _RecordingDiagnostics()
    decision = await ResearchDecompositionStrategy(
        llm=llm,
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    assert [child.id for child in decision.children] == ["root-route-a", "root-route-b"]
    assert llm.calls == 3
    attempts = [
        fields["attempt"]
        for event, fields in diagnostics.events
        if event == "planner_protocol_attempt"
    ]
    assert attempts == [0, 1, 2]
    outcome = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_protocol_outcome"
    )
    assert outcome == {
        "node_id": "root",
        "outcome": "retry_recovered",
        "successful_attempt": 2,
        "attempts": 3,
    }


@pytest.mark.asyncio
async def test_planner_protocol_retry_is_bounded_before_execute_direct_fallback() -> None:
    class _LLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            self.calls += 1
            return _fragmented_decompose_response()

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    llm = _LLM()
    planner = HTNPlanner(
        strategy=ResearchDecompositionStrategy(llm=llm),
        max_depth=2,
        max_children=4,
    )

    children = await planner.decompose(tree, "root")

    assert children == []
    assert root.needs_decomposition is False
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_planner_audits_each_subtask_and_revises_only_flagged_text() -> None:
    class _PlannerLLM:
        model = "planner"
        reasoning_effort = "high"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return _tool_response(
                "decompose",
                rationale="Discover candidates, then reconcile them.",
                subtasks=[
                    {
                        "id": "acquirer-route",
                        "title": "Trace acquirer founder",
                        "task": (
                            "Find brands acquired by a company whose founder graduated "
                            "during 1900-1999. Return a short candidate ledger."
                        ),
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "reconcile",
                        "title": "Reconcile candidates",
                        "task": "Reconcile candidates against the original root constraints.",
                        "depends_on": ["acquirer-route"],
                        "needs_decomposition": True,
                    },
                ],
            )

    class _AuditLLM:
        model = "auditor"
        reasoning_effort = "low"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def chat(self, messages, tools=None):  # noqa: ANN001
            assert [tool["function"]["name"] for tool in tools] == [
                "audit_subtask_fidelity"
            ]
            prompt = messages[1]["content"]
            self.prompts.append(prompt)
            assert "acquirer-route" in prompt
            assert "reconcile" in prompt
            target = prompt.rsplit("## Target Subtask", 1)[1]
            if "1900-1999" in target:
                assert "1900, 1999" in prompt
                assert (
                    "Reconcile candidates against the original root constraints."
                    not in prompt
                )
                return _tool_response(
                    "audit_subtask_fidelity",
                    verdict="REVISE",
                    issue_type="constraint_narrowing",
                    authoritative_context_quote="graduated in the 1900s",
                    subtask_quote="graduated during 1900-1999",
                    feedback=(
                        "Preserve 'in the 1900s' literally instead of choosing a range."
                    ),
                )
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="graduated in the 1900s",
                subtask_quote=(
                    "Reconcile candidates against the original root constraints."
                    if "Reconcile candidates" in target
                    else "graduated in the 1900s"
                ),
                feedback="",
            )

    class _RevisionLLM:
        model = "reviser"
        reasoning_effort = "high"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            assert [tool["function"]["name"] for tool in tools] == [
                "revise_subtask"
            ]
            prompt = messages[1]["content"]
            assert "graduated in the 1900s" in prompt
            assert "constraint_narrowing" in prompt
            assert "reconcile" in prompt
            assert (
                "Reconcile candidates against the original root constraints."
                not in prompt
            )
            assert "not a correction memo" in messages[0]["content"]
            return _tool_response(
                "revise_subtask",
                title="Trace acquirer founder",
                task=(
                    "Find brands acquired by a company whose founder graduated in the "
                    "1900s. Return a short candidate ledger."
                ),
                revision_summary="Restored the root's literal temporal wording.",
            )

    root = HTNTaskNode(
        id="root",
        task=(
            "Identify the brand acquired by a company whose founder graduated in the "
            "1900s."
        ),
        needs_decomposition=True,
    )
    tree = HTNTree(root_id="root", nodes={"root": root})
    auditor = _AuditLLM()
    reviser = _RevisionLLM()
    diagnostics = _RecordingDiagnostics()
    decision = await ResearchDecompositionStrategy(
        llm=_PlannerLLM(),
        audit_llm=auditor,
        revision_llm=reviser,
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    discover, reconcile = decision.children
    assert "graduated in the 1900s" in discover.task
    assert "1900-1999" not in discover.task
    assert discover.id == "root-acquirer-route"
    assert discover.depends_on == []
    assert reconcile.id == "root-reconcile"
    assert reconcile.depends_on == [discover.id]
    assert reconcile.needs_decomposition is True
    assert len(auditor.prompts) == 3  # two initial audits plus one re-audit
    assert reviser.calls == 1
    summary = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_audit_summary"
    )
    assert summary["subtask_count"] == 2
    assert summary["initial_revise_count"] == 1
    assert summary["revision_success_count"] == 1
    assert summary["post_revision_pass_count"] == 1


@pytest.mark.asyncio
async def test_science_overlong_revision_retries_with_count_and_keeps_complete_retry() -> None:
    first_revision_task = "FIRST-CORRECTED " + ("x" * 1_100)
    retry_revision_task = "SECOND-CORRECTED " + ("y" * 1_025)

    class _PlannerLLM:
        model = "planner"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return _tool_response(
                "decompose",
                rationale="Use two routes and reconcile them.",
                subtasks=[
                    {
                        "id": "route-a",
                        "title": "First route",
                        "task": "KNOWN-BAD narrowing: solve the related surface quantity.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "route-b",
                        "title": "Second route",
                        "task": "Solve the requested perceived displacement independently.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "reconcile",
                        "title": "Reconcile routes",
                        "task": "Compare both routes against the root quantity.",
                        "depends_on": ["route-a", "route-b"],
                        "needs_decomposition": False,
                    },
                ],
            )

    class _AuditLLM:
        model = "auditor"
        reasoning_effort = "high"

        async def chat(self, messages, tools=None):  # noqa: ANN001
            assert tools[0]["function"]["name"] == "audit_subtask_fidelity"
            target = messages[1]["content"].rsplit("## Target Subtask", 1)[1]
            if "KNOWN-BAD" in target:
                return _tool_response(
                    "audit_subtask_fidelity",
                    verdict="REVISE",
                    issue_type="constraint_narrowing",
                    authoritative_context_quote="perceived displacement",
                    subtask_quote="related surface quantity",
                    feedback="Restore the requested perceived displacement.",
                )
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="perceived displacement",
                subtask_quote=(
                    "SECOND-CORRECTED"
                    if "SECOND-CORRECTED" in target
                    else "requested perceived displacement"
                ),
                feedback="",
            )

    class _RevisionLLM:
        model = "reviser"
        reasoning_effort = "high"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            assert tools[0]["function"]["name"] == "revise_subtask"
            prompt = messages[1]["content"]
            if self.calls == 1:
                assert "Length-only protocol correction" not in prompt
                task = first_revision_task
            else:
                assert self.calls == 2
                assert "Length-only protocol correction" in prompt
                assert f"{len(first_revision_task)} characters" in prompt
                assert "at most 1000" in prompt
                task = retry_revision_task
            return _tool_response(
                "revise_subtask",
                title="First route",
                task=task,
                revision_summary="Restored the requested root quantity.",
            )

    root = HTNTaskNode(
        id="root",
        task="Determine the perceived displacement.",
        needs_decomposition=True,
    )
    tree = HTNTree(root_id="root", nodes={"root": root})
    reviser = _RevisionLLM()
    diagnostics = _RecordingDiagnostics()
    decision = await ResearchDecompositionStrategy(
        llm=_PlannerLLM(),
        task_strategy=get_task_strategy("science_olympiad").planning,
        audit_llm=_AuditLLM(),
        revision_llm=reviser,
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=1,
            max_children=4,
        ),
    )

    assert reviser.calls == 2
    assert len(retry_revision_task) > 1_000
    assert decision.children[0].task == retry_revision_task
    assert "KNOWN-BAD" not in decision.children[0].task
    retry_end = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_revision_length_retry_end"
    )
    assert retry_end["status"] == "over_limit_accepted"
    assert retry_end["initial_task_chars"] == len(first_revision_task)
    assert retry_end["final_task_chars"] == len(retry_revision_task)
    summary = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_audit_summary"
    )
    assert summary["revision_success_count"] == 1
    assert summary["revision_failure_count"] == 0
    assert summary["post_revision_pass_count"] == 1


@pytest.mark.asyncio
async def test_planner_malformed_subtask_audit_retries_once_then_fails_open() -> None:
    class _PlannerLLM:
        model = "planner"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return _tool_response(
                "decompose",
                rationale="Use one bounded discovery route.",
                subtasks=[
                    {
                        "id": "discover",
                        "title": "Discover candidates",
                        "task": "Return a short candidate ledger.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    }
                ],
            )

    class _MalformedAuditLLM:
        model = "auditor"
        reasoning_effort = "low"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return LLMResponse(content="PASS", tool_calls=[])

    class _ForbiddenRevisionLLM:
        model = "reviser"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            raise AssertionError("malformed fail-open audit must not trigger revision")

    root = HTNTaskNode(id="root", task="Identify the item.", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    diagnostics = _RecordingDiagnostics()
    decision = await ResearchDecompositionStrategy(
        llm=_PlannerLLM(),
        audit_llm=_MalformedAuditLLM(),
        revision_llm=_ForbiddenRevisionLLM(),
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.children[0].task == "Return a short candidate ledger."
    summary = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_audit_summary"
    )
    assert summary["initial_invalid_fail_open_count"] == 1
    assert summary["initial_revise_count"] == 0
    retry_events = [
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_audit_retry_end"
    ]
    assert len(retry_events) == 1
    assert retry_events[0]["status"] == "invalid_fail_open"


@pytest.mark.asyncio
async def test_planner_protocol_retry_requires_nonempty_pass_quotes() -> None:
    class _PlannerLLM:
        async def chat(self, _messages, tools=None):  # noqa: ANN001
            return _tool_response(
                "decompose",
                rationale="Use one bounded route.",
                subtasks=[
                    {
                        "id": "discover",
                        "title": "Discover candidates",
                        "task": "Try four targeted queries and return at most five candidates.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    }
                ],
            )

    class _AuditLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return _tool_response(
                    "audit_subtask_fidelity",
                    verdict="PASS",
                    issue_type="none",
                    authoritative_context_quote="",
                    subtask_quote="",
                    feedback="",
                )
            assert "Protocol correction only" in messages[-1]["content"]
            assert "omitted a required non-empty quote" in messages[-1]["content"]
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="Identify the item.",
                subtask_quote="Try four targeted queries",
                feedback="",
            )

    root = HTNTaskNode(id="root", task="Identify the item.", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    diagnostics = _RecordingDiagnostics()
    auditor = _AuditLLM()
    decision = await ResearchDecompositionStrategy(
        llm=_PlannerLLM(),  # type: ignore[arg-type]
        audit_llm=auditor,  # type: ignore[arg-type]
        diagnostics=diagnostics,  # type: ignore[arg-type]
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.children[0].task.startswith("Try four targeted queries")
    assert auditor.calls == 2
    audit_end = next(
        fields
        for event, fields in diagnostics.events
        if event == "planner_subtask_audit_end"
    )
    assert audit_end["status"] == "valid_after_protocol_retry"


def _dependent_tree() -> tuple[HTNTree, HTNTaskNode]:
    root = HTNTaskNode(
        id="root",
        title="Question",
        task="Identify the item from several clues.",
        children=["route-a", "route-b", "reconcile"],
    )
    route_a = HTNTaskNode(
        id="route-a",
        parent_id="root",
        title="Route A",
        task="Discover candidates by route A.",
        status=HTNTaskStatus.COMPLETED,
        result_summary="Finding: candidate X. Unresolved: publication date.",
        ref="/tmp/full-route-a-secret.md",
    )
    route_b = HTNTaskNode(
        id="route-b",
        parent_id="root",
        title="Route B",
        task="Discover candidates by route B.",
        status=HTNTaskStatus.COMPLETED,
        result_summary="Finding: candidate Y. Contradicted: candidate X author clue.",
        ref="/tmp/full-route-b-secret.md",
    )
    reconcile = HTNTaskNode(
        id="reconcile",
        parent_id="root",
        title="Reconcile candidates",
        task="Resolve candidate conflicts against every original constraint.",
        depends_on=["route-a", "route-b"],
        needs_decomposition=True,
    )
    tree = HTNTree(
        root_id="root",
        nodes={
            node.id: node
            for node in (root, route_a, route_b, reconcile)
        },
    )
    return tree, reconcile


@pytest.mark.asyncio
async def test_planner_receives_bounded_result_summaries_not_refs() -> None:
    class _LLM:
        model = "test"

        def __init__(self) -> None:
            self.messages = []
            self.tools = []

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.messages = messages
            self.tools = tools
            return _tool_response(
                "execute_direct",
                rationale="One adaptive reconciliation leaf is sufficient.",
            )

    tree, reconcile = _dependent_tree()
    llm = _LLM()
    budget = DecompositionBudget.for_node(
        current_depth=1,
        max_depth=2,
        max_children=4,
    )

    decision = await ResearchDecompositionStrategy(llm=llm).decompose(
        tree,
        reconcile,
        budget,
    )

    prompt = llm.messages[1]["content"]
    assert decision.action == "execute_direct"
    assert "candidate X" in prompt
    assert "candidate Y" in prompt
    assert 'relation="direct"' in prompt
    assert "full-route-a-secret.md" not in prompt
    assert {tool["function"]["name"] for tool in llm.tools} == {
        "decompose",
        "execute_direct",
    }
    system_prompt = llm.messages[0]["content"]
    assert "Planning controls search; it does not nominate answers" in system_prompt
    assert "Do not turn such a list into a closed fan-out" in system_prompt
    assert "candidate-agnostic discovery or reconciliation route" in system_prompt
    assert "join evidence against ALL decisive ROOT constraints" in system_prompt


def test_precursor_context_is_bounded_and_direct_results_have_priority() -> None:
    tree, reconcile = _dependent_tree()
    tree.get_node("route-a").result_summary = "A" * 200
    tree.get_node("route-b").result_summary = "B" * 200

    rendered = _render_precursor_evidence(
        tree,
        reconcile,
        max_total_chars=100,
        max_summary_chars=70,
    )

    assert "precursor summary truncated" in rendered
    assert "A" * 100 not in rendered
    assert rendered.count('relation="direct"') == 2
    summary_payload = "".join(
        block.split("\n", 1)[1].rsplit("\n", 1)[0]
        for block in rendered.split("\n\n<precursor")
    )
    assert len(summary_payload) <= 100


def test_descendant_planning_receives_completed_ancestor_precursors() -> None:
    route = HTNTaskNode(
        id="route",
        parent_id="root",
        title="Discovery",
        status=HTNTaskStatus.COMPLETED,
        result_summary="Finding: candidate Z.",
    )
    parent = HTNTaskNode(
        id="reconcile",
        parent_id="root",
        title="Reconcile",
        depends_on=["route"],
        children=["deep-check"],
    )
    child = HTNTaskNode(
        id="deep-check",
        parent_id="reconcile",
        title="Deep check",
        needs_decomposition=True,
    )
    root = HTNTaskNode(
        id="root",
        children=["route", "reconcile"],
        task="Question",
    )
    tree = HTNTree(
        root_id="root",
        nodes={node.id: node for node in (root, route, parent, child)},
    )

    rendered = _render_precursor_evidence(
        tree,
        child,
        max_total_chars=1000,
        max_summary_chars=1000,
    )

    assert 'relation="ancestor"' in rendered
    assert "candidate Z" in rendered


@pytest.mark.asyncio
async def test_explicit_execute_direct_marks_node_executable() -> None:
    class _Strategy:
        async def decompose(self, tree, node, budget):  # noqa: ANN001, ARG002
            return DecompositionDecision(
                action="execute_direct",
                rationale="The remaining work is one coherent verification.",
            )

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner = HTNPlanner(strategy=_Strategy(), max_depth=2, max_children=4)

    children = await planner.decompose(tree, "root")

    assert children == []
    assert root.needs_decomposition is False
    assert tree.ready_for_execution() == [root]


@pytest.mark.asyncio
async def test_result_informed_decompose_attaches_targeted_children() -> None:
    class _LLM:
        model = "test"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            assert "candidate X" in messages[1]["content"]
            assert "candidate Y" in messages[1]["content"]
            return _tool_response(
                "decompose",
                rationale="The reports conflict on identity and publication date.",
                subtasks=[
                    {
                        "id": "identity-conflict",
                        "title": "Resolve identity conflict",
                        "task": "Independently test candidates X and Y against the author clue.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "date-conflict",
                        "title": "Resolve date conflict",
                        "task": "Verify the exact publication date from a primary source.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                ],
            )

    tree, reconcile = _dependent_tree()
    planner = HTNPlanner(
        strategy=ResearchDecompositionStrategy(llm=_LLM()),
        max_depth=2,
        max_children=4,
    )

    children = await planner.decompose(tree, reconcile.id)

    assert [child.id for child in children] == [
        "reconcile-identity-conflict",
        "reconcile-date-conflict",
    ]
    assert all(child.parent_id == "reconcile" for child in children)
    assert all(not child.needs_decomposition for child in children)


@pytest.mark.asyncio
async def test_unbounded_result_informed_subtask_is_locally_revised_with_precursors() -> None:
    unsafe = [
        {
            "id": "catalogue",
            "title": "Global catalogue",
            "task": "Search globally and enumerate every possible candidate.",
            "depends_on": [],
            "needs_decomposition": False,
        }
    ]
    class _PlannerLLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            self.calls += 1
            return _tool_response(
                "decompose",
                rationale="Search all candidates.",
                subtasks=unsafe,
            )

    class _AuditLLM:
        model = "audit"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            assert tools[0]["function"]["name"] == "audit_subtask_fidelity"
            assert "candidate X" in messages[1]["content"]
            assert "candidate Y" in messages[1]["content"]
            if self.calls == 1:
                return _tool_response(
                    "audit_subtask_fidelity",
                    verdict="REVISE",
                    issue_type="operationally_unbounded",
                    authoritative_context_quote="(no authorizing text)",
                    subtask_quote="Search globally and enumerate every possible candidate.",
                    feedback=(
                        "Add a finite stopping condition while preserving an open route."
                    ),
                )
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="Identify the item from several clues.",
                subtask_quote="open candidate-discovery route",
                feedback="",
            )

    class _RevisionLLM:
        model = "revision"

        async def chat(self, messages, tools=None):  # noqa: ANN001
            assert tools[0]["function"]["name"] == "revise_subtask"
            assert "candidate X" in messages[1]["content"]
            assert "candidate Y" in messages[1]["content"]
            assert "Do not delete a candidate-agnostic discovery route" in messages[0][
                "content"
            ]
            return _tool_response(
                "revise_subtask",
                title="Global catalogue",
                task=(
                    "Use an open candidate-discovery route for five distinct query "
                    "formulations, then return at most five ranked candidates."
                ),
                revision_summary="Added a finite stopping condition.",
            )

    tree, reconcile = _dependent_tree()
    budget = DecompositionBudget.for_node(
        current_depth=1,
        max_depth=2,
        max_children=4,
    )

    planner_llm = _PlannerLLM()
    audit_llm = _AuditLLM()
    decision = await ResearchDecompositionStrategy(
        llm=planner_llm,
        audit_llm=audit_llm,
        revision_llm=_RevisionLLM(),
    ).decompose(
        tree,
        reconcile,
        budget,
    )

    assert decision.action == "decompose"
    assert [child.id for child in decision.children] == ["reconcile-catalogue"]
    assert "open candidate-discovery route" in decision.children[0].task
    assert "at most five" in decision.children[0].task
    assert planner_llm.calls == 1
    assert audit_llm.calls == 2


@pytest.mark.asyncio
async def test_subtask_audit_accepts_negated_finite_enumeration_without_revision() -> None:
    finite_flagged = [
        {
            "id": "shortlist",
            "title": "Build finite shortlist",
            "task": (
                "Try these six targeted queries. Do NOT attempt an exhaustive global "
                "enumeration. Return at most five ranked candidates."
            ),
            "depends_on": [],
            "needs_decomposition": False,
        }
    ]

    class _PlannerLLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            return _tool_response(
                "decompose",
                rationale="Use a finite candidate-producing route.",
                subtasks=finite_flagged,
            )

    class _AuditLLM:
        model = "audit"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            assert tools[0]["function"]["name"] == "audit_subtask_fidelity"
            assert "Do NOT attempt an exhaustive global enumeration" in messages[1]["content"]
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="Identify the item.",
                subtask_quote="Do NOT attempt an exhaustive global enumeration",
                feedback="",
            )

    root = HTNTaskNode(id="root", task="Identify the item.", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner_llm = _PlannerLLM()
    audit_llm = _AuditLLM()
    decision = await ResearchDecompositionStrategy(
        llm=planner_llm,
        audit_llm=audit_llm,
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    assert [child.id for child in decision.children] == ["root-shortlist"]
    assert planner_llm.calls == 1
    assert audit_llm.calls == 1
    assert not decision.rationale.startswith("Repaired structurally invalid decomposition.")


@pytest.mark.asyncio
async def test_unbounded_open_route_revision_preserves_candidate_discovery() -> None:
    unbounded = [
        {
            "id": "catalogue",
            "title": "Global catalogue",
            "task": (
                "Keep an open route to find a better-supported candidate not named by "
                "siblings, with no stopping condition."
            ),
            "depends_on": [],
            "needs_decomposition": False,
        }
    ]
    class _PlannerLLM:
        model = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            return _tool_response(
                "decompose",
                rationale="Keep an open discovery route.",
                subtasks=unbounded,
            )

    class _AuditLLM:
        model = "audit"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                assert "open route" in messages[1]["content"]
                return _tool_response(
                    "audit_subtask_fidelity",
                    verdict="REVISE",
                    issue_type="operationally_unbounded",
                    authoritative_context_quote="(no authorizing text)",
                    subtask_quote="with no stopping condition",
                    feedback=(
                        "Add a finite stopping condition; preserve the ability to find "
                        "an unnamed candidate."
                    ),
                )
            return _tool_response(
                "audit_subtask_fidelity",
                verdict="PASS",
                issue_type="none",
                authoritative_context_quote="Identify the item.",
                subtask_quote="Keep an open route to find a better-supported candidate",
                feedback="",
            )

    class _RevisionLLM:
        model = "revision"

        async def chat(self, messages, tools=None):  # noqa: ANN001
            assert "Do not delete a candidate-agnostic discovery route" in messages[0][
                "content"
            ]
            return _tool_response(
                "revise_subtask",
                title="Global catalogue",
                task=(
                    "Keep an open route to find a better-supported candidate not named by "
                    "siblings. Try five distinct query formulations and return at most four "
                    "new candidates."
                ),
                revision_summary="Bounded the open route without closing it.",
            )

    root = HTNTaskNode(id="root", task="Identify the item.", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner_llm = _PlannerLLM()
    audit_llm = _AuditLLM()
    decision = await ResearchDecompositionStrategy(
        llm=planner_llm,
        audit_llm=audit_llm,
        revision_llm=_RevisionLLM(),
    ).decompose(
        tree,
        root,
        DecompositionBudget.for_node(
            current_depth=0,
            max_depth=2,
            max_children=4,
        ),
    )

    assert decision.action == "decompose"
    assert [child.id for child in decision.children] == ["root-catalogue"]
    assert "not named by siblings" in decision.children[0].task
    assert "at most four new candidates" in decision.children[0].task
    assert "no new candidates" not in decision.children[0].task
    assert planner_llm.calls == 1
    assert audit_llm.calls == 2
    assert not decision.rationale.startswith("Repaired structurally invalid decomposition.")


@pytest.mark.asyncio
async def test_valid_dependencies_force_a_later_informed_planning_decision() -> None:
    class _LLM:
        model = "test"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return _tool_response(
                "decompose",
                rationale="Discover first, then reconcile from the actual result.",
                subtasks=[
                    {
                        "id": "discover",
                        "title": "Discover candidates",
                        "task": "Return a short candidate ledger.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "reconcile",
                        "title": "Reconcile candidates",
                        "task": "Decide what follow-up the discovered candidates require.",
                        "depends_on": ["discover"],
                        # The runtime must override this because the task shape
                        # depends on a result that does not exist yet.
                        "needs_decomposition": False,
                    },
                ],
            )

    root = HTNTaskNode(id="root", task="Question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    budget = DecompositionBudget.for_node(
        current_depth=0,
        max_depth=2,
        max_children=4,
    )

    decision = await ResearchDecompositionStrategy(llm=_LLM()).decompose(
        tree,
        root,
        budget,
    )

    discover, reconcile = decision.children
    assert discover.needs_decomposition is False
    assert reconcile.depends_on == [discover.id]
    assert reconcile.needs_decomposition is True


@pytest.mark.asyncio
async def test_dependency_does_not_exceed_the_depth_ceiling() -> None:
    class _LLM:
        model = "test"

        async def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return _tool_response(
                "decompose",
                rationale="Final-depth dependent work.",
                subtasks=[
                    {
                        "id": "source",
                        "title": "Find source",
                        "task": "Find the source.",
                        "depends_on": [],
                        "needs_decomposition": False,
                    },
                    {
                        "id": "extract",
                        "title": "Extract detail",
                        "task": "Extract the requested detail from that source.",
                        "depends_on": ["source"],
                        "needs_decomposition": True,
                    },
                ],
            )

    root = HTNTaskNode(id="root", task="Question", children=["parent"])
    parent = HTNTaskNode(
        id="parent",
        parent_id="root",
        task="Final split",
        needs_decomposition=True,
    )
    tree = HTNTree(root_id="root", nodes={"root": root, "parent": parent})
    budget = DecompositionBudget.for_node(
        current_depth=1,
        max_depth=2,
        max_children=4,
    )

    decision = await ResearchDecompositionStrategy(llm=_LLM()).decompose(
        tree,
        parent,
        budget,
    )

    assert all(not child.needs_decomposition for child in decision.children)
