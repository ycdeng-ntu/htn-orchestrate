"""Behavioral tests for the report DAG, delivery session and rebuild boundaries."""

from __future__ import annotations

import asyncio
import json
import re
import time
from types import SimpleNamespace

import pytest

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    AgentResultRecord,
    ExecutionBudget,
    OrchestratorRequest,
)
from workflows.dynamic_htn_informed_research.htn.planner import DecompositionDecision, HTNPlanner
from workflows.dynamic_htn_informed_research.htn.task_node import HTNCompletionOutcome, HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.task_strategies import get_task_strategy
from workflows.dynamic_htn_informed_research.task_strategies.report_research.delivery import (
    ReportDeliveryOrchestrator,
    parse_review,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.topology import (
    ReportPlanningStrategy,
    delivery_root_summary,
)


class PlanAuditor:
    model = "test-auditor"

    def __init__(self, removed=(), responses=()):
        self.removed = set(removed)
        self.responses = list(responses)
        self.calls = []  # Delivery-classification calls, including its protocol retries.
        self.all_calls = []
        self.last_review = {"verdict": "PASS", "issues": []}

    async def chat(self, messages, tools=None):
        name = tools[0]["function"]["name"]
        self.all_calls.append((messages, tools))
        if name == "audit_report_plan":
            self.calls.append((messages, tools))
            nodes = [{"id": node_id} for node_id in re.findall(r"^ID: (.+)$", messages[1]["content"], re.M)]
            payload = {"has_delivery": any(n["id"] in self.removed for n in nodes),
                       "plan_review": {"verdict": "PASS", "issues": []}, "nodes": [{
                "id": node["id"],
                "kind": "report_delivery" if node["id"] in self.removed else "research",
                "reason": "Classified the task objective",
            } for node in nodes]}
            if self.responses:
                payload = self.responses.pop(0)
                if isinstance(payload, BaseException):
                    raise payload
            self.last_review = payload.get("plan_review", {"verdict": "PASS", "issues": []})
        elif name == "synthesize_report_plan_audit":
            payload = {"plan_review": self.last_review, "dispositions": [
                {"source_id": f"delivery:{index}", "issue_numbers": [index],
                 "reason": "The candidate task confirms the reported defect"}
                for index, _ in enumerate(self.last_review["issues"], 1)
            ]}
        else:
            assert name in {"audit_report_fidelity", "audit_report_decomposition"}
            payload = {"plan_review": {"verdict": "PASS", "issues": []}}
        return LLMResponse(tool_calls=[{"function": {
            "name": name, "arguments": json.dumps(payload),
        }}])


class Plan:
    model = "test-planner"

    def __init__(self, mode="decompose", nodes=None):
        self.mode = mode
        self.nodes = nodes
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        if self.mode == "failure":
            raise ValueError("Planner transport failure")
        nodes = self.nodes
        if nodes is None:
            nodes = [HTNTaskNode(id="a", title="Derive", task="Derive the quantity")]
            if self.mode != "direct":
                nodes.append(HTNTaskNode(id="b", title="Check", task="Check the limiting case",
                                         depends_on=["a"]))
        payload = {"nodes": [{"id": n.id, "title": n.title or n.id, "task": n.task,
                               "depends_on": n.depends_on} for n in nodes],
                   "rationale": "One coherent derivation and necessary independent checks"}
        return LLMResponse(tool_calls=[{"function": {
            "name": tools[0]["function"]["name"], "arguments": json.dumps(payload),
        }}])


def report_strategy(planner, auditor, reviser=None, **kwargs):
    return ReportPlanningStrategy(
        planner_llm=planner, audit_llm=auditor,
        revision_llm=reviser or Plan("failure"),
        planner_system=get_task_strategy("report_research").planning.decomposition_system,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["decompose", "direct", "failure"])
async def test_planner_always_builds_shallow_dependency_gated_delivery(mode):
    root = HTNTaskNode(id="root", task="Public question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root}, summarizer=delivery_root_summary)
    planner = HTNPlanner(strategy=report_strategy(Plan(mode), PlanAuditor()), max_depth=1, max_children=3)
    children = await planner.decompose(tree, "root")
    delivery = children[-1]
    assert delivery.execution_kind == "report_delivery"
    assert set(delivery.depends_on) == {node.id for node in children[:-1]}
    assert all(tree.depth_of(node.id) == 1 and not node.needs_decomposition for node in children)
    assert tree.validate() == []
    assert not tree.is_ready(delivery.id)
    for node in children[:-1]:
        assert tree.is_ready(node.id)
        tree.mark_running(node.id)
        # Unresolved research material must still reach the full-answer verifier.
        await tree.mark_completed(
            node.id,
            "Supported derivation and an explicit gap",
            completion_outcome=HTNCompletionOutcome.EXHAUSTED_UNRESOLVED,
        )
    assert tree.is_ready(delivery.id)
    tree.mark_running(delivery.id)
    await tree.mark_completed(delivery.id, "EXACT REPORT\nWith verified conditional results")
    assert tree.root().result_summary == "EXACT REPORT\nWith verified conditional results"
    assert tree.root().completion_outcome is HTNCompletionOutcome.RESOLVED


class Writer:
    model = "test-writer"

    def __init__(self, action="draft"):
        self.histories = []
        self.action = action

    async def chat(self, messages, tools=None):
        self.histories.append(list(messages))
        draft = (
            "Initial report: cost = 3" if len(self.histories) == 1 else "Corrected report: cost = 6"
        )
        return LLMResponse(
            tool_calls=[
                {
                    "function": {
                        "name": "compose_report",
                        "arguments": json.dumps(
                            {
                                "action": self.action,
                                "report": draft,
                                "reason": "Need the missing derivation"
                                if self.action == "rebuild"
                                else "",
                                "affected_node_ids": ["research"],
                            }
                        ),
                    }
                }
            ]
        )


def record(agent, text, quality="submitted"):
    return AgentResultRecord(
        agent.agent_id,
        agent.role_id,
        agent.task_summary,
        text,
        execution_quality=quality,
        submitted=quality == "submitted",
    )


def review(verdict, action="revise", *, material=None):
    material = material or {
        "question": "Calculate total cost and explain all assumptions",
        "draft": "Initial report: cost = 3", "previous_findings": [],
    }
    prior = material["previous_findings"]
    return {
        "verdict": verdict,
        "checks": [{"id": "cost", "question_quote": "Calculate total cost",
                    "report_quote": material["draft"], "method": "Recompute total cost",
                    "result": "2 * 3 = 6; compared against current report",
                    "status": "supported" if verdict == "pass" else "contradicted"}],
        "findings": [] if verdict == "pass" else [{
            "finding_id": prior[0]["finding_id"] if prior else "",
            "issue": "Cost table omitted a factor of two", "action": action,
            "affected_node_ids": ["research"], "check_ids": ["cost"],
        }],
        "finding_reviews": [{
            "finding_id": f["finding_id"],
            "status": "resolved" if verdict == "pass" else "unresolved",
            "check_ids": ["cost"], "reason": "Compared current cost against 2 * 3 = 6",
        } for f in prior],
    }


def verification_material(instruction):
    return json.loads(instruction.split("\n\n", 1)[1])


def delivery_fixture(tmp_path, writer, dispatch, budget=None, deadline=None):
    root = HTNTaskNode(
        id="root", task="Calculate total cost and explain all assumptions", needs_decomposition=True
    )
    tree = HTNTree(root_id="root", nodes={"root": root}, summarizer=delivery_root_summary)
    tree.decompose(
        "root",
        [
            HTNTaskNode(id="research", task="Derive costs"),
            HTNTaskNode(id="delivery", execution_kind="report_delivery", depends_on=["research"]),
        ],
    )
    req = OrchestratorRequest(
        tree=tree,
        leaf=tree.get_node("delivery"),
        available_agents=[],
        execution_budget=ExecutionBudget(max_rounds=4),
        output_dir=tmp_path,
        llm=writer,
        dispatch=dispatch,
        dispatch_budget=budget,
        diagnostics=DiagnosticRecorder(tmp_path),
        task_strategy=get_task_strategy("report_research"),
    )
    source = AgentResultRecord(
        "source", "researcher", "Derive costs", "COMPLETE_EVIDENCE " + "x" * 9000
    )
    return ReportDeliveryOrchestrator(
        req,
        research_orchestrators={"research": SimpleNamespace(results=(source,))},
        deadline=deadline or time.monotonic() + 10,
    )


@pytest.mark.asyncio
async def test_local_revision_retains_context_and_reverifies_exact_revised_draft(tmp_path):
    writer = Writer()
    instructions = []

    async def dispatch(agent, instruction):
        instructions.append(instruction)
        return record(agent, json.dumps(review("revise" if len(instructions) == 1 else "pass",
                                              material=verification_material(instruction))))

    budget = QuestionDispatchBudget(limit=2)
    owner = delivery_fixture(tmp_path, writer, dispatch, budget)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert result.result_summary == "Corrected report: cost = 6"
    assert len(instructions) == budget.used == 2
    assert all("COMPLETE_EVIDENCE" in text and "x" * 9000 in text for text in instructions)
    assert "Initial report: cost = 3" in instructions[0]
    assert "Corrected report: cost = 6" in instructions[1]
    history = str(writer.histories[1])
    assert "Initial report: cost = 3" in history and "omitted a factor of two" in history
    assert len(owner.results) == 2
    first, second = [verification_material(i) for i in instructions]
    assert first["previous_findings"] == first["previous_reviews"] == []
    assert second["previous_findings"][0]["finding_id"] == "e0-r0-f1"
    assert second["previous_reviews"][0]["review"]["findings"][0]["finding_id"] == "e0-r0-f1"
    assert owner.review_state["findings"][0]["last_assessment"]["status"] == "resolved"
    assert (tmp_path / "report_delivery/epoch-0/review-state.json").exists()


@pytest.mark.asyncio
async def test_research_finding_returns_to_planner_without_local_research_or_rewrite(tmp_path):
    writer = Writer()
    calls = []

    async def dispatch(agent, instruction):
        calls.append(agent.role_id)
        # Even a mislabelled revise verdict cannot hide an explicit research requirement.
        return record(agent, json.dumps(review("revise", action="research")))

    owner = delivery_fixture(tmp_path, writer, dispatch)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert result.result_summary == "Initial report: cost = 3"
    assert calls == ["dynir_report_final_verifier"]
    assert len(writer.histories) == 1
    assert "omitted a factor of two" in owner.rebuild_feedback
    assert set(owner.request.tree.nodes) == {"root", "research", "delivery"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,quality",
    [
        ("CONFIRMED", "submitted"),
        (json.dumps(review("pass")), "best_effort"),
        (json.dumps({"verdict": "pass", "checks": [], "findings": []}), "submitted"),
        (json.dumps({**review("rebuild", "research"), "verdict": "pass"}), "submitted"),
    ],
)
async def test_incomplete_or_contradictory_review_cannot_pass(tmp_path, content, quality):
    async def dispatch(agent, instruction):
        return record(agent, content, quality)

    owner = delivery_fixture(tmp_path, Writer(), dispatch)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert owner.rebuild_feedback


@pytest.mark.asyncio
async def test_no_verifier_quota_keeps_draft_unverified(tmp_path):
    async def forbidden(*args):
        pytest.fail("Exhausted quota must not launch a worker")

    budget = QuestionDispatchBudget(limit=1)
    budget.try_acquire()
    owner = delivery_fixture(tmp_path, Writer(), forbidden, budget)
    result = await owner.run()
    assert (
        result.result_summary
        and result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    )
    assert budget.used == 1


@pytest.mark.asyncio
async def test_cancellation_quiesces_final_verifier_and_keeps_draft(tmp_path):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def dispatch(agent, instruction):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    owner = delivery_fixture(tmp_path, Writer(), dispatch)
    task = asyncio.create_task(owner.run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert owner.partial_result().result_summary == "Initial report: cost = 3"


@pytest.mark.asyncio
async def test_writer_can_request_rebuild_but_cannot_skip_verification_to_pass(tmp_path):
    async def forbidden(*args):
        pytest.fail("Rebuild must not launch another worker in the delivery leaf")

    owner = delivery_fixture(tmp_path, Writer(action="rebuild"), forbidden)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert "missing derivation" in owner.rebuild_feedback


def test_fenced_json_review_is_accepted_without_accepting_prose():
    context = dict(question="Calculate total cost", draft="Initial report: cost = 3",
                   previous_findings=[], research_node_ids={"research"})
    assert parse_review("```json\n" + json.dumps(review("pass")) + "\n```", **context)["verdict"] == "pass"
    with pytest.raises(ValueError):
        parse_review("The answer looks fine. " + json.dumps(review("pass")), **context)


@pytest.mark.asyncio
async def test_deadline_cancels_verifier_without_losing_draft(tmp_path):
    cancelled = asyncio.Event()

    async def dispatch(agent, instruction):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    owner = delivery_fixture(tmp_path, Writer(), dispatch, deadline=time.monotonic() + 0.05)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert not owner.rebuild_requested
    assert cancelled.is_set()
    assert owner.partial_result().result_summary == "Initial report: cost = 3"


@pytest.mark.asyncio
async def test_blank_draft_cannot_be_verified(tmp_path):
    class BlankWriter(Writer):
        async def chat(self, messages, tools=None):
            response = await super().chat(messages, tools)
            call = response.tool_calls[0]["function"]
            payload = json.loads(call["arguments"])
            payload["report"] = "   "
            call["arguments"] = json.dumps(payload)
            return response

    async def forbidden(*args):
        pytest.fail("A blank report must not reach a verifier")

    owner = delivery_fixture(tmp_path, BlankWriter(), forbidden)
    with pytest.raises(ValueError):
        await owner.run()
    assert not owner.draft
    assert owner.partial_result().completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED


async def audited_tree(nodes, auditor, *, diagnostics=None, deadline=None, epoch_id=0,
                       reviser=None):
    root = HTNTaskNode(id="root", task="Derive both requested quantities", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root}, epoch_id=epoch_id)
    planner = HTNPlanner(strategy=report_strategy(
        Plan(nodes=nodes), auditor, reviser or Plan(nodes=nodes),
        diagnostics=diagnostics, deadline=deadline,
    ), max_depth=1, max_children=len(nodes) + 1)
    children = await planner.decompose(tree, "root")
    assert tree.validate() == []
    assert len([c for c in children if c.execution_kind == "report_delivery"]) == 1
    assert children[-1].depends_on == [c.id for c in children[:-1]]
    return tree, children


@pytest.mark.asyncio
async def test_audit_removes_terminal_delivery_without_changing_research_dependencies(tmp_path):
    nodes = [
        HTNTaskNode(id="a", task="Derive quantity A"),
        HTNTaskNode(id="b", task="Calculate quantity B", depends_on=["a"]),
        HTNTaskNode(id="report-delivery", task="Write final report", depends_on=["a", "b"]),
    ]
    auditor = PlanAuditor(["report-delivery"])
    tree, children = await audited_tree(nodes, auditor, diagnostics=DiagnosticRecorder(tmp_path),
                                        epoch_id=2)
    assert [c.id for c in children] == ["a", "b", "report-delivery"]
    assert tree.get_node("b").depends_on == ["a"]
    assert not tree.is_ready("b")
    assert len(auditor.calls) == 1
    material = auditor.calls[0][0][1]["content"]
    assert tree.root().task in material
    assert re.findall(r"^ID: (.+)$", material, re.M) == [n.id for n in nodes]
    artifact = json.loads((tmp_path / "report_delivery/epoch-2/plan-audit-1.json").read_text())
    assert artifact["has_delivery"]
    assert artifact["plan_review"]["verdict"] == "PASS"


@pytest.mark.asyncio
@pytest.mark.parametrize("successor_is_report", [False, True])
async def test_report_successors_use_reviser_without_recalling_planner(tmp_path, successor_is_report):
    bad = [
        HTNTaskNode(id="a", task="Derive A"),
        HTNTaskNode(id="draft", task="Write final report", depends_on=["a"]),
        HTNTaskNode(id="later", task="Work after draft", depends_on=["draft"]),
    ]
    good = [HTNTaskNode(id="fresh", task="Derive and check all required quantities")]
    producer, reviser = Plan(nodes=bad), Plan(nodes=good)
    auditor = PlanAuditor(["draft", "later"] if successor_is_report else ["draft"])
    root = HTNTaskNode(id="root", task="Original public question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root}, restart_context="PRIOR_EVIDENCE")
    planner = HTNPlanner(strategy=report_strategy(
        producer, auditor, reviser, diagnostics=DiagnosticRecorder(tmp_path),
    ), max_depth=1, max_children=4)
    children = await planner.decompose(tree, "root")
    assert [c.id for c in children] == ["fresh", "report-delivery"]
    assert children[-1].depends_on == ["fresh"]
    assert tree.validate() == []
    assert len(producer.calls) == len(reviser.calls) == 1
    assert len(auditor.calls) == 1
    assert len(auditor.all_calls) == 4
    ip = {"task": root.task, "recovery_capsule": "PRIOR_EVIDENCE"}
    revision_input = reviser.calls[0][0][1]["content"]
    assert all(value in revision_input for value in ip.values())
    assert revision_input.split("## Audit feedback")[0].rstrip() == auditor.calls[0][0][1]["content"]
    assert "Delivery nodes have successors" in revision_input
    assert "Plan verdict: REVISE" in revision_input
    assert reviser.calls[0][1][0]["function"]["name"] == "revise_report_plan"
    assert tree.restart_context == "PRIOR_EVIDENCE"
    assert bad[-1].depends_on == ["draft"]
    for messages, _ in [*producer.calls, *auditor.calls, *reviser.calls]:
        assert all(value in messages[1]["content"] for value in ip.values())
    assert not (tmp_path / "report_delivery/epoch-0/plan-audit-2.json").exists()
    validation = json.loads((tmp_path / "report_delivery/epoch-0/revision-validation.json").read_text())
    assert validation == {"valid": True, "issues": [], "semantic_audit_performed": False}


@pytest.mark.asyncio
async def test_revision_does_not_reuse_old_delivery_classifications():
    auditor = PlanAuditor(["draft"])
    _, children = await audited_tree([
        HTNTaskNode(id="draft", task="Write final report"),
        HTNTaskNode(id="later", task="Work after draft", depends_on=["draft"]),
    ], auditor)
    assert len(auditor.calls) == 1
    assert [c.id for c in children] == ["draft", "later", "report-delivery"]
    assert all(c.execution_kind == "standard" for c in children[:-1])
    assert children[1].depends_on == ["draft"]


@pytest.mark.asyncio
async def test_audit_keeps_local_evidence_memos_even_with_reserved_delivery_id():
    nodes = [HTNTaskNode(id="report-delivery", title="Report measured results",
                         task="Calculate the value and return a local evidence memo")]
    auditor = PlanAuditor()
    _, children = await audited_tree(nodes, auditor)
    assert children[0].task == nodes[0].task
    assert children[0].execution_kind == "standard"
    assert children[-1].id == "report-delivery-final"
    assert len(auditor.calls) == 1


@pytest.mark.asyncio
async def test_all_delivery_nodes_need_revision_and_revision_failure_uses_fallback():
    _, children = await audited_tree(
        [HTNTaskNode(id="draft", task="Write the complete report")], PlanAuditor(["draft"]),
        reviser=Plan("failure"),
    )
    assert [c.id for c in children] == ["report-research", "report-delivery"]
    assert "Do not write" in children[0].task
    assert "Derive both requested quantities" in children[0].task


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"nodes": []},
    {"nodes": [{"id": "invented", "kind": "research", "reason": "Wrong ID"}]},
    {"nodes": [{"id": "a", "kind": "research", "reason": "Repeated"}] * 2},
    {"nodes": [{"id": "a", "kind": "other", "reason": "Unknown kind"}]},
    {"nodes": [{"id": "a", "kind": "research", "reason": " "}]},
])
async def test_invalid_audit_retries_then_discards_unaudited_plan(payload):
    auditor = PlanAuditor(responses=[payload, payload, payload])
    _, children = await audited_tree([HTNTaskNode(id="a", task="Untrusted task")], auditor)
    assert len(auditor.calls) == 3
    assert children[0].id == "report-research"
    assert "Untrusted task" not in children[0].task


@pytest.mark.asyncio
async def test_audit_protocol_correction_can_remove_delivery_node():
    auditor = PlanAuditor(["draft"], responses=[{"nodes": []}])
    _, children = await audited_tree([
        HTNTaskNode(id="a", task="Derive all requested quantities"),
        HTNTaskNode(id="draft", task="Write report", depends_on=["a"]),
    ], auditor)
    assert len(auditor.calls) == 2
    assert [c.id for c in children] == ["a", "report-delivery"]


@pytest.mark.asyncio
async def test_audit_transport_failure_does_not_execute_unaudited_plan():
    auditor = PlanAuditor(responses=[RuntimeError("Transport failed")])
    _, children = await audited_tree([HTNTaskNode(id="draft", task="Write report")], auditor)
    assert len(auditor.calls) == 1
    assert children[0].id == "report-research"


@pytest.mark.asyncio
async def test_audit_respects_deadline_and_cancellation():
    nodes = [HTNTaskNode(id="draft", task="Write report")]
    auditor = PlanAuditor()
    _, children = await audited_tree(nodes, auditor, deadline=time.monotonic() - 1)
    assert auditor.calls == []
    assert children[0].id == "report-research"
    with pytest.raises(asyncio.CancelledError):
        await audited_tree(nodes, PlanAuditor(responses=[asyncio.CancelledError()]))


@pytest.mark.asyncio
async def test_revision_cannot_overrun_planning_deadline():
    class HangingReviser:
        def __init__(self):
            self.calls = 0
            self.cancelled = False

        async def chat(self, messages, tools=None):
            self.calls += 1
            assert tools[0]["function"]["name"] == "revise_report_plan"
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    producer = Plan(nodes=[HTNTaskNode(id="draft", task="Write report"),
                           HTNTaskNode(id="after", task="Work after report", depends_on=["draft"])])
    reviser = HangingReviser()
    root = HTNTaskNode(id="root", task="Public question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner = HTNPlanner(strategy=report_strategy(
        producer, PlanAuditor(["draft"]), reviser, deadline=time.monotonic() + 0.1,
    ), max_depth=1, max_children=3)
    children = await asyncio.wait_for(planner.decompose(tree, "root"), timeout=1)
    assert len(producer.calls) == reviser.calls == 1
    assert reviser.cancelled
    assert [c.id for c in children] == ["report-research", "report-delivery"]


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["forgot_finding", "stale_draft_check"])
async def test_second_review_cannot_pass_with_missing_closure_or_old_draft(tmp_path, defect):
    calls = []

    async def dispatch(agent, instruction):
        material = verification_material(instruction)
        calls.append(material)
        payload = review("revise" if len(calls) == 1 else "pass", material=material)
        if len(calls) >= 2:
            if defect == "forgot_finding":
                payload["finding_reviews"] = []
            else:
                payload["checks"][0]["report_quote"] = calls[0]["draft"]
        return record(agent, json.dumps(payload))

    owner = delivery_fixture(tmp_path, Writer(), dispatch)
    result = await owner.run()
    assert len(calls) == 3
    assert result.result_summary == "Corrected report: cost = 6"
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert len(owner.review_state["reviews"]) == 1  # Invalid review cannot close the issue.


@pytest.mark.asyncio
async def test_writer_can_challenge_wrong_finding_and_verifier_can_refute_it(tmp_path):
    class DefendingWriter(Writer):
        async def chat(self, messages, tools=None):
            response = await super().chat(messages, tools)
            fn = response.tool_calls[0]["function"]
            payload = json.loads(fn["arguments"])
            payload["report"] = "Corrected report: cost = 6"
            if len(self.histories) > 1:
                payload["reason"] = "The earlier reviewer counted only one item; 2 * 3 = 6."
            fn["arguments"] = json.dumps(payload)
            return response

    calls = []

    async def dispatch(agent, instruction):
        material = verification_material(instruction)
        calls.append(material)
        payload = review("revise" if len(calls) == 1 else "pass", material=material)
        if len(calls) == 1:
            payload["findings"][0]["issue"] = "Reviewer wrongly expects the cost of one item"
        else:
            assert "2 * 3 = 6" in material["writer_reason"]
            payload["finding_reviews"][0].update(
                status="refuted", reason="The question asks for both items; 2 * 3 = 6 is correct")
        return record(agent, json.dumps(payload))

    owner = delivery_fixture(tmp_path, DefendingWriter(), dispatch)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert calls[0]["draft"] == calls[1]["draft"]  # No forced change to a correct report.
    assert owner.review_state["findings"][0]["last_assessment"]["status"] == "refuted"


@pytest.mark.asyncio
async def test_citation_repair_reuses_draft_and_preserves_review(tmp_path):
    from test_report_quote_alignment import Aligner
    calls = []

    async def dispatch(agent, instruction):
        material = verification_material(instruction)
        calls.append(material)
        p = review('pass', material=material)
        if len(calls) == 1:
            p['checks'][0]['report_quote'] = 'Cost = 999'
        else:
            assert material['review_correction']['previous_review']
            assert material['review_correction']['citations'][0]['status'] == 'mismatched'
        return record(agent, json.dumps(p))

    owner = delivery_fixture(tmp_path, Writer(), dispatch)
    owner.alignment_llm = Aligner('mismatched')
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.RESOLVED
    assert len(calls) == 2 and len(owner.messages) == 3  # One draft, no writer revision.
    assert calls[0]['draft'] == calls[1]['draft'] == result.result_summary
    assert not owner.rebuild_requested
    assert len(owner.review_state['reviews']) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['uncertain', 'transport', 'mismatched', 'dispatch_error'])
async def test_alignment_failure_keeps_draft_without_research_rebuild(tmp_path, failure):
    from test_report_quote_alignment import Aligner
    calls = []

    async def dispatch(agent, instruction):
        calls.append(agent)
        if failure == 'dispatch_error':
            raise ValueError('Invalid transport')
        p = review('pass', material=verification_material(instruction))
        p['checks'][0]['report_quote'] = 'Cost = 999'
        return record(agent, json.dumps(p))

    owner = delivery_fixture(tmp_path, Writer(), dispatch)
    if failure == 'transport':
        class Broken:
            async def chat(self, *args, **kwargs):
                raise ConnectionError('test only')
        owner.alignment_llm = Broken()
    else:
        owner.alignment_llm = Aligner(failure)
    result = await owner.run()
    assert result.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
    assert result.result_summary == 'Initial report: cost = 3'
    assert not owner.rebuild_requested and not owner.review_state['reviews']
    assert owner.delivery_status == 'verification_incomplete'
    assert len(calls) == (2 if failure == 'mismatched' else 1)
    assert not list(tmp_path.rglob('rebuild-feedback.md'))


def plan_review_payload(nodes, *, delivery=(), revise=False):
    return {
        "has_delivery": bool(delivery),
        "nodes": [{"id": n.id, "kind": "report_delivery" if n.id in delivery else "research",
                   "reason": "Classified from the actual task"} for n in nodes],
        "plan_review": {"verdict": "REVISE" if revise else "PASS", "issues": [{
            "node_ids": [n.id for n in nodes if n.id not in delivery],
            "issue": "Inventory and unit conversion fragment a single derivation",
            "repair": "Merge the coupled calculation; preserve the independent model check",
        }] if revise else []},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["", '{"material":"FULL_RECOVERY_EVIDENCE","gap":"check units"}'])
@pytest.mark.parametrize("has_delivery", [False, True])
async def test_whole_plan_revision_merges_nodes_and_preserves_exact_inputs(
    tmp_path, recovery, has_delivery,
):
    bad = [HTNTaskNode(id="inventory", task="List given quantities"),
           HTNTaskNode(id="convert", task="Convert the computed units", depends_on=["inventory"]),
           HTNTaskNode(id="check", task="Independently test the physical model", depends_on=["convert"])]
    if has_delivery:
        bad.append(HTNTaskNode(id="draft", task="Write the report", depends_on=["check"]))
    good = [HTNTaskNode(id="complete", task="Derive the result with all units and intermediate steps"),
            HTNTaskNode(id="independent", task="Test the alternative physical model", depends_on=["complete"])]
    producer, reviser = Plan(nodes=bad), Plan(nodes=good)
    first_audit = plan_review_payload(bad, delivery=["draft"] if has_delivery else [], revise=True)
    auditor = PlanAuditor(responses=[first_audit])
    root = HTNTaskNode(id="root", task="Derive the result and compare physical models", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root}, epoch_id=1 if recovery else 0,
                   restart_context=recovery)
    planner = HTNPlanner(strategy=report_strategy(producer, auditor, reviser,
                                                 diagnostics=DiagnosticRecorder(tmp_path)),
                         max_depth=1, max_children=5)
    children = await planner.decompose(tree, "root")
    assert [n.id for n in children] == ["complete", "independent", "report-delivery"]
    assert children[1].depends_on == ["complete"]
    assert children[-1].depends_on == ["complete", "independent"]
    assert tree.validate() == []
    assert not ({n.id for n in bad} & tree.nodes.keys())
    assert len(producer.calls) == len(reviser.calls) == 1
    assert len(auditor.calls) == 1
    assert len(auditor.all_calls) == 4
    pi = producer.calls[0][0][1]["content"]
    ai = auditor.calls[0][0][1]["content"]
    ri = reviser.calls[0][0][1]["content"]
    expected = {"task": root.task, **({"recovery_capsule": recovery} if recovery else {})}
    assert ai.split("## Candidate plan")[0].rstrip() == pi
    assert ri.split("## Candidate plan")[0].rstrip() == pi
    for material in (pi, ai, ri):
        assert all(value in material for value in expected.values())
        assert "Maximum research nodes: 4" in material
        assert "max_depth" not in material
    assert ri.split("## Audit feedback")[0].rstrip() == ai
    for issue in first_audit["plan_review"]["issues"]:
        assert issue["issue"] in ri and issue["repair"] in ri
    # Diagnostic objects stay structured, while the actual model message is Markdown.
    recorded = json.loads((tmp_path / f"report_delivery/epoch-{tree.epoch_id}/revised-plan.json").read_text())
    assert recorded["input"]["task_context"] == expected
    assert all(recorded["input"]["audit_feedback"][key] == value for key, value in first_audit.items())
    assert set(recorded["input"]["audit_feedback"]["dimension_audits"]) == {"delivery", "fidelity", "decomposition"}
    assert recorded["rendered_input"] == ri
    assert tree.restart_context == recovery
    assert tree.epoch_id == (1 if recovery else 0)


@pytest.mark.asyncio
async def test_delivery_only_defect_never_calls_reviser():
    nodes = [HTNTaskNode(id="derive", task="Derive all quantities"),
             HTNTaskNode(id="draft", task="Write final report", depends_on=["derive"])]
    reviser = Plan("failure")
    _, children = await audited_tree(nodes, PlanAuditor(["draft"]), reviser=reviser)
    assert [n.id for n in children] == ["derive", "report-delivery"]
    assert reviser.calls == []


@pytest.mark.asyncio
async def test_revision_is_not_classified_by_titles_or_stale_delivery_ids():
    bad = [HTNTaskNode(id="inventory", task="List given quantities"),
           HTNTaskNode(id="draft", task="Write final report", depends_on=["inventory"])]
    revised = [HTNTaskNode(id="complete", task="Derive and check all quantities"),
               HTNTaskNode(id="draft", title="Report independent evidence",
                           task="Calculate an independent observable", depends_on=["complete"])]
    auditor = PlanAuditor(["draft"], responses=[plan_review_payload(bad, delivery=["draft"], revise=True)])
    _, children = await audited_tree(bad, auditor, reviser=Plan(nodes=revised))
    assert len(auditor.calls) == 1
    assert len(auditor.all_calls) == 4
    assert [n.id for n in children] == ["complete", "draft", "report-delivery"]
    assert children[1].execution_kind == "standard"


@pytest.mark.asyncio
async def test_valid_revision_is_adopted_without_second_semantic_verdict():
    nodes = [HTNTaskNode(id="inventory", task="List given quantities")]
    verdict = plan_review_payload(nodes, revise=True)
    reviser = Plan(nodes=nodes)
    auditor = PlanAuditor(responses=[verdict, verdict])
    _, children = await audited_tree(nodes, auditor, reviser=reviser)
    assert len(reviser.calls) == 1
    assert len(auditor.calls) == 1
    assert len(auditor.all_calls) == 4
    assert auditor.responses == [verdict]  # A second semantic verdict is never requested.
    assert [n.id for n in children] == ["inventory", "report-delivery"]


@pytest.mark.asyncio
@pytest.mark.parametrize("nodes", [
    [HTNTaskNode(id="a", task="Derive", depends_on=["missing"])],
    [HTNTaskNode(id="a", task="Derive", depends_on=["b"]),
     HTNTaskNode(id="b", task="Check", depends_on=["a"])],
    [HTNTaskNode(id="a", task="Derive"), HTNTaskNode(id="a", task="Check")],
    [HTNTaskNode(id="root", task="Overwrite root")],
])
async def test_structurally_invalid_plan_cannot_bypass_reviser_with_model_pass(nodes):
    reviser = Plan(nodes=[HTNTaskNode(id="fixed", task="Derive and check all quantities")])
    _, children = await audited_tree(nodes, PlanAuditor(), reviser=reviser)
    assert [n.id for n in children] == ["fixed", "report-delivery"]
    material = reviser.calls[0][0][1]["content"]
    assert "### Auditor verdict before runtime checks\n\nPlan verdict: PASS" in material
    assert "## Audit feedback\n\nContains delivery nodes: No\n\nPlan verdict: REVISE" in material
    assert "### Runtime structural findings\n\n#### Finding 1" in material


@pytest.mark.asyncio
async def test_reviser_title_task_patch_is_not_accepted_as_a_complete_tree():
    class PatchReviser:
        async def chat(self, messages, tools=None):
            return LLMResponse(tool_calls=[{"function": {
                "name": "revise_report_plan",
                "arguments": json.dumps({"title": "Fixed", "task": "Complete the calculation"}),
            }}])

    nodes = [HTNTaskNode(id="inventory", task="List quantities")]
    auditor = PlanAuditor(responses=[plan_review_payload(nodes, revise=True)])
    _, children = await audited_tree(nodes, auditor, reviser=PatchReviser())
    assert [n.id for n in children] == ["report-research", "report-delivery"]
    assert len(auditor.calls) == 1  # No malformed replacement reaches another audit or execution.


@pytest.mark.asyncio
async def test_serial_milestones_remain_siblings_and_keep_execution_dependencies():
    nodes = [
        HTNTaskNode(id="model", task="Establish the physical model"),
        HTNTaskNode(id="spectrum", task="Determine the spectrum using the model", depends_on=["model"]),
        HTNTaskNode(id="response", task="Derive the observable response from the spectrum", depends_on=["spectrum"]),
        HTNTaskNode(id="design", task="Determine the experimental design from the response", depends_on=["response"]),
    ]
    producer, auditor, reviser = Plan(nodes=nodes), PlanAuditor(), Plan("failure")
    root = HTNTaskNode(id="root", task="Model, spectrum, response and experiment", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner = HTNPlanner(strategy=report_strategy(producer, auditor, reviser), max_depth=1, max_children=5)
    children = await planner.decompose(tree, "root")
    assert [n.id for n in children] == [n.id for n in nodes] + ["report-delivery"]
    assert all(n.parent_id == "root" and tree.depth_of(n.id) == 1 for n in children)
    assert all(not n.children and not n.needs_decomposition for n in children)
    assert tree.validate() == []
    assert reviser.calls == []
    for messages, _ in producer.calls + auditor.calls:
        assert "Maximum research nodes: 4" in messages[1]["content"]
        assert "Runtime owns final delivery: Yes" in messages[1]["content"]
        assert "max_depth" not in messages[1]["content"]
    for i, node in enumerate(children):
        assert tree.is_ready(node.id)
        assert not any(tree.is_ready(later.id) for later in children[i + 1:])
        tree.mark_running(node.id)
        await tree.mark_completed(node.id, "Checked result")


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"children": ["nested"]}, {"parent_id": "other"},
                                  {"needs_decomposition": True}])
async def test_nested_hierarchy_fields_are_rejected_before_semantic_audit(extra):
    class NestedPlan(Plan):
        async def chat(self, messages, tools=None):
            response = await super().chat(messages, tools)
            call = response.tool_calls[0]["function"]
            payload = json.loads(call["arguments"])
            payload["nodes"][0].update(extra)
            call["arguments"] = json.dumps(payload)
            return response

    producer, auditor = NestedPlan("direct"), PlanAuditor()
    root = HTNTaskNode(id="root", task="Public question", needs_decomposition=True)
    tree = HTNTree(root_id="root", nodes={"root": root})
    planner = HTNPlanner(strategy=report_strategy(producer, auditor), max_depth=1, max_children=5)
    children = await planner.decompose(tree, "root")
    assert len(producer.calls) == 3
    assert auditor.calls == []
    assert [n.id for n in children] == ["report-research", "report-delivery"]
    assert all(n.parent_id == "root" and not n.children for n in children)
    assert tree.validate() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["unknown", "self", "cycle", "duplicate", "root", "limit",
                                    "empty", "schema", "blank_task", "blank_title", "space_id"])
async def test_invalid_replacement_tree_is_blocked_without_second_audit(tmp_path, defect):
    initial = [HTNTaskNode(id="a", task="Derive all requested results"),
               HTNTaskNode(id="b", task="Check the physical model", depends_on=["a"])]

    class InvalidRevision(Plan):
        async def chat(self, messages, tools=None):
            response = await super().chat(messages, tools)
            call = response.tool_calls[0]["function"]
            payload = json.loads(call["arguments"])
            nodes = payload["nodes"]
            if defect == "unknown":
                nodes[0]["depends_on"] = ["absent"]
            elif defect == "self":
                nodes[0]["depends_on"] = ["a"]
            elif defect == "cycle":
                nodes[0]["depends_on"] = ["b"]
            elif defect == "duplicate":
                nodes[1]["id"] = "a"
            elif defect == "root":
                nodes[1]["id"] = "root"
            elif defect == "limit":
                nodes.append({"id": "c", "title": "Extra", "task": "Extra work", "depends_on": []})
            elif defect == "empty":
                payload["nodes"] = []
            elif defect == "schema":
                nodes[0]["children"] = ["b"]
            elif defect == "blank_task":
                nodes[0]["task"] = " "
            elif defect == "blank_title":
                nodes[0]["title"] = " "
            elif defect == "space_id":
                nodes[0]["id"] = " a "
            call["arguments"] = json.dumps(payload)
            return response

    auditor = PlanAuditor(responses=[plan_review_payload(initial, revise=True)])
    reviser = InvalidRevision(nodes=initial)
    _, children = await audited_tree(initial, auditor, reviser=reviser,
                                     diagnostics=DiagnosticRecorder(tmp_path))
    assert [n.id for n in children] == ["report-research", "report-delivery"]
    assert len(auditor.all_calls) == 4
    assert len(reviser.calls) == (3 if defect in {"empty", "schema"} else 1)
    artifact_dir = tmp_path / "report_delivery/epoch-0"
    acceptance = json.loads((artifact_dir / "plan-acceptance.json").read_text())
    assert acceptance["fallback"] and not acceptance["accepted"]
    assert not acceptance["accepted_by_audit"]
    assert not acceptance["revision_mechanically_validated"]
    assert not (artifact_dir / "plan-audit-2.json").exists()
    if defect not in {"empty", "schema"}:
        validation = json.loads((artifact_dir / "revision-validation.json").read_text())
        assert not validation["valid"] and validation["issues"]
        assert not validation["semantic_audit_performed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("revise", [False, True])
async def test_acceptance_distinguishes_audited_and_mechanically_validated_plans(tmp_path, revise):
    nodes = [HTNTaskNode(id="a", task="Derive and check all requested results")]
    auditor = PlanAuditor(responses=[plan_review_payload(nodes, revise=revise)])
    _, children = await audited_tree(nodes, auditor, diagnostics=DiagnosticRecorder(tmp_path))
    assert [n.id for n in children] == ["a", "report-delivery"]
    directory = tmp_path / "report_delivery/epoch-0"
    acceptance = json.loads((directory / "plan-acceptance.json").read_text())
    assert acceptance == {
        "accepted": True,
        "acceptance_method": "revision_mechanical" if revise else "initial_audit",
        "accepted_by_audit": not revise,
        "revision_mechanically_validated": revise,
        "fallback": False,
    }
    assert len(auditor.all_calls) == (4 if revise else 3)
    assert len(list(directory.glob("plan-audit-*.json"))) == 1
