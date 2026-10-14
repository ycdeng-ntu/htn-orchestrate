"""Report protocol correction, raw-response isolation, and fixed stage budgets."""

import asyncio
from copy import deepcopy
import json
import time

import pytest

from agent_harness.core.llm import LLMResponse as NativeResponse
from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.runtime.llm import NativeHTNClient, raw_tool_envelope
from workflows.dynamic_htn_informed_research.task_strategies.report_research.protocol import (
    report_call,
    tool_payload,
)
from workflows.dynamic_htn_informed_research.task_strategies.report_research.premise_review import (
    PREMISE_TOOL,
    parse_premise_checks,
)

NAME = "derive_premise_checks"
QUESTION = "Calculate six times seven."
MESSAGES = [
    {"role": "system", "content": "Read only the public question."},
    {"role": "user", "content": QUESTION},
]
GOOD = {
    "checks": [
        {
            "question_quote": QUESTION,
            "constraint": "Use the given operands.",
            "independent_reasoning": "Six groups of seven total 42.",
            "test": "Compare with repeated addition.",
        }
    ]
}


def native(arguments, *, name=NAME, reasoning=""):
    return NativeResponse(
        tool_calls=[
            {
                "id": "call-recorded",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
        reasoning_content=reasoning,
    )


class RecordedClient:
    model = "recorded"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


async def call(client, tmp_path, **kwargs):
    return await report_call(
        llm=client,
        messages=MESSAGES,
        tool=PREMISE_TOOL,
        parse=lambda r: parse_premise_checks(tool_payload(r, NAME), QUESTION),
        stage="independent_premises",
        deadline=kwargs.pop("deadline", time.monotonic() + 5),
        diagnostics=DiagnosticRecorder(tmp_path),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        '{"checks">[\n: ',
        '{"checks":[',
        json.dumps({"checks": []}),
        json.dumps({"checks": [{**GOOD["checks"][0], "test": 123}]}),
        json.dumps({"checks": [{**GOOD["checks"][0], "question_quote": "Invented question"}]}),
    ],
)
async def test_raw_failures_get_one_correction_with_full_context_and_real_error(tmp_path, bad):
    transport = RecordedClient(
        native(bad, reasoning="REASONING_MUST_NOT_BE_FEEDBACK"), native(json.dumps(GOOD))
    )
    response, checks = await call(NativeHTNClient(transport), tmp_path)
    assert len(checks) == 1
    assert len(transport.calls) == 2
    retry = transport.calls[1]
    assert retry[:-1] == MESSAGES
    feedback = retry[-1]["content"]
    assert "Validation error:" in feedback
    assert bad in json.loads(feedback.split("\n")[-1])["tool_calls"][0]["function"]["arguments"]
    assert "REASONING_MUST_NOT_BE_FEEDBACK" not in json.dumps(retry)
    assert tool_payload(response, NAME) == GOOD
    files = list((tmp_path / "report_protocol").glob("*-invalid.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["response"]["tool_calls"][0]["function"]["arguments"] == bad


@pytest.mark.asyncio
async def test_repeat_failure_is_not_silently_salvaged_from_reasoning(tmp_path):
    bad = native('{"checks">[\n: ', reasoning=json.dumps({"name": NAME, "arguments": GOOD}))
    transport = RecordedClient(bad, bad)
    with pytest.raises(ValueError, match="invalid JSON"):
        await call(NativeHTNClient(transport), tmp_path)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_wrong_tool_and_missing_field_have_actionable_feedback(tmp_path):
    transport = RecordedClient(native(json.dumps(GOOD), name="wrong"), native("{}"))
    with pytest.raises(ValueError, match="required property"):
        await call(NativeHTNClient(transport), tmp_path)
    assert f"expected tool {NAME}" in transport.calls[1][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", [2, 3])
async def test_transport_failure_is_not_retried_by_protocol_layer(tmp_path, max_attempts):
    client = RecordedClient(RuntimeError("transport failure"))
    with pytest.raises(RuntimeError):
        await call(client, tmp_path, max_attempts=max_attempts)
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", [2, 3])
async def test_protocol_cancellation_propagates(tmp_path, max_attempts):
    client = RecordedClient(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await call(client, tmp_path, max_attempts=max_attempts)
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", [2, 3])
async def test_all_attempts_share_one_stage_deadline(tmp_path, max_attempts):
    class Slow:
        calls = 0
        cancelled = False

        async def chat(self, *args, **kwargs):
            self.calls += 1
            if self.calls < max_attempts:
                await asyncio.sleep(0.015)
                return LLMResponse()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    client = Slow()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await call(client, tmp_path, deadline=start + 0.08, max_attempts=max_attempts)
    assert client.calls == max_attempts and client.cancelled
    assert time.monotonic() - start < 0.3


@pytest.mark.asyncio
async def test_expired_stage_does_not_call_model(tmp_path):
    client = RecordedClient()
    with pytest.raises(TimeoutError):
        await call(client, tmp_path, deadline=time.monotonic() - 1)
    assert not client.calls


@pytest.mark.asyncio
async def test_raw_mode_is_context_local_and_restored_after_failure(tmp_path):
    # Existing non-report behavior can losslessly close this truncated object;
    # report calls must ask for a complete response instead.
    text = '{"checks": []'
    transport = RecordedClient(native(text), native(text), native(text), native(text))
    client = NativeHTNClient(transport)

    async def regular():
        return await client.chat(MESSAGES, tools=[PREMISE_TOOL])

    async def report():
        with pytest.raises(ValueError, match="invalid JSON"):
            await call(client, tmp_path)

    normal, _ = await asyncio.gather(regular(), report())
    assert tool_payload(normal, NAME) == {"checks": []}
    assert tool_payload(await regular(), NAME) == {"checks": []}


@pytest.mark.asyncio
async def test_exhausted_native_transport_retry_is_not_mistaken_for_empty_output(tmp_path):
    transport = RecordedClient(RuntimeError("transport exhausted"))
    with pytest.raises(RuntimeError, match="transport exhausted"):
        await call(NativeHTNClient(transport, attempts=1), tmp_path)
    assert len(transport.calls) == 1


def wrapped(payload=GOOD, name=NAME):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": payload}) + "\n</tool_call>"


@pytest.mark.asyncio
async def test_single_explicit_content_wrapper_is_validated_without_an_extra_call(tmp_path):
    transport = RecordedClient(NativeResponse(content=wrapped()))
    response, checks = await call(NativeHTNClient(transport), tmp_path)
    assert len(transport.calls) == len(checks) == 1
    assert tool_payload(response, NAME) == GOOD
    assert response.content == wrapped()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    "Explanation: " + wrapped(), wrapped() + " trailing prose", wrapped() + wrapped(),
    "```json\n" + wrapped() + "\n```", wrapped(name="wrong"), wrapped({"checks": []}),
    '<tool_call>{"name":"wrong","name":"derive_premise_checks","arguments":{}}</tool_call>',
    '<tool_call>{"name":"derive_premise_checks","arguments":{},"extra":1}</tool_call>',
    '<tool_call>{"name":"derive_premise_checks","arguments":',
])
async def test_ambiguous_or_invalid_text_wrappers_still_require_correction(tmp_path, content):
    transport = RecordedClient(NativeResponse(content=content), native(json.dumps(GOOD)))
    _, checks = await call(NativeHTNClient(transport), tmp_path)
    assert len(transport.calls) == 2 and len(checks) == 1


@pytest.mark.asyncio
async def test_text_wrapper_does_not_override_native_tool_arguments(tmp_path):
    first = native("{}")
    first.content = wrapped()
    transport = RecordedClient(first, native(json.dumps(GOOD)))
    await call(NativeHTNClient(transport), tmp_path)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_first_attempt_can_use_the_whole_stage_budget(tmp_path, monkeypatch):
    recorded = []
    original = asyncio.wait_for

    async def record_timeout(awaitable, timeout):
        recorded.append(timeout)
        return await original(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", record_timeout)
    await call(NativeHTNClient(RecordedClient(native(json.dumps(GOOD)))), tmp_path,
               deadline=time.monotonic() + 10)
    assert recorded[0] > 9  # Previously cancelled at half the same stage budget.


@pytest.mark.asyncio
async def test_correction_reports_multiple_missing_fields(tmp_path):
    checks = deepcopy(GOOD["checks"] * 2)
    del checks[0]["question_quote"]
    del checks[1]["test"]
    transport = RecordedClient(native(json.dumps({"checks": checks})), native(json.dumps(GOOD)))
    await call(NativeHTNClient(transport), tmp_path)
    feedback = transport.calls[1][-1]["content"]
    assert "checks.0" in feedback and "question_quote" in feedback
    assert "checks.1" in feedback and "'test'" in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["planner", "delivery", "fidelity", "decomposition", "synthesis", "reviser"])
@pytest.mark.parametrize("third_response_valid", [True, False])
async def test_planning_roles_allow_two_corrections_with_unchanged_inputs(
    tmp_path, role, third_response_valid,
):
    from workflows.dynamic_htn_informed_research.htn.planner import DecompositionBudget
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_audit import (
        call_plan_audit, PLAN_AUDIT_TOOLS, plan_constraints, produce_report_plan,
    )

    context = {"task": QUESTION, "recovery_capsule": "Keep the checked operands; close the gap."}
    plan = {"nodes": [{"id": "research", "title": "Checked result", "task": QUESTION,
                       "depends_on": []}], "rationale": "One complete milestone"}
    audit = {"has_delivery": False,
             "nodes": [{"id": "research", "kind": "research", "reason": "Research result"}],
             "plan_review": {"verdict": "REVISE", "issues": [{
                 "node_ids": ["research"], "issue": "Missing completion check",
                 "repair": "Include an independent check within the milestone",
             }]}}
    is_auditor = role in PLAN_AUDIT_TOOLS
    tool_name = (PLAN_AUDIT_TOOLS[role]["function"]["name"] if is_auditor else
                 {"planner": "plan_report_research", "reviser": "revise_report_plan"}[role])
    good = (audit if role == "delivery" else {"plan_review": audit["plan_review"]}) if is_auditor else plan
    if role == "synthesis":
        good["dispositions"] = [{"source_id": "fidelity:1", "issue_numbers": [1], "reason": "Confirmed missing check"}]
    bad = deepcopy(good)
    if is_auditor:
        del bad["plan_review"]["issues"][0]["repair"]
    else:
        del bad["nodes"][0]["depends_on"]
    transport = RecordedClient(
        native(json.dumps(bad), name=tool_name),
        native(json.dumps(bad), name=tool_name),
        native(json.dumps(good if third_response_valid else bad), name=tool_name),
        native(json.dumps(good), name=tool_name),  # Must never use a fourth attempt.
    )
    budget = DecompositionBudget.for_node(current_depth=0, max_depth=1, max_children=4)
    common = dict(planner_input=context, llm=NativeHTNClient(transport),
                  deadline=time.monotonic() + 5, diagnostics=DiagnosticRecorder(tmp_path),
                  epoch_id=1)
    if is_auditor:
        from workflows.dynamic_htn_informed_research.task_strategies.report_research.plan_rendering import render_planning_input
        pending = call_plan_audit(
            dimension=role, expected_ids=["research"],
            rendered_input=render_planning_input(planner_input=context, candidate_plan=plan,
                                                 constraints=plan_constraints(budget)),
            source_ids=["fidelity:1"] if role == "synthesis" else None,
            **{key: value for key, value in common.items() if key not in {"planner_input", "epoch_id"}},
        )
    else:
        pending = produce_report_plan(
            constraints=plan_constraints(budget), system="Test the planning protocol",
            previous_plan=plan if role == "reviser" else None,
            audit=audit if role == "reviser" else None, **common,
        )
    if third_response_valid:
        assert await pending == good
    else:
        with pytest.raises(ValueError, match="required property"):
            await pending
    assert len(transport.calls) == 3
    original = transport.calls[0]
    for retry in transport.calls[1:]:
        assert retry[:-1] == original
        assert "required property" in retry[-1]["content"]
    material = original[1]["content"]
    assert material.startswith("# Planning input\n")
    assert all(value in material for value in context.values())
    assert "Maximum research nodes: 4" in material
    if is_auditor or role == "reviser":
        assert "## Candidate plan" in material
        assert plan["rationale"] in material
        assert "ID: research" in material
        assert "Depends on: None" in material
    if role == "reviser":
        assert "## Audit feedback" in material
        assert audit["nodes"][0]["reason"] in material
        assert audit["plan_review"]["issues"][0]["issue"] in material
        assert audit["plan_review"]["issues"][0]["repair"] in material
    assert len(list((tmp_path / "report_protocol").glob("*-invalid.json"))) == (
        2 if third_response_valid else 3
    )


@pytest.mark.asyncio
async def test_planning_normalization_keeps_exact_complete_payload_and_records_raw_response(tmp_path):
    arguments = json.dumps(GOOD, ensure_ascii=False)
    raw = arguments + ' }\n }\t'
    transport = RecordedClient(native(raw))
    response, checks = await call(NativeHTNClient(transport), tmp_path, normalize_surplus_closers=True)
    assert len(checks) == len(transport.calls) == 1
    assert response.tool_calls[0]['function']['arguments'] == arguments
    artifacts = list((tmp_path / 'report_protocol').glob('*-normalized.json'))
    assert len(artifacts) == 1
    saved = json.loads(artifacts[0].read_text())
    assert saved['original_response']['tool_calls'][0]['function']['arguments'] == raw
    assert saved['normalized_response']['tool_calls'][0]['function']['arguments'] == arguments


@pytest.mark.asyncio
async def test_surplus_closer_recovery_is_opt_in(tmp_path):
    transport = RecordedClient(native(json.dumps(GOOD) + '}'), native(json.dumps(GOOD)))
    await call(NativeHTNClient(transport), tmp_path)
    assert len(transport.calls) == 2
    assert not list((tmp_path / 'report_protocol').glob('*-normalized.json'))


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [
    json.dumps(GOOD) + ' explanation',
    json.dumps(GOOD) + '} explanation',
    json.dumps(GOOD) + ' {}',
    json.dumps(GOOD) + '} {}',
    json.dumps(GOOD) + ']',
    json.dumps(GOOD)[:-1],
    'prefix ' + json.dumps(GOOD) + '}',
    '{"checks": [], "checks": ' + json.dumps(GOOD['checks']) + '}}',
    '{"checks": [], "x": NaN}}',
])
async def test_normalization_does_not_accept_ambiguous_or_incomplete_arguments(tmp_path, arguments):
    transport = RecordedClient(native(arguments), native(json.dumps(GOOD)))
    await call(NativeHTNClient(transport), tmp_path, normalize_surplus_closers=True)
    assert len(transport.calls) == 2
    assert not list((tmp_path / 'report_protocol').glob('*-normalized.json'))


@pytest.mark.asyncio
async def test_normalized_arguments_still_need_all_required_fields(tmp_path):
    bad = deepcopy(GOOD)
    del bad['checks'][0]['test']
    transport = RecordedClient(native(json.dumps(bad) + '}'), native(json.dumps(GOOD)))
    await call(NativeHTNClient(transport), tmp_path, normalize_surplus_closers=True)
    assert len(transport.calls) == 2
    assert "'test' is a required property" in transport.calls[1][-1]['content']
    assert 'separate JSON keys' in transport.calls[1][-1]['content']
    assert len(list((tmp_path / 'report_protocol').glob('*-normalized.json'))) == 1


def test_normalization_preserves_literal_braces_numbers_and_source_text_without_mutation():
    from workflows.dynamic_htn_informed_research.task_strategies.report_research.protocol import _trim_surplus_object_closers
    text = '  {"evidence":"literal } and { plus \\"quoted\\" 中文", "number": 1.2300e-40}'
    original = LLMResponse(tool_calls=[{'function': {'name': NAME, 'arguments': text + ' }'}}])
    normalized = _trim_surplus_object_closers(original, NAME)
    assert normalized.tool_calls[0]['function']['arguments'] == text
    assert original.tool_calls[0]['function']['arguments'] == text + ' }'


@pytest.mark.asyncio
async def test_native_length_metadata_reaches_control_without_reasoning_recovery():
    transport = RecordedClient(NativeResponse(content="", reasoning_content="PRIVATE_DRAFT",
                                              finish_reason="length"))
    with raw_tool_envelope():
        response = await NativeHTNClient(transport).chat(MESSAGES, tools=[PREMISE_TOOL])
    assert response.finish_reason == "length"
    assert response.content == "" and response.tool_calls == []
    assert not hasattr(response, "reasoning_content")
