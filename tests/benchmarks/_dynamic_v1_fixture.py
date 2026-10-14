"""Exercise the production entry, native envelopes, routing and report delivery offline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness.core.llm import LLMResponse
from workflows.dynamic_htn_informed_research.diagnostics import scope_snapshot
from workflows.dynamic_htn_informed_research.htn import AgentResultRecord
from workflows.dynamic_htn_informed_research.nodes.dynamic_solve import dynamic_solve_node
from workflows.dynamic_htn_informed_research.profile import load_profile_selection, resolve_task_profile
from workflows.dynamic_htn_informed_research.runtime.context import override_runtime
from workflows.dynamic_htn_informed_research.task_strategies import get_active_task_strategy

SOURCE = "https://example.org/evidence"
SMALL_CONFIG = {
    "llm": {"high_effort_timeout_s": 0},
    "agent": {
        "max_depth": 1,
        "max_children": 2,
        "max_rounds": 3,
        "max_agents_per_round": 1,
        "max_iterations": 3,
        "subagent_max_turns": 3,
        "progressive_fan_in": False,
        "fan_in_grace_s": 0,
        "leaf_timeout_s": 10,
        "question_timeout_s": 30,
        "finalization_reserve_s": 5,
        "answer_format_timeout_s": 2,
        "max_rebuilds": 0,
        "recovery_timeout_s": 2,
        "min_rebuild_remaining_s": 2,
        "forced_finalization_high_timeout_s": 2,
    },
}
ROUTES = [
    ("research", None),
    ("science_olympiad", None),
    ("science_mcq", None),
    ("report_research", "sourced_report"),
    ("report_research", "scientific_report"),
]


class ScriptedModel:
    model = "offline-fixture"

    def __init__(self, strategy, report_kind=None):
        self.strategy, self.report_kind = strategy, report_kind
        self.calls, self.dispatched, self.unexpected = [], set(), []
        self.worker_roles = []
        self.finalizer_inputs = []

    @property
    def report(self):
        if self.report_kind == "sourced_report":
            return f"# Result\n\nThe result is 42. [Fixture source]({SOURCE})"
        return "# Derivation\n\nUsing $x=6\\times7$, we obtain $x=42$.\n\nThe result is 42."

    def response(self, name, args):
        # These native id/type fields must be removed by the HTN adapter for
        # strict report parsers, and retained by the worker's native loop.
        return LLMResponse(
            tool_calls=[
                {
                    "id": f"call-{len(self.calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 5},
            model=self.model,
        )

    async def chat(self, messages, *, tools=None, **kwargs):
        await asyncio.sleep(0)  # Deliberately interleave concurrent tasks.
        names = {tool["function"]["name"] for tool in tools or []}
        scope = scope_snapshot()
        self.calls.append((names, scope))
        if "route_task" in names:
            args = {"strategy_name": self.strategy, "rationale": "Fixture route."}
            if self.report_kind:
                args["report_kind"] = self.report_kind
            return self.response("route_task", args)
        if "execute_direct" in names:
            return self.response("execute_direct", {"rationale": "One bounded task."})
        if "dispatch" in names:
            leaf = scope.get("leaf_id", "root")
            if leaf not in self.dispatched:
                self.dispatched.add(leaf)
                role = get_active_task_strategy().workers.role_ids[0]
                return self.response(
                    "dispatch",
                    {
                        "role_id": role,
                        "task_summary": "Establish the result with evidence.",
                        "rationale": "Acquire evidence for the requested result.",
                    },
                )
            return self.response("terminate", {})
        if "derive_requirements" in names:
            return self.response(
                "derive_requirements",
                {
                    "requirements": [
                        {
                            "description": "Explain the result.",
                            "acceptance_criteria": ["State the result."],
                            "priority": 1,
                            "kind": "fact",
                        }
                    ]
                },
            )
        if "extract_evidence" in names:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            requirement = payload["candidate_requirements"][0]
            record = payload["worker_records"][0]
            return self.response(
                "extract_evidence",
                {
                    "cards": [
                        {
                            "record_id": record["record_id"],
                            "supports": [
                                {
                                    "requirement_id": requirement["id"],
                                    "acceptance_criterion_id": requirement["acceptance_criteria"][
                                        0
                                    ]["id"],
                                }
                            ],
                            "claim": "The result is 42.",
                            "source_url": SOURCE if self.report_kind == "sourced_report" else None,
                            "source_title": "Fixture source"
                            if self.report_kind == "sourced_report"
                            else None,
                            "support_summary": "The submitted evidence establishes the result as 42.",
                            "confidence": 0.9,
                        }
                    ]
                },
            )
        if "plan_report_outline" in names:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            self.finalizer_inputs.append(payload)
            return self.response(
                "plan_report_outline",
                {
                    "sections": [
                        {
                            "title": "Result",
                            "requirement_ids": [r["id"] for r in payload["requirements"]],
                        }
                    ]
                },
            )
        if "draft_report" in names:
            return self.response("draft_report", {"content": self.report})
        if "verify_report" in names:
            return self.response("verify_report", {"findings": []})
        if "finalize_root_answer" in names:
            return self.response(
                "finalize_root_answer",
                {
                    "requested_output_fields": ["result"],
                    "evidence_summary": "The evidence gives 42.",
                    "answer": self.report
                    if self.report_kind
                    else "B"
                    if self.strategy == "science_mcq"
                    else "42",
                    "completion_outcome": "resolved",
                },
            )
        if "finalize_answer" in names:
            return self.response(
                "finalize_answer",
                {
                    "result_summary": "The result is 42.",
                    "completion_outcome": "resolved",
                },
            )
        if "submit_report" in names:
            self.worker_roles.append(get_active_task_strategy().workers.role_ids[0])
            return self.response("submit_report", {"content": self.report, "confidence": 0.9})
        if not names:
            return LLMResponse(
                content="Establish the result and submit the supporting evidence.",
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            )
        self.unexpected.append(sorted(names))
        return LLMResponse(content="Unexpected fixture call.")

    async def dispatch(self, agent, instruction):
        self.worker_roles.append(agent.role_id)
        return AgentResultRecord(
            agent_id=agent.agent_id,
            role_id=agent.role_id,
            task_summary=agent.task_summary,
            result_summary=f"The result is 42, from 6 times 7. [Fixture source]({SOURCE})",
            execution_quality="submitted",
            submitted=True,
            observed_source_index=f'- [exact_observed] "Fixture source" — {SOURCE}',
        )
