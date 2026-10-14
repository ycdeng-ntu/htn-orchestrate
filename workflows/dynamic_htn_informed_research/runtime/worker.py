"""One HTN dispatch runs one native worker; the orchestrator owns redispatch."""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_harness.core.llm import LLMResponse
from agent_harness.core.loop_types import BaseObserver, Intervention, LoopConfig, LoopPolicy
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.loop.model_profile import HistoryPolicy, ModelProfile, infer_thinking_format

from ..diagnostics import AgentLoopDiagnosticObserver, diagnostic_scope, serialize_messages
from ..htn import AgentResultRecord
from ..markup import strip_leaked_tool_markup
from ..source_index import build_observed_source_index
from ..task_strategies import get_active_task_strategy
from .context import current_runtime
from .evidence_memory import EvidenceMemory
from .llm import ResearchToolCallParser
from .tool_call_recovery import WorkerToolCallRecovery
from .worker_budget import DEFAULT_WORKER_TOKEN_BUDGET, InformedWorkerBudgetObserver

logger = logging.getLogger(__name__)


class SubmissionObserver(BaseObserver):
    critical = True

    def __init__(self, tools):
        self.tools = tools
        self.accepted = False

    async def on_tool_result(self, ctx, result):
        if result.name != "submit_report" or result.is_error or not self.tools.report_latch.submitted:
            return None
        self.accepted = True
        ctx.metadata["final_answer"] = self.tools.report_latch.report
        ctx.metadata["answer_confidence"] = self.tools.report_latch.confidence
        return None

    async def on_turn_end(self, ctx):
        # on_tool_result only accepts ToolResult replacements. Stop decisions
        # must be returned from an Intervention-bearing observer hook.
        if self.accepted:
            return Intervention(stop_reason="submitted_report")
        return None


class _SingleTurnPolicyObserver(BaseObserver):
    """Seed the final-turn policy when no preceding turn exists to arm it."""
    critical = True

    async def on_llm_response(self, ctx):
        recorded = next(
            (m.get("tool_calls", []) for m in reversed(ctx.messages) if m.get("role") == "assistant"), [],
        )
        parser = ResearchToolCallParser()
        calls = parser.parse_unfiltered(
            LLMResponse(content=ctx.ai_text, tool_calls=recorded),
        )
        # Recovered text calls have already acquired a native history envelope;
        # retain inspection of unknown text companions in the original content.
        calls.extend(parser.parse_text_unfiltered(ctx.ai_text))
        requested = {call["name"] for call in calls}
        ctx.metadata["tool_policy"] = {"turn": ctx.turn, "allowed_tools": ["submit_report"]}
        if requested - {"submit_report"}:
            ctx.metadata["tool_policy_violation"] = {
                "turn": ctx.turn, "allowed_tools": ["submit_report"],
                "disallowed_tools": sorted(requested - {"submit_report"}),
                "mixed_with_allowed": "submit_report" in requested,
                "call_formats": ["native" if recorded else "text_fallback"],
            }
            return Intervention(stop_reason="tool_policy_violation")
        return None


def worker_capability_instruction(tools, *, network_enabled: bool) -> str:
    """Describe runtime permissions, after role/final-turn tool filtering."""
    names = {tool.name for tool in tools}
    if network_enabled:
        return ""  # Keep existing online worker instructions unchanged.
    abilities = []
    if "run_python_code" in names:
        abilities.append("offline Python computation")
    if "read_file" in names:
        abilities.append("file readback from this worker's artifacts, task reports and sandbox")
    if "submit_report" in names:
        abilities.append("report submission via submit_report")
    return ("\n\nRuntime capabilities: only " + ", ".join(abilities or ["no tools"]) +
            " are available. External retrieval is disabled, including from Python. "
            "Do not search or download external information. Use the supplied question, "
            "your reasoning and results actually produced by available tools. "
            "Never present proposed or unexecuted computations as measured results.")


def build_dispatch(
    parent_task_id: str,
    *,
    max_turns: int,
    sampling: dict[str, Any],
    provider: str,
    model: str,
    language: str | None = None,
    ref_dir: Path | None = None,
    job_timeout_s: float | None = None,
    diagnostics=None,
    malformed_trace_summarizer=None,
    reasoning_effort=None,
):
    if max_turns < 1:
        raise ValueError("Worker max_turns must be positive")
    if job_timeout_s is not None and job_timeout_s <= 0:
        raise ValueError("Worker job timeout must be positive")
    runtime = current_runtime()
    if runtime.overrides.dispatch is not None:
        return runtime.overrides.dispatch
    report_dir = (ref_dir or runtime.artifact_dir / "refs").resolve()
    strategy = get_active_task_strategy()
    if strategy.name == "report_research":
        from ..task_strategies.report_research import report_strategy_for_access

        strategy = report_strategy_for_access(
            network_enabled=(runtime.web_policy or {}).get("network_enabled", True),
        )
    elif strategy.name == "science_olympiad":
        from ..task_strategies.science_olympiad import science_strategy_for_access

        strategy = science_strategy_for_access(
            network_enabled=(runtime.web_policy or {}).get("network_enabled", True),
        )
    elif strategy.name == "science_mcq":
        from ..task_strategies.science_mcq import mcq_strategy_for_access

        strategy = mcq_strategy_for_access(
            network_enabled=(runtime.web_policy or {}).get("network_enabled", True),
        )
    definitions = {definition.role_id: definition for definition in strategy.workers.agent_definitions}

    async def dispatch(agent, instruction: str):
        runtime.raise_if_quota_failed()
        from ..adapters import _AttemptOutcome, _classify_attempt
        from ..tools import ResearchToolConfig, build_research_tools

        if agent.role_id not in definitions:
            raise ValueError(f"Worker role {agent.role_id!r} does not belong to strategy {strategy.name}")
        job_id = re.sub(r"[^A-Za-z0-9._+-]", "_", agent.agent_id)[:120] + "-" + uuid.uuid4().hex[:12]
        worker_dir = runtime.artifact_dir / "workers" / job_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        if language:
            instruction += f"\n\n(Write the answer in: {language}.)"
        tool_config = ResearchToolConfig.from_profile(
            runtime.profile,
            web_policy=runtime.web_policy,
            usage_sink=runtime.record_external_usage,
            scrape_cache=runtime.scrape_cache,
            summary_llm=runtime.resolved_summary_config(),
            summary_environment=runtime.summary_environment(),
            **runtime.resolved_tool_config(),
        )
        # Reports are shared within this dispatch's question, independently of
        # each worker's tool artifacts. Authorize only that report directory.
        tool_config.read_roots = (*tool_config.read_roots, report_dir)
        tool_factory = runtime.overrides.tool_factory or build_research_tools
        agent_cfg = runtime.profile.get("agent") or {}
        cfg = LoopConfig(
            task_id=job_id, role_id=agent.role_id, max_turns=max_turns,
            max_tool_calls_per_turn=max(1, int(agent_cfg.get("max_tool_calls_per_turn", 50))),
            max_llm_retries=max(1, int(agent_cfg.get("max_llm_retries", 5))),
            llm_timeout=int(agent_cfg.get("llm_timeout_s", 600)),
            # Summary extraction has bounded 300s requests plus retries and a
            # fallback candidate. The worker deadline owns its overall budget;
            # an unrelated 120s outer cap must not preempt those mechanisms.
            tool_timeout=int(agent_cfg.get("tool_timeout_s", job_timeout_s or 7200)),
            compact_after_turns=max(100, int(agent_cfg.get("compact_after_turns", 100))),
            context_token_limit=int(agent_cfg.get("context_token_limit", 120000)),
            loop_policy=LoopPolicy(
                no_tool_behavior="nudge", terminal_tool_names=("submit_report",),
                no_tool_nudge_message=(
                    "Do not return the report as an ordinary chat message. If your "
                    "research is complete, call `submit_report` using a numeric "
                    "confidence between 0 and 1 that reflects the strength and "
                    "completeness of your evidence now. Otherwise call a research tool and continue."
                ),
            ),
        )
        tool_runtime = tool_factory(
            agent.role_id, task_id=parent_task_id, artifact_dir=worker_dir,
            config=tool_config,
        )
        if inspect.isawaitable(tool_runtime):
            tool_runtime = await tool_runtime
        if not tool_config.network_enabled:
            tool_runtime.config.read_roots = (report_dir,)
        submission = SubmissionObserver(tool_runtime)
        budget = InformedWorkerBudgetObserver(
            max_tokens=int(agent_cfg.get("worker_token_budget", DEFAULT_WORKER_TOKEN_BUDGET)),
        )
        parser = WorkerToolCallRecovery()
        observers = [
            submission, budget, parser,
        ]
        selected_tools = tool_runtime.tools
        if max_turns == 1:
            selected_tools = [t for t in selected_tools if t.name == "submit_report"]
            observers.insert(0, _SingleTurnPolicyObserver())
            instruction += "\n\nThis is your final turn. Only submit_report is available; submit your best supported report with numeric confidence."
        instruction += worker_capability_instruction(
            selected_tools, network_enabled=tool_config.network_enabled,
        )
        artifact_dir = getattr(tool_runtime, "artifact_dir", None)
        evidence_memory = EvidenceMemory(
            Path(artifact_dir) if artifact_dir is not None else worker_dir,
            keep_tool_results=int(agent_cfg.get("keep_tool_results", 10)),
            can_read=artifact_dir is not None and any(t.name == "read_file" for t in selected_tools),
        )
        cfg.compactor = evidence_memory
        observers.append(evidence_memory)
        if diagnostics:
            observers.append(AgentLoopDiagnosticObserver(diagnostics, job_id, agent.role_id, reasoning_effort))
        last_messages: list = []
        last_metadata: dict = {}

        class CaptureTrace(BaseObserver):
            critical = True

            async def on_llm_response(self, ctx):
                nonlocal last_messages, last_metadata
                # Keep the live list so results appended during execution are
                # available even if this turn times out before its checkpoint.
                last_messages, last_metadata = ctx.messages, ctx.metadata

        observers.append(CaptureTrace())

        async def checkpoint(turn, messages, metadata):
            nonlocal last_messages, last_metadata
            last_messages, last_metadata = messages, metadata
            runtime.write_json(f"workers/{job_id}/checkpoint.json", {
                "turn": turn, "messages": serialize_messages(messages), "metadata": metadata,
            })

        result, error, failure_stop_reason = None, None, None
        started = time.monotonic()
        with diagnostic_scope(parent_task_id=parent_task_id, agent_id=agent.agent_id,
                              role_id=agent.role_id, job_id=job_id, call_kind="worker"):
            try:
                loop = run_agent_loop(
                    system_prompt=definitions[agent.role_id].system_prompt,
                    user_message=instruction,
                    llm=runtime.native_llm(provider, model, sampling),
                    tools=selected_tools, config=cfg, observers=observers,
                    parser=parser, on_turn_complete=checkpoint,
                    # Echo native reasoning with each assistant turn so tool
                    # continuations do not depend on upstream reconstruction.
                    history_policy=HistoryPolicy(preserve_reasoning_fields=True),
                    model_profile=ModelProfile(
                        model_id=model, provider=provider,
                        thinking_format=agent_cfg.get("thinking_format") or infer_thinking_format(model),
                    ),
                    scope_metadata={"parent_task_id": parent_task_id, "artifact_dir": str(worker_dir)},
                )
                result = await asyncio.wait_for(loop, timeout=job_timeout_s) if job_timeout_s else await loop
            except asyncio.CancelledError:
                runtime.write_json(f"workers/{job_id}/cancelled.json", {
                    "messages": serialize_messages(last_messages), "metadata": last_metadata,
                    "submitted": submission.accepted,
                })
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failure_stop_reason = (
                    "timeout" if isinstance(exc, TimeoutError) else "execution_error"
                )
                if not last_messages and parser.last_raw_response is not None:
                    raw = parser.last_raw_response
                    last_messages = [
                        {"role": "system", "content": definitions[agent.role_id].system_prompt},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": raw.content, "tool_calls": raw.tool_calls},
                    ]
            finally:
                try:
                    await tool_runtime.aclose()
                except Exception as exc:
                    last_metadata["cleanup_error"] = type(exc).__name__
                    logger.warning("Worker tool cleanup failed: %s", type(exc).__name__)

            metadata = dict(result.metadata or {}) if result is not None else last_metadata
            submitted = submission.accepted
            content = tool_runtime.report_latch.report if submitted else (getattr(result, "final_content", "") or "")
            full = strip_leaked_tool_markup(content)
            stopped = getattr(result, "stopped_by", None) or failure_stop_reason
            metadata["stopped_by"] = stopped
            success = error is None and stopped != "llm_error"
            if not success and error is None:
                error = str(metadata.get("llm_error") or "worker model call failed")
            attempt = _AttemptOutcome(content, submitted, job_id, success, error, metadata, (time.monotonic() - started) * 1000)
            quality, reason = _classify_attempt(attempt, full)
            if not submitted and stopped == "malformed_tool_calls":
                quality = "malformed"
                reason = metadata["tool_call_correction"]["error"]
            recovery = None
            if quality == "malformed" and malformed_trace_summarizer:
                recovery = await malformed_trace_summarizer(
                    result or SimpleNamespace(messages=last_messages), job_id,
                    SimpleNamespace(question=instruction, role_id=agent.role_id),
                )
            report = full
            if recovery:
                report = "[TRACE RECOVERY MEMO — generated from an unsubmitted malformed run; this is not a completed report]\n\n" + recovery
            target = report_dir / f"{job_id}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(full, encoding="utf-8")
            records = getattr(tool_runtime, "evidence_cards", [])
            runtime.write_json(f"workers/{job_id}/result.json", {
                "execution_quality": quality, "quality_reason": reason,
                "submitted": submitted, "stopped_by": stopped,
                "metadata": metadata, "error": error,
                "messages": serialize_messages(getattr(result, "messages", None) or last_messages),
                "evidence_cards": records,
            })
            return AgentResultRecord(
                agent_id=agent.agent_id, role_id=agent.role_id,
                task_summary=agent.task_summary, result_summary=report,
                ref=str(target), execution_quality=quality, quality_reason=reason,
                submitted=submitted, stop_reason=stopped, trace_recovery=bool(recovery),
                observed_source_index=build_observed_source_index(report, records),
            )
    return dispatch
