"""Dynamic HTN driver, with all model/profile state isolated per question.

The epoch/recovery/report control flow is ported from the frozen Miro driver.
Only configuration, dispatch, resources and artifacts use AgentHarness adapters.
"""

from __future__ import annotations
import asyncio
import logging
import os
import sys
import re
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Awaitable, Callable
from agent_harness.core.atomic_files import atomic_write_text
from workflows.dynamic_htn_informed_research.adapters import (
    HighEffortTimeoutFallbackLLMClient,
    build_malformed_trace_summarizer,
    build_summarizer,
    build_worker_llm,
)
from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder, diagnostic_scope
from workflows.dynamic_htn_informed_research.finalize import _best_fallback_answer
from workflows.dynamic_htn_informed_research.profile import (
    resolve_reasoning_efforts,
    sampling_with_reasoning_effort,
)
from workflows.dynamic_htn_informed_research.task_strategies import (
    AUTO_TASK_STRATEGY_NAME,
    get_active_task_strategy,
    get_task_strategy,
    is_auto_task_strategy,
    normalize_task_strategy_name,
    set_active_task_strategy,
)
from workflows.dynamic_htn_informed_research.task_strategies.contracts import SynthesisStrategy
from workflows.dynamic_htn_informed_research.task_strategies.report_research.policy import (
    SCIENTIFIC_REPORT_POLICY,
    ReportPolicy,
)
from workflows.dynamic_htn_informed_research.task_router import TaskRoute, route_task
from workflows.dynamic_htn_informed_research.htn import (
    AgentDescriptor,
    HTNCompletionOutcome,
    HTNTaskStatus,
)
from workflows.dynamic_htn_informed_research.htn.recovery import (
    CompletedNodeEvidence,
    FailedNodeEvidence,
    SupportedRecoveryFact,
    TimeoutRecoveryEvidence,
    annotate_recovery_answer_values,
    assess_timeout_finalizability,
    extract_supported_recovery_facts,
    render_recovery_evidence,
    render_supported_fact_capsule,
)
from workflows.dynamic_htn_informed_research.htn.epoch import EpochRunResult, run_epoch
from workflows.dynamic_htn_informed_research.htn.dispatch_budget import QuestionDispatchBudget
from workflows.dynamic_htn_informed_research.htn.orchestrator import (
    ForcedFinalizationAudit,
    audit_forced_finalization,
    finalize_root_from_evidence,
    select_forced_final_answer,
)
from copy import deepcopy
from agent_harness.models.node_context import NodeContext
from ..runtime.context import TaskRuntime, _redact
from ..runtime.worker import build_dispatch
from ..profile import load_profile_selection
import tempfile

logger = logging.getLogger(__name__)
LEGACY_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 65536,
                   "extra_body": {"provider": {"only": ["deepseek"], "allow_fallbacks": False}}}
LEGACY_WORKER_MODEL = "deepseek-v4.1-flash"
LEGACY_WORKER_PROVIDER = "openrouter"
_RECOVERY_CAPSULE_MAX_TOKENS = 32768
_RECOVERY_CAPSULE_RETRY_EFFORT = "low"
RECOVERY_MAX_INPUT_CHARS = 500000
FORCED_FINALIZATION_HIGH_TIMEOUT = 600


def resolve_report_policy_for_route(route: TaskRoute) -> ReportPolicy | None:
    """Resolve report behavior solely from the native LLM routing result."""
    if route.strategy_name != "report_research":
        return None
    return SCIENTIFIC_REPORT_POLICY


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _profile_int(section: dict[str, Any], key: str, env: str, default: int) -> int:
    """Resolve an integer from task profile, then legacy env, then default."""
    if key not in section and os.getenv(env, "").strip():
        return _int_env(env, default)
    try:
        return int(section.get(key, default))
    except (TypeError, ValueError):
        return default


def _resolve_worker_limit(section: dict[str, Any], default: int = 0) -> int:
    """Resolve the new admission cap without silently disabling invalid limits."""
    value = section.get(
        "max_total_workers",
        os.getenv("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_TOTAL_WORKERS", "").strip() or default,
    )
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Dynamic HTN max_total_workers must be a non-negative integer")
    return value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _profile_bool(section: dict[str, Any], key: str, env: str, default: bool) -> bool:
    """Resolve a boolean from task profile, then legacy env, then default."""
    value = os.getenv(env)
    return _as_bool(section[key], default) if key in section else _as_bool(value, default)


def _recovery_route(
    *,
    should_run_gate: bool,
    gate_result: bool | None,
    rebuild_available: bool,
) -> str:
    """Choose one mutually exclusive recovery branch."""
    if should_run_gate and gate_result is True:
        return "finalize"
    if rebuild_available:
        return "rebuild"
    return "best_effort"


def _should_run_recovery_gate(
    *,
    epoch_id: int,
    semantic_exhaustion: bool,
    timeout_like: bool,
) -> bool:
    """The gate is exclusively a first-epoch wall-clock recovery step."""
    return timeout_like and not semantic_exhaustion and epoch_id == 0


def _recovery_capsule_retry_reason(status: str, capsule: str) -> str | None:
    """Return why a failed 32K/high capsule extraction needs one fresh retry."""
    if status == "malformed":
        return "malformed_extraction"
    if status == "timeout":
        return "high_timeout"
    if status.startswith("error:"):
        return "high_error"
    if status == "completed" and not capsule:
        return "empty_validated_capsule"
    return None


def _render_recovery_for_review(
    evidence: TimeoutRecoveryEvidence,
    *,
    restart_answer_context: str,
    max_input_chars: int | None = None,
) -> str:
    """Prioritize the answer view for terminal review within one total budget."""
    if max_input_chars is None:
        max_input_chars = RECOVERY_MAX_INPUT_CHARS
    if not restart_answer_context:
        return render_recovery_evidence(
            evidence,
            max_chars=max_input_chars,
        )
    prefix = (
        "# Prior epoch answer-field context (untrusted evidence)\n"
        "This precision-preserving view is derived from the same validated facts used "
        "for rebuild planning. It is not a selected answer or completed dependency.\n\n"
        f"{restart_answer_context}"
    )
    separator = "\n\n---\n\n"
    remaining = max(
        0,
        max_input_chars - len(prefix) - len(separator),
    )
    raw = render_recovery_evidence(evidence, max_chars=remaining)
    return f"{prefix}{separator}{raw}" if raw else prefix


def _question(state: dict[str, Any]) -> str:
    for key in ("original_question", "task_question", "question"):
        val = state.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


async def _ensure_answer_marker(
    llm,
    question: str,
    answer: str,
    *,
    force_best_effort: bool = False,
    answer_pattern: str = "",
) -> str:
    """Serialize an already-selected root answer without making a semantic choice."""
    del llm, question, force_best_effort
    explicit_answer = _best_fallback_answer(
        answer,
        answer_pattern=answer_pattern,
    )
    if explicit_answer:
        return answer
    line = "# Answer: (no answer could be determined)"
    return f"{answer}\n\n{line}" if answer.strip() else line


async def _run_forced_finalization_stage(
    *,
    stage: str,
    high_llm: Any,
    low_llm: Any,
    invoke: Callable[[Any], Awaitable[Any]],
    high_timeout_s: float,
    question_deadline: float,
    reserve_s: float,
    diagnostics: DiagnosticRecorder,
    input_chars: int,
) -> Any | None:
    """Try one forced-finalization stage at high, then fresh low.

    High has its own bounded timeout. Low has no independent per-call cap; the
    only remaining bound is the shared question wall clock and final formatting
    reserve. Empty or structurally invalid results are represented by a falsey
    return value from ``invoke`` and receive the same fresh-low retry as an
    exception or high timeout.
    """

    previous_status: str | None = None
    for effort, llm in (("high", high_llm), ("low", low_llm)):
        remaining = max(0.0, question_deadline - time.monotonic() - reserve_s)
        if remaining < 1.0:
            diagnostics.emit(
                f"forced_final_{stage}_{effort}_end",
                status="skipped_no_time",
                success=False,
                input_chars=input_chars,
            )
            return None
        timeout_s = min(high_timeout_s, remaining) if effort == "high" else remaining
        diagnostics.emit(
            f"forced_final_{stage}_{effort}_start",
            reasoning_effort=effort,
            timeout_s=(round(timeout_s, 3) if effort == "high" else None),
            shared_wall_clock_remaining_s=round(remaining, 3),
            input_chars=input_chars,
            retry_reason=previous_status if effort == "low" else None,
        )
        started = time.monotonic()
        status = "completed"
        value: Any | None = None
        try:
            with diagnostic_scope(
                call_kind=f"forced_final_{stage}_{effort}",
                forced_final_stage=stage,
                forced_final_effort=effort,
            ):
                value = await asyncio.wait_for(invoke(llm), timeout=timeout_s)
            if not value:
                status = "invalid"
        except (TimeoutError, asyncio.TimeoutError):
            status = "timeout"
        except Exception as exc:  # noqa: BLE001 - retry fresh at low
            status = f"error:{type(exc).__name__}"
            logger.warning(
                "dynamic_solve: forced final %s %s attempt failed: %s",
                stage,
                effort,
                exc,
            )
        output_chars = len(
            value.rendered if isinstance(value, ForcedFinalizationAudit) else str(value or "")
        )
        diagnostics.emit(
            f"forced_final_{stage}_{effort}_end",
            reasoning_effort=effort,
            status=status,
            success=status == "completed",
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            input_chars=input_chars,
            output_chars=output_chars,
        )
        if status == "completed":
            return value
        previous_status = status
    return None


async def _audit_then_select_root_answer(
    *,
    question: str,
    evidence_text: str,
    epoch_id: int,
    raw_control_llm: Callable[[str | None], Any],
    question_deadline: float,
    reserve_s: float,
    diagnostics: DiagnosticRecorder,
    strategy: SynthesisStrategy | None = None,
    audit_deadline: float | None = None,
    high_timeout_s: int = FORCED_FINALIZATION_HIGH_TIMEOUT,
) -> tuple[str, str]:
    """Reuse the root-bound audit/select path without inventing a child answer."""
    strategy = strategy or get_active_task_strategy().synthesis
    diagnostics.emit(
        "recovery_root_finalize_start",
        epoch_id=epoch_id,
        best_effort=True,
        mode="audit_then_select",
        high_timeout_s=high_timeout_s,
        input_chars=len(evidence_text),
    )
    audit = await _run_forced_finalization_stage(
        stage="audit",
        high_llm=raw_control_llm("high"),
        low_llm=raw_control_llm("low"),
        invoke=lambda llm: audit_forced_finalization(
            llm,
            question=question,
            evidence_text=evidence_text,
            strategy=strategy,
        ),
        high_timeout_s=high_timeout_s,
        question_deadline=audit_deadline or question_deadline,
        reserve_s=reserve_s,
        diagnostics=diagnostics,
        input_chars=len(evidence_text),
    )
    candidate = ""
    if isinstance(audit, ForcedFinalizationAudit):
        diagnostics.write_text(
            f"recovery/epoch-{epoch_id}-forced-final-audit.json",
            audit.rendered,
        )
        diagnostics.emit(
            "forced_final_audit_summary",
            epoch_id=epoch_id,
            candidate_count=audit.candidate_count,
            contradicted_candidate_count=audit.contradicted_candidate_count,
            role_binding_issue_count=audit.role_binding_issue_count,
            output_chars=len(audit.rendered),
        )
        candidate = (
            await _run_forced_finalization_stage(
                stage="select",
                high_llm=raw_control_llm("high"),
                low_llm=raw_control_llm("low"),
                invoke=lambda llm: select_forced_final_answer(
                    llm,
                    question=question,
                    audit=audit,
                    strategy=strategy,
                ),
                high_timeout_s=high_timeout_s,
                question_deadline=question_deadline,
                reserve_s=reserve_s,
                diagnostics=diagnostics,
                input_chars=len(audit.rendered),
            )
            or ""
        )
    if candidate.strip():
        branch_status = "completed_two_stage"
    else:
        # Preserve the established low-effort semantic fallback if either
        # structured stage remains unusable. The marker layer below never
        # chooses from child summaries.
        remaining = max(
            0.0,
            question_deadline - time.monotonic() - reserve_s,
        )
        diagnostics.emit(
            "forced_final_legacy_fallback_start",
            epoch_id=epoch_id,
            reasoning_effort="low",
            shared_wall_clock_remaining_s=round(remaining, 3),
            input_chars=len(evidence_text),
        )
        started = time.monotonic()
        legacy_status = "skipped_no_time"
        if remaining >= 1.0:
            legacy_status = "completed"
            try:
                with diagnostic_scope(
                    epoch_id=epoch_id,
                    call_kind="forced_final_legacy_low",
                ):
                    candidate = await asyncio.wait_for(
                        finalize_root_from_evidence(
                            raw_control_llm("low"),
                            question=question,
                            evidence_text=evidence_text,
                            best_effort=True,
                            strategy=strategy,
                        ),
                        timeout=remaining,
                    )
                if not _best_fallback_answer(
                    candidate,
                    answer_pattern=strategy.root_answer_pattern,
                ):
                    legacy_status = "invalid"
                    candidate = ""
            except (TimeoutError, asyncio.TimeoutError):
                legacy_status = "timeout"
            except Exception as exc:  # noqa: BLE001 - preserve evidence fallback
                legacy_status = f"error:{type(exc).__name__}"
                logger.warning(
                    "dynamic_solve: legacy forced final fallback failed: %s",
                    exc,
                )
        diagnostics.emit(
            "forced_final_legacy_fallback_end",
            epoch_id=epoch_id,
            reasoning_effort="low",
            status=legacy_status,
            success=legacy_status == "completed",
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            output_chars=len(candidate),
        )
        branch_status = f"legacy_{legacy_status}"
    diagnostics.emit(
        "recovery_root_finalize_end",
        epoch_id=epoch_id,
        best_effort=True,
        status=branch_status,
        output_chars=len(candidate),
    )
    return candidate, branch_status


class _QuestionDriver:
    def __init__(self, profile_name, profile, task_runtime, overrides=None, profiles=None):
        self.profiles = profiles
        self.profile_name = profile_name
        self.task_runtime = task_runtime
        self.overrides = overrides or {}
        self._ACTIVE_PROFILE = profile
        self._ACTIVE_BEHAVIOR_PROFILE = self._ACTIVE_PROFILE.get("behavior") or {}
        self.CONFIGURED_TASK_STRATEGY_NAME = normalize_task_strategy_name(
            self._ACTIVE_BEHAVIOR_PROFILE.get("strategy") or AUTO_TASK_STRATEGY_NAME
        )
        self.AUTO_ROUTE_ENABLED = bool(profiles and profiles.is_meta) or is_auto_task_strategy(self.CONFIGURED_TASK_STRATEGY_NAME)
        self._configure_execution_profile(profile)
        self.TASK_ROUTER_TIMEOUT_S = _profile_int(
            self._BEHAVIOR_PROFILE,
            "router_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_ROUTER_TIMEOUT_S",
            600,
        )
        self.TASK_ROUTER_MAX_TOKENS = _profile_int(
            self._BEHAVIOR_PROFILE,
            "router_max_tokens",
            "DYNAMIC_HTN_INFORMED_RESEARCH_ROUTER_MAX_TOKENS",
            int(self._LLM_PROFILE.get("max_tokens", 16384)) if profiles and profiles.is_meta else 16384,
        )
        router_effort = self._BASE_REASONING_EFFORT if profiles and profiles.is_meta else "low"
        self.TASK_ROUTER_SAMPLING = sampling_with_reasoning_effort(self.SAMPLING, router_effort)
        self.TASK_ROUTER_SAMPLING["max_tokens"] = self.TASK_ROUTER_MAX_TOKENS

    def _configure_execution_profile(self, profile):
        """Rebuild model/sampling/budget defaults from the selected execution file."""
        self._PROFILE = profile
        self._LLM_PROFILE = self._PROFILE.get("llm") or {}
        self._PLANNER_PROFILE = self._PROFILE.get("planner") or {}
        self._AGENT_PROFILE = self._PROFILE.get("agent") or {}
        self._WEB_PROFILE = self._PROFILE.get("web") or {}
        self._SUMMARY_PROFILE = self._PROFILE.get("summary_llm") or {}
        self._BEHAVIOR_PROFILE = self._PROFILE.get("behavior") or {}
        self.TASK_STRATEGY = get_task_strategy(
            "research" if self.AUTO_ROUTE_ENABLED else self.CONFIGURED_TASK_STRATEGY_NAME
        )
        self.WORKER_PROVIDER = str(self._LLM_PROFILE.get("provider") or LEGACY_WORKER_PROVIDER)
        self.WORKER_MODEL = str(self._LLM_PROFILE.get("model") or LEGACY_WORKER_MODEL)
        self.SAMPLING = {
            key: self._LLM_PROFILE[key]
            for key in ("temperature", "top_p", "max_tokens", "repetition_penalty", "extra_body")
            if key in self._LLM_PROFILE
        } or LEGACY_SAMPLING
        self._BASE_EXTRA_BODY = self.SAMPLING.get("extra_body")
        self._BASE_REASONING_EFFORT = (
            self._BASE_EXTRA_BODY.get("reasoning_effort")
            if isinstance(self._BASE_EXTRA_BODY, dict)
            else None
        )
        self.REASONING_EFFORTS = resolve_reasoning_efforts(
            self._PROFILE, fallback=self._BASE_REASONING_EFFORT
        )
        self.HIGH_EFFORT_TIMEOUT_S = _profile_int(
            self._LLM_PROFILE,
            "high_effort_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_HIGH_EFFORT_TIMEOUT_S",
            0,
        )
        self.WORKER_SAMPLING = sampling_with_reasoning_effort(
            self.SAMPLING, self.REASONING_EFFORTS["worker"]
        )
        self.SUMMARY_PROVIDER = str(self._SUMMARY_PROFILE.get("provider") or self.WORKER_PROVIDER)
        self.SUMMARY_MODEL = str(self._SUMMARY_PROFILE.get("model") or self.WORKER_MODEL)
        self.MAX_DEPTH = _profile_int(
            self._AGENT_PROFILE, "max_depth", "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_DEPTH", 2
        )
        self.MAX_CHILDREN = _profile_int(
            self._AGENT_PROFILE, "max_children", "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_CHILDREN", 4
        )
        self.PLANNER_MAX_PRECURSOR_CHARS = _profile_int(
            self._PLANNER_PROFILE,
            "max_precursor_chars",
            "DYNAMIC_HTN_INFORMED_RESEARCH_PLANNER_MAX_PRECURSOR_CHARS",
            24000,
        )
        self.PLANNER_MAX_SINGLE_SUMMARY_CHARS = _profile_int(
            self._PLANNER_PROFILE,
            "max_single_summary_chars",
            "DYNAMIC_HTN_INFORMED_RESEARCH_PLANNER_MAX_SINGLE_SUMMARY_CHARS",
            6000,
        )
        self.MAX_ROUNDS = _profile_int(
            self._AGENT_PROFILE, "max_rounds", "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_ROUNDS", 12
        )
        self.MAX_TOTAL_WORKERS = _resolve_worker_limit(self._AGENT_PROFILE)
        self.MAX_AGENTS_PER_ROUND = _profile_int(
            self._AGENT_PROFILE,
            "max_agents_per_round",
            "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_AGENTS_PER_ROUND",
            3,
        )
        self.PROGRESSIVE_FAN_IN = _profile_bool(
            self._AGENT_PROFILE,
            "progressive_fan_in",
            "DYNAMIC_HTN_INFORMED_RESEARCH_PROGRESSIVE_FAN_IN",
            True,
        )
        self.FAN_IN_GRACE_S = _profile_int(
            self._AGENT_PROFILE,
            "fan_in_grace_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_FAN_IN_GRACE_S",
            30,
        )
        self.SUBAGENT_MAX_TURNS = _profile_int(
            self._AGENT_PROFILE,
            "subagent_max_turns",
            "DYNAMIC_HTN_INFORMED_RESEARCH_SUBAGENT_MAX_TURNS",
            70,
        )
        self.MAX_ITERATIONS = _profile_int(
            self._AGENT_PROFILE,
            "max_iterations",
            "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_ITERATIONS",
            24,
        )
        self.QUESTION_TIMEOUT = _profile_int(
            self._AGENT_PROFILE,
            "question_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_QUESTION_TIMEOUT",
            28800,
        )
        self.LEAF_TIMEOUT = _profile_int(
            self._AGENT_PROFILE,
            "leaf_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_LEAF_TIMEOUT",
            7200,
        )
        self.ANSWER_FORMAT_TIMEOUT = _profile_int(
            self._AGENT_PROFILE,
            "answer_format_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_ANSWER_FORMAT_TIMEOUT",
            120,
        )
        self.FINALIZATION_RESERVE = _profile_int(
            self._AGENT_PROFILE,
            "finalization_reserve_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_FINALIZATION_RESERVE",
            1800,
        )
        self.MAX_REBUILDS = _profile_int(
            self._AGENT_PROFILE, "max_rebuilds", "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_REBUILDS", 1
        )
        self.RECOVERY_TIMEOUT = _profile_int(
            self._AGENT_PROFILE,
            "recovery_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_TIMEOUT",
            300,
        )
        self.FORCED_FINALIZATION_HIGH_TIMEOUT = _profile_int(
            self._AGENT_PROFILE,
            "forced_finalization_high_timeout_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_FORCED_FINALIZATION_HIGH_TIMEOUT",
            600,
        )
        self.MIN_REBUILD_REMAINING = _profile_int(
            self._AGENT_PROFILE,
            "min_rebuild_remaining_s",
            "DYNAMIC_HTN_INFORMED_RESEARCH_MIN_REBUILD_REMAINING",
            1800,
        )
        self.RECOVERY_MAX_INPUT_CHARS = _profile_int(
            self._AGENT_PROFILE,
            "recovery_max_input_chars",
            "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_MAX_INPUT_CHARS",
            500000,
        )
        self.RECOVERY_CAPSULE_CHARS = _profile_int(
            self._AGENT_PROFILE,
            "recovery_capsule_chars",
            "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_CAPSULE_CHARS",
            24000,
        )
        self.CONTENT_CACHE_ENABLED = _profile_bool(
            self._WEB_PROFILE,
            "content_cache_enabled",
            "DYNAMIC_HTN_INFORMED_RESEARCH_CONTENT_CACHE_ENABLED",
            False,
        )

    def _runtime_profile_for_route(self, route: TaskRoute) -> tuple[str, dict[str, Any]]:
        """Bind a routed strategy/report kind to its preloaded execution profile."""
        if self.profiles is not None and self.profiles.is_meta:
            return self.profiles.select(route.strategy_name)
        return (self.profile_name or "legacy", self._PROFILE)

    def _resolve_task_runtime(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Resolve per-question limits without mutating process-global settings."""
        llm = profile.get("llm") or {}
        planner = profile.get("planner") or {}
        agent = profile.get("agent") or {}
        agent_ints = {
            "max_depth": ("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_DEPTH", self.MAX_DEPTH),
            "max_children": ("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_CHILDREN", self.MAX_CHILDREN),
            "max_rounds": ("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_ROUNDS", self.MAX_ROUNDS),
            "max_agents_per_round": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_MAX_AGENTS_PER_ROUND",
                self.MAX_AGENTS_PER_ROUND,
            ),
            "fan_in_grace_s": ("DYNAMIC_HTN_INFORMED_RESEARCH_FAN_IN_GRACE_S", self.FAN_IN_GRACE_S),
            "subagent_max_turns": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_SUBAGENT_MAX_TURNS",
                self.SUBAGENT_MAX_TURNS,
            ),
            "max_iterations": ("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_ITERATIONS", self.MAX_ITERATIONS),
            "question_timeout_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_QUESTION_TIMEOUT",
                self.QUESTION_TIMEOUT,
            ),
            "leaf_timeout_s": ("DYNAMIC_HTN_INFORMED_RESEARCH_LEAF_TIMEOUT", self.LEAF_TIMEOUT),
            "answer_format_timeout_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_ANSWER_FORMAT_TIMEOUT",
                self.ANSWER_FORMAT_TIMEOUT,
            ),
            "finalization_reserve_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_FINALIZATION_RESERVE",
                self.FINALIZATION_RESERVE,
            ),
            "max_rebuilds": ("DYNAMIC_HTN_INFORMED_RESEARCH_MAX_REBUILDS", self.MAX_REBUILDS),
            "recovery_timeout_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_TIMEOUT",
                self.RECOVERY_TIMEOUT,
            ),
            "forced_finalization_high_timeout_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_FORCED_FINALIZATION_HIGH_TIMEOUT",
                self.FORCED_FINALIZATION_HIGH_TIMEOUT,
            ),
            "min_rebuild_remaining_s": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_MIN_REBUILD_REMAINING",
                self.MIN_REBUILD_REMAINING,
            ),
            "recovery_max_input_chars": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_MAX_INPUT_CHARS",
                self.RECOVERY_MAX_INPUT_CHARS,
            ),
            "recovery_capsule_chars": (
                "DYNAMIC_HTN_INFORMED_RESEARCH_RECOVERY_CAPSULE_CHARS",
                self.RECOVERY_CAPSULE_CHARS,
            ),
        }
        runtime = {
            key: _profile_int(agent, key, env, default)
            for key, (env, default) in agent_ints.items()
        }
        runtime["max_total_workers"] = _resolve_worker_limit(agent, self.MAX_TOTAL_WORKERS)
        runtime.update(
            {
                "reasoning_efforts": resolve_reasoning_efforts(
                    profile, fallback=self._BASE_REASONING_EFFORT
                ),
                "high_effort_timeout_s": _profile_int(
                    llm,
                    "high_effort_timeout_s",
                    "DYNAMIC_HTN_INFORMED_RESEARCH_HIGH_EFFORT_TIMEOUT_S",
                    self.HIGH_EFFORT_TIMEOUT_S,
                ),
                "planner_max_precursor_chars": _profile_int(
                    planner,
                    "max_precursor_chars",
                    "DYNAMIC_HTN_INFORMED_RESEARCH_PLANNER_MAX_PRECURSOR_CHARS",
                    self.PLANNER_MAX_PRECURSOR_CHARS,
                ),
                "planner_max_single_summary_chars": _profile_int(
                    planner,
                    "max_single_summary_chars",
                    "DYNAMIC_HTN_INFORMED_RESEARCH_PLANNER_MAX_SINGLE_SUMMARY_CHARS",
                    self.PLANNER_MAX_SINGLE_SUMMARY_CHARS,
                ),
                "progressive_fan_in": _profile_bool(
                    agent,
                    "progressive_fan_in",
                    "DYNAMIC_HTN_INFORMED_RESEARCH_PROGRESSIVE_FAN_IN",
                    self.PROGRESSIVE_FAN_IN,
                ),
            }
        )
        return runtime

    async def run(self, state: dict[str, Any], ctx: TaskRuntime) -> dict[str, Any]:
        question_started = time.monotonic()
        question = _question(state)
        language = state.get("language") or None
        empty_out = {
            "report": {"content": "", "title": "", "question": question},
            "final_answer": "",
            "final_content": "",
            "task_tree": {},
        }
        if not question:
            logger.warning("dynamic_solve: empty question")
            return empty_out
        out_dir = self.task_runtime.artifact_dir
        diagnostic_run_id = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        diagnostics = DiagnosticRecorder(
            out_dir / "diagnostics" / diagnostic_run_id,
            task_id=str(ctx.task_id or ""),
            diagnostic_run_id=diagnostic_run_id,
        )
        logger.info("dynamic_htn_informed_research diagnostics: %s", diagnostics.root)
        route_started = time.monotonic()
        if self.AUTO_ROUTE_ENABLED:
            router_llm = build_worker_llm(
                self.WORKER_PROVIDER,
                self.WORKER_MODEL,
                self.TASK_ROUTER_SAMPLING,
                diagnostics=diagnostics,
                purpose="router" if self.profiles is not None and self.profiles.is_meta else "main",
            )
            with diagnostic_scope(call_kind="task_router"):
                route = await route_task(router_llm, question, timeout_s=self.TASK_ROUTER_TIMEOUT_S)
            if self.profiles is not None and self.profiles.is_meta and route.used_fallback:
                self.task_runtime.write_json("routing.json", {
                    "route": asdict(route), "execution_profile": None,
                    "meta_profile": str(self.profiles.path), "status": "failed",
                })
                raise ValueError("Task routing failed after protocol retries; no execution profile selected")
            task_strategy = get_task_strategy(route.strategy_name)
            report_policy = resolve_report_policy_for_route(route)
            route_source = "question_router"
        else:
            task_strategy = self.TASK_STRATEGY
            route = TaskRoute(
                strategy_name=task_strategy.name,
                rationale="Selected explicitly by the active workflow profile.",
                attempt_count=0,
            )
            report_policy = resolve_report_policy_for_route(route)
            route_source = "explicit_profile"
        network_enabled = (self.task_runtime.web_policy or {}).get("network_enabled", True)
        if report_policy is not None:
            from ..task_strategies.report_research import report_strategy_for_access

            task_strategy = report_strategy_for_access(network_enabled=network_enabled)
            report_policy = replace(report_policy, network_enabled=network_enabled)
        elif task_strategy.name == "science_olympiad":
            from ..task_strategies.science_olympiad import science_strategy_for_access

            task_strategy = science_strategy_for_access(network_enabled=network_enabled)
        elif task_strategy.name == "science_mcq":
            from ..task_strategies.science_mcq import mcq_strategy_for_access

            task_strategy = mcq_strategy_for_access(network_enabled=network_enabled)
        runtime_profile_name, runtime_profile = self._runtime_profile_for_route(route)
        self._configure_execution_profile(runtime_profile)
        self.task_runtime.profile = deepcopy(runtime_profile)
        runtime = self._resolve_task_runtime(runtime_profile)
        if task_strategy.delivery.output_mode == "report":
            runtime["max_depth"] = 1
            runtime["max_children"] = max(2, runtime["max_children"])
            self.task_runtime.profile.setdefault("agent", {}).update(
                max_depth=1, max_children=runtime["max_children"])
        _validate_runtime(runtime)
        # Persist the effective default/environment value as well as explicit
        # profile overrides, so evaluations can identify this strategy change.
        self.task_runtime.profile.setdefault("agent", {})["max_total_workers"] = runtime["max_total_workers"]
        self.task_runtime.write_json(
            "effective_config.json",
            {"profile": runtime_profile_name, "configuration": _redact(self.task_runtime.profile),
             "meta_profile": str(self.profiles.path) if self.profiles is not None and self.profiles.is_meta else None,
             "task_strategy": route.strategy_name},
        )
        self.task_runtime.write_json("routing.json", {
            "route": asdict(route), "execution_profile": runtime_profile_name,
            "meta_profile": str(self.profiles.path) if self.profiles is not None and self.profiles.is_meta else None,
        })
        dispatch_budget = QuestionDispatchBudget(runtime["max_total_workers"])
        self.dispatch_budget = dispatch_budget
        reasoning_efforts = runtime["reasoning_efforts"]
        worker_sampling = sampling_with_reasoning_effort(self.SAMPLING, reasoning_efforts["worker"])
        set_active_task_strategy(task_strategy)
        blocked_source_directive = ""
        diagnostics.emit(
            "question_start",
            question_chars=len(question),
            diagnostics_root=str(diagnostics.root),
            worker_provider=self.WORKER_PROVIDER,
            worker_model=self.WORKER_MODEL,
            task_strategy=task_strategy.name,
            configured_task_strategy=self.CONFIGURED_TASK_STRATEGY_NAME,
            runtime_profile=runtime_profile_name,
            task_route_source=route_source,
            report_policy_key=report_policy.key if report_policy is not None else None,
            trace_recovery_provider=self.SUMMARY_PROVIDER,
            trace_recovery_model=self.SUMMARY_MODEL,
            task_session_id=self.task_runtime.task_id,
            reasoning_efforts={
                **reasoning_efforts,
                "router": (self.TASK_ROUTER_SAMPLING.get("extra_body") or {}).get("reasoning_effort"),
                "summary": str(self._SUMMARY_PROFILE.get("reasoning_effort") or "default"),
            },
            summary_retry_thinking_disabled=bool(
                self._SUMMARY_PROFILE.get("retry_empty_or_length_with_thinking_disabled")
            ),
            max_depth=runtime["max_depth"],
            max_children=runtime["max_children"],
            max_rounds=runtime["max_rounds"],
            max_total_workers=runtime["max_total_workers"],
            max_live_agents=runtime["max_agents_per_round"],
            subagent_max_turns=runtime["subagent_max_turns"],
            question_timeout_s=runtime["question_timeout_s"],
            leaf_timeout_s=runtime["leaf_timeout_s"],
            finalization_reserve_s=runtime["finalization_reserve_s"],
            max_rebuilds=runtime["max_rebuilds"],
            recovery_timeout_s=runtime["recovery_timeout_s"],
            forced_finalization_high_timeout_s=runtime["forced_finalization_high_timeout_s"],
            min_rebuild_remaining_s=runtime["min_rebuild_remaining_s"],
            recovery_max_input_chars=runtime["recovery_max_input_chars"],
            recovery_capsule_chars=runtime["recovery_capsule_chars"],
            high_effort_timeout_s=runtime["high_effort_timeout_s"],
            router_timeout_s=self.TASK_ROUTER_TIMEOUT_S,
            router_max_tokens=self.TASK_ROUTER_MAX_TOKENS,
        )
        diagnostics.emit(
            "task_route_decision",
            strategy_name=route.strategy_name,
            rationale=route.rationale,
            runtime_profile=runtime_profile_name,
            auto=self.AUTO_ROUTE_ENABLED,
            attempt_count=route.attempt_count,
            failed_attempt_count=len(route.attempt_failure_reasons),
            attempt_failure_reasons=list(route.attempt_failure_reasons),
            used_fallback=route.used_fallback,
            timeout_s=self.TASK_ROUTER_TIMEOUT_S,
            max_tokens=self.TASK_ROUTER_MAX_TOKENS,
            route_source=route_source,
            report_policy_key=report_policy.key if report_policy is not None else None,
            duration_ms=round((time.monotonic() - route_started) * 1000, 3),
        )
        diagnostics.write_text("question.txt", question)
        raw_control_llms: dict[str | None, Any] = {}
        bounded_high_control_llm: HighEffortTimeoutFallbackLLMClient | None = None

        def raw_control_llm(effort: str | None) -> Any:
            if effort not in raw_control_llms:
                raw_control_llms[effort] = build_worker_llm(
                    self.WORKER_PROVIDER,
                    self.WORKER_MODEL,
                    sampling_with_reasoning_effort(self.SAMPLING, effort),
                    diagnostics=diagnostics,
                )
            return raw_control_llms[effort]

        def control_llm(role: str) -> Any:
            nonlocal bounded_high_control_llm
            effort = reasoning_efforts[role]
            if effort == "high" and runtime["high_effort_timeout_s"] > 0:
                if bounded_high_control_llm is None:
                    bounded_high_control_llm = HighEffortTimeoutFallbackLLMClient(
                        raw_control_llm("high"),
                        raw_control_llm("low"),
                        timeout_s=runtime["high_effort_timeout_s"],
                        diagnostics=diagnostics,
                    )
                return bounded_high_control_llm
            return raw_control_llm(effort)

        recovery_llm = control_llm("recovery")
        trace_recovery_llm = build_worker_llm(
            self.SUMMARY_PROVIDER,
            self.SUMMARY_MODEL,
            sampling_with_reasoning_effort(
                {"temperature": 0.3, "max_tokens": 8192}, reasoning_efforts["recovery"]
            ),
            diagnostics=diagnostics,
            purpose="summary",
        )
        capsule_recovery_llm = build_worker_llm(
            self.SUMMARY_PROVIDER,
            self.SUMMARY_MODEL,
            sampling_with_reasoning_effort(
                {"temperature": 0.3, "max_tokens": _RECOVERY_CAPSULE_MAX_TOKENS},
                reasoning_efforts["recovery"],
            ),
            diagnostics=diagnostics,
            purpose="summary",
        )
        capsule_retry_llm = build_worker_llm(
            self.SUMMARY_PROVIDER,
            self.SUMMARY_MODEL,
            sampling_with_reasoning_effort(
                {"temperature": 0.3, "max_tokens": _RECOVERY_CAPSULE_MAX_TOKENS},
                _RECOVERY_CAPSULE_RETRY_EFFORT,
            ),
            diagnostics=diagnostics,
            purpose="summary",
        )
        malformed_trace_summarizer = build_malformed_trace_summarizer(
            trace_recovery_llm, diagnostics=diagnostics, strategy=task_strategy.recovery
        )
        dispatch = build_dispatch(
            ctx.task_id,
            max_turns=runtime["subagent_max_turns"],
            sampling=worker_sampling,
            provider=self.WORKER_PROVIDER,
            model=self.WORKER_MODEL,
            language=language,
            ref_dir=out_dir / "refs",
            job_timeout_s=runtime["leaf_timeout_s"] + 300,
            diagnostics=diagnostics,
            malformed_trace_summarizer=malformed_trace_summarizer,
            reasoning_effort=reasoning_efforts["worker"],
        )
        available_agents = [
            AgentDescriptor(role_id=role_id) for role_id in task_strategy.workers.role_ids
        ]
        question_deadline = time.monotonic() + runtime["question_timeout_s"]
        terminal_reserve = min(
            max(runtime["finalization_reserve_s"], runtime["answer_format_timeout_s"]),
            max(0, runtime["question_timeout_s"] - 1),
        )
        research_deadline = question_deadline - terminal_reserve
        report_tree = task_strategy.delivery.output_mode == "report"
        report_alignment_llm = (build_worker_llm(
            self.WORKER_PROVIDER, self.WORKER_MODEL,
            sampling_with_reasoning_effort({**self.SAMPLING, "max_tokens": 8192}, "low"),
            diagnostics=diagnostics,
        ) if report_tree else None)
        previous_report = ""
        restart_context = ""
        restart_answer_context = ""
        restart_facts: tuple[SupportedRecoveryFact, ...] = ()
        recovery_candidate = ""
        forced_best_effort = False
        epoch_id = 0
        epoch_run: EpochRunResult | None = None
        execution_failures: list[dict[str, Any]] = []

        def _root_is_exhausted(run: EpochRunResult) -> bool:
            root = run.tree.root()
            return (
                root.status is HTNTaskStatus.COMPLETED
                and root.completion_outcome is HTNCompletionOutcome.EXHAUSTED_UNRESOLVED
            )

        def _collect_recovery_evidence(
            run: EpochRunResult, *, trigger_kind: str | None = None
        ) -> TimeoutRecoveryEvidence:
            all_exhaustion_incidents = [
                *run.exhaustion_incidents,
                *run.nonterminal_exhaustion_incidents,
            ]
            semantic_exhaustion = bool(all_exhaustion_incidents) or _root_is_exhausted(run)
            completed = [
                CompletedNodeEvidence(
                    node_id=node.id,
                    title=node.title or node.id,
                    result_summary=node.result_summary or "",
                    ref=node.ref,
                )
                for node in run.tree.nodes.values()
                if node.status is HTNTaskStatus.COMPLETED
                and node.result_summary
                and (node.id != run.tree.root_id or semantic_exhaustion)
            ]
            records = [
                record
                for orchestrator in run.orchestrators.values()
                for record in orchestrator._results
            ]
            incidents = run.timeout_incidents
            failures = getattr(run, "execution_failure_incidents", [])
            failed = [
                node for node in run.tree.nodes.values() if node.status is HTNTaskStatus.FAILED
            ]
            return TimeoutRecoveryEvidence(
                question=question,
                epoch_id=run.tree.epoch_id,
                trigger_kind=trigger_kind
                or (
                    "question_deadline"
                    if run.question_deadline_exhausted and (not incidents)
                    else "timeout"
                    if incidents
                    else "execution_failure"
                    if failures
                    else "semantic_exhaustion"
                    if run.exhaustion_incidents
                    else "question_dispatch_budget_exhausted"
                    if run.dispatch_budget_exhausted
                    else "tree_complete_unresolved"
                ),
                timed_out_leaf_ids=[incident.leaf_id for incident in incidents]
                or [incident.node_id for incident in failures]
                or [incident.node_id for incident in all_exhaustion_incidents]
                or [node.id for node in failed],
                timed_out_tasks=[incident.leaf_task for incident in incidents]
                or [incident.node_task for incident in failures]
                or [incident.node_task for incident in all_exhaustion_incidents]
                or [node.task for node in failed],
                completed_nodes=completed,
                agent_results=records,
                failed_nodes=[
                    FailedNodeEvidence(
                        node_id=incident.node_id, task=incident.node_task,
                        error_type=incident.error_type, error=incident.error,
                        partial_summary=run.tree.get_node(incident.node_id).result_summary or "",
                        ref=incident.ref,
                    )
                    for incident in failures
                ],
            )

        while True:
            planner_role = "planner_epoch_0" if epoch_id == 0 else "planner_rebuild"
            tree_synthesis_llm = control_llm("tree_synthesis")
            answer_context_ready = False
            cached_answer_context = ""

            async def root_answer_context_provider() -> str:
                """Build Call 2 lazily and only at a ROOT-synthesis boundary."""
                nonlocal answer_context_ready, cached_answer_context
                if answer_context_ready:
                    return cached_answer_context
                answer_context_ready = True
                if not restart_facts:
                    return ""
                started = time.monotonic()
                diagnostics.emit(
                    "recovery_answer_annotation_start",
                    epoch_id=epoch_id,
                    fact_count=len(restart_facts),
                )
                status = "completed"
                try:
                    with diagnostic_scope(
                        epoch_id=epoch_id, call_kind="root_recovery_answer_annotation"
                    ):
                        annotated = await annotate_recovery_answer_values(
                            tree_synthesis_llm, question=question, facts=restart_facts
                        )
                    if annotated is None:
                        status = "malformed_fallback_unprioritized"
                        annotated = restart_facts
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "dynamic_solve: recovery answer annotation failed; using unprioritized facts: %s",
                        exc,
                    )
                    status = f"error_fallback_unprioritized:{type(exc).__name__}"
                    annotated = restart_facts
                cached_answer_context = render_supported_fact_capsule(
                    annotated, max_chars=runtime["recovery_capsule_chars"], view="answer"
                )
                diagnostics.write_text(
                    f"recovery/epoch-{epoch_id}-root-answer-capsule.md", cached_answer_context
                )
                diagnostics.emit(
                    "recovery_answer_annotation_end",
                    epoch_id=epoch_id,
                    status=status,
                    fact_count=len(annotated),
                    answer_value_count=sum((bool(fact.answer_shaped_value) for fact in annotated)),
                    capsule_chars=len(cached_answer_context),
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                )
                return cached_answer_context

            epoch_root_answer_context_provider = (
                None
                if task_strategy.delivery.output_mode == "report"
                else root_answer_context_provider
            )
            with diagnostic_scope(epoch_id=epoch_id):
                epoch_run = await run_epoch(
                    question=question,
                    epoch_id=epoch_id,
                    restart_context=restart_context,
                    restart_answer_context=restart_answer_context,
                    research_deadline=question_deadline if report_tree else research_deadline,
                    report_research_deadline=research_deadline if report_tree else None,
                    report_network_enabled=network_enabled,
                    max_iterations=runtime["max_iterations"],
                    max_depth=runtime["max_depth"],
                    max_children=runtime["max_children"],
                    max_precursor_chars=runtime["planner_max_precursor_chars"],
                    max_single_summary_chars=runtime["planner_max_single_summary_chars"],
                    max_rounds=runtime["max_rounds"],
                    max_agents_per_round=runtime["max_agents_per_round"],
                    progressive_fan_in=runtime["progressive_fan_in"],
                    fan_in_grace_s=runtime["fan_in_grace_s"],
                    leaf_timeout_s=runtime["leaf_timeout_s"],
                    available_agents=available_agents,
                    llm=recovery_llm,
                    dispatch=dispatch,
                    dispatch_budget=dispatch_budget,
                    summarizer=build_summarizer(
                        tree_synthesis_llm,
                        epoch_id=epoch_id,
                        restart_answer_context=restart_answer_context,
                        root_answer_context_provider=epoch_root_answer_context_provider,
                        strategy=task_strategy.synthesis,
                    ),
                    output_dir=out_dir,
                    diagnostics=diagnostics,
                    ctx=ctx,
                    interrupt_on_semantic_exhaustion=epoch_id < runtime["max_rebuilds"],
                    planner_llm=control_llm(planner_role),
                    planner_audit_llm=control_llm("planner_audit"),
                    planner_revision_llm=control_llm("planner_revision"),
                    coordinator_llm=control_llm("coordinator"),
                    instruction_llm=control_llm("instruction_generation"),
                    leaf_synthesis_llm=control_llm("leaf_synthesis"),
                    report_alignment_llm=report_alignment_llm,
                    root_answer_context_provider=epoch_root_answer_context_provider,
                    task_strategy=task_strategy,
                    blocked_source_directive=blocked_source_directive,
                )
            tree = epoch_run.tree
            failure_incidents = getattr(epoch_run, "execution_failure_incidents", [])
            execution_failed = bool(failure_incidents)
            execution_failures.extend(
                {"epoch_id": epoch_id, **asdict(incident)} for incident in failure_incidents
            )
            if report_tree:
                from ..task_strategies.report_research.topology import rebuild_context
                delivery = next((n for n in tree.nodes.values()
                                 if n.execution_kind == "report_delivery"), None)
                if delivery is not None and delivery.result_summary:
                    previous_report = delivery.result_summary
                if (tree.root().status is HTNTaskStatus.COMPLETED
                        and tree.root().completion_outcome is HTNCompletionOutcome.RESOLVED):
                    break
                delivery_owner = epoch_run.orchestrators.get(delivery.id) if delivery else None
                if delivery_owner is not None and not getattr(delivery_owner, "rebuild_requested", True):
                    # Failed review protocol/transport is not a scientific research gap.
                    diagnostics.emit("report_tree_review", epoch_id=epoch_id,
                                     action="verification_incomplete", delivery_node_id=delivery.id)
                    break
                # The delivery verdict returns directly to Planner, not a second repair engine.
                restart_context = rebuild_context(epoch_run)
                diagnostics.write_text(f"recovery/epoch-{epoch_id}-report-rebuild.json", restart_context)
                can_rebuild = (epoch_id < runtime["max_rebuilds"]
                               and not dispatch_budget.exhausted
                               and research_deadline - time.monotonic() >= runtime["min_rebuild_remaining_s"])
                diagnostics.emit("report_tree_review", epoch_id=epoch_id,
                                 action="rebuild" if can_rebuild else "partial",
                                 delivery_node_id=delivery.id if delivery else None)
                if can_rebuild:
                    diagnostics.emit("epoch_rebuild", from_epoch=epoch_id, to_epoch=epoch_id + 1,
                                     trigger_kind="report_tree_unresolved",
                                     capsule_chars=len(restart_context))
                    epoch_id += 1
                    continue
                break
            root_exhausted = _root_is_exhausted(epoch_run)
            dispatch_stopped = epoch_run.dispatch_budget_exhausted and not (
                tree.root().status is HTNTaskStatus.COMPLETED
                and tree.root().completion_outcome is HTNCompletionOutcome.RESOLVED
            )
            if (
                not epoch_run.timeout_incidents
                and not execution_failed
                and (not epoch_run.exhaustion_incidents)
                and (not epoch_run.question_deadline_exhausted)
                and (not root_exhausted)
                and (not dispatch_stopped)
            ):
                break
            recovery_evidence = _collect_recovery_evidence(epoch_run)
            rendered_recovery = _render_recovery_for_review(
                recovery_evidence,
                restart_answer_context=restart_answer_context,
                max_input_chars=runtime["recovery_max_input_chars"],
            )
            diagnostics.write_text(f"recovery/epoch-{epoch_id}-input.md", rendered_recovery)
            terminal_semantic_exhaustion = bool(epoch_run.exhaustion_incidents)
            semantic_exhaustion = terminal_semantic_exhaustion or root_exhausted
            timeout_like = (
                bool(epoch_run.timeout_incidents) or epoch_run.question_deadline_exhausted
                or execution_failed
            )
            trigger_kind = (
                "question_dispatch_budget_exhausted"
                if dispatch_stopped
                else "question_deadline"
                if epoch_run.question_deadline_exhausted and (not epoch_run.timeout_incidents)
                else "timeout"
                if epoch_run.timeout_incidents
                else "execution_failure"
                if execution_failed
                else "semantic_exhaustion"
                if terminal_semantic_exhaustion
                else "tree_complete_unresolved"
            )
            remaining_before_gate = max(0.0, question_deadline - time.monotonic())
            remaining_research_before_gate = max(0.0, research_deadline - time.monotonic())
            allow_rebuild = (
                (terminal_semantic_exhaustion or bool(epoch_run.timeout_incidents) or execution_failed)
                and not dispatch_budget.exhausted
                and epoch_id < runtime["max_rebuilds"]
                and (remaining_research_before_gate >= runtime["min_rebuild_remaining_s"])
            )
            recovery_budget = min(
                runtime["recovery_timeout_s"],
                max(0.0, remaining_before_gate - runtime["answer_format_timeout_s"]),
            )
            recovery_deadline = time.monotonic() + recovery_budget
            diagnostics.emit(
                "recovery_review_start",
                epoch_id=epoch_id,
                trigger_kind=trigger_kind,
                allow_rebuild=allow_rebuild,
                budget_s=recovery_budget,
                remaining_research_s=remaining_research_before_gate,
                input_chars=len(rendered_recovery),
                completed_node_count=len(recovery_evidence.completed_nodes),
                unique_report_count=len(recovery_evidence.unique_agent_results()),
                execution_failure_count=len(failure_incidents),
            )
            should_run_gate = not dispatch_budget.exhausted and _should_run_recovery_gate(
                epoch_id=epoch_id,
                semantic_exhaustion=semantic_exhaustion,
                timeout_like=timeout_like,
            )
            gate_budget = min(recovery_budget / 2, recovery_budget) if should_run_gate else 0.0
            gate_result: bool | None = None
            gate_status = (
                "skipped_question_dispatch_budget"
                if dispatch_budget.exhausted
                else "skipped_semantic_exhaustion"
                if semantic_exhaustion
                else "skipped_post_rebuild"
                if epoch_id > 0
                else "skipped_no_time"
            )
            if should_run_gate:
                diagnostics.emit(
                    "recovery_finalize_gate_start",
                    epoch_id=epoch_id,
                    budget_s=gate_budget,
                    input_chars=len(rendered_recovery),
                )
            if should_run_gate and gate_budget >= 1.0:
                try:
                    with diagnostic_scope(epoch_id=epoch_id, call_kind="timeout_finalize_gate"):
                        gate_result = await asyncio.wait_for(
                            assess_timeout_finalizability(trace_recovery_llm, rendered_recovery),
                            timeout=gate_budget,
                        )
                    gate_status = "completed" if gate_result is not None else "malformed"
                except (TimeoutError, asyncio.TimeoutError):
                    gate_status = "timeout"
                except Exception as exc:
                    logger.warning("dynamic_solve: timeout finalize gate failed: %s", exc)
                    gate_status = f"error:{type(exc).__name__}"
            if should_run_gate:
                diagnostics.emit(
                    "recovery_finalize_gate_end",
                    epoch_id=epoch_id,
                    status=gate_status,
                    can_finalize=gate_result,
                    remaining_question_s=max(0.0, question_deadline - time.monotonic()),
                )
            else:
                diagnostics.emit(
                    "recovery_finalize_gate_skipped",
                    epoch_id=epoch_id,
                    reason=gate_status,
                    trigger_kind=trigger_kind,
                )
            action = "best_effort"
            branch_status = "skipped_no_time"
            capsule = ""
            answer_capsule = ""
            facts: tuple[SupportedRecoveryFact, ...] = ()
            recovery_candidate = ""
            rebuild_available = (
                allow_rebuild
                and research_deadline - time.monotonic() >= runtime["min_rebuild_remaining_s"]
            )
            recovery_route = _recovery_route(
                should_run_gate=should_run_gate,
                gate_result=gate_result,
                rebuild_available=rebuild_available,
            )
            if recovery_route == "finalize":
                action = "finalize"
                finalize_budget = min(
                    max(0.0, recovery_deadline - time.monotonic()),
                    max(0.0, question_deadline - time.monotonic()),
                )
                diagnostics.emit(
                    "recovery_root_finalize_start",
                    epoch_id=epoch_id,
                    best_effort=False,
                    budget_s=finalize_budget,
                    input_chars=len(rendered_recovery),
                )
                if finalize_budget >= 1.0:
                    try:
                        with diagnostic_scope(
                            epoch_id=epoch_id, call_kind="timeout_root_finalize"
                        ):
                            recovery_candidate = await asyncio.wait_for(
                                finalize_root_from_evidence(
                                    recovery_llm,
                                    question=question,
                                    evidence_text=rendered_recovery,
                                ),
                                timeout=finalize_budget,
                            )
                        branch_status = "completed"
                    except (TimeoutError, asyncio.TimeoutError):
                        branch_status = "timeout"
                    except Exception as exc:
                        logger.warning("dynamic_solve: timeout root finalize failed: %s", exc)
                        branch_status = f"error:{type(exc).__name__}"
                if not recovery_candidate.strip():
                    recovery_candidate = rendered_recovery
                diagnostics.emit(
                    "recovery_root_finalize_end",
                    epoch_id=epoch_id,
                    best_effort=False,
                    status=branch_status,
                    output_chars=len(recovery_candidate),
                )
            elif recovery_route == "rebuild":
                action = "rebuild"
                rebuild_slack = max(
                    0.0, research_deadline - time.monotonic() - runtime["min_rebuild_remaining_s"]
                )
                capsule_budget = min(max(0.0, recovery_deadline - time.monotonic()), rebuild_slack)
                diagnostics.emit(
                    "recovery_capsule_start",
                    epoch_id=epoch_id,
                    budget_s=capsule_budget,
                    input_chars=len(rendered_recovery),
                    max_tokens=_RECOVERY_CAPSULE_MAX_TOKENS,
                    reasoning_effort=reasoning_efforts["recovery"],
                )
                fact_count = 0
                capsule_retry_attempted = False
                capsule_retry_reason: str | None = None
                if capsule_budget >= 1.0:
                    for attempt in range(2):
                        is_retry = attempt == 1
                        remaining_rebuild_slack = max(
                            0.0,
                            research_deadline
                            - time.monotonic()
                            - runtime["min_rebuild_remaining_s"],
                        )
                        if is_retry:
                            attempt_budget = min(
                                runtime["recovery_timeout_s"], remaining_rebuild_slack
                            )
                        else:
                            attempt_budget = min(
                                max(0.0, recovery_deadline - time.monotonic()),
                                remaining_rebuild_slack,
                            )
                        if attempt_budget < 1.0:
                            break
                        if is_retry:
                            capsule_retry_attempted = True
                            diagnostics.emit(
                                "recovery_capsule_retry_start",
                                epoch_id=epoch_id,
                                reason=capsule_retry_reason,
                                budget_s=attempt_budget,
                                input_chars=len(rendered_recovery),
                                max_tokens=_RECOVERY_CAPSULE_MAX_TOKENS,
                                reasoning_effort=_RECOVERY_CAPSULE_RETRY_EFFORT,
                                fresh_recovery_budget=True,
                            )
                        attempt_status = "completed"
                        attempt_facts = ()
                        attempt_capsule = ""
                        try:
                            call_kind = f"{trigger_kind}_supported_fact_extraction"
                            if is_retry:
                                call_kind += "_retry"
                            with diagnostic_scope(epoch_id=epoch_id, call_kind=call_kind):
                                extracted = await asyncio.wait_for(
                                    extract_supported_recovery_facts(
                                        capsule_retry_llm if is_retry else capsule_recovery_llm,
                                        rendered_recovery,
                                        max_capsule_chars=runtime["recovery_capsule_chars"],
                                    ),
                                    timeout=attempt_budget,
                                )
                            if extracted is None:
                                attempt_status = "malformed"
                            else:
                                attempt_facts, attempt_capsule = extracted
                        except (TimeoutError, asyncio.TimeoutError):
                            attempt_status = "timeout"
                        except Exception as exc:
                            logger.warning(
                                "dynamic_solve: recovery fact extraction%s failed: %s",
                                " retry" if is_retry else "",
                                exc,
                            )
                            attempt_status = f"error:{type(exc).__name__}"
                        if is_retry:
                            diagnostics.emit(
                                "recovery_capsule_retry_end",
                                epoch_id=epoch_id,
                                status=attempt_status,
                                fact_count=len(attempt_facts),
                                capsule_chars=len(attempt_capsule),
                                reason=capsule_retry_reason,
                            )
                        branch_status = attempt_status
                        if attempt_status == "completed":
                            facts = attempt_facts
                            capsule = attempt_capsule
                            fact_count = len(facts)
                        if is_retry:
                            break
                        capsule_retry_reason = _recovery_capsule_retry_reason(
                            attempt_status, attempt_capsule
                        )
                        if capsule_retry_reason is None:
                            break
                diagnostics.emit(
                    "recovery_capsule_end",
                    epoch_id=epoch_id,
                    status=branch_status,
                    fact_count=fact_count,
                    capsule_chars=len(capsule),
                    answer_capsule_chars=len(answer_capsule),
                    retry_attempted=capsule_retry_attempted,
                    retry_reason=capsule_retry_reason,
                )
            else:
                answer_capsule = await root_answer_context_provider()
                rendered_recovery = _render_recovery_for_review(
                    recovery_evidence,
                    restart_answer_context=answer_capsule,
                    max_input_chars=runtime["recovery_max_input_chars"],
                )
                recovery_candidate, branch_status = await _audit_then_select_root_answer(
                    question=question,
                    evidence_text=rendered_recovery,
                    epoch_id=epoch_id,
                    raw_control_llm=raw_control_llm,
                    question_deadline=question_deadline,
                    reserve_s=runtime["answer_format_timeout_s"],
                    diagnostics=diagnostics,
                    strategy=task_strategy.synthesis,
                    high_timeout_s=runtime["forced_finalization_high_timeout_s"],
                )
                if not recovery_candidate.strip():
                    recovery_candidate = rendered_recovery
                forced_best_effort = True
            diagnostics.write_text(f"recovery/epoch-{epoch_id}-capsule.md", capsule)
            diagnostics.write_text(f"recovery/epoch-{epoch_id}-answer-capsule.md", answer_capsule)
            remaining_after_review = max(0.0, question_deadline - time.monotonic())
            diagnostics.emit(
                "recovery_review_end",
                epoch_id=epoch_id,
                status=branch_status,
                gate_status=gate_status,
                can_finalize=gate_result,
                decision=action,
                trigger_kind=trigger_kind,
                answer_candidate_chars=len(recovery_candidate),
                capsule_chars=len(capsule),
                answer_capsule_chars=len(answer_capsule),
                unresolved_count=0,
                remaining_question_s=remaining_after_review,
            )
            if action == "rebuild":
                restart_context = capsule
                restart_facts = facts
                restart_answer_context = ""
                diagnostics.emit(
                    "epoch_rebuild",
                    from_epoch=epoch_id,
                    to_epoch=epoch_id + 1,
                    trigger_kind=trigger_kind,
                    capsule_chars=len(restart_context),
                    answer_capsule_chars=len(restart_answer_context),
                    remaining_question_s=remaining_after_review,
                )
                epoch_id += 1
                continue
            diagnostics.emit(
                "epoch_finalize",
                epoch_id=epoch_id,
                recovery_action=action,
                candidate_chars=len(recovery_candidate),
            )
            break
        if epoch_run is None:
            raise RuntimeError("dynamic HTN epoch driver did not start")
        root_node = tree.root()
        if report_tree:
            # Root passes through the delivery leaf; never synthesize a second report.
            answer = root_node.result_summary or previous_report
            if not answer.strip():
                memos = [n for n in tree.nodes.values() if n.id != tree.root_id and n.result_summary]
                answer = "# Partial research\n\nReport delivery did not complete.\n\n" + "\n\n".join(
                    f"## {n.title or n.id}\n{n.result_summary}" for n in memos)
            report_title = "Research Report"
            delivery_status = ("verified" if root_node.status is HTNTaskStatus.COMPLETED
                               and root_node.completion_outcome is HTNCompletionOutcome.RESOLVED
                               else "partial")
            delivery_node = next((n for n in tree.nodes.values()
                                  if n.execution_kind == "report_delivery"), None)
            delivery_owner = epoch_run.orchestrators.get(delivery_node.id) if delivery_node else None
            if delivery_status != "verified" and delivery_owner is not None:
                delivery_status = getattr(delivery_owner, "delivery_status", "partial")
            diagnostics.write_text("report_delivery/final.md", answer)
            diagnostics.write_json("report_delivery/result.json", {
                "status": delivery_status, "content": answer, "epoch_id": epoch_id,
                "workflow": "shallow_report_tree",
            })
        else:
            if (
                not recovery_candidate.strip()
                and root_node.completion_outcome is HTNCompletionOutcome.RESOLVED
                and (
                    not _best_fallback_answer(
                        root_node.result_summary or "",
                        answer_pattern=task_strategy.synthesis.root_answer_pattern,
                    )
                )
            ):
                recovery_evidence = _collect_recovery_evidence(
                    epoch_run, trigger_kind="root_answer_contract_failure"
                )
                restart_answer_context = await root_answer_context_provider()
                rendered_recovery = _render_recovery_for_review(
                    recovery_evidence,
                    restart_answer_context=restart_answer_context,
                    max_input_chars=runtime["recovery_max_input_chars"],
                )
                diagnostics.write_text(
                    f"recovery/epoch-{epoch_id}-root-answer-contract-input.md", rendered_recovery
                )
                remaining = max(0.0, question_deadline - time.monotonic())
                diagnostics.emit(
                    "root_answer_contract_failure",
                    epoch_id=epoch_id,
                    root_outcome=root_node.completion_outcome.value,
                    root_summary_chars=len(root_node.result_summary or ""),
                    evidence_chars=len(rendered_recovery),
                    remaining_question_s=round(remaining, 3),
                )
                recovery_candidate, contract_status = await _audit_then_select_root_answer(
                    question=question,
                    evidence_text=rendered_recovery,
                    epoch_id=epoch_id,
                    raw_control_llm=raw_control_llm,
                    question_deadline=question_deadline,
                    reserve_s=0.0,
                    audit_deadline=time.monotonic() + remaining / 2,
                    diagnostics=diagnostics,
                    strategy=task_strategy.synthesis,
                    high_timeout_s=runtime["forced_finalization_high_timeout_s"],
                )
                diagnostics.emit(
                    "root_answer_contract_recovery_end",
                    epoch_id=epoch_id,
                    status=contract_status,
                    success=bool(
                        _best_fallback_answer(
                            recovery_candidate,
                            answer_pattern=task_strategy.synthesis.root_answer_pattern,
                        )
                    ),
                    output_chars=len(recovery_candidate),
                )
            candidate = recovery_candidate or root_node.result_summary or ""
            done = [
                n
                for n in tree.nodes.values()
                if n.status is HTNTaskStatus.COMPLETED and n.id != "root" and n.result_summary
            ]
            evidence = "\n\n".join((f"## {n.title or n.id}\n{n.result_summary}" for n in done))
            if not candidate.strip():
                candidate = evidence
            answer_format_started = time.monotonic()
            answer_format_budget = min(
                runtime["answer_format_timeout_s"], max(question_deadline - time.monotonic(), 1.0)
            )
            diagnostics.emit(
                "answer_format_start",
                budget_s=answer_format_budget,
                candidate_chars=len(candidate),
                had_explicit_answer=bool(
                    _best_fallback_answer(
                        candidate, answer_pattern=task_strategy.synthesis.root_answer_pattern
                    )
                ),
            )
            try:
                with diagnostic_scope(call_kind="ensure_answer_marker"):
                    answer = await asyncio.wait_for(
                        _ensure_answer_marker(
                            recovery_llm,
                            question,
                            candidate,
                            force_best_effort=forced_best_effort,
                            answer_pattern=task_strategy.synthesis.root_answer_pattern,
                        ),
                        timeout=answer_format_budget,
                    )
                answer_format_status = "completed"
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "dynamic_solve: answer formatting timed out; preserving explicit candidate"
                )
                explicit = _best_fallback_answer(
                    candidate, answer_pattern=task_strategy.synthesis.root_answer_pattern
                )
                answer = (
                    f"{candidate}\n\n# Answer: {explicit}"
                    if explicit
                    else "# Answer: (no answer could be determined)"
                )
                answer_format_status = "timeout"
            diagnostics.emit(
                "answer_format_end",
                duration_ms=round((time.monotonic() - answer_format_started) * 1000, 3),
                status=answer_format_status,
                output_chars=len(answer or ""),
            )
            report_title = root_node.title or "Answer"
        await ctx.emit_event(
            "dynamic_htn_informed_research_done",
            {
                "epoch_id": epoch_id,
                "node_count": len(tree.nodes),
                "root_completed": root_node.status.value,
            },
        )
        report = {"content": answer, "title": report_title, "question": question}
        diagnostics.write_json(
            diagnostics.artifact_path("tree_snapshots", f"epoch-{epoch_id}-final", ".json"),
            {nid: node.model_dump() for nid, node in tree.nodes.items()},
        )
        diagnostics.emit(
            "question_end",
            duration_ms=round((time.monotonic() - question_started) * 1000, 3),
            final_epoch_id=epoch_id,
            node_count=len(tree.nodes),
            root_status=root_node.status.value,
            answer_chars=len(answer or ""),
            dispatch_budget=dispatch_budget.snapshot(),
        )
        return {
            "final_answer": answer,
            "final_content": answer,
            "report": report,
            "task_tree": {nid: n.model_dump() for nid, n in tree.nodes.items()},
            "delivery_mode": task_strategy.delivery.output_mode,
            "task_strategy": task_strategy.name,
            "dispatch_budget": dispatch_budget.snapshot(),
            "execution_failures": execution_failures,
        }


def _validate_runtime(runtime):
    _resolve_worker_limit({"max_total_workers": runtime["max_total_workers"]})
    for key in (
        "max_children",
        "max_rounds",
        "max_agents_per_round",
        "subagent_max_turns",
        "max_iterations",
        "question_timeout_s",
        "leaf_timeout_s",
        "answer_format_timeout_s",
    ):
        if runtime[key] <= 0:
            raise ValueError(f"Dynamic HTN {key} must be positive")
    for key in ("max_depth", "max_rebuilds", "fan_in_grace_s", "finalization_reserve_s"):
        if runtime[key] < 0:
            raise ValueError(f"Dynamic HTN {key} must be non-negative")


def prepare_profile_selection(metadata=None):
    """Preflight every candidate before router or worker requests can start."""
    profiles = load_profile_selection(metadata)
    if profiles.is_meta:
        router = _QuestionDriver(profiles.name, profiles.initial_profile, None, profiles=profiles)
        if router.TASK_ROUTER_TIMEOUT_S <= 0 or router.TASK_ROUTER_MAX_TOKENS <= 0:
            raise ValueError("Router timeout and max tokens must be positive")
        for key in profiles.routes:
            name, profile = profiles.select(key)
            candidate = _QuestionDriver(name, profile, None)
            _validate_runtime(candidate._resolve_task_runtime(profile))
    return profiles


async def dynamic_solve_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    """Execute through the same entry point used by the pipeline/benchmark runner."""
    metadata = state.get("metadata") or {}
    profiles = prepare_profile_selection(metadata)
    profile_name, profile = profiles.name, profiles.initial_profile
    safe_id = re.sub(r"[^A-Za-z0-9._+-]", "_", str(ctx.task_id or "task"))
    trial_dir = metadata.get("_trial_dir")
    root = (
        Path(trial_dir) / "agent" / "dynir"
        if trial_dir
        else Path(tempfile.gettempdir()) / "agentharness-outputs" / safe_id / "dynir"
    )
    run_dir = root / uuid.uuid4().hex[:12]
    runtime = TaskRuntime(profile, str(ctx.task_id or "task"), run_dir,
                          router_profile=profile if profiles.is_meta else None,
                          web_policy=metadata.get("benchmark_web_policy"))
    previous_strategy = get_active_task_strategy()
    driver = None
    with runtime:
        try:
            runtime.write_json("profile_selection.json", _redact(profiles.snapshot()))
            runtime.write_json("benchmark_web_policy.json", runtime.web_policy)
            driver = _QuestionDriver(profile_name, profile, runtime, metadata.get("dynir_config"), profiles)
            result = await runtime.run_until_quota(driver.run(state, runtime))
            result["usage_summary"] = runtime.usage_summary()
            result["artifact_dir"] = str(run_dir)
            runtime.write_json("result.json", result)
            runtime.write_json("usage.json", runtime.usage_records)
            atomic_write_text(run_dir / "report.md", result.get("final_content", ""))
            return result
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            try:
                artifacts = [("usage.json", runtime.usage_records)]
                if driver is not None and getattr(driver, "dispatch_budget", None) is not None:
                    artifacts.append(("dispatch_budget.json", driver.dispatch_budget.snapshot()))
                for name, payload in artifacts:
                    try:
                        runtime.write_json(name, payload)
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
                        logger.warning("Could not persist %s during cleanup: %s", name, type(exc).__name__)
                try:
                    await runtime.aclose()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
                    logger.warning("Task resource cleanup failed: %s", type(exc).__name__)
            finally:
                set_active_task_strategy(previous_strategy)
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error
