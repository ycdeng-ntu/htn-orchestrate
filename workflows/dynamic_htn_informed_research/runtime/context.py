"""Per-question resources; no workflow state is installed on global services."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.core.llm_errors import LLMQuotaError
from agent_harness.core.atomic_files import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeOverrides:
    """Optional embedding/test seams. Builders receive task-local configuration."""

    llm_factory: Callable[[str, str, dict[str, Any]], Any] | None = None
    tool_factory: Callable[..., Any] | None = None
    dispatch: Callable[..., Any] | None = None
    event_sink: Callable[..., Any] | None = None


_OVERRIDES: ContextVar[RuntimeOverrides] = ContextVar("dynir_overrides", default=RuntimeOverrides())
_CURRENT: ContextVar[TaskRuntime | None] = ContextVar("dynir_runtime", default=None)


@contextmanager
def override_runtime(**kwargs: Any):
    """Inject fake transports or embedding resources without editing process env."""
    token = _OVERRIDES.set(RuntimeOverrides(**kwargs))
    try:
        yield
    finally:
        _OVERRIDES.reset(token)


def current_runtime() -> TaskRuntime:
    runtime = _CURRENT.get()
    if runtime is None:
        raise RuntimeError("Dynamic HTN requires a task runtime")
    return runtime


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[redacted]" if any(s in str(k).lower() for s in ("api_key", "authorization", "secret", "password", "headers")) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_redact(v) for v in value]
    return value


class TaskRuntime:
    def __init__(self, profile: dict[str, Any], task_id: str, artifact_dir: Path, *,
                 router_profile: dict[str, Any] | None = None,
                 web_policy: dict | str | None = None):
        from ..tools.scrape_cache import _ScrapeCache

        from agent_harness.core.web_policy import validate_web_policy
        self.web_policy = validate_web_policy(web_policy) if web_policy is not None else None
        self.profile = deepcopy(profile)
        self.router_profile = deepcopy(router_profile)
        self.task_id = task_id
        self.artifact_dir = artifact_dir
        self.overrides = _OVERRIDES.get()
        self.clients: dict[str, Any] = {}
        self.owned_clients: list[Any] = []
        self.usage_records: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._token = None
        self._close_task = None
        self.close_errors: list[str] = []
        self._credentials: dict[str, Any] | None = None
        self.scrape_cache = _ScrapeCache()
        self.quota_failure: LLMQuotaError | None = None
        self._quota_failed = asyncio.Event()

    def raise_if_quota_failed(self) -> None:
        if self.quota_failure is not None:
            raise self.quota_failure

    def fail_quota(self, error: LLMQuotaError, provider: str, model: str, purpose: str) -> None:
        if self.quota_failure is not None:
            return
        self.quota_failure = error
        self._quota_failed.set()
        try:
            self.write_json("quota_error.json", {
                "error_code": error.error_code, "status_code": error.status_code,
                "provider": provider, "model": model, "purpose": purpose,
            })
        except OSError as exc:
            logger.warning("Could not persist quota error: %s", type(exc).__name__)

    async def run_until_quota(self, awaitable):
        """Stop this question even if a strategy catches the transport exception.

        Cancel and settle the driver so its tree/worker cleanup preserves existing
        evidence. The latch is question-local, never shared with unrelated tasks.
        """
        work = asyncio.create_task(awaitable, name="dynir-question")
        signal = asyncio.create_task(self._quota_failed.wait(), name="dynir-quota-watch")
        try:
            await asyncio.wait((work, signal), return_when=asyncio.FIRST_COMPLETED)
            if self.quota_failure is not None:
                if not work.done():
                    work.cancel()
                settled = await asyncio.gather(work, return_exceptions=True)
                if isinstance(settled[0], dict):
                    try:
                        self.write_json("quota_interrupted_result.json", settled[0])
                    except OSError as exc:
                        logger.warning("Could not persist interrupted result: %s", type(exc).__name__)
                raise self.quota_failure
            return await work
        finally:
            for task in (work, signal):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, signal, return_exceptions=True)

    def __enter__(self):
        if self._token is not None or self._close_task is not None:
            raise RuntimeError("A task runtime cannot be entered twice")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._token = _CURRENT.set(self)
        self.write_json("effective_config.json", _redact(self.profile))
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        try:
            await self.aclose()
        finally:
            self.__exit__(*exc)

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_clients())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await asyncio.shield(self._close_task)
            raise

    async def _close_clients(self) -> None:
        try:
            self.write_json("fetch_retry_state.json", self.scrape_cache.snapshot())
        except OSError as exc:
            logger.warning("Could not persist fetch retry state: %s", type(exc).__name__)
        seen: set[int] = set()
        for client in self.owned_clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is None:
                close = getattr(getattr(client, "_client", None), "close", None)
            if close is None:
                continue
            try:
                outcome = close()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as exc:
                self.close_errors.append(type(exc).__name__)
                logger.warning("Dynamic HTN client cleanup failed: %s", type(exc).__name__)
        self.owned_clients.clear()
        self.clients.clear()
        self._credentials = None

    def write_json(self, name: str, payload: Any) -> None:
        path = self.artifact_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    async def emit_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "task_id": self.task_id, **payload}
        self.events.append(record)
        with (self.artifact_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.overrides.event_sink is not None:
            result = self.overrides.event_sink(event, payload)
            if inspect.isawaitable(result):
                await result

    def record_usage(
        self, provider: str, model: str, response: Any, *, status: str = "completed",
        purpose: str = "main", usage_known: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        from ..diagnostics import scope_snapshot

        usage = getattr(response, "usage", None) or {}
        if not isinstance(usage, dict):
            usage = {}
        known = any(k in usage for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"))
        self.usage_records.append({
            **scope_snapshot(), "task_id": self.task_id,
            "provider": provider, "purpose": purpose,
            "reasoning_effort": reasoning_effort,
            "request_id": uuid.uuid4().hex,
            "model": getattr(response, "model", "") or model,
            "usage": dict(usage), "status": status,
            "usage_known": known if usage_known is None else usage_known,
        })

    def record_external_usage(
        self, *, provider: str, model: str, status: str = "completed",
        usage_known: bool = True, **counts: Any,
    ) -> None:
        """Receive one event per actual summary-tool request, outside native loops."""
        from agent_harness.core.llm import LLMResponse
        from ..diagnostics import diagnostic_scope

        if status == "LLMQuotaError":
            self.fail_quota(LLMQuotaError(int(counts.get("http_status") or 402)),
                            provider, model, "summary")

        usage = {
            key: int(counts[key] or 0)
            for key in ("prompt_tokens", "completion_tokens", "cache_read_tokens")
            if key in counts
        }
        with diagnostic_scope(call_kind="tool_summary"):
            self.record_usage(
                provider, model, LLMResponse(usage=usage), status=status,
                purpose="summary", usage_known=usage_known,
            )

    def usage_summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.usage_records),
            "unknown_usage_calls": sum(not r["usage_known"] for r in self.usage_records),
            "prompt_tokens": sum(int(r["usage"].get("prompt_tokens", r["usage"].get("input_tokens", 0)) or 0) for r in self.usage_records),
            "completion_tokens": sum(int(r["usage"].get("completion_tokens", r["usage"].get("output_tokens", 0)) or 0) for r in self.usage_records),
        }

    def _credential_source(self) -> dict[str, Any]:
        from dotenv import dotenv_values

        if self._credentials is None:
            self._credentials = {
                **dotenv_values(Path(__file__).resolve().parents[3] / ".env"), **os.environ,
            }
        return self._credentials

    def resolved_tool_config(self) -> dict[str, Any]:
        """Bridge standard .env backend fields into one worker's explicit config."""
        source = self._credential_source()
        web = self.profile.get("web") or {}
        fields = {
            "serper_api_key": "SERPER_API_KEY", "serper_base_url": "SERPER_BASE_URL",
            "jina_api_key": "JINA_API_KEY", "jina_base_url": "JINA_BASE_URL",
            "e2b_api_key": "E2B_API_KEY", "e2b_template": "E2B_TEMPLATE",
            "ncbi_api_key": "NCBI_API_KEY", "unpaywall_email": "UNPAYWALL_EMAIL",
        }
        result = {}
        for field, env_name in fields.items():
            value = web[field] if field in web else source.get(env_name)
            if value is not None:
                result[field] = value
        if "e2b_timeout" in web or "E2B_TIMEOUT" in source:
            setting = "web.e2b_timeout" if "e2b_timeout" in web else "E2B_TIMEOUT"
            raw = web["e2b_timeout"] if "e2b_timeout" in web else source["E2B_TIMEOUT"]
            timeout = raw if type(raw) is int else None
            if isinstance(raw, str) and raw.strip().isascii() and raw.strip().isdecimal():
                timeout = int(raw.strip())
            if timeout is None or timeout <= 0:
                raise ValueError(f"{setting} must be a positive integer number of seconds")
            result["e2b_timeout"] = timeout
        return result

    def _resolve_endpoint(
        self, config: dict[str, Any], provider: str, source: dict[str, Any],
        *, summary_prefix: str = "",
    ) -> dict[str, Any]:
        result = deepcopy(config)
        prefix = provider.upper().replace("-", "_")
        summary_key = source.get(summary_prefix + "API_KEY") if summary_prefix else None
        summary_url = source.get(summary_prefix + "BASE_URL") if summary_prefix else None
        result["provider"] = provider
        result["api_key"] = result.get("api_key") or summary_key or source.get(prefix + "_API_KEY") or ""
        result["base_url"] = result.get("base_url") or summary_url or source.get(prefix + "_BASE_URL") or {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }.get(provider, "")
        return result

    def resolved_summary_config(self) -> dict[str, Any]:
        """Resolve tool extraction credentials without substituting the main client.

        SummaryExtractor requires an endpoint as well as a model to select a
        profile candidate. Resolve provider defaults before handing it the config
        so a valid profile without base_url is not silently replaced by env.
        """
        requested = deepcopy(self.profile.get("summary_llm") or {})
        if not requested:
            return {}
        source = self._credential_source()
        provider = str(requested.get("provider") or (self.profile.get("llm") or {}).get("provider") or "openai")
        resolved = self._resolve_endpoint(requested, provider, source, summary_prefix="SUMMARY_LLM_")
        fallback = requested.get("fallback")
        if isinstance(fallback, dict):
            fallback_provider = str(fallback.get("provider") or provider)
            resolved["fallback"] = self._resolve_endpoint(
                fallback, fallback_provider, source, summary_prefix="SUMMARY_LLM_FALLBACK_",
            )
        return resolved

    def summary_environment(self) -> dict[str, Any]:
        """Only summary-specific env fields cross the tool extraction boundary."""
        return {
            key: value for key, value in self._credential_source().items()
            if key.startswith("SUMMARY_LLM_") and value is not None
        }

    def native_llm(
        self, provider: str, model: str, sampling: dict[str, Any], *, purpose: str = "main",
    ) -> Any:
        from .llm import MeteredLLM

        if self._close_task is not None:
            raise RuntimeError("Task runtime is closed")
        if purpose not in {"main", "summary", "router"}:
            raise ValueError(f"Unknown LLM purpose: {purpose}")
        key = json.dumps([purpose, provider, model, sampling], sort_keys=True, default=str)
        if key in self.clients:
            return self.clients[key]
        if self.overrides.llm_factory:
            client = self.overrides.llm_factory(provider, model, deepcopy(sampling))
        else:
            from agent_harness.infra.openai_client import OpenAIClient

            source = self._credential_source()
            requested_profile = self.router_profile if purpose == "router" else self.profile
            if requested_profile is None:
                raise ValueError("Router LLM requires an explicit router profile")
            requested = requested_profile.get("summary_llm" if purpose == "summary" else "llm") or {}
            # Credentials and endpoints belong to the selected provider. A
            # fallback to a different provider must not inherit the main key.
            cfg = requested if requested.get("provider", provider) == provider else {}
            prefix = provider.upper().replace("-", "_")
            cfg = self._resolve_endpoint(
                cfg, provider, source,
                summary_prefix="SUMMARY_LLM_" if purpose == "summary" and cfg is requested else "",
            )
            api_key, base_url = cfg["api_key"], cfg["base_url"]
            if not api_key:
                raise ValueError(f"Missing {prefix}_API_KEY for the selected Dynamic HTN provider")
            if provider not in {"openai", "openrouter"} and not base_url:
                raise ValueError(f"Provider {provider!r} requires an explicit OpenAI-compatible base_url")
            extra = deepcopy(sampling.get("extra_body") or {})
            for field in ("top_p", "repetition_penalty"):
                if sampling.get(field) is not None:
                    extra[field] = sampling[field]
            client = OpenAIClient(
                model, api_key=str(api_key), base_url=base_url,
                temperature=sampling.get("temperature"),
                max_completion_tokens=sampling.get("max_tokens"),
                timeout=float(cfg.get("timeout_s", 600)),
                default_headers=deepcopy(cfg.get("headers") or None), extra_body=extra,
            )
        if isinstance(client, MeteredLLM):
            if client.runtime is not self:
                raise ValueError("Cannot reuse a metered LLM from another task runtime")
            wrapped = client
        else:
            self.owned_clients.append(client)
            wrapped = MeteredLLM(
                client, self, provider, model, purpose=purpose,
                reasoning_effort=(sampling.get("extra_body") or {}).get("reasoning_effort"),
            )
        self.clients[key] = wrapped
        return wrapped
