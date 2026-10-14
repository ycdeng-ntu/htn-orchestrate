"""Shared judge infrastructure — LLM client, retry helpers, type aliases.

The judge endpoint is read from ``JUDGE_API_KEY`` / ``JUDGE_BASE_URL`` env
(falling back to ``OPENAI_*``). The judge **model** and reasoning effort
are pinned per benchmark in the per-family modules — not configurable
via env.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Literal
from contextlib import contextmanager
from contextvars import ContextVar

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Verdict = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "SCORED"]
JudgeFn = Callable[[str, str, str], Awaitable[Verdict]]

_DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"
_JUDGE_CONFIG: ContextVar[dict | None] = ContextVar("benchmark_judge_config", default=None)


@contextmanager
def judge_scope(config: dict | None):
    from benchmarks.core.profile import validate_judge
    token = _JUDGE_CONFIG.set(validate_judge(config) if config is not None else None)
    try:
        yield
    finally:
        _JUDGE_CONFIG.reset(token)


def judge_parameters(default_model: str) -> dict:
    config = _JUDGE_CONFIG.get() or {}
    result = {"model": config.get("model") or default_model}
    if config.get("reasoning_effort"):
        result["reasoning_effort"] = config["reasoning_effort"]
    return result


def judge_transport() -> tuple[str | None, str | None]:
    config = _JUDGE_CONFIG.get() or {}
    provider = config.get("provider", "judge")
    if provider == "judge":
        return (os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    prefix = provider.upper()
    if provider == "openai" and os.environ.get("NEW_API_API_KEY"):
        prefix = "NEW_API"
        if not os.environ.get("NEW_API_BASE_URL"):
            raise ValueError("NEW_API_BASE_URL is required with NEW_API_API_KEY")
    key = os.environ.get(prefix + "_API_KEY")
    if not key:
        raise ValueError(f"Missing {prefix}_API_KEY for benchmark judge")
    base = os.environ.get(prefix + "_BASE_URL") or {
        "openrouter": "https://openrouter.ai/api/v1", "openai": "https://api.openai.com/v1",
    }[provider]
    return key, base



def _make_client() -> AsyncOpenAI:
    api_key, base_url = judge_transport()
    if base_url:
        base_url = base_url.rstrip("/").removesuffix("/chat/completions")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _build_judge_kwargs(
    prompt: str, *, max_completion_tokens: int,
    model_override: str | None = None,
    default_model: str | None = None,
    default_reasoning_effort: str | None = None,
) -> dict:
    """Common kwargs for ``chat.completions.create`` on judge calls.

    A task-local benchmark profile is authoritative. Without one, use
    model_override, default_model, then the global fallback; effort follows
    default_reasoning_effort.
    """
    model = model_override or default_model or _DEFAULT_JUDGE_MODEL
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    config = _JUDGE_CONFIG.get()
    if config is not None:
        kwargs.update(judge_parameters(model))
    elif default_reasoning_effort:
        kwargs["reasoning_effort"] = default_reasoning_effort
    return kwargs


def _model_supports_reasoning_effort(model: str) -> bool:
    """Whether the model accepts the ``reasoning_effort`` kwarg.

    OpenAI o-series (o1 / o3 / o4) and gpt-5 family accept it; classic
    chat models (gpt-4.1 / gpt-4o / gpt-4) reject it with a 400.
    """
    m = (model or "").lower()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    if m.startswith(("o1", "o3", "o4")):
        return True
    return m.startswith("gpt-5")


def _effort_chain(model: str, initial: str) -> list[str]:
    """Cascade of ``reasoning_effort`` levels to try (``initial`` → medium → low).

    For non-reasoning models returns ``[""]`` so the kwarg is never sent.
    """
    if not _model_supports_reasoning_effort(model):
        return [""]
    chain: list[str] = []
    for e in [(initial or "").lower(), "medium", "low"]:
        if e and e not in chain:
            chain.append(e)
    return chain or [""]


async def _judge_call_with_effort_fallback(
    kwargs: dict, *, label: str, client: AsyncOpenAI | None = None,
    allow_effort_fallback: bool = True,
) -> str:
    """Run ``chat.completions.create`` with ``reasoning_effort`` cascade.

    Cascades the initial effort (from ``kwargs["reasoning_effort"]`` if set)
    through medium → low, retrying once at each level. Returns the completion
    string, or ``""`` if every attempt yielded empty content or errored.
    """
    initial = (kwargs.get("reasoning_effort") or "").lower()
    # Some published evaluation protocols require a fixed reasoning effort.
    # Transport/empty retries may repeat that setting, but must not lower it.
    config = _JUDGE_CONFIG.get()
    # Explicit null in a benchmark profile means no reasoning_effort parameter.
    # Direct legacy calls still retain their family fallback behavior.
    if config is not None and config.get("reasoning_effort") is None:
        chain = [""]
    else:
        chain = _effort_chain(kwargs.get("model", ""), initial) if allow_effort_fallback else [initial]
    content = ""
    for attempt_effort in chain:
        if attempt_effort:
            kwargs["reasoning_effort"] = attempt_effort
        else:
            kwargs.pop("reasoning_effort", None)
        for attempt_idx in range(2):
            try:
                active_client = client if client is not None else _make_client()
                resp = await active_client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                # API error (rate limit, 5xx, network). Sleep + continue so
                # the surrounding effort×attempt loop actually retries; the
                # earlier ``return ""`` here defeated the retry budget by
                # giving up after a single transient failure.
                logger.warning("%s judge call error (%s, effort=%s, attempt=%d): %s",
                               label, kwargs.get("model"), attempt_effort, attempt_idx, exc)
                await asyncio.sleep(1.0)
                continue
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if attempt_effort != initial or attempt_idx > 0:
                    logger.info(
                        "%s judge: empty at effort=%s, recovered at effort=%s (retry=%d)",
                        label, initial, attempt_effort, attempt_idx,
                    )
                return content
            await asyncio.sleep(1.0)
    return content
