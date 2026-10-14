"""Task-local Miro summary policy over an injected OpenAI-compatible HTTP client.

Provider credentials must already be resolved by the workflow. This module never
changes environment variables or global clients. Failure intentionally returns
the original text (at most 20,000 characters), as the source extractor does.
"""
from __future__ import annotations

from agent_harness.core.llm_errors import LLMQuotaError, quota_error

import asyncio
import json
import logging
import os
from copy import deepcopy
from typing import Any, Callable, Mapping

import httpx

from ._summary_prompts import _build_payload, _EXTRACT_INFO_PROMPT, _EXTRACT_INFO_PROMPT_SCIENCE

logger = logging.getLogger(__name__)
FALLBACK_TRUNCATE = 20_000


def raw_fallback(content: str) -> str:
    return content[:FALLBACK_TRUNCATE] + "\n\n[Content truncated...]" if len(content) > FALLBACK_TRUNCATE else content


def _environment_candidate(prefix: str, environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = os.environ if environment is None else environment
    raw_routing = str(source.get(prefix + "PROVIDER_ROUTING") or "")
    try:
        routing = json.loads(raw_routing) if raw_routing else None
    except ValueError:
        routing = None
    return {
        "base_url": source.get(prefix + "BASE_URL") or "",
        "model": source.get(prefix + "MODEL") or source.get(prefix + "MODEL_NAME") or "",
        "api_key": source.get(prefix + "API_KEY") or "",
        "reasoning_effort": source.get(prefix + "REASONING_EFFORT") or "",
        "retry_empty_or_length_with_thinking_disabled": str(source.get(prefix + "RETRY_THINKING_DISABLED") or "").lower() in {"1", "true", "yes", "on"},
        "provider_routing": routing if isinstance(routing, dict) else None,
        "provider": prefix.lower().rstrip("_"),
    }


def summary_candidates(config: Mapping[str, Any] | None, *, use_environment: bool = True,
                       environment: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Primary profile/env, profile fallback, then environment fallback."""
    override = deepcopy(dict(config or {}))
    primary = override
    allow_environment = use_environment and override.get("allow_environment_fallback", True)
    if not (primary.get("base_url") and primary.get("model")):
        primary = _environment_candidate("SUMMARY_LLM_", environment) if allow_environment else {}
    items = [primary, override.get("fallback")]
    if allow_environment:
        items.append(_environment_candidate("SUMMARY_LLM_FALLBACK_", environment))
    return [dict(c) for c in items if isinstance(c, dict) and c.get("base_url") and c.get("model")]


class SummaryExtractor:
    def __init__(self, client: httpx.AsyncClient, config: Mapping[str, Any] | None = None, *,
                 extract_mode: str = "", use_environment: bool = True,
                 usage_sink: Callable[..., Any] | None = None, retry_delay: float = 1.0,
                 environment: Mapping[str, Any] | None = None):
        self.client = client
        self.candidates = summary_candidates(config, use_environment=use_environment, environment=environment)
        self.prompt = _EXTRACT_INFO_PROMPT_SCIENCE if extract_mode == "science" else _EXTRACT_INFO_PROMPT
        self.usage_sink = usage_sink
        self.retry_delay = retry_delay

    async def _request(self, candidate: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str]:
        endpoint = str(candidate["base_url"]).rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if candidate.get("api_key"):
            headers["Authorization"] = f"Bearer {candidate['api_key']}"
        usage: dict[str, Any] = {}
        status = "success"
        http_status = None
        try:
            response = await self.client.post(endpoint, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Summary provider returned a non-object response")
            usage = data.get("usage") or {}
        except BaseException as exc:
            terminal = quota_error(exc)
            status = type(terminal or exc).__name__
            if terminal is not None:
                http_status = terminal.status_code
                raise terminal from None
            raise
        finally:
            if self.usage_sink:
                self.usage_sink(model=candidate["model"], provider=candidate.get("_provider_label") or candidate.get("provider") or "summary_llm",
                                prompt_tokens=int(usage.get("prompt_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0),
                                cache_read_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                                status=status, usage_known=bool(usage),
                                **({"http_status": http_status} if http_status else {}))
        choices = data.get("choices") or [{}]
        choice = choices[0]
        return str((choice.get("message") or {}).get("content") or ""), str(choice.get("finish_reason") or "")

    async def __call__(self, content: str, focus: str) -> str:
        for candidate in self.candidates:
            payload = _build_payload(str(candidate["model"]), self.prompt.format(content=content, focus=focus),
                                     reasoning_effort=str(candidate.get("reasoning_effort") or ""),
                                     provider_routing=candidate.get("provider_routing"))
            current_content = content
            for attempt in range(4):
                try:
                    summary, finish = await self._request(candidate, payload)
                    if candidate.get("retry_empty_or_length_with_thinking_disabled") and (not summary or finish == "length"):
                        disabled = _build_payload(str(candidate["model"]), payload["messages"][0]["content"],
                                                  thinking_disabled=True, provider_routing=candidate.get("provider_routing"))
                        try:
                            rescued, _ = await self._request(candidate, disabled)
                            if rescued:
                                return rescued
                        except LLMQuotaError:
                            raise
                        except Exception as exc:
                            logger.warning("Summary thinking-disabled retry failed: %s", type(exc).__name__)
                    if summary:
                        return summary
                    break
                except LLMQuotaError:
                    raise
                except httpx.HTTPStatusError as exc:
                    text = exc.response.text
                    if "maximum context length" in text or "longer than the model's context length" in text:
                        remove = 40_960 * (attempt + 1)
                        if remove >= len(current_content):
                            break
                        current_content = content[:-remove] + "[...truncated]"
                        payload["messages"][0]["content"] = self.prompt.format(content=current_content, focus=focus)
                        continue
                    if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                        break
                except Exception as exc:
                    logger.warning("Summary attempt failed: %s", type(exc).__name__)
                if attempt < 3:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        return raw_fallback(content)
