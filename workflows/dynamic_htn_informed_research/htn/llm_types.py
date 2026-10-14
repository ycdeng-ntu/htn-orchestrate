"""Backend-agnostic LLM interface for the ported HTN core.

This preserves the Miro HTN control protocol while the workflow adapter uses
AgentHarness's native clients. The pure logic files (``planner`` / ``strategies`` /
``orchestrator``) import only this small surface and stay independent of concrete
providers. Control responses deliberately omit native tool-call IDs; worker
message histories retain their native IDs in the agent loop.

Contract expected by the ported code:
- ``LLMResponse.tool_calls`` is a list of RAW OpenAI tool-call dicts, i.e.
  ``[{"function": {"name": str, "arguments": <json-string>}}, ...]`` — the
  parse helpers do ``json.loads(tc["function"]["arguments"])``.
- ``LLMClient.chat(messages, tools=None)`` returns an ``LLMResponse``. ``tools``
  is a list of OpenAI function schemas (or ``None`` for a plain completion).
- ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]


def system_msg(content: str) -> Message:
    return {"role": "system", "content": content}


def user_msg(content: str) -> Message:
    return {"role": "user", "content": content}


@dataclass
class LLMResponse:
    """Minimal response envelope the ported HTN code reads."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """A model handle bound to one provider/model."""

    model: str

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        ...
