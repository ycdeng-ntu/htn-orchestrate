"""Keep assistant requests and tool responses paired across loop recovery."""
from __future__ import annotations

import json

from agent_harness.core.llm import LLMResponse
from agent_harness.core.messages import Message, is_assistant_msg, is_tool_msg, tool_msg


def _call_ids(messages: list[Message]) -> set[str]:
    return {
        call["id"]
        for message in messages
        for call in message.get("tool_calls", [])
        if call.get("id")
    }


def _unique_id(preferred: str, used: set[str], fallback: str) -> str:
    candidate = preferred or fallback
    suffix = 1
    while candidate in used:
        candidate = f"{fallback}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def prepare_native_tool_call_ids(
    response: LLMResponse, messages: list[Message], attempt: int,
) -> None:
    """Assign missing/duplicate IDs before both history normalization and parsing."""
    used = _call_ids(messages)
    response.tool_calls = [
        {
            **call,
            "id": _unique_id(call.get("id", ""), used, f"call_{attempt}_{index}"),
        }
        for index, call in enumerate(response.tool_calls)
    ]


def record_parsed_tool_calls(
    history_msg: Message, parsed_calls: list[dict], messages: list[Message], attempt: int,
) -> None:
    """Write recovered text calls into the assistant envelope before dispatch.

    Native calls retain their original arguments. Recovered IDs are made unique
    across turns, since text parsers often restart their counters on each call.
    """
    recorded = list(history_msg.get("tool_calls") or [])
    native_ids = {call["id"] for call in recorded}
    used = _call_ids(messages)
    parsed_ids: set[str] = set()
    for index, call in enumerate(parsed_calls):
        call_id = call.get("id") or ""
        if call_id in native_ids and call_id not in parsed_ids:
            parsed_ids.add(call_id)
            continue
        call_id = _unique_id(call_id, used, f"call_{attempt}_recovered_{index}")
        call["id"] = call_id
        parsed_ids.add(call_id)
        recorded.append({
            "type": "function",
            "id": call_id,
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
            },
        })
    if recorded:
        history_msg["tool_calls"] = recorded


def answer_dropped_tool_calls(
    messages: list[Message], history_msg: Message,
    pending_calls: list[dict], tool_names: set[str], *, reason: str = "",
) -> None:
    """Answer calls omitted by parsing, dispatch caps, or an early-stop policy.

    ``pending_calls`` will receive real responses later. This helper never runs
    tools and must not count as successful tool execution in workflow observers.
    """
    answered = {message.get("tool_call_id") for message in messages if is_tool_msg(message)}
    answered.update(call.get("id") for call in pending_calls)
    for call in history_msg.get("tool_calls") or []:
        call_id = call.get("id")
        if not call_id or call_id in answered:
            continue
        name = (call.get("function") or {}).get("name", "")
        detail = reason or (
            f"unknown tool '{name}' is not available"
            if name not in tool_names else
            "this tool call was not dispatched; re-issue it if still needed"
        )
        messages.append(tool_msg(f"[tool call not executed] {detail}", call_id))
        answered.add(call_id)


def pop_last_assistant_turn(messages: list[Message]) -> None:
    """Roll back the assistant request together with its trailing tool replies."""
    index = len(messages)
    while index > 0 and is_tool_msg(messages[index - 1]):
        index -= 1
    if index > 0 and is_assistant_msg(messages[index - 1]):
        del messages[index - 1:]
