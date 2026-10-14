"""Compact tool messages into durable, readable evidence references.

This observer must see results before the message postprocessor truncates them.
It retains unarchived results only until they leave the recent-results window;
archived records contain a short preview and references, never the full body.
Only runtime metadata can authorize reuse of an existing artifact. Tool output
text is untrusted and is never parsed for paths or successful execution status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_harness.core.atomic_files import atomic_write_text
from agent_harness.core.loop_types import BaseObserver, ToolResult, TurnContext
from agent_harness.core.messages import Message, text_of

_PREVIEW_CHARS = 320
_REF_FIELDS = (
    "char_count", "start_char", "end_char", "total_chars", "start_line", "end_line",
)


@dataclass
class _Evidence:
    tool_name: str
    tool_call_id: str
    content: Any
    content_format: str = "text"
    args: dict[str, Any] = field(default_factory=dict)
    is_error: bool | None = None
    preview: str = ""
    reference: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    result_truncated: bool = False
    original_result_chars: int | None = None
    evidence_id: str = field(default_factory=lambda: f"evidence-{uuid4().hex}")
    index: str | None = None


class EvidenceMemory(BaseObserver):
    """Observer and compactor for workers allowed to use ``read_file``.

    ``keep_tool_results=-1`` disables compaction. Without ``can_read``, results
    are preserved in full rather than replaced with inaccessible references.
    Storage failures propagate: an unavailable archive must never authorize
    removal of the original message. Input messages are never mutated.
    """

    critical = True

    def __init__(
        self, artifact_dir: Path, keep_tool_results: int, can_read: bool = True,
    ) -> None:
        if keep_tool_results < -1:
            raise ValueError("keep_tool_results must be >= -1")
        self._root = Path(artifact_dir).resolve()
        self._keep = keep_tool_results
        self._can_read = can_read
        self._records: dict[str, _Evidence] = {}

    def _trusted_reference(self, value: Any) -> dict[str, Any] | None:
        """Validate even trusted metadata against the worker's artifact root."""
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            return None
        kind = value.get("kind")
        if kind not in {"tool_output", "source", "read_source"}:
            return None
        try:
            path = Path(value["path"])
            if not path.is_absolute():
                path = self._root / path
            path = path.resolve(strict=True)
            if not path.is_relative_to(self._root) or not path.is_file():
                return None
        except (OSError, ValueError, RuntimeError):
            return None
        reference: dict[str, Any] = {"path": str(path), "kind": kind}
        for name in _REF_FIELDS:
            number = value.get(name)
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                reference[name] = number
        if kind == "read_source":
            read_args = value.get("read_args")
            if isinstance(read_args, dict):
                permitted = {"start_line", "end_line", "start_char", "max_chars"}
                if set(read_args) - permitted:
                    return None
                if any(
                    item is not None and (
                        not isinstance(item, int) or isinstance(item, bool) or item < 0
                    ) for item in read_args.values()
                ):
                    return None
                reference["read_args"] = {
                    name: number for name, number in read_args.items() if number is not None
                }
            else:
                # A reread can reuse a file only when its page is known;
                # otherwise preserve the actual returned text separately.
                start = reference.get("start_char")
                end = reference.get("end_char")
                total = reference.get("total_chars")
                if start is None or end is None or end < start:
                    return None
                if total is not None and end > total:
                    return None
        return reference

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> None:
        if not self._can_read or self._keep == -1:
            return
        metadata = getattr(result, "metadata", {}) or {}
        references = metadata.get("artifact_refs", [])
        references = references if isinstance(references, list) else []
        claimed_outputs = [
            item for item in references if isinstance(item, dict) and item.get("kind") == "tool_output"
        ]
        references = [
            reference for item in references
            if (reference := self._trusted_reference(item)) is not None
        ]
        read_ref = metadata.get("read_ref")
        if isinstance(read_ref, dict):
            reference = self._trusted_reference({**read_ref, "kind": "read_source"})
            if reference is not None:
                references.append(reference)
        full_output = next(
            (ref for ref in references if ref["kind"] == "tool_output"), None,
        )
        read_source = next(
            (ref for ref in references if ref["kind"] == "read_source"), None,
        )
        # A failed read may contain diagnostic information absent from its
        # requested source. Preserve the actual error result in that case.
        reference = full_output or (read_source if not result.is_error else None)
        original_chars = metadata.get("original_result_chars")
        if not isinstance(original_chars, int) or isinstance(original_chars, bool):
            original_chars = next((
                ref["char_count"] for ref in claimed_outputs
                if isinstance(ref.get("char_count"), int) and not isinstance(ref["char_count"], bool)
            ), None)
        self._records[result.tool_call_id] = _Evidence(
            tool_name=result.name,
            tool_call_id=result.tool_call_id,
            content=None if reference else result.result,
            args={} if reference else dict(result.args),
            is_error=result.is_error,
            preview=text_of(result.result)[:_PREVIEW_CHARS],
            reference=reference,
            sources=[ref for ref in references if ref["kind"] == "source"],
            # A rejected native output reference may have accompanied only
            # an inline prefix. Never relabel that prefix a complete result.
            result_truncated=(bool(metadata.get("result_truncated")) or bool(claimed_outputs))
            and full_output is None,
            original_result_chars=original_chars,
        )

    def _archive(self, record: _Evidence) -> None:
        if record.reference is not None:
            # Do not publish a newly stale path if the native artifact was
            # removed after registration. The full result may no longer exist
            # in memory, so explicit failure is the only lossless response.
            if self._trusted_reference(record.reference) is None:
                raise FileNotFoundError(f"Evidence artifact unavailable: {record.reference['path']}")
            return
        path = (self._root / f"{record.evidence_id}.json").resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("Evidence archive path escapes the artifact root")
        content = record.content
        content_format = record.content_format
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
            content_format = "json"
        payload = {
            "evidence_id": record.evidence_id,
            "tool_name": record.tool_name,
            "tool_call_id": record.tool_call_id,
            "is_error": record.is_error,
            "args": record.args,
            "content": content,
            "content_format": content_format,
            "source_refs": record.sources,
            "result_truncated": record.result_truncated,
            "original_result_chars": record.original_result_chars,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        atomic_write_text(path, serialized)
        # Commit the reference only after a complete, atomic write succeeds.
        record.reference = {
            "path": str(path), "kind": "tool_output", "char_count": len(content),
        }
        record.content = None
        record.args = {}

    @staticmethod
    def _index(record: _Evidence) -> str:
        if record.reference is None:
            raise ValueError("Cannot index evidence before it has been archived")
        reference = record.reference
        status = "unknown" if record.is_error is None else (
            "error" if record.is_error else "success"
        )
        lines = [
            f"[Archived tool evidence: {record.evidence_id}]",
            f"Tool: {record.tool_name}",
            f"Execution status: {status} (not a scientific correctness judgment)",
            f"Preview (verbatim prefix): {json.dumps(record.preview, ensure_ascii=False)}",
        ]
        if reference["kind"] == "read_source":
            read_args = reference.get("read_args")
            if read_args is None:
                read_args = {
                    "start_char": reference["start_char"],
                    "max_chars": reference["end_char"] - reference["start_char"],
                }
            lines.append("Previously read source excerpt; this file is the source, "
                         "not a newly generated tool result.")
            lines.append("Read again with read_file: " + json.dumps(
                {"path": reference["path"], **read_args}, ensure_ascii=False,
            ))
        else:
            label = "Captured partial tool result (complete output unavailable)" if (
                record.result_truncated
            ) else "Full tool result"
            lines.append(label + "; read_file: " + json.dumps({
                "path": reference["path"],
            }, ensure_ascii=False))
        for source in record.sources:
            lines.append("Related source (distinct from tool output); read_file: " + json.dumps({
                "path": source["path"],
            }, ensure_ascii=False))
        lines.append("Read the artifact to recover details; follow pagination when needed.")
        return "\n".join(lines)

    async def compact(self, messages: list[Message], keep_recent: int) -> list[Message]:
        if not self._can_read or self._keep == -1:
            return messages
        live_keys = {
            msg.get("tool_call_id") or f"message:{id(msg)}"
            for msg in messages if msg.get("role") == "tool"
        }
        # Discard results removed by an overflow guard or rollback, including
        # when the retained history is too short to require compaction.
        self._records = {key: value for key, value in self._records.items() if key in live_keys}
        tool_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "tool"]
        old_indices = tool_indices[:-self._keep] if self._keep else tool_indices
        if not old_indices:
            return messages
        out = list(messages)
        for index in old_indices:
            message = messages[index]
            call_id = message.get("tool_call_id", "")
            key = call_id or f"message:{id(message)}"
            record = self._records.get(key)
            if record is None:
                # Replay/legacy messages might not have an observed result.
                # Preserve exactly the content available; never infer success
                # or authorize a path based on arbitrary output text.
                record = _Evidence(
                    tool_name=message.get("name", "unknown"),
                    tool_call_id=call_id,
                    content=message.get("content", ""),
                    preview=text_of(message.get("content", ""))[:_PREVIEW_CHARS],
                )
                self._records[key] = record
            if record.index is not None and message.get("content") == record.index:
                # An already-published index needs no new filesystem work on
                # every turn; read_file handles later artifact availability.
                continue
            self._archive(record)
            if record.index is None:
                record.index = self._index(record)
            if message.get("content") != record.index:
                out[index] = {**message, "content": record.index}
        # Results dropped by an overflow guard or rollback must not retain
        # large bodies indefinitely. Recent messages retain at most N bodies.
        return out
