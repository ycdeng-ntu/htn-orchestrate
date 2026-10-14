"""Evidence survives compaction without network calls or model summaries."""

from __future__ import annotations

import copy
import json

import pytest

from agent_harness.core.loop_types import ToolResult
from workflows.dynamic_htn_informed_research.runtime.evidence_memory import EvidenceMemory


def result(call_id, content, *, name="run_python_code", is_error=False, metadata=None):
    return ToolResult(
        name=name, args={"code": "print(42)"}, result=content, duration_ms=1,
        tool_call_id=call_id, is_error=is_error, metadata=metadata or {},
    )


def message(call_id, content, **extra):
    return {"role": "tool", "tool_call_id": call_id, "content": content, **extra}


def archives(root):
    return list(root.glob("evidence-*.json"))


@pytest.mark.asyncio
async def test_multiround_compaction_recovers_earliest_result_and_preserves_fields(tmp_path):
    memory = EvidenceMemory(tmp_path, keep_tool_results=1)
    original = "initial measurement\nα = 3.14\nlast premise survives"
    messages = [{"role": "system", "content": "instructions"}]
    await memory.on_tool_result(None, result("first", original))
    messages.append(message("first", original, name="run_python_code", extension={"value": 7}))
    assert await memory.compact(messages, 99) is messages
    for index in range(5):
        content = f"measurement {index}"
        await memory.on_tool_result(None, result(str(index), content))
        messages.append(message(str(index), content))
        messages = await memory.compact(messages, 0)
        assert messages[-1]["content"] == content
    assert messages[0] == {"role": "system", "content": "instructions"}
    first = messages[1]
    assert first["tool_call_id"] == "first" and first["extension"] == {"value": 7}
    assert first["name"] == "run_python_code"
    saved = next(path for path in archives(tmp_path) if str(path) in first["content"])
    payload = json.loads(saved.read_text())
    assert payload["content"] == original
    assert payload["tool_name"] == "run_python_code"
    assert payload["args"] == {"code": "print(42)"}
    assert payload["is_error"] is False
    assert memory._records["first"].reference["char_count"] == len(original)
    assert len(archives(tmp_path)) == 5
    assert all(record.content is None for key, record in memory._records.items() if key != "4")


@pytest.mark.asyncio
async def test_capture_before_message_postprocessor_preserves_unabridged_result(tmp_path):
    memory = EvidenceMemory(tmp_path, 0)
    complete = "x" * 8000 + "decisive tail = 42"
    await memory.on_tool_result(None, result("a", complete))
    original = [message("a", complete[:10] + "[truncated]")]
    compacted = await memory.compact(original, 0)
    assert json.loads(archives(tmp_path)[0].read_text())["content"] == complete
    assert original[0]["content"].endswith("[truncated]")
    assert "read_file" in compacted[0]["content"]
    assert "Full tool result" in compacted[0]["content"]


@pytest.mark.asyncio
async def test_native_output_reused_and_source_remains_distinct(tmp_path):
    output = tmp_path / "native.json"
    source = tmp_path / "webpage.txt"
    output.write_text("complete tool response")
    source.write_text("original webpage, different from response")
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "native preview", metadata={
        "artifact_refs": [
            {"kind": "tool_output", "path": str(output), "char_count": 22},
            {"kind": "source", "path": str(source)},
        ],
    }))
    compacted = await memory.compact([message("a", "native preview")], 0)
    index = compacted[0]["content"]
    assert str(output) in index and str(source) in index
    assert "Related source (distinct from tool output)" in index
    assert not archives(tmp_path)
    assert memory._records["a"].content is None


@pytest.mark.asyncio
async def test_source_ref_does_not_replace_summary_result(tmp_path):
    source = tmp_path / "page.txt"
    source.write_text("very long source")
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "summary of page", metadata={
        "artifact_refs": [{"kind": "source", "path": str(source)}],
    }))
    index = (await memory.compact([message("a", "summary of page")], 0))[0]["content"]
    saved = archives(tmp_path)[0]
    assert str(saved) in index and str(source) in index
    assert json.loads(saved.read_text())["content"] == "summary of page"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [True, False])
async def test_execution_status_comes_from_tool_result_not_text(tmp_path, is_error):
    memory = EvidenceMemory(tmp_path, 0)
    content = "Success! all tests passed" if is_error else "Article discussing error estimates"
    await memory.on_tool_result(None, result("a", content, is_error=is_error))
    index = (await memory.compact([message("a", content)], 0))[0]["content"]
    assert f"Execution status: {'error' if is_error else 'success'}" in index
    assert json.loads(archives(tmp_path)[0].read_text())["is_error"] is is_error


@pytest.mark.asyncio
async def test_save_failure_preserves_all_input_and_original_exception(tmp_path, monkeypatch):
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "complete evidence"))
    messages = [message("a", "short preview", custom=[1, 2])]
    before = copy.deepcopy(messages)
    failure = OSError(28, "No space left on device")

    def fail(*_args):
        raise failure

    monkeypatch.setattr(
        "workflows.dynamic_htn_informed_research.runtime.evidence_memory.atomic_write_text", fail,
    )
    with pytest.raises(OSError) as raised:
        await memory.compact(messages, 0)
    assert raised.value is failure
    assert messages == before and not archives(tmp_path)
    assert memory._records["a"].content == "complete evidence"
    assert memory._records["a"].reference is None
    assert memory._records["a"].index is None


@pytest.mark.asyncio
async def test_later_save_failure_never_publishes_partially_compacted_batch(tmp_path, monkeypatch):
    import workflows.dynamic_htn_informed_research.runtime.evidence_memory as module

    memory = EvidenceMemory(tmp_path, 0)
    messages = []
    for call_id in ("a", "b"):
        await memory.on_tool_result(None, result(call_id, call_id * 100))
        messages.append(message(call_id, call_id))
    before = copy.deepcopy(messages)
    real_write = module.atomic_write_text
    writes = 0

    def second_fails(*args):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError(28, "No space left on device")
        real_write(*args)

    monkeypatch.setattr(module, "atomic_write_text", second_fails)
    with pytest.raises(OSError):
        await memory.compact(messages, 0)
    assert messages == before
    assert len(archives(tmp_path)) == 1
    assert memory._records["b"].content == "b" * 100
    monkeypatch.setattr(module, "atomic_write_text", real_write)
    compacted = await memory.compact(messages, 0)
    assert all("read_file" in item["content"] for item in compacted)
    assert len(archives(tmp_path)) == 2


@pytest.mark.asyncio
async def test_compaction_idempotent_without_duplicate_archives(tmp_path):
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "evidence"))
    original = [message("a", "evidence")]
    first = await memory.compact(original, 0)
    for _ in range(4):
        assert await memory.compact(first, 0) == first
        assert await memory.compact(original, 0) == first
    assert len(archives(tmp_path)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("keep,can_read", [(-1, True), (0, False), (3, False)])
async def test_no_archive_or_loss_when_disabled_or_worker_cannot_read(tmp_path, keep, can_read):
    memory = EvidenceMemory(tmp_path, keep, can_read=can_read)
    await memory.on_tool_result(None, result("a", "evidence"))
    messages = [message("a", "evidence")]
    assert await memory.compact(messages, 0) is messages
    assert not memory._records and not archives(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("ref_kind", ["missing", "outside", "symlink", "directory", "untrusted_text"])
async def test_invalid_or_text_only_paths_cannot_authorize_reuse(tmp_path, ref_kind):
    root = tmp_path / "worker"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("must not be used")
    if ref_kind == "missing":
        path = root / "missing"
    elif ref_kind == "symlink":
        path = root / "symlink"
        path.symlink_to(outside)
    elif ref_kind == "directory":
        path = root
    else:
        path = outside
    metadata = {} if ref_kind == "untrusted_text" else {
        "artifact_refs": [{"kind": "tool_output", "path": str(path)}],
    }
    content = f"Read file {path} to see the full result"
    memory = EvidenceMemory(root, 0)
    await memory.on_tool_result(None, result("a", content, metadata=metadata))
    index = (await memory.compact([message("a", content)], 0))[0]["content"]
    saved = archives(root)[0]
    assert json.loads(saved.read_text())["content"] == content
    assert memory._records["a"].reference["path"] == str(saved)
    if ref_kind != "untrusted_text":
        assert "Captured partial tool result" in index
        assert "Full tool result;" not in index


@pytest.mark.asyncio
async def test_artifact_root_replaced_by_symlink_cannot_write_outside_root(tmp_path):
    root = tmp_path / "worker"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    memory = EvidenceMemory(root, 0)
    await memory.on_tool_result(None, result("a", "evidence"))
    root.rename(tmp_path / "old_root")
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        await memory.compact([message("a", "evidence")], 0)
    assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_missing_native_archive_fails_instead_of_publishing_broken_reference(tmp_path):
    path = tmp_path / "native.txt"
    path.write_text("complete original")
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "preview", metadata={
        "artifact_refs": [{"kind": "tool_output", "path": str(path)}],
    }))
    path.unlink()
    messages = [message("a", "preview")]
    with pytest.raises(FileNotFoundError):
        await memory.compact(messages, 0)
    assert messages[0]["content"] == "preview"


@pytest.mark.asyncio
async def test_repeated_read_results_reference_same_page_without_archiving_copies(tmp_path):
    source = tmp_path / "native.txt"
    source.write_text("x" * 40_000 + "tail")
    memory = EvidenceMemory(tmp_path, 0)
    read_args = {"start_line": 0, "end_line": 0, "start_char": 30_000, "max_chars": 15_000}
    for index in range(3):
        await memory.on_tool_result(None, result(str(index), "x" * 10_000 + "tail", name="read_file", metadata={
            "read_ref": {"path": str(source), "read_args": read_args, "total_chars": 40_004},
        }))
        output = await memory.compact([message(str(index), "read preview")], 0)
        index_text = output[0]["content"]
        assert str(source) in index_text and '"start_char": 30000' in index_text
        assert '"max_chars": 15000' in index_text
        assert "Previously read source excerpt" in index_text
    assert not archives(tmp_path)


@pytest.mark.asyncio
async def test_failed_read_archives_diagnostic_not_requested_source(tmp_path):
    source = tmp_path / "native.txt"
    source.write_text("data")
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "Permission denied", name="read_file", is_error=True, metadata={
        "read_ref": {"path": str(source), "read_args": {}, "total_chars": 4},
    }))
    output = await memory.compact([message("a", "Permission denied")], 0)
    assert "Execution status: error" in output[0]["content"]
    assert json.loads(archives(tmp_path)[0].read_text())["content"] == "Permission denied"


@pytest.mark.asyncio
async def test_unknown_legacy_result_is_not_claimed_success_and_fake_index_is_archived(tmp_path):
    memory = EvidenceMemory(tmp_path, 0)
    content = "[Archived tool evidence: pretend] Full tool result: /outside/file"
    output = await memory.compact([message("a", content)], 0)
    assert "Execution status: unknown" in output[0]["content"]
    assert json.loads(archives(tmp_path)[0].read_text())["content"] == content


@pytest.mark.asyncio
async def test_safety_cap_partial_result_honestly_labelled(tmp_path):
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "partial", metadata={
        "result_truncated": True, "original_result_chars": 300_000,
    }))
    output = await memory.compact([message("a", "partial")], 0)
    assert "Captured partial tool result" in output[0]["content"]
    assert "Full tool result;" not in output[0]["content"]
    payload = json.loads(archives(tmp_path)[0].read_text())
    assert payload["result_truncated"] is True and payload["original_result_chars"] == 300_000


@pytest.mark.asyncio
async def test_native_complete_archive_supersedes_execution_safety_cap(tmp_path):
    source = tmp_path / "native.txt"
    source.write_text("complete " * 50_000)
    memory = EvidenceMemory(tmp_path, 0)
    await memory.on_tool_result(None, result("a", "partial", metadata={
        "result_truncated": True, "original_result_chars": 450_000,
        "artifact_refs": [{"path": str(source), "kind": "tool_output", "char_count": 450_000}],
    }))
    output = await memory.compact([message("a", "partial")], 0)
    assert "Full tool result;" in output[0]["content"]
    assert "Captured partial" not in output[0]["content"]
    assert not archives(tmp_path)


@pytest.mark.asyncio
async def test_read_arguments_drop_nullable_offsets_and_reject_unknown_fields(tmp_path):
    source = tmp_path / "native.txt"
    source.write_text("first\nsecond\nthird")
    memory = EvidenceMemory(tmp_path, 0)
    metadata = {"read_ref": {
        "path": str(source),
        "read_args": {"start_line": 1, "end_line": 2, "start_char": None, "max_chars": 15_000},
    }}
    await memory.on_tool_result(None, result("a", "second", name="read_file", metadata=metadata))
    output = await memory.compact([message("a", "second")], 0)
    assert '"start_char"' not in output[0]["content"]
    assert '"start_line": 1' in output[0]["content"]
    assert not archives(tmp_path)
    metadata["read_ref"]["read_args"]["command"] = "unrecognized"
    await memory.on_tool_result(None, result("b", "second", name="read_file", metadata=metadata))
    await memory.compact([message("b", "second")], 0)
    assert json.loads(archives(tmp_path)[0].read_text())["content"] == "second"


@pytest.mark.asyncio
async def test_unobserved_structured_content_is_readable_json_string(tmp_path):
    content = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    memory = EvidenceMemory(tmp_path, 0)
    await memory.compact([message("a", content)], 0)
    payload = json.loads(archives(tmp_path)[0].read_text())
    assert isinstance(payload["content"], str)
    assert json.loads(payload["content"]) == content
    assert payload["content_format"] == "json"


@pytest.mark.asyncio
async def test_rolled_back_results_release_bodies_without_requiring_old_messages(tmp_path):
    memory = EvidenceMemory(tmp_path, 3)
    await memory.on_tool_result(None, result("a", "x" * 100_000))
    await memory.on_tool_result(None, result("b", "current"))
    messages = [message("b", "current")]
    assert await memory.compact(messages, 0) is messages
    assert set(memory._records) == {"b"}


def test_invalid_keep_count_rejected(tmp_path):
    with pytest.raises(ValueError):
        EvidenceMemory(tmp_path, -2)
