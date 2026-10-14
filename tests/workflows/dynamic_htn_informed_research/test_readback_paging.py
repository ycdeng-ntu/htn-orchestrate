from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows.dynamic_htn_informed_research.tools.readback import paginate_text


def _page(text: str) -> tuple[dict[str, str], str]:
    header, body = text.split("\n", 1)
    assert header.startswith("[read_file chars ")
    assert header.endswith("]")
    fields = dict(field.split("=", 1) for field in header[17:-1].split())
    return fields, body


def test_default_long_single_line_can_recover_its_tail() -> None:
    content = "a" * 30_000 + "FINAL SCIENTIFIC CONCLUSION"
    first, first_body = _page(paginate_text(content))
    second, second_body = _page(paginate_text(content, start_char=int(first["next_start_char"])))
    final, final_body = _page(paginate_text(content, start_char=int(second["next_start_char"])))

    assert first["total"] == str(len(content))
    assert first["start"] == "0"
    assert first["end"] == "15000"
    assert final["next_start_char"] == "null"
    assert final_body == "FINAL SCIENTIFIC CONCLUSION"
    assert first_body + second_body + final_body == content


def test_unicode_pages_round_trip_without_byte_offset_gaps() -> None:
    content = ("浓度🧪e\u0301\r\nμ=1.23\u2028" * 9) + "终"
    parts = []
    position = 0
    while True:
        metadata, body = _page(paginate_text(content, start_char=position, max_chars=7))
        assert int(metadata["start"]) == position
        assert int(metadata["end"]) - position == len(body)
        assert len(body) <= 7
        parts.append(body)
        if metadata["next_start_char"] == "null":
            break
        position = int(metadata["next_start_char"])

    assert "".join(parts) == content


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "first\r\nsecond\nthird"),
        ({"start_line": 2, "end_line": 2}, "[lines 2-2 of 3]\nsecond\n"),
        ({"start_line": 2}, "[lines 2-3 of 3]\nsecond\nthird"),
        ({"end_line": 2}, "[lines 1-2 of 3]\nfirst\r\nsecond\n"),
        ({"start_line": 1, "end_line": 3}, "first\r\nsecond\nthird"),
        ({"end_line": 100}, "first\r\nsecond\nthird"),
    ],
)
def test_short_text_and_line_ranges_keep_existing_output(kwargs: dict, expected: str) -> None:
    assert paginate_text("first\r\nsecond\nthird", **kwargs) == expected


def test_long_line_range_advertises_absolute_offsets_and_its_stop_boundary() -> None:
    prefix, selected, suffix = "prefix\r\n", "🧪" * 30_001 + "\n", "unrequested suffix"
    content = prefix + selected + suffix
    first, body = _page(paginate_text(content, start_line=2, end_line=2))
    assert first["start"] == str(len(prefix))
    assert first["end"] == str(len(prefix) + 15_000)
    assert first["total"] == str(len(content))
    assert first["line_range_end_char"] == str(len(prefix) + len(selected))
    assert first["lines"] == "2-2/3"

    parts = [body]
    position = int(first["next_start_char"])
    stop = int(first["line_range_end_char"])
    while position < stop:
        metadata, body = _page(
            paginate_text(content, start_char=position, max_chars=min(15_000, stop - position))
        )
        parts.append(body)
        position = int(metadata["end"])

    assert "".join(parts) == selected
    assert suffix not in "".join(parts)


def test_max_chars_boundary_and_final_empty_character_page() -> None:
    assert paginate_text("x" * 15_000) == "x" * 15_000
    metadata, body = _page(paginate_text("abc", start_char=3))
    assert metadata == {"total": "3", "start": "3", "end": "3", "next_start_char": "null"}
    assert body == ""
    assert paginate_text("") == ""
    assert _page(paginate_text("", start_char=0))[1] == ""


def test_line_header_does_not_consume_selected_content_budget() -> None:
    metadata, body = _page(paginate_text("first\nlast", start_line=2, max_chars=4))
    assert body == "last"
    assert metadata["next_start_char"] == "null"
    assert metadata["line_range_end_char"] == "10"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_char": -1},
        {"start_char": 4},
        {"start_char": True},
        {"start_char": 0.0},
        {"start_char": "0"},
        {"start_char": 0, "start_line": 1},
        {"start_char": 0, "end_line": 1},
        {"start_line": -1},
        {"start_line": True},
        {"start_line": 0.0},
        {"end_line": -1},
        {"end_line": False},
        {"end_line": "1"},
        {"start_line": 2, "end_line": 1},
        {"max_chars": 0},
        {"max_chars": -1},
        {"max_chars": 15_001},
        {"max_chars": True},
        {"max_chars": 1.5},
        {"max_chars": None},
    ],
)
def test_invalid_paging_arguments_raise_actionable_errors(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="start_char|start_line|end_line|max_chars"):
        paginate_text("abc", **kwargs)


def test_read_file_schema_advertises_optional_character_pagination() -> None:
    root = Path(__file__).resolve().parents[3]
    schemas = json.loads(
        (root / "workflows/dynamic_htn_informed_research/tools/schemas.json").read_text()
    )
    parameters = schemas["read_file"]["function"]["parameters"]
    assert parameters["required"] == ["path"]
    assert parameters["properties"]["start_char"]["minimum"] == 0
    assert parameters["properties"]["max_chars"]["maximum"] == 15_000
    assert parameters["properties"]["max_chars"]["minimum"] == 1
