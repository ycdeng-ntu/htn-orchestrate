"""Lossless, character-addressable views of archived research output."""

from __future__ import annotations


MAX_READ_CHARS = 15_000


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _character_page(
    content: str,
    *,
    start: int,
    stop: int,
    max_chars: int,
    line_range: str = "",
) -> str:
    end = min(start + max_chars, stop)
    next_start = str(end) if end < stop else "null"
    metadata = (
        f"[read_file chars total={len(content)} start={start} end={end} "
        f"next_start_char={next_start}"
    )
    if line_range:
        metadata += f" {line_range} line_range_end_char={stop}"
    return metadata + "]\n" + content[start:end]


def paginate_text(
    content: str,
    *,
    start_line: int = 0,
    end_line: int = 0,
    start_char: int | None = None,
    max_chars: int = MAX_READ_CHARS,
) -> str:
    """Read a bounded text page without losing the continuation position.

    Line ranges retain their existing one-based, inclusive semantics. Character
    offsets refer to the decoded, complete content, use Python Unicode character
    indices, and have an exclusive end. ``max_chars`` limits the content payload;
    the small metadata header is additional. Explicit character reads always
    include that header, including the last/empty page.

    A truncated line range advertises its absolute ``line_range_end_char``.
    Continue with ``start_char`` only, and keep ``max_chars`` at or below the
    remaining distance to that boundary when limiting reads to that line range.
    ``next_start_char=null`` means the requested range has been fully returned.
    """
    _nonnegative_integer("start_line", start_line)
    _nonnegative_integer("end_line", end_line)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError(f"max_chars must be an integer between 1 and {MAX_READ_CHARS}")
    if not 1 <= max_chars <= MAX_READ_CHARS:
        raise ValueError(f"max_chars must be an integer between 1 and {MAX_READ_CHARS}")
    if start_line > 0 and end_line > 0 and end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")

    if start_char is not None:
        _nonnegative_integer("start_char", start_char)
        if start_line != 0 or end_line != 0:
            raise ValueError("start_char cannot be combined with start_line or end_line")
        if start_char > len(content):
            raise ValueError("start_char must not exceed the total number of characters")
        return _character_page(
            content, start=start_char, stop=len(content), max_chars=max_chars
        )

    start, stop = 0, len(content)
    line_label = ""
    line_range = ""
    if start_line > 0 or end_line > 0:
        lines = content.splitlines(keepends=True)
        first_line = start_line - 1 if start_line > 0 else 0
        last_line = end_line if end_line > 0 else len(lines)
        start = sum(map(len, lines[:first_line]))
        selected = "".join(lines[first_line:last_line])
        stop = start + len(selected)
        line_range = f"lines={first_line + 1}-{min(last_line, len(lines))}/{len(lines)}"
        if first_line > 0 or last_line < len(lines):
            line_label = f"[lines {first_line + 1}-{min(last_line, len(lines))} of {len(lines)}]\n"

    rendered = line_label + content[start:stop]
    if len(rendered) <= max_chars:
        return rendered
    return _character_page(
        content, start=start, stop=stop, max_chars=max_chars, line_range=line_range
    )
