"""Bounded, cancellable isolation for native HTML extraction.

The child only receives downloaded text over stdin. It imports trafilatura after
process isolation, so a native abort cannot terminate a research worker.
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import re
import sys
import threading

MAX_INPUT_CHARS = 3_000_000
MAX_OUTPUT_BYTES = 3_000_000
MAX_CONCURRENT_EXTRACTIONS = 4
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_EXTRACTIONS)
_HTML_MARKUP = re.compile(
    rb"<(?:!doctype\s+html|html|head|body|title|main|article|section|div|p|h[1-6]|table)(?:\s|>)",
    re.IGNORECASE,
)
_BINARY_TYPES = {
    "application/pdf", "application/x-pdf", "application/zip", "application/gzip",
    "application/x-gzip", "application/x-7z-compressed", "application/x-rar-compressed",
    "application/wasm", "application/msword", "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
}
_BINARY_MAGIC = (
    b"%PDF-", b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"PK\x03\x04",
    b"\x1f\x8b", b"\x7fELF", b"\xd0\xcf\x11\xe0", b"RIFF", b"7z\xbc\xaf\x27\x1c",
)


class HtmlExtractionError(ValueError):
    def __init__(self, reason: str, returncode: int | None = None, *, diagnostic: str = ""):
        self.reason = reason
        self.returncode = returncode
        self.diagnostic = diagnostic
        super().__init__(f"html_{reason}" + (f"_{returncode}" if returncode is not None else ""))


def is_html_response(content: bytes, content_type: str) -> bool:
    """Accept declared HTML or recognizable markup, excluding known binaries.

Non-HTML responses are skipped because this fallback extracts HTML; this is
not a claim that JSON or another particular input caused a native crash.
"""
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type in _BINARY_TYPES or media_type.startswith((
        "image/", "audio/", "video/", "font/", "application/vnd.openxmlformats-",
    )):
        return False
    sample = content[:8192].lstrip().removeprefix(b"\xef\xbb\xbf").lstrip()
    if sample.startswith(_BINARY_MAGIC) or b"\x00" in sample:
        return False
    return media_type in {"text/html", "application/xhtml+xml"} or bool(_HTML_MARKUP.search(sample))


def _worker_environment() -> dict[str, str]:
    # Explicit allowlist: no tokens, proxy credentials, PYTHONPATH or user home.
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


async def _start_worker(max_output_bytes: int) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-X", "faulthandler", str(Path(__file__).resolve()),
        "--worker", str(max_output_bytes),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=_worker_environment(),
        limit=64 * 1024,
    )


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
    data = bytearray()
    while chunk := await stream.read(min(64 * 1024, limit + 1 - len(data))):
        data.extend(chunk)
        if len(data) > limit:
            raise HtmlExtractionError("output_limit")
    return bytes(data)


async def _exchange(process: asyncio.subprocess.Process, content: bytes, max_output_bytes: int):
    async def send() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(content)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # A crashed child is classified using its exit status.
        finally:
            process.stdin.close()

    assert process.stdout is not None and process.stderr is not None
    tasks = [asyncio.create_task(coro) for coro in (
        send(), _read_limited(process.stdout, max_output_bytes),
        _read_limited(process.stderr, 64 * 1024), process.wait(),
    )]
    try:
        _, output, diagnostic, returncode = await asyncio.gather(*tasks)
        return output, diagnostic, returncode
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _reap(spawn: asyncio.Task, exchange: asyncio.Task | None) -> None:
    try:
        process = await spawn
    except Exception:
        return
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    if exchange is not None:
        exchange.cancel()
        await asyncio.gather(exchange, return_exceptions=True)
    # Drain any buffered pipes as well as waiting: Process.wait alone can wait
    # forever for a paused pipe transport after a noisy child has been killed.
    await process.communicate()


async def _finish_cleanup(task: asyncio.Task) -> None:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def extract_html(content: str, *, timeout: float = 30.0,
                       max_output_bytes: int = MAX_OUTPUT_BYTES) -> str:
    if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes <= 0:
        raise ValueError("HTML extraction limits must be positive and finite")
    acquired = False
    spawn = exchange = None
    try:
        async with asyncio.timeout(timeout):
            # A process-wide semaphore also works across independently created
            # event loops. Acquiring it never blocks an event-loop thread.
            while not _SLOTS.acquire(blocking=False):
                await asyncio.sleep(0.025)
            acquired = True
            spawn = asyncio.create_task(_start_worker(max_output_bytes))
            # Keep ownership if cancellation arrives while subprocess creation
            # is in progress, then finish creation and kill/reap in finally.
            process = await asyncio.shield(spawn)
            exchange = asyncio.create_task(_exchange(
                process, content[:MAX_INPUT_CHARS].encode("utf-8"), max_output_bytes,
            ))
            output, diagnostic, returncode = await asyncio.shield(exchange)
            if returncode:
                raise HtmlExtractionError(
                    "output_limit" if returncode == 3 else "worker_exit", returncode,
                    diagnostic=diagnostic.decode("utf-8", errors="replace"),
                )
            return output.decode("utf-8")
    except TimeoutError as exc:
        raise HtmlExtractionError("timeout") from exc
    except OSError as exc:
        raise HtmlExtractionError("worker_start") from exc
    finally:
        try:
            if spawn is not None:
                await _finish_cleanup(asyncio.create_task(_reap(spawn, exchange)))
        finally:
            if acquired:
                _SLOTS.release()


def _worker() -> None:
    # Avoid large core files for malformed-page crashes; faulthandler stderr is
    # bounded by the parent. This child has no reason to spawn further children.
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import trafilatura

    content = sys.stdin.buffer.read(MAX_INPUT_CHARS * 4 + 1)
    if len(content) > MAX_INPUT_CHARS * 4:
        raise SystemExit(2)
    output = (trafilatura.extract(content.decode("utf-8"), include_links=True) or "").encode("utf-8")
    if len(output) > int(sys.argv[2]):
        raise SystemExit(3)
    sys.stdout.buffer.write(output)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _worker()
