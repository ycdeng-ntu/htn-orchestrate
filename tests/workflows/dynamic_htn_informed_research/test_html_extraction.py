"""Offline extraction contracts, including real disposable child processes."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading

import httpx
import pytest

from workflows.dynamic_htn_informed_research.tools import html_extract
from workflows.dynamic_htn_informed_research.tools import runtime as runtime_module
from workflows.dynamic_htn_informed_research.tools.runtime import (
    ResearchToolConfig, ResearchToolError, build_research_tools,
)


ARTICLE = """<!DOCTYPE html><html><body><article><h1>Research result</h1><p>
Researchers measured the samples at regular intervals and documented their
observations. The final report includes detailed experimental methods and
independently verified measurements. Further experiments confirmed the findings.
</p><p>The original study and supplementary data provide sufficient information
for other researchers to reproduce the experiment and compare results.</p>
</article></body></html>"""


@pytest.mark.asyncio
async def test_real_html_extraction():
    extracted = await html_extract.extract_html(ARTICLE, timeout=10)
    assert "Researchers measured the samples" in extracted
    assert "<article>" not in extracted


@pytest.mark.parametrize(("content_type", "body", "accepted"), [
    ("text/html; charset=utf-8", b"<p>Example</p>", True),
    ("application/xhtml+xml", b"<html>Example</html>", True),
    ("", b"<!DOCTYPE html><html>Example</html>", True),
    ("text/plain", b"<article>Example</article>", True),
    ("application/octet-stream", b"<html>Example</html>", True),
    ("application/pdf", b"<html>Example</html>", False),
    ("text/html", b"%PDF-1.7 binary payload", False),
    ("", b"\xef\xbb\xbf  %PDF-1.7 binary payload", False),
    ("image/png", b"<html>Example</html>", False),
    ("text/html", b"PK\x03\x04zip archive", False),
    ("text/html", b"\x89PNG\x00binary", False),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"document", False),
    ("application/json", b'{"result":"plain JSON response"}', False),
    ("text/plain", b"plain text without HTML", False),
])
def test_content_type_and_binary_filter(content_type, body, accepted):
    assert html_extract.is_html_response(body, content_type) is accepted


def replace_worker(monkeypatch, script):
    started = asyncio.Event()
    processes = []

    async def start(max_output_bytes):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", script,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=html_extract._worker_environment(),
        )
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(html_extract, "_start_worker", start)
    return started, processes


def assert_reaped(process):
    assert process.returncode is not None
    if sys.platform != "win32":
        # The asyncio child watcher must already have consumed the exit status.
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.asyncio
async def test_timeout_kills_and_reaps_real_process(monkeypatch):
    _, processes = replace_worker(monkeypatch, "import time; time.sleep(60)")
    with pytest.raises(html_extract.HtmlExtractionError) as error:
        await html_extract.extract_html(ARTICLE, timeout=0.3)
    assert error.value.reason == "timeout"
    assert len(processes) == 1
    assert_reaped(processes[0])


@pytest.mark.asyncio
async def test_cancellation_kills_and_reaps_real_process(monkeypatch):
    started, processes = replace_worker(monkeypatch, "import time; time.sleep(60)")
    task = asyncio.create_task(html_extract.extract_html(ARTICLE))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert_reaped(processes[0])


@pytest.mark.asyncio
async def test_cancellation_during_spawn_retains_ownership(monkeypatch):
    created = asyncio.Event()
    finish_spawn = asyncio.Event()
    processes = []

    async def delayed_spawn(limit):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", "import time; time.sleep(60)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        created.set()
        await finish_spawn.wait()
        return process

    monkeypatch.setattr(html_extract, "_start_worker", delayed_spawn)
    task = asyncio.create_task(html_extract.extract_html(ARTICLE))
    await asyncio.wait_for(created.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()  # Repeated shutdown cancellation must not abandon cleanup.
    finish_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert_reaped(processes[0])


@pytest.mark.asyncio
async def test_stdout_limit_terminates_noisy_process(monkeypatch):
    _, processes = replace_worker(monkeypatch,
        "import os,time; os.write(1,b'x'*200000); time.sleep(60)")
    with pytest.raises(html_extract.HtmlExtractionError) as error:
        await asyncio.wait_for(html_extract.extract_html(ARTICLE, max_output_bytes=100), 2)
    assert error.value.reason == "output_limit"
    assert_reaped(processes[0])


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal exit status")
@pytest.mark.parametrize("signum", [signal.SIGABRT, signal.SIGSEGV])
async def test_native_exit_is_confined_to_real_child(monkeypatch, signum):
    # Only the disposable child signals itself; the test runner never receives
    # a signal and core files are disabled before the child exits.
    _, processes = replace_worker(monkeypatch,
        "import os,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
        "os.write(2,b'child diagnostic'); "
        f"os.kill(os.getpid(), {int(signum)})")
    with pytest.raises(html_extract.HtmlExtractionError) as error:
        await html_extract.extract_html(ARTICLE)
    assert error.value.reason == "worker_exit"
    assert error.value.returncode == -signum
    assert error.value.diagnostic == "child diagnostic"
    assert_reaped(processes[0])


@pytest.mark.asyncio
async def test_process_environment_excludes_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-sensitive-token")
    monkeypatch.setenv("HTTPS_PROXY", "https://credential:secret@proxy.invalid")
    monkeypatch.setenv("PYTHONPATH", "/not/an/import/path")
    replace_worker(monkeypatch, "import json,os; print(json.dumps(dict(os.environ)))")
    environment = json.loads(await html_extract.extract_html(ARTICLE))
    assert "OPENAI_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "PYTHONPATH" not in environment


@pytest.mark.asyncio
async def test_shared_process_limit_and_event_loop_remains_responsive(monkeypatch):
    monkeypatch.setattr(html_extract, "_SLOTS", threading.BoundedSemaphore(2))
    processes = []
    max_alive = 0

    async def start(limit):
        nonlocal max_alive
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", "import time; time.sleep(0.2); print('complete')",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        max_alive = max(max_alive, sum(p.returncode is None for p in processes))
        return process

    monkeypatch.setattr(html_extract, "_start_worker", start)
    tasks = [asyncio.create_task(html_extract.extract_html(ARTICLE)) for _ in range(5)]
    await asyncio.sleep(0.05)
    assert len(processes) == 2
    assert await asyncio.wait_for(asyncio.gather(*tasks), 3) == ["complete\n"] * 5
    assert max_alive == 2
    for process in processes:
        assert_reaped(process)


def tool_config(**kwargs):
    return ResearchToolConfig(summary_use_environment=False, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [-6, -11])
async def test_direct_native_failure_records_error_and_continues_fallback(tmp_path, monkeypatch, exit_code):
    async def extract(content):
        raise html_extract.HtmlExtractionError("worker_exit", exit_code, diagnostic="isolated child stack")

    monkeypatch.setattr(runtime_module, "extract_html", extract)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"}),
    )) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=tool_config(), http_client=http) as runtime:
            async def jina(url):
                return "Recovered content. " * 100 if url.endswith("/alternative") else ""

            async def alternative(url):
                return "https://source.example/alternative"

            monkeypatch.setattr(runtime, "_jina", jina)
            monkeypatch.setattr(runtime.academic, "oa_url", alternative)
            content, actual_url = await runtime._fetch_content("https://source.example/article")
            assert content.startswith("Recovered content.")
            assert actual_url.endswith("/alternative")
            failure, = runtime.backend_failures
            assert failure["reason"] == "extraction_error"
            assert failure["error"] == f"html_worker_exit_{exit_code}"
            diagnostic, = runtime.provenance
            assert diagnostic["tool_name"] == "html_extraction_error"
            assert diagnostic["worker_exit_code"] == exit_code
            assert json.loads(next(runtime.artifact_dir.glob("*.json")).read_text())["content"] == "isolated child stack"


@pytest.mark.asyncio
async def test_direct_binary_response_skips_parser(tmp_path, monkeypatch):
    async def forbidden(content):
        pytest.fail("binary response was passed to HTML parser")

    monkeypatch.setattr(runtime_module, "extract_html", forbidden)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}),
    )) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=tool_config(), http_client=http) as runtime:
            assert await runtime._direct("https://source.example/file") == ""
            assert runtime.backend_failures[0]["reason"] == "unsupported_content_type"


@pytest.mark.asyncio
async def test_direct_keeps_redirect_source_policy(tmp_path, monkeypatch):
    async def forbidden(content):
        pytest.fail("blocked response was passed to HTML parser")

    monkeypatch.setattr(runtime_module, "extract_html", forbidden)

    def response(request):
        if request.url.host == "allowed.example":
            return httpx.Response(302, headers={"location": "https://blocked.example/article"})
        return httpx.Response(200, text=ARTICLE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                config=tool_config(blocked_needles=("blocked.example",)), http_client=http) as runtime:
            with pytest.raises(ResearchToolError, match="source policy"):
                await runtime._direct("https://allowed.example/article")
