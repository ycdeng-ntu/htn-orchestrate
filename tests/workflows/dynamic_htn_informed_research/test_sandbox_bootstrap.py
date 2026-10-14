"""Research sandbox preparation and ownership, without E2B or package downloads."""
from __future__ import annotations

import asyncio
import json
import shlex
import threading
from types import SimpleNamespace

import pytest
from e2b.sandbox.commands.command_handle import CommandExitException

from workflows.dynamic_htn_informed_research.tools import (
    ResearchToolConfig,
    build_research_tools,
)
from workflows.dynamic_htn_informed_research.tools.sandbox import WorkerSandbox, e2b_factory
from workflows.dynamic_htn_informed_research.tools.sandbox_bootstrap import (
    PROBE_PREFIX,
    RESEARCH_PACKAGES,
    SandboxPreparationError,
)


def command_result(*, stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def probe_result(missing=()):
    report = {"missing": list(missing), "python": "3.12.13", "executable": "/usr/bin/python3"}
    return command_result(stdout="package import diagnostic\n" + PROBE_PREFIX + json.dumps(report) + "\n")


class SandboxDouble:
    def __init__(self, responses, *, install_gate=None, kill_error=None):
        self.responses = list(responses)
        self.calls = []
        self.writes = []
        self.kills = 0
        self.install_gate = install_gate
        self.kill_error = kill_error
        self.commands = SimpleNamespace(run=self.run)
        self.files = SimpleNamespace(write=self.write)

    def run(self, command, timeout):
        self.calls.append((command, timeout))
        if " -m pip install " in command and self.install_gate:
            entered, release = self.install_gate
            entered.set()
            assert release.wait(5), "test never released sandbox package installation"
        assert self.responses, f"unexpected sandbox command: {command}"
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def write(self, path, content):
        self.writes.append((path, content))

    def kill(self):
        self.kills += 1
        if self.kill_error:
            raise self.kill_error


def patch_create(monkeypatch, sandbox, *, create_gate=None, create_error=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if create_gate:
            entered, release = create_gate
            entered.set()
            assert release.wait(5), "test never released sandbox creation"
        if create_error:
            raise create_error
        return sandbox

    monkeypatch.setattr("e2b_code_interpreter.Sandbox.create", create)
    return calls


def make_sandbox():
    return e2b_factory(api_key="fake-key", template="base", timeout=1800)


def test_prepares_missing_dependencies_before_returning_sandbox(monkeypatch):
    missing = ["requests", "bs4", "PIL"]
    sandbox = SandboxDouble([probe_result(missing), command_result(), probe_result()])
    creates = patch_create(monkeypatch, sandbox)

    assert make_sandbox() is sandbox

    assert creates == [{"api_key": "fake-key", "template": "base", "timeout": 1800,
                        "request_timeout": 30}]
    assert len(sandbox.calls) == 3
    first_probe, installation, final_probe = sandbox.calls
    assert first_probe == final_probe
    probe_args = shlex.split(first_probe[0])
    assert probe_args[:6] == ["timeout", "-s", "KILL", "60s", "python3", "-c"]
    assert first_probe[1] == 65
    install_args = shlex.split(installation[0])
    assert install_args[:7] == ["timeout", "-s", "KILL", "180s", "python3", "-m", "pip"]
    assert "--no-input" in install_args and "--disable-pip-version-check" in install_args
    assert install_args[install_args.index("--retries") + 1] == "1"
    assert install_args[install_args.index("--timeout") + 1] == "30"
    assert installation[1] == 185
    assert {"requests", "beautifulsoup4", "Pillow"}.issubset(install_args)
    assert "numpy" not in install_args  # Preinstalled dependencies do not get reinstalled.
    assert sandbox.kills == 0 and not sandbox.writes and not sandbox.responses


def test_preinstalled_template_skips_installation(monkeypatch):
    sandbox = SandboxDouble([probe_result()])
    patch_create(monkeypatch, sandbox)

    assert make_sandbox() is sandbox

    assert len(sandbox.calls) == 1
    assert " -m pip install " not in sandbox.calls[0][0]
    assert sandbox.kills == 0


@pytest.mark.parametrize("phase", ["probe", "install", "verify"])
@pytest.mark.parametrize("failure_kind", ["returned_exit", "sdk_exit", "transport"])
def test_command_failure_never_returns_unprepared_sandbox(monkeypatch, phase, failure_kind):
    if failure_kind == "sdk_exit":
        failure = CommandExitException(stderr="command diagnostic", stdout="", exit_code=1, error=None)
    elif failure_kind == "transport":
        failure = TimeoutError("SDK command deadline")
    else:
        failure = command_result(stderr="command diagnostic", exit_code=1)
    preceding = {"probe": [], "install": [probe_result(["requests"])],
                 "verify": [probe_result(["requests"]), command_result()]}[phase]
    sandbox = SandboxDouble([*preceding, failure])
    patch_create(monkeypatch, sandbox)

    with pytest.raises(SandboxPreparationError, match=phase) as caught:
        make_sandbox()

    if failure_kind == "transport":
        assert "TimeoutError" in str(caught.value)
    else:
        assert "exit code 1" in str(caught.value)
    assert len(sandbox.calls) == len(preceding) + 1
    assert sandbox.kills == 1 and not sandbox.writes


@pytest.mark.parametrize("stdout", ["", "pip succeeded", PROBE_PREFIX + "not json",
                                     PROBE_PREFIX + '{"missing": "requests"}'])
def test_invalid_probe_is_failure_instead_of_implicit_readiness(monkeypatch, stdout):
    sandbox = SandboxDouble([command_result(stdout=stdout)])
    patch_create(monkeypatch, sandbox)

    with pytest.raises(SandboxPreparationError):
        make_sandbox()

    assert len(sandbox.calls) == 1 and sandbox.kills == 1


def test_successful_pip_still_requires_successful_imports(monkeypatch):
    sandbox = SandboxDouble([probe_result(["requests"]), command_result(), probe_result(["requests"])])
    patch_create(monkeypatch, sandbox)

    with pytest.raises(SandboxPreparationError, match="requests"):
        make_sandbox()

    assert len(sandbox.calls) == 3 and sandbox.kills == 1


def test_create_failure_does_not_attempt_to_kill_an_unacquired_sandbox(monkeypatch):
    sandbox = SandboxDouble([])
    error = RuntimeError("creation failed")
    patch_create(monkeypatch, sandbox, create_error=error)

    with pytest.raises(RuntimeError) as caught:
        make_sandbox()

    assert caught.value is error
    assert not sandbox.calls and sandbox.kills == 0


def test_cleanup_failure_does_not_replace_preparation_error(monkeypatch):
    sandbox = SandboxDouble([command_result(exit_code=1)], kill_error=RuntimeError("kill failed"))
    patch_create(monkeypatch, sandbox)

    with pytest.raises(SandboxPreparationError):
        make_sandbox()

    assert sandbox.kills == 1


@pytest.mark.asyncio
async def test_concurrent_worker_get_prepares_once_and_reuses_sandbox(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    sandbox = SandboxDouble([probe_result(["requests"]), command_result(), probe_result()],
                            install_gate=(entered, release))
    creates = patch_create(monkeypatch, sandbox)
    worker = WorkerSandbox(make_sandbox)
    callers = [asyncio.create_task(worker.get()) for _ in range(3)]
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not any(caller.done() for caller in callers)
        release.set()
        assert await asyncio.gather(*callers) == [sandbox] * 3
        assert await worker.get() is sandbox
        assert len(creates) == 1 and len(sandbox.calls) == 3
    finally:
        release.set()
        await asyncio.gather(*callers, return_exceptions=True)
        await worker.aclose()
        await worker.aclose()
    assert sandbox.kills == 1


@pytest.mark.asyncio
async def test_failed_worker_preparation_is_reused_without_duplicate_kill(monkeypatch):
    sandbox = SandboxDouble([command_result(exit_code=1)])
    creates = patch_create(monkeypatch, sandbox)
    worker = WorkerSandbox(make_sandbox)
    try:
        for _ in range(2):
            with pytest.raises(SandboxPreparationError):
                await worker.get()
    finally:
        await worker.aclose()
        await worker.aclose()
    assert len(creates) == 1 and sandbox.kills == 1


@pytest.mark.asyncio
async def test_python_tool_waits_for_preparation_and_reuses_ready_environment(monkeypatch, tmp_path):
    sandbox = SandboxDouble([probe_result(["requests"]), command_result(), probe_result(),
                             command_result(stdout="first result"), command_result(stdout="second result")])
    creates = patch_create(monkeypatch, sandbox)
    config = ResearchToolConfig(e2b_api_key="fake-key", e2b_template="base", summary_use_environment=False)
    async with build_research_tools("dynir_researcher", task_id="prepared-python", artifact_dir=tmp_path,
                                     config=config) as runtime:
        assert await runtime.run_python_code("import requests\nprint('first result')") == "first result"
        assert await runtime.run_python_code("import requests\nprint('second result')") == "second result"
    assert len(creates) == 1 and sandbox.kills == 1
    assert len(sandbox.calls) == 5 and len(sandbox.writes) == 2
    assert all("/tmp/exec_" not in command for command, _ in sandbox.calls[:3])
    assert all("/tmp/exec_" in command for command, _ in sandbox.calls[3:])


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["create", "install"])
@pytest.mark.parametrize("installation_fails", [False, True])
async def test_cancellation_during_preparation_releases_without_running_code(
    monkeypatch, tmp_path, blocked_phase, installation_fails,
):
    entered, release = threading.Event(), threading.Event()
    installation = command_result(exit_code=1 if installation_fails else 0)
    responses = [probe_result(["requests"]), installation]
    if not installation_fails:
        responses.append(probe_result())
    sandbox = SandboxDouble(responses, install_gate=(entered, release) if blocked_phase == "install" else None)
    patch_create(monkeypatch, sandbox, create_gate=(entered, release) if blocked_phase == "create" else None)
    config = ResearchToolConfig(e2b_api_key="fake-key", e2b_template="base", summary_use_environment=False)
    runtime = build_research_tools("dynir_researcher", task_id="prepare-cancel", artifact_dir=tmp_path,
                                   config=config)
    research = asyncio.create_task(runtime.run_python_code("import requests\nprint('research started')"))
    cleanup = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        research.cancel()
        with pytest.raises(asyncio.CancelledError):
            await research
        cleanup = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        await asyncio.wait_for(cleanup, timeout=3)
    finally:
        release.set()
        await asyncio.gather(research, return_exceptions=True)
        if cleanup is not None:
            await cleanup
        await runtime.aclose()
    assert sandbox.kills == 1 and not sandbox.writes
    assert all("/tmp/exec_" not in command for command, _ in sandbox.calls)


def test_research_dependency_contract_covers_original_stack_and_graphs():
    assert set(RESEARCH_PACKAGES) == {
        "matplotlib", "numpy", "pandas", "seaborn", "scipy", "Pillow", "markdown", "sympy",
        "plotly", "tabulate", "mplfonts", "beautifulsoup4", "requests", "openpyxl", "pypdf",
        "PyMuPDF", "pdfplumber", "networkx",
    }
    assert RESEARCH_PACKAGES["Pillow"] == "PIL"
    assert RESEARCH_PACKAGES["beautifulsoup4"] == "bs4"
    assert RESEARCH_PACKAGES["PyMuPDF"] == "fitz"
