"""Prepare the Python used by research tools before handing out an E2B sandbox."""
from __future__ import annotations

import json
import shlex
from typing import Any


# Miro's research packages, plus networkx promised by the frozen tool schema.
# Distribution names differ from import names for Pillow, Beautiful Soup and
# PyMuPDF. Installation and verification must use the same python3 as execution.
RESEARCH_PACKAGES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "Pillow": "PIL",
    "markdown": "markdown",
    "sympy": "sympy",
    "plotly": "plotly",
    "tabulate": "tabulate",
    "mplfonts": "mplfonts",
    "beautifulsoup4": "bs4",
    "requests": "requests",
    "openpyxl": "openpyxl",
    "pypdf": "pypdf",
    "PyMuPDF": "fitz",
    "pdfplumber": "pdfplumber",
    "networkx": "networkx",
}
PROBE_PREFIX = "AGENTHARNESS_SANDBOX_PROBE:"
PROBE_TIMEOUT_S = 60
INSTALL_TIMEOUT_S = 180


class SandboxPreparationError(RuntimeError):
    """The sandbox's research dependencies could not be made usable."""


def _run(sandbox: Any, command: str, *, timeout: int, phase: str) -> str:
    # The SDK timeout only bounds its wait. Kill the remote process on timeout
    # too; the owning factory releases the entire sandbox on any failure.
    try:
        result = sandbox.commands.run(
            f"timeout -s KILL {timeout}s {command}", timeout=timeout + 5,
        )
    except Exception as exc:
        code = getattr(exc, "exit_code", None)
        detail = f"exit code {code}" if isinstance(code, int) else type(exc).__name__
        # Provider/pip output can include endpoint credentials. Keep the error
        # actionable without surfacing raw transport responses or install logs.
        raise SandboxPreparationError(f"Sandbox dependency {phase} failed ({detail})") from None
    if getattr(result, "exit_code", None) != 0:
        raise SandboxPreparationError(
            f"Sandbox dependency {phase} failed (exit code {getattr(result, 'exit_code', None)})"
        )
    return str(getattr(result, "stdout", "") or "")


def _probe(sandbox: Any, *, phase: str) -> dict:
    # find_spec is insufficient: installed modules can fail to import because
    # of missing shared libraries or incompatible binary dependencies.
    script = (
        "import importlib, importlib.metadata, json, sys\n"
        f"packages = {RESEARCH_PACKAGES!r}\n"
        "missing, versions = [], {}\n"
        "for package, module in packages.items():\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except Exception:\n"
        "        missing.append(module)\n"
        "    else:\n"
        "        try:\n"
        "            versions[package] = importlib.metadata.version(package)\n"
        "        except importlib.metadata.PackageNotFoundError:\n"
        "            versions[package] = 'unknown'\n"
        f"print({PROBE_PREFIX!r} + json.dumps({{'missing': missing, 'versions': versions, "
        "'python': sys.version.split()[0], 'executable': sys.executable}))\n"
    )
    output = _run(
        sandbox, f"python3 -c {shlex.quote(script)}", timeout=PROBE_TIMEOUT_S, phase=phase,
    )
    try:
        line = next(line for line in reversed(output.splitlines()) if line.startswith(PROBE_PREFIX))
        report = json.loads(line[len(PROBE_PREFIX):])
        missing = report["missing"]
        if not isinstance(missing, list) or any(
            not isinstance(module, str) or module not in RESEARCH_PACKAGES.values()
            for module in missing
        ):
            raise ValueError("Invalid missing module list")
    except (StopIteration, ValueError, TypeError, KeyError):
        raise SandboxPreparationError(
            f"Sandbox dependency {phase} returned an invalid import report"
        ) from None
    return report


def prepare_research_sandbox(sandbox: Any) -> dict:
    """Verify, install missing packages once, and verify again before execution.

    Templates that already satisfy the import contract do not invoke pip.
    Supplied test sandbox factories own their own preparation; this helper is
    called only by the production E2B factory.
    """
    report = _probe(sandbox, phase="probe")
    if not report["missing"]:
        return report
    missing = set(report["missing"])
    packages = [package for package, module in RESEARCH_PACKAGES.items() if module in missing]
    _run(
        sandbox,
        "python3 -m pip install --disable-pip-version-check --no-input "
        "--retries 1 --timeout 30 --quiet " + shlex.join(packages),
        timeout=INSTALL_TIMEOUT_S, phase="install",
    )
    report = _probe(sandbox, phase="verify")
    if report["missing"]:
        raise SandboxPreparationError(
            "Sandbox dependency verify failed; Python modules still unavailable: "
            + ", ".join(report["missing"])
        )
    return report
