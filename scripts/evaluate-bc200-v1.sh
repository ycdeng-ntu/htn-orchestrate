#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use a Python environment with the project's dependencies installed.
# Extra runner arguments override these defaults (e.g. --concurrency 10).
exec "${PYTHON_BIN:-python}" - "$@" <<'PY'
import runpy
import sys
from datetime import datetime

from scripts.benchmark_credentials import configure

configure()
sys.argv = [
    "benchmarks.runner.run_subprocess",
    "--benchmark_profile", "bc200_semantic",
    "--pipeline", "dynamic_htn_informed_research_v1",
    "--profile", "deepseek_v4_1_flash_bc200",
    "--concurrency", "1",
    "--seed", "42",
    "--out", "results/bc200_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    *sys.argv[1:],
]
runpy.run_module("benchmarks.runner.run_subprocess", run_name="__main__")
PY
