#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON_BIN:-python}" - "$@" <<'PY'
import runpy
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.env')
sys.argv = [
    'benchmarks.runner.run_subprocess', '--benchmark_profile', 'prl_bench',
    '--pipeline', 'dynamic_htn_informed_research_v1',
    '--profile', 'deepseek_v4_1_flash_prl_offline',
    '--concurrency', '16', '--seed', '42',
    '--out', 'results/prl_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
    *sys.argv[1:],
]
runpy.run_module('benchmarks.runner.run_subprocess', run_name='__main__')
PY
