#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# BC200 without epoch rebuilds; keep automatic routing and all other budgets.
# Additional runner arguments override defaults, e.g. --out results/<run>.
exec "${PYTHON_BIN:-.venv/bin/python}" -m benchmarks.runner.run_subprocess \
  --benchmark_profile bc200 \
  --pipeline dynamic_htn_informed_research \
  --profile workflows/dynamic_htn_informed_research/profiles/ablations/no_replanning/auto.yaml \
  --runs 1 --concurrency 10 --seed 42 --no-shuffle \
  --out "results/bc200_no_replanning_c10_$(date +%Y%m%d_%H%M%S)" \
  "$@"
