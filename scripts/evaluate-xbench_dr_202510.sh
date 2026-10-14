#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/docker_agent.py run \
  --benchmark_profile xbench_dr_202510 \
  --pipeline react_base \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_xbench_dr_202510


# uv run python -m benchmarks.runner.check_progress ./results/$(date +%F)_xbench_dr_202510
