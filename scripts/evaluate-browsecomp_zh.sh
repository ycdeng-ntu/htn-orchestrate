#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/docker_agent.py run \
  --benchmark_profile browsecomp_zh \
  --pipeline dynamic_htn_informed_research \
  --profile keep5 \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_browsecomp_zh


# uv run python -m benchmarks.runner.check_progress ./results/$(date +%F)_browsecomp_zh
