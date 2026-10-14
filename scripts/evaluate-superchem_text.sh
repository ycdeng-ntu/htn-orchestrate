#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/docker_agent.py run \
  --benchmark_profile superchem_text \
  --pipeline dynamic_htn_informed_research \
  --profile auto \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_superchem_text


# uv run python -m benchmarks.runner.check_progress ./results/$(date +%F)_superchem_text
