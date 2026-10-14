# HTN-Orchestrate

This repository contains the implementation of **HTN-Orchestrate**, a two-layer orchestration framework for LLM-powered multi-agent systems described in the following paper:

> **HTN-Orchestrate: A Hierarchical Task Network Framework for Multi-Agent Orchestration**

HTN-Orchestrate separates global task planning from local multi-agent execution through a hierarchical task network (HTN). The Planner expands the task network incrementally as prerequisite results become available, so that each decomposition decision reflects findings obtained during execution. Each executable leaf subtask is delegated to an Orchestrator that schedules workers and uses their reports to guide investigation, verification, and synthesis. Results propagate upward through the task network under AND semantics, and a rebuilding mechanism constructs a new task network conditioned on partial results when the current decomposition cannot resolve the task.

This codebase is developed on top of [AgentHarness](https://github.com/ApodexAI/AgentHarness), an open-source evaluation framework for LLM agents on public benchmarks.

## Repository Structure

```
agent_harness/          Core runtime (scheduling, LLM clients, agent loop)
workflows/
  dynamic_htn_informed_research/
    htn/                HTN runtime (planner, orchestrator, task tree, epoch runner)
    task_strategies/    Strategy-specific coordination and synthesis
    tools/              Worker tool backends (web search, code execution, file I/O)
    profiles/           Solver configuration (model, budgets, reasoning effort)
benchmarks/             Evaluation runner, dataset loading, judging
tests/                  Unit and integration tests
scripts/                Utility scripts
```

See the [workflow README](workflows/dynamic_htn_informed_research/README.md) for a detailed description of the HTN-Orchestrate implementation.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
uv sync --python 3.12
```

Copy `.env.example` to `.env` and fill in the required API keys:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM provider |
| `SERPER_API_KEY` | Web search |
| `JINA_API_KEY` | Web content extraction |
| `E2B_API_KEY` | Code execution sandbox |

## Usage

**Full evaluation run**

```bash
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark_profile fso100 \
  --pipeline dynamic_htn_informed_research \
  --profile auto --concurrency 10
```

**Single-question smoke test**

```bash
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark_profile /path/to/smoke.yaml \
  --pipeline dynamic_htn_informed_research \
  --profile research_smoke --concurrency 1
```

**Check progress**

```bash
uv run python -m benchmarks.runner.check_progress ./results
```

See `benchmarks/profiles/` for dataset and judge configuration.

## Tests

```bash
uv run --with pytest --with pytest-asyncio python -m pytest tests -q
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{htn-orchestrate2026,
  title  = {HTN-Orchestrate: A Hierarchical Task Network Framework
            for Multi-Agent Orchestration},
  author = {},
  year   = {2026}
}
```

## Acknowledgments

This project is built on [AgentHarness](https://github.com/ApodexAI/AgentHarness) by [Apodex AI](https://github.com/ApodexAI). We thank the AgentHarness authors for providing the evaluation infrastructure that made this work possible.

## License

Apache 2.0. See [LICENSE](./LICENSE) for details.
