# HTN-Orchestrate

Implementation of the HTN-Orchestrate framework described in the accompanying paper.
HTN-Orchestrate is a two-layer orchestration framework that separates global task
planning from local multi-agent execution through a hierarchical task network (HTN).

## Structure

```
htn/                  Core HTN runtime
  task_node.py          Task node model (hierarchy, dependencies, status)
  task_tree.py          Task tree with AND-semantic completion propagation
  planner.py            Planner with decomposition strategies and budgets
  orchestrator.py       Leaf orchestrator (dispatch, fan-in, synthesis)
  epoch.py              Epoch runner (planning loop → execution wave → settle)
  strategies.py         Decomposition strategy implementations
  recovery.py           Task-network rebuilding with evidence carryover
nodes/                Entry point and question-level driver
tools/                Worker tool backends (web search, code execution, file I/O)
task_strategies/      Strategy-specific coordination, synthesis, and planning
  research/             Open-ended web research
  science_olympiad/     Science olympiad reasoning
  science_mcq/          Science multiple-choice
  report_research/      Scientific report writing and verification
profiles/             Solver configuration (model, budgets, reasoning effort)
runtime/              Worker runtime and context management
```

## Setup

Dependencies are managed by the root package. From the repository root:

```bash
uv sync
```

Configure API keys in `.env` or the process environment:

| Key | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM provider |
| `SERPER_API_KEY` | Web search |
| `JINA_API_KEY` | Web content extraction |
| `E2B_API_KEY` | Python code execution sandbox |

## Running

Solver configuration is set with `--profile`; dataset, judge model, and web
restrictions are set with `--benchmark_profile`. See the
`benchmarks/profiles/` for available
benchmark profiles.

```bash
# Full evaluation run
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark_profile fso100 \
  --pipeline dynamic_htn_informed_research \
  --profile auto --concurrency 10

# Single-question smoke test (reduced budgets)
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark_profile /path/to/smoke.yaml \
  --pipeline dynamic_htn_informed_research \
  --profile research_smoke --concurrency 1
```

The default `--profile auto` loads the meta profile
([auto.yaml](profiles/auto.yaml)), which routes each question to a
task-strategy-specific solver profile:

```yaml
router:
  profile: ./router.yaml
routes:
  research: ./research.yaml
  science_olympiad: ./science_olympiad.yaml
  science_mcq: ./science_mcq.yaml
  report_research: ./report_research.yaml
```

A single named profile (e.g., `--profile research_smoke`) bypasses the router
and uses that profile directly.

## Python API

```python
import asyncio
from benchmarks.core.kernel_adapter import BenchmarkSession

async def main():
    async with BenchmarkSession() as session:
        state = await session.run(
            "Your research question",
            pipeline_id="dynamic_htn_informed_research",
            meta={"profile": "auto"},
        )
        print(state["final_content"])

asyncio.run(main())
```

## Tests

```bash
uv run --with pytest --with pytest-asyncio python -m pytest tests -q
```
