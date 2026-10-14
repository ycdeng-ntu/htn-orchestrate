# Docker Runner

Run benchmark evaluations inside a shared Docker container. The host machine only
needs Python 3 and a working Docker CLI. The container reuses a pre-built local
image (`frontieragent-eval-local:latest`), which must already exist; it is not
pulled or built automatically.

## Start the container

```bash
python3 scripts/docker_agent.py up
python3 scripts/docker_agent.py status
```

The container is named `agentharness-benchmarks`. All benchmarks run through
separate runner processes inside this single container. Each question still uses
its own worker subprocess.

## Submit an evaluation run

```bash
python3 scripts/docker_agent.py run \
  --benchmark frontier_science_research \
  --pipeline dynamic_htn_informed_research \
  --profile auto \
  --subset /host/path/dataset.jsonl \
  --concurrency 16 \
  --out results/fsr_docker_run
```

The `run` command submits in the background. Closing the terminal does not stop
the evaluation. It returns a job ID and log path. A successful submission does
not guarantee a successful run; check `status.json` for the outcome.

- Container `results/<run>` maps to host `results/docker/<run>`.
- Logs and status are in `results/docker/.jobs/<job>/runner.log` and `status.json`.
- `--profile`, `--concurrency`, and scoring parameters pass through to the runner.
- `--subset` accepts a host JSONL file, copied into the container. Datasets with
  attachments require a separately mounted read-only data directory.
- Omitting `--out` creates a unique output directory.

Check progress:

```bash
python3 scripts/docker_agent.py exec python -m benchmarks.runner.check_progress results/fsr_docker_run
```

## File isolation

The container runs as UID/GID `65534:65534` with no elevated capabilities. Code,
data, tests, scripts, `pyproject.toml`, and `.env` are mounted read-only. Only
`results/docker` is writable. The Docker socket, host root, and other projects
are not mounted. Container temp files are in `/tmp` and discarded on removal.

Source code is mounted live, not snapshotted. Changes on the host are visible to
new processes, so avoid modifying code during active runs.

## Stop and verify

```bash
# Stop all tasks and remove the container (host results are preserved)
python3 scripts/docker_agent.py stop

# Recreate the container (does not resume stopped tasks)
python3 scripts/docker_agent.py up

# Offline verification of mounts, identity, dotenv, result writes, and workers
python3 scripts/docker_agent.py exec python scripts/docker_smoke.py
```

Stopping marks running jobs as interrupted. Resubmitting the same `--out`
directory uses the runner's per-question resume logic.
