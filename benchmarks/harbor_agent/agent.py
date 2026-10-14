"""AgentHarnessAgent — Harbor BaseAgent adapter.

Bridges Harbor's agent protocol to AgentHarness's research pipeline,
enabling benchmark evaluation via the Harbor framework.
"""

from __future__ import annotations

import json
import logging
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from benchmarks.core.contracts import extract_prediction, report_content, solver_metadata

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "harbor is required for benchmark runs but isn't installed. "
        "Run: uv sync --extra benchmark"
    ) from exc

logger = logging.getLogger(__name__)

# ── TOML metadata parser (regex, no dependency) ──────────────────────

_METADATA_SECTION_RE = re.compile(
    r"^\[metadata\]\s*$",
    re.MULTILINE,
)
_KV_RE = re.compile(
    r'^(\w+)\s*=\s*"([^"]*)"',
)


def parse_task_metadata(toml_text: str) -> dict[str, Any]:
    """Read solver-visible TOML metadata, with a legacy malformed-text fallback."""
    try:
        parsed = tomllib.loads(toml_text).get("metadata", {})
        result = {key: str(value) for key, value in solver_metadata(parsed).items()}
        if "dynir_config" in result:
            result["dynir_config"] = json.loads(result["dynir_config"])
        return result
    except tomllib.TOMLDecodeError:
        pass
    try:
        m = _METADATA_SECTION_RE.search(toml_text)
        if not m:
            return {}

        result: dict[str, str] = {}
        for line in toml_text[m.end():].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Stop at next section header
            if line.startswith("["):
                break
            kv = _KV_RE.match(line)
            if kv:
                result[kv.group(1)] = kv.group(2)
        return solver_metadata(result)
    except Exception:
        return {}


# ── AgentHarnessAgent ─────────────────────────────────────────────────


class AgentHarnessAgent(BaseAgent):
    """Harbor-compatible agent that delegates to AgentHarness pipelines."""

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Populated on first ``setup()``; reused across trials. Harbor
        # never calls a teardown hook, so the session lives until process
        # exit. Tests should drive ``BenchmarkSession`` via ``async with``
        # directly so the registry snapshot is restored cleanly.
        self._session: Any = None

    @staticmethod
    def name() -> str:
        return "AgentHarness"

    def version(self) -> str | None:
        return "0.2.0"

    async def setup(
        self,
        environment: BaseEnvironment,  # noqa: ARG002  (Harbor compat)
        *,
        db_path: str | None = None,  # noqa: ARG002  (Harbor compat)
    ) -> None:
        """Bootstrap the AgentHarness runtime (reused across trials)."""
        if self._session is not None:
            return
        from benchmarks.core.kernel_adapter import BenchmarkSession

        session = BenchmarkSession()
        await session.__aenter__()
        self._session = session
        logger.info("AgentHarness session bootstrapped")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute a single benchmark question through the pipeline."""
        # Capture logs_dir immediately — concurrent tasks share the same agent
        # instance and may mutate self.logs_dir between coroutine yields.
        logs_dir = self.logs_dir
        t0 = time.monotonic()
        agent_dir = logs_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Read task metadata ────────────────────────────────
        meta = self._read_task_metadata()
        answer_type = meta.get("answer_type", "exactMatch")
        pipeline_id = meta.get("pipeline_id", "react_base")
        image_path = meta.get("image_path", "")
        file_path = meta.get("file_path", "") or image_path
        question_id = meta.get("question_id", "")

        # Pass trial dir to nodes that write per-task artifacts (e.g. SWE trajectory)
        meta = solver_metadata(meta)
        meta["_trial_dir"] = str(logs_dir)
        meta["file_path"] = file_path
        meta["image_path"] = image_path

        # ── 2. Execute pipeline via session ──────────────────────
        if self._session is None:
            raise RuntimeError("setup() was not called before run()")
        state: dict[str, Any] | None = None
        pipeline_error: str | None = None
        try:
            state = await self._session.run(
                instruction, meta=meta, pipeline_id=pipeline_id,
            )
        except Exception as exc:
            pipeline_error = str(exc)
            logger.error("Pipeline error: %s", exc)
        task_id = state.get("task_id", "") if state else ""
        logger.info(
            "Task %s ran question %s",
            task_id, question_id or "(unknown)",
        )

        # ── 5. Extract answer ────────────────────────────────────
        predicted = _extract_predicted_answer(state, answer_type)

        if not predicted:
            predicted = "<ANSWER_NOT_FOUND>"

        duration = time.monotonic() - t0

        # ── 6. Write final_answer.txt ────────────────────────────
        (agent_dir / "final_answer.txt").write_text(
            predicted, encoding="utf-8",
        )
        raw_content = report_content(state)
        if raw_content:
            (agent_dir / "final_content.md").write_text(raw_content, encoding="utf-8")

        # ── 7. Write error.log if pipeline errored ───────────────
        errors = state.get("errors", []) if state else []
        if pipeline_error or errors:
            lines = []
            if pipeline_error:
                lines.append(f"Pipeline error: {pipeline_error}")
            for e in errors:
                lines.append(str(e))
            (agent_dir / "error.log").write_text(
                "\n".join(lines), encoding="utf-8",
            )

        # ── 8. Populate context.metadata ─────────────────────────
        if context.metadata is None:
            context.metadata = {}
        context.metadata.update({
            "question_id": question_id,
            "answer_type": answer_type,
            "predicted_answer": predicted,
            "duration_seconds": round(duration, 2),
            "pipeline_id": pipeline_id,
        })

        logger.info(
            "Completed question %s in %.1fs → %s",
            question_id or task_id,
            duration,
            predicted[:60],
        )

    # ── Private helpers ──────────────────────────────────────────

    def _read_task_metadata(self) -> dict[str, Any]:
        """Read task metadata from the trial's config.json → task.toml.

        Harbor writes a config.json in logs_dir with a 'task' key
        pointing to the task source directory, which contains task.toml.
        Falls back to empty dict if any file is missing.
        """
        config_path = self.logs_dir / "config.json"
        if not config_path.exists():
            logger.debug("No config.json at %s", config_path)
            return {}

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read config.json: %s", exc)
            return {}

        task_source = config.get("task", {})
        # task_source may be a dict with a 'path' key or a string path
        if isinstance(task_source, dict):
            task_path = task_source.get("path", "")
        else:
            task_path = str(task_source)

        if not task_path:
            return {}

        toml_path = Path(task_path) / "task.toml"
        if not toml_path.exists():
            logger.debug("No task.toml at %s", toml_path)
            return {}

        try:
            toml_text = toml_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read task.toml: %s", exc)
            return {}

        return parse_task_metadata(toml_text)


def _extract_predicted_answer(
    state: dict[str, Any] | None,
    answer_type: str,
) -> str:
    """Honor short-answer versus full-report benchmark output contracts."""
    return extract_prediction(state, answer_type)
