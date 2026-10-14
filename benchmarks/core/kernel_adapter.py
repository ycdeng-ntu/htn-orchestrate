"""Benchmark-side kernel adapter — exposes ``BenchmarkSession``.

``BenchmarkSession`` is an async context manager that bootstraps the runtime
components the benchmark stack needs (event bus, agent registry, scheduler,
etc.) inside a scoped registry snapshot, then tears it down on exit.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.core.llm import LLMClient

    from agent_harness.scheduling.pipeline_registry import PipelineRegistry
    from agent_harness.scheduling.process_manager import ProcessManager
    from agent_harness.scheduling.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _discover_pipeline_specs(
    pipeline_reg: "PipelineRegistry",
    scan_dirs: list[tuple[str, bool]] | None = None,
) -> None:
    """Scan ``workflows/*/spec.py`` and register module-level ``PipelineSpec``s."""
    from agent_harness.models.pipeline_spec import PipelineSpec

    default_dirs = [("workflows", True)]
    for base, overwrite in (scan_dirs or default_dirs):
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        patterns = ["*/spec.py"] if base_path.name == "workflows" else ["*.py"]
        for pat in patterns:
            for spec_file in sorted(base_path.glob(pat)):
                if spec_file.name.startswith("_"):
                    continue
                mod_name = str(spec_file).replace("/", ".").removesuffix(".py")
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as exc:
                    logger.debug("Skipping %s: %s", mod_name, exc)
                    continue
                for attr in dir(mod):
                    obj = getattr(mod, attr, None)
                    if isinstance(obj, PipelineSpec):
                        if overwrite or not pipeline_reg.has(obj.pipeline_id):
                            pipeline_reg.register(obj)


class BenchmarkSession:
    """Scoped runtime container for benchmark execution.

    Async context manager: snapshots the global service registry on enter,
    bootstraps runtime components inside the snapshot, restores on exit.

        async with BenchmarkSession() as session:
            state = await session.run(question, meta=meta, pipeline_id="react_base")
            # ``state['final_answer']`` holds the agent's reply.
    """

    def __init__(self) -> None:
        self._scheduler: Scheduler | None = None
        self._pm: ProcessManager | None = None
        self._llm: LLMClient | None = None
        self._registry_snapshot: dict[type, Any] | None = None

    @property
    def scheduler(self) -> "Scheduler":
        if self._scheduler is None:
            raise RuntimeError("BenchmarkSession not entered")
        return self._scheduler

    @property
    def pm(self) -> "ProcessManager":
        if self._pm is None:
            raise RuntimeError("BenchmarkSession not entered")
        return self._pm

    @property
    def llm(self) -> "LLMClient":
        if self._llm is None:
            raise RuntimeError("BenchmarkSession not entered")
        return self._llm

    async def __aenter__(self) -> "BenchmarkSession":
        from agent_harness.core.runtime import registry

        if self._registry_snapshot is not None:
            raise RuntimeError("BenchmarkSession already entered")

        self._registry_snapshot = registry.snapshot()
        try:
            await self._bootstrap()
        except Exception:
            registry.restore(self._registry_snapshot)
            self._registry_snapshot = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        from agent_harness.core.runtime import registry

        if self._registry_snapshot is None:
            return
        registry.restore(self._registry_snapshot)
        self._registry_snapshot = None
        self._scheduler = None
        self._pm = None
        self._llm = None

    async def run(
        self,
        instruction: str,
        *,
        meta: dict[str, Any] | None = None,
        pipeline_id: str,
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one benchmark question and return the final pipeline state."""
        task = await self.pm.create_task(instruction)
        task_id = str(task.id)

        from benchmarks.core.contracts import solver_metadata
        meta = solver_metadata(meta)

        input_data: dict[str, Any] = {
            "task_id": task_id,
            "original_question": instruction,
            "file_path": (meta or {}).get("file_path", "")
            or (meta or {}).get("image_path", ""),
            "metadata": dict(meta or {}),
            "clarified_questions": [],
            "evidence_cards": [],
            "assertions": [],
            "react_steps": [],
            "report": None,
            "current_phase": "",
            "errors": [],
            "messages": [],
            "language": "auto",
        }
        if extra_input:
            input_data.update(extra_input)
        input_data["metadata"] = solver_metadata(input_data.get("metadata"))

        async for _mode, _chunk in self.scheduler.execute(
            task.id, input_data, pipeline_id=pipeline_id,
        ):
            pass
        state = await self.scheduler.get_state(task.id)
        return state or {}

    async def _bootstrap(self) -> None:
        """Wire up the runtime components inside the scoped registry."""
        from agent_harness.components.middleware.llm import (
            LLMMiddlewareChain,
            SummarizationMiddleware,
        )
        from agent_harness.core.runtime import registry
        from agent_harness.core.runtime.dag.graph_builder import DynamicGraphBuilder
        from agent_harness.core.runtime.events.bus import EventBus
        from agent_harness.core.runtime.registries.agents import AgentRegistry
        from agent_harness.core.runtime.resources.manager import ResourceManager
        from agent_harness.infra.config import get_config
        from agent_harness.infra.llm_adapter import create_llm
        from agent_harness.scheduling.pipeline_registry import PipelineRegistry
        from agent_harness.scheduling.process_manager import ProcessManager
        from agent_harness.scheduling.scheduler import Scheduler
        from agent_harness.state.event_store.sqlite import EventStore
        from plugins.tools import get_builtin_tools

        config = get_config()
        os.makedirs("data", exist_ok=True)

        # In-memory EventStore + ProcessManager (no DB writes).
        event_store = EventStore()
        registry.register(EventStore, event_store)
        registry.register(EventBus, EventBus())

        pm = ProcessManager(event_store, session_factory=None)
        registry.register(ProcessManager, pm)

        agent_reg = AgentRegistry()
        registry.register(AgentRegistry, agent_reg)

        llm = create_llm(config)
        tools_map = dict(get_builtin_tools())

        registry.register(
            ResourceManager,
            ResourceManager(llm=llm, tools=tools_map),
        )

        # Outer safety net; ``react_base`` does its own in-loop compaction.
        llm_mw_chain = LLMMiddlewareChain()
        llm_mw_chain.add(SummarizationMiddleware(threshold=80_000, keep_recent=10))
        registry.register(LLMMiddlewareChain, llm_mw_chain)

        pipeline_reg = PipelineRegistry()
        from agent_harness.scheduling.workflow_loader import load_workflow_plugins
        load_workflow_plugins(pipeline_reg, agent_reg)
        _discover_pipeline_specs(pipeline_reg)
        registry.register(PipelineRegistry, pipeline_reg)

        graph_builder = DynamicGraphBuilder()
        registry.register(DynamicGraphBuilder, graph_builder)

        scheduler = Scheduler(
            None,
            pm, event_store,
            graph_builder=graph_builder,
            pipeline_registry=pipeline_reg,
            checkpointer=None,
        )
        registry.register(Scheduler, scheduler)

        self._scheduler = scheduler
        self._pm = pm
        self._llm = llm


__all__ = ["BenchmarkSession"]
