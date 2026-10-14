"""One lazily created E2B sandbox per worker, with cancellation-safe release.

Only the small files/commands/kill interface is required by injected fakes.
No host execution fallback and no application/session/task-board dependencies.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from .sandbox_bootstrap import prepare_research_sandbox

logger = logging.getLogger(__name__)


async def call_adapter(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    value = await asyncio.to_thread(fn, *args, **kwargs)
    return await value if inspect.isawaitable(value) else value


class WorkerSandbox:
    def __init__(self, factory: Callable[..., Any]):
        self.factory = factory
        self._creation: asyncio.Task[Any] | None = None
        self._close: asyncio.Task[None] | None = None
        self.closed = False
        self.command_lock = asyncio.Lock()

    async def get(self) -> Any:
        if self.closed:
            raise RuntimeError("Worker sandbox has been closed")
        if self._creation is None:
            self._creation = asyncio.create_task(call_adapter(self.factory))
        # Cancellation of a tool caller must not discard a resource being created
        # in a background thread. aclose waits for it and kills the eventual result.
        return await asyncio.shield(self._creation)

    async def _release(self) -> None:
        if self._creation is None:
            return
        try:
            sandbox = await self._creation
        except Exception:
            return  # No resource was acquired.
        await call_adapter(sandbox.kill)

    async def aclose(self) -> None:
        self.closed = True
        if self._close is None:
            self._close = asyncio.create_task(self._release())
        try:
            await asyncio.shield(self._close)
        except asyncio.CancelledError:
            # Finish releasing before propagating a second cancellation.
            await self._close
            raise


def e2b_factory(*, api_key: str, template: str, timeout: int, allow_internet_access: bool = True) -> Any:
    if not api_key:
        raise RuntimeError("E2B_API_KEY is required for run_python_code and sandbox read_file")
    from e2b_code_interpreter import Sandbox
    sandbox = Sandbox.create(
        api_key=api_key, template=template, timeout=timeout, request_timeout=30,
        **({"allow_internet_access": False} if not allow_internet_access else {}),
    )
    try:
        if allow_internet_access:
            prepare_research_sandbox(sandbox)
        else:
            def probe_ready() -> bool:
                try:
                    result = sandbox.commands.run(
                        "python3 -c 'import numpy, scipy, sympy, matplotlib'", timeout=60,
                    )
                    return result.exit_code == 0
                except Exception as exc:
                    if type(exc).__name__ == "CommandExitException" and hasattr(exc, "exit_code"):
                        return False
                    raise

            if not probe_ready():
                # Controlled setup only: no model code has run or can run yet.
                # A preinstalled template skips this temporary network window.
                try:
                    sandbox.update_network({"allow_internet_access": True})
                    install = sandbox.commands.run(
                        "python3 -m pip install --disable-pip-version-check --no-input --quiet "
                        "numpy==2.2.6 scipy==1.15.3 sympy==1.14.0 matplotlib==3.10.3",
                        timeout=180,
                    )
                    if install.exit_code != 0:
                        raise RuntimeError("Offline scientific dependency preparation failed")
                finally:
                    sandbox.update_network({"allow_internet_access": False})
                if not probe_ready():
                    raise RuntimeError("Offline sandbox requires preinstalled numpy/scipy/sympy/matplotlib")

    except BaseException:
        # WorkerSandbox cannot release a failed factory's return value. Once
        # create succeeds, this factory owns cleanup until preparation passes.
        try:
            sandbox.kill()
        except Exception as exc:
            logger.warning("Sandbox cleanup after preparation failure failed: %s", type(exc).__name__)
        raise
    return sandbox
