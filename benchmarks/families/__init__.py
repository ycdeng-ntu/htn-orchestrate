"""Per-family DatasetConfig registry.

Each family module exposes ``CONFIGS: list[tuple[str, DatasetConfig]]``.
This package imports them all and builds a single ``REGISTRY`` dict
keyed by benchmark slug (e.g. ``"browsecomp"``).

Adding a new benchmark:
    - If its family already has a file here, append a tuple to that
      file's ``CONFIGS``.
    - Otherwise create a new ``benchmarks/families/<family>.py`` exposing
      ``CONFIGS: list[tuple[str, DatasetConfig]]``. It's picked up
      automatically — no edits here needed.
"""

from __future__ import annotations

import importlib
import pkgutil

from benchmarks.core.registry import DatasetConfig

REGISTRY: dict[str, DatasetConfig] = {}

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    for _key, _cfg in getattr(_mod, "CONFIGS", []):
        if _key in REGISTRY:
            raise RuntimeError(
                f"Duplicate benchmark key {_key!r} in families/"
            )
        REGISTRY[_key] = _cfg

__all__ = ["REGISTRY"]
