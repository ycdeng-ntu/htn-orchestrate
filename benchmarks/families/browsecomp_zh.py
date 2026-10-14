"""BrowseComp-ZH (Chinese)."""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("browsecomp_zh", DatasetConfig(
        name="BrowseComp-ZH", key="BrowseComp-ZH",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
