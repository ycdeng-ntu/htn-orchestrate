"""BrowseComp (English)."""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("browsecomp", DatasetConfig(
        name="BrowseComp", key="BrowseComp",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
    ("browsecomp_200", DatasetConfig(
        name="BrowseComp-200", key="BrowseComp-200",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
