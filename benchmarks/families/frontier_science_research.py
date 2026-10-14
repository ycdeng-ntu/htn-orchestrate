"""FrontierScience-Research (rubric-graded science questions)."""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("frontier_science_research", DatasetConfig(
        name="FrontierScience-Research", key="FrontierScience-Research",
        default_answer_type="report",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
