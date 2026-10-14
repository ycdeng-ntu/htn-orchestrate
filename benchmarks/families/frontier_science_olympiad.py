"""FrontierScience-Olympiad (answer-match science questions)."""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("frontier_science_olympiad", DatasetConfig(
        name="FrontierScience-Olympiad", key="FrontierScience-Olympiad",
        default_answer_type="single_answer",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
