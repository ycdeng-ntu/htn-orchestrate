"""SUPERChem-Text — single-letter MCQ (A-J), graded with the HLE judge schema."""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("superchem_text", DatasetConfig(
        name="SUPERChem-Text", key="SUPERChem-Text",
        default_answer_type="multipleChoice",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
