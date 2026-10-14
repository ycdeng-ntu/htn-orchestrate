"""WideSearch — structural F1 scorer (not LLM judge).

``ground_truth`` is a JSON blob carrying the eval_spec + gold_table; the
structural scorer parses it at scoring time.
"""

from __future__ import annotations

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("widesearch", DatasetConfig(
        name="WideSearch", key="WideSearch",
        default_pipeline="react_base", **STD_SCHEMA,
    )),
]
