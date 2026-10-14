"""PRL-Bench: one research problem per paper, retaining all subtasks."""
from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS = [("prl_bench", DatasetConfig(
    name="PRL-Bench", key="PRL-Bench", default_answer_type="report",
    default_pipeline="dynamic_htn_informed_research",
    extra_metadata_fields=("subfield", "source_revision", "scoring_protocol"),
    **STD_SCHEMA,
))]
