"""DRB-II: official idx identity and full reports, graded out of band."""

from benchmarks.core.registry import DatasetConfig
from benchmarks.families._schema import STD_SCHEMA

CONFIGS = [
    ("deepresearch_bench_ii", DatasetConfig(
        name="DeepResearch Bench II", key="deepresearch_bench_ii",
        default_pipeline="dynamic_htn_informed_research",
        default_answer_type="report",
        extra_metadata_fields=("task_id", "language", "theme", "idx", "description", "judge_task"),
        **{**STD_SCHEMA, "id_field": "idx"},
    )),
]
