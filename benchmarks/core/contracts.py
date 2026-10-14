"""Boundary between solver inputs, raw deliverables and evaluation inputs."""

from __future__ import annotations

from typing import Any

# Explicit solver-visible fields. Everything else remains in the evaluation
# snapshot; adding a dataset field never silently exposes a rubric to an agent.
SOLVER_METADATA_FIELDS = frozenset({
    "question_id", "answer_type", "pipeline_id", "benchmark", "dataset",
    "profile", "dynir_config", "benchmark_web_policy", "category", "language", "file_path", "file_name", "image_path",
    "task_id", "idx", "_trial_dir",
    "instance_id", "repo_name", "base_commit", "hints_text", "swe_profile",
})


def solver_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (metadata or {}).items()
            if key in SOLVER_METADATA_FIELDS}


def report_content(state: dict[str, Any] | None) -> str:
    """The complete deliverable, preserved separately from the scored span."""
    if not state:
        return ""
    report = state.get("report")
    if isinstance(report, dict) and isinstance(report.get("content"), str):
        if report["content"].strip():
            return report["content"]
    return state.get("final_content") or state.get("final_answer") or ""


def extract_prediction(state: dict[str, Any] | None, answer_type: str) -> str:
    """Report benchmarks grade the whole deliverable; short tasks extract ROOT's answer."""
    if not state:
        return ""
    if answer_type == "report":
        return report_content(state)
    report = state.get("report")
    if isinstance(report, dict) and report.get("content"):
        from benchmarks.core.answer_extractor import extract_answer_regex

        return extract_answer_regex(report, answer_type)
    # Existing react_base emits an already-extracted final_answer.
    return (state.get("final_answer") or "").strip()


def is_graded(result: dict[str, Any]) -> bool:
    """Legacy boolean results are graded; explicit null/deferred results are not."""
    if result.get("judge_method") == "rubric":
        return (not result.get("scoring_error") and
                type(result.get("rubric_score")) in (int, float) and
                0 <= result["rubric_score"] <= 100)
    if result.get("benchmark") == "prl_bench":
        return False
    return (result.get("judge_method") != "not_graded"
            and result.get("is_correct") is not None)
