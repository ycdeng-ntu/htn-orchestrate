"""Export DRB-II reports and invoke a separately installed official evaluator.

Generation stays ``not_graded``. This module has no official-evaluator import or
model dependency: ``export`` writes a reviewable bundle; ``grade`` explicitly
invokes scripts from a user-supplied official checkout. Official criterion and
aggregate scores remain in that bundle, separate from binary benchmark rewards.

    python -m benchmarks.grade_drb export --run-dir results/drb --out-dir results/drb/export
    python -m benchmarks.grade_drb grade --run-dir results/drb --out-dir results/drb/export \
        --official-dir /path/to/DeepResearch-Bench-II
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from benchmarks.core.question import BenchmarkQuestion
from benchmarks.core.registry import load_questions

logger = logging.getLogger(__name__)
_CITATION_MAP_RE = re.compile(r"\n*<!--\s*citation-map:.*?-->\s*", re.DOTALL)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_judge_config(config: dict) -> dict[str, str]:
    """Only non-secret evaluator settings may enter the exported manifest."""
    if not isinstance(config, dict):
        raise ValueError("DRB judge must be a mapping")
    clean = {key: config.get(key) for key in ("provider", "model", "reasoning_effort")}
    if any(not isinstance(value, str) or not value.strip() for value in clean.values()):
        raise ValueError("DRB judge provider/model/reasoning_effort must be nonempty strings")
    clean = {key: value.strip() for key, value in clean.items()}
    if clean["provider"] not in {"openrouter", "openai"}:
        raise ValueError("DRB judge provider must be openrouter or openai")
    if clean["reasoning_effort"] not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("Invalid DRB judge reasoning_effort")
    return clean


def resolve_benchmark_judge(profile: str | Path | None = None, *, run_dir: Path | None = None) -> dict:
    from benchmarks.core.profile import load_benchmark_profile, snapshot_config

    snapshot_path = run_dir / "benchmark_profile.json" if run_dir else None
    if profile is None and snapshot_path and snapshot_path.is_file():
        config = snapshot_config(json.loads(snapshot_path.read_text()), "deepresearch_bench_ii")
    else:
        config = load_benchmark_profile(profile or "deepresearch_bench_ii").configuration
        if config["benchmark"] != "deepresearch_bench_ii":
            raise ValueError("DRB grading requires a deepresearch_bench_ii benchmark profile")
    return _safe_judge_config(config["judge"])


def official_environment(judge: dict, *, source: dict | None = None) -> dict[str, str]:
    """Translate the selected provider's credential pair for the GPT evaluator.

    Do not modify the parent process or fall back to another provider's key.
    """
    from dotenv import dotenv_values

    judge = _safe_judge_config(judge)
    source = ({**dotenv_values(_PROJECT_ROOT / ".env"), **os.environ}
              if source is None else source)
    prefix = judge["provider"].upper()
    key = source.get(prefix + "_API_KEY")
    if not key:
        raise ValueError(f"Missing {prefix}_API_KEY for DRB grading")
    api_url = source.get(prefix + "_API_URL")
    if not api_url:
        base = source.get(prefix + "_BASE_URL") or {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
        }[judge["provider"]]
        api_url = base.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
    return {
        **os.environ,
        "OPENAI_API_URL": str(api_url), "OPENAI_API_KEY": str(key),
        "OPENAI_MODEL": judge["model"],
        "OPENAI_REASONING_EFFORT": judge["reasoning_effort"],
        "OPENAI_MAX_OUTPUT_TOKENS": "32768", "OPENAI_TIMEOUT": "600",
    }


def _run_dirs(run_dir: Path) -> list[Path]:
    runs = sorted(p for p in run_dir.glob("run_*") if (p / "trials").is_dir())
    if not runs and (run_dir / "trials").is_dir():
        runs = [run_dir]
    if not runs:
        raise ValueError(f"No trials/ or run_*/trials/ under {run_dir}")
    return runs


def load_run_questions(run_dir: Path, dataset: str) -> list[BenchmarkQuestion]:
    """Prefer saved evaluation inputs, preserving the exact run cohort and rubrics."""
    by_id: dict[str, BenchmarkQuestion] = {}
    for run in _run_dirs(run_dir):
        for path in sorted((run / "tasks").glob("*/tests/evaluation.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            q = BenchmarkQuestion(
                id=str(row["id"]), question=row["question"],
                ground_truth=row["ground_truth"], answer_type=row["answer_type"],
                metadata=row.get("metadata", {}),
            )
            if q.id in by_id and by_id[q.id] != q:
                raise ValueError(f"Conflicting evaluation inputs for idx {q.id}")
            by_id[q.id] = q
    return list(by_id.values()) if by_id else load_questions(dataset)


def _read_answer(trial: Path) -> str:
    for name in ("final_content.md", "final_answer.txt"):
        path = trial / "agent" / name
        if path.is_file():
            value = path.read_text(encoding="utf-8")
            if value.strip() and value.strip() != "<ANSWER_NOT_FOUND>":
                return value
    path = trial / "result.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8")).get("predicted_answer") or ""
        if value.strip() != "<ANSWER_NOT_FOUND>":
            return value
    return ""


def export_run(
    run_dir: Path, out_dir: Path, *, model_name: str = "dynamic_htn_informed_research",
    dataset: str = "deepresearch_bench_ii", questions: list[BenchmarkQuestion] | None = None,
) -> int:
    """Write official ``<model>/idx-<N>.md`` files, one model bucket per run."""
    if not model_name or Path(model_name).name != model_name or model_name in {".", ".."}:
        raise ValueError("model_name must be a directory basename")
    questions = questions if questions is not None else load_run_questions(run_dir, dataset)
    id_to_idx: dict[str, int] = {}
    for q in questions:
        idx = int(q.metadata["idx"])
        id_to_idx[q.id] = idx
        if q.metadata.get("task_id"):
            id_to_idx[str(q.metadata["task_id"])] = idx
    runs = _run_dirs(run_dir)
    count = 0
    for run in runs:
        model = f"{model_name}_{run.name}" if len(runs) > 1 else model_name
        bucket = out_dir / model
        bucket.mkdir(parents=True, exist_ok=True)
        for trial in sorted((run / "trials").iterdir()):
            if not trial.is_dir() or trial.name not in id_to_idx:
                continue
            if content := _read_answer(trial):
                content = _CITATION_MAP_RE.sub("", content).rstrip() + "\n"
                (bucket / f"idx-{id_to_idx[trial.name]}.md").write_text(content, encoding="utf-8")
                count += 1
    return count


def build_tasks_file(
    dataset: str, dest: Path, questions: list[BenchmarkQuestion] | None = None,
    *, allow_prompt_fallback: bool = False,
) -> int:
    """Reconstruct the official judge input; refuse silent judge-task substitution."""
    rows = []
    for q in questions if questions is not None else load_questions(dataset):
        gt = json.loads(q.ground_truth)
        if not isinstance(gt, dict) or "rubric" not in gt:
            raise ValueError(f"Question {q.id} has no official rubric")
        judge_task = q.metadata.get("judge_task")
        if not judge_task:
            if not allow_prompt_fallback:
                raise ValueError(f"Question {q.id} has no judge_task; supply official content.task")
            logger.warning("Question %s: using solver prompt as nonofficial judge context", q.id)
            judge_task = q.question
        rows.append({
            "idx": int(q.metadata["idx"]), "language": q.metadata.get("language", ""),
            "theme": q.metadata.get("theme", ""),
            "content": {"task": judge_task, "rubric": gt["rubric"],
                        "blocked": gt.get("blocked", {})},
        })
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return len(rows)


def official_commands(
    official_dir: Path, bundle_dir: Path, *, python_executable: str = sys.executable,
) -> list[list[str]]:
    """Explicit official CLI interface, using argv arrays (no shell interpolation)."""
    official_dir, bundle_dir = official_dir.resolve(), bundle_dir.resolve()
    return [
        [python_executable, str(official_dir / "run_evaluation.py"),
         "--pdf_dir", str(bundle_dir / "report"),
         "--out_jsonl", str(bundle_dir / "result.jsonl"),
         "--tasks_jsonl", str(bundle_dir / "tasks_and_rubrics.jsonl"),
         "--log_file", str(bundle_dir / "run_evaluation.log")],
        [python_executable, str(official_dir / "aggregate_scores.py"),
         "--input", str(bundle_dir / "result.jsonl"),
         "--output-prefix", str(bundle_dir / "agg_scores"),
         "--tasks-file", str(bundle_dir / "tasks_and_rubrics.jsonl")],
    ]


def export_bundle(
    run_dir: Path, out_dir: Path, *, model_name: str = "dynamic_htn_informed_research",
    dataset: str = "deepresearch_bench_ii", official_dir: Path | None = None,
    allow_prompt_fallback: bool = False,
    benchmark_profile: str | Path | None = None,
) -> dict:
    judge = resolve_benchmark_judge(benchmark_profile, run_dir=run_dir)
    questions = load_run_questions(run_dir, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / "report").glob("*/idx-*.md")):
        raise ValueError(f"Export already contains reports: {out_dir}; choose a fresh bundle directory")
    n_tasks = build_tasks_file(dataset, out_dir / "tasks_and_rubrics.jsonl", questions,
                               allow_prompt_fallback=allow_prompt_fallback)
    n_reports = export_run(run_dir, out_dir / "report", model_name=model_name,
                           dataset=dataset, questions=questions)
    if not n_reports:
        raise ValueError("No nonempty reports exported; nothing to grade")
    manifest = {
        "benchmark": dataset, "status": "not_graded", "reports": n_reports, "tasks": n_tasks,
        "run_dir": str(run_dir.resolve()), "model_name": model_name,
        "judge_context": "prompt_fallback_allowed" if allow_prompt_fallback else "official_task",
        "official_dir_required": official_dir is None,
        "commands": official_commands(official_dir or Path("/path/to/DeepResearch-Bench-II"),
                                      out_dir),
        "benchmark_profile": str(benchmark_profile) if benchmark_profile else "run snapshot or default", "judge": judge,
        "judge_configuration": ["OPENAI_API_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
                                "OPENAI_REASONING_EFFORT"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return manifest


def run_official(official_dir: Path, bundle_dir: Path) -> None:
    """Run an explicitly requested external grading; never convert it to binary reward."""
    commands = official_commands(official_dir, bundle_dir)
    for command in commands:
        if not Path(command[1]).is_file():
            raise FileNotFoundError(f"Official evaluator script missing: {command[1]}")
    if not (official_dir / "gpt_client.py").is_file():
        raise FileNotFoundError("DRB grading requires the GPT evaluator checkout with gpt_client.py")
    result = bundle_dir / "result.jsonl"
    if result.exists():
        raise ValueError(f"Official evaluator appends to {result}; use a fresh bundle")
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Use the exported snapshot, even if the source profile changes afterwards.
    if "judge" not in manifest:
        raise ValueError("Bundle has no saved GPT judge configuration; export a fresh bundle")
    env = official_environment(manifest["judge"])
    subprocess.run(commands[0], check=True, cwd=official_dir, env=env)
    if not result.is_file() or not result.stat().st_size:
        raise RuntimeError("Official evaluator produced no scores; submission remains not_graded")
    subprocess.run(commands[1], check=True, cwd=official_dir)
    manifest.update(status="official_scores_exported", result_jsonl=str(result.resolve()))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("action", choices=["export", "grade"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="dynamic_htn_informed_research")
    parser.add_argument("--dataset", default="deepresearch_bench_ii")
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--benchmark_profile", help="Benchmark profile name/path; defaults to saved run configuration")
    parser.add_argument("--allow-prompt-fallback", action="store_true")
    args = parser.parse_args()
    if args.action == "grade" and args.official_dir is None:
        parser.error("grade requires --official-dir with separately installed official dependencies")
    manifest = export_bundle(args.run_dir, args.out_dir, model_name=args.model_name,
                             dataset=args.dataset, official_dir=args.official_dir,
                             allow_prompt_fallback=args.allow_prompt_fallback,
                             benchmark_profile=args.benchmark_profile)
    if args.action == "grade":
        run_official(args.official_dir, args.out_dir)
        manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
