"""Harbor task directory generator.

Converts benchmark JSONL datasets into Harbor task directories.
Each question becomes a directory with instruction.md, task.toml, and tests/test.sh.

Usage:
    uv run python -m benchmarks.harbor_task_generator \\
        --benchmark_profile browsecomp \\
        --out benchmarks/tasks-generated \\
        --verifier-image agent-harness/verifier:v1 \\
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.core.contracts import solver_metadata

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TEST_SH = "#!/bin/bash\nset -euo pipefail\npython /opt/score.py\n"

_DEFAULT_PIPELINE = "react_base"
_DEFAULT_VERIFIER_IMAGE = "agent-harness/verifier:v1"


# ── TOML helpers ──────────────────────────────────────────────────────────────


def _escape_toml_string(s: str) -> str:
    """Escape special characters for TOML double-quoted string values.

    Handles backslash, double-quote, and common control characters.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_task_toml(
    question_id: str,
    ground_truth: str,
    answer_type: str,
    pipeline_id: str,
    verifier_image: str,
    image_path: str | None,
    category: str,
    extra_metadata: dict[str, str] | None = None,
) -> str:
    """Render a task.toml string for a single Harbor task.

    extra_metadata is appended to the [metadata] section verbatim — used for
    pipeline-specific fields (e.g. SWE's instance_id, repo_name, base_commit).
    """
    escaped_gt = _escape_toml_string(ground_truth)
    escaped_answer_type = _escape_toml_string(answer_type)
    escaped_pipeline = _escape_toml_string(pipeline_id)
    escaped_category = _escape_toml_string(category)
    escaped_qid = _escape_toml_string(question_id)

    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "{escaped_pipeline}/{escaped_qid}"',
        f'description = "Benchmark task {escaped_qid}"',
        'authors = [{name = "AgentHarness"}]',
        f'keywords = ["{escaped_pipeline}", "benchmark"]',
        "",
        "[environment]",
        f'docker_image = "{_escape_toml_string(verifier_image)}"',
        "cpus = 1",
        "memory_mb = 1024",
        "",
        "[agent]",
        "timeout_sec = 7200",
        "",
        "[verifier]",
        "timeout_sec = 120",
        "",
        "[verifier.env]",
        f'GROUND_TRUTH = "{escaped_gt}"',
        f'ANSWER_TYPE = "{escaped_answer_type}"',
        "",
        "[metadata]",
        f'question_id = "{escaped_qid}"',
        f'answer_type = "{escaped_answer_type}"',
        f'pipeline_id = "{escaped_pipeline}"',
        f'category = "{escaped_category}"',
    ]

    if image_path is not None:
        lines.append(f'image_path = "{_escape_toml_string(image_path)}"')

    if extra_metadata:
        for key, value in solver_metadata(extra_metadata).items():
            if key in {"question_id", "answer_type", "pipeline_id", "category", "image_path"}:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f'{key} = "{_escape_toml_string(str(value))}"')

    lines.append("")
    return "\n".join(lines)


def _render_instruction(question: str, image_path: str | None) -> str:
    """Render instruction.md for a single Harbor task."""
    lines = ["# Task", "", question, ""]
    if image_path is not None:
        lines += [
            "## Attached Image",
            "",
            f"Image path: `{image_path}`",
            "",
        ]
    return "\n".join(lines)


# ── Main generator ─────────────────────────────────────────────────────────────


def generate_task_dirs(
    questions: list[dict[str, Any]],
    output_dir: Path,
    *,
    verifier_image: str = _DEFAULT_VERIFIER_IMAGE,
    pipeline_id: str = _DEFAULT_PIPELINE,
) -> None:
    """Write Harbor task directories for a list of questions.

    Uses atomic write: builds all files in a temporary directory, then
    replaces output_dir in a single rename operation.

    Args:
        questions: List of question dicts with keys: id, question, answer,
                   answer_type, category, image_path.
        output_dir: Destination directory (will be replaced atomically).
        verifier_image: Docker image tag for the verifier container.
        pipeline_id: AgentHarness pipeline to use for evaluation.
    """
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp dir in the same filesystem for atomic rename.
    with tempfile.TemporaryDirectory(dir=parent, prefix=".harbor_tmp_") as tmp_str:
        tmp_dir = Path(tmp_str)

        for q in questions:
            qid = str(q["id"])
            question_text = q["question"]
            answer = q.get("answer", "")
            ground_truth = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            answer_type = q.get("answer_type", "exactMatch")
            category = q.get("category", "")
            image_path: str | None = q.get("image_path")

            task_dir = tmp_dir / qid
            tests_dir = task_dir / "tests"
            tests_dir.mkdir(parents=True)

            # instruction.md
            (task_dir / "instruction.md").write_text(
                _render_instruction(question_text, image_path), encoding="utf-8"
            )

            # task.toml — extra_metadata flows through pipeline-specific fields
            # (e.g. SWE: instance_id, repo_name, base_commit, hints_text, swe_profile)
            extra_meta = {
                k: v for k, v in solver_metadata(q.get("metadata")).items()
                if k != "category" and v
            }
            extra_meta.pop("benchmark_web_policy", None)
            if q.get("benchmark_profile") is not None:
                from benchmarks.core.profile import snapshot_config
                policy = snapshot_config(q["benchmark_profile"])["web_policy"]
                restricted = (not policy["network_enabled"] or policy["blocked_domains"] or policy["blocked_terms"])
                if not policy["network_enabled"] and pipeline_id != "dynamic_htn_informed_research":
                    raise ValueError(f"{pipeline_id} does not support benchmark-owned offline execution")
                if restricted and pipeline_id not in {
                    "dynamic_htn_informed_research",
                }:
                    raise ValueError(f"{pipeline_id} does not support benchmark web_policy; select a supported pipeline")
                extra_meta["benchmark_web_policy"] = policy
            for field in ("file_path", "file_name"):
                if q.get(field):
                    extra_meta[field] = q[field]
            (task_dir / "task.toml").write_text(
                _render_task_toml(
                    question_id=qid,
                    ground_truth=ground_truth,
                    answer_type=answer_type,
                    pipeline_id=pipeline_id,
                    verifier_image=verifier_image,
                    image_path=image_path,
                    category=category,
                    extra_metadata=extra_meta,
                ),
                encoding="utf-8",
            )

            # tests/test.sh
            test_sh_path = tests_dir / "test.sh"
            test_sh_path.write_text(TEST_SH, encoding="utf-8")
            test_sh_path.chmod(0o755)

            # Verifier-only snapshot: never part of instruction.md or solver
            # metadata. Workers reuse this exact row rather than reload a
            # potentially different dataset/subset after task generation.
            (tests_dir / "evaluation.json").write_text(json.dumps({
                "id": qid, "question": question_text,
                "ground_truth": ground_truth, "answer_type": answer_type,
                "category": category,
                "metadata": q.get("evaluation_metadata") or q.get("metadata") or {},
                "judge_variant": q.get("judge_variant"),
                "benchmark_profile": q.get("benchmark_profile"),
                "benchmark": q.get("benchmark"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Copy to staging dir outside the temp context before it gets deleted.
        staging = parent / f".harbor_stage_{output_dir.name}"
        shutil.copytree(tmp_str, str(staging))

    # Now tmp context is closed; rename staging to final destination.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    logger.info("Generated %d task dirs in %s", len(questions), output_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False,
        description="Convert HLE/GAIA JSONL dataset into Harbor task directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark_profile", required=True,
                   help="Benchmark profile name or YAML path")
    p.add_argument(
        "--out",
        default="benchmarks/tasks-generated",
        help="Output directory for generated task dirs.",
    )
    p.add_argument(
        "--pipeline",
        default=None,
        help="AgentHarness pipeline ID (default: from dataset config).",
    )
    p.add_argument(
        "--verifier-image",
        dest="verifier_image",
        default=_DEFAULT_VERIFIER_IMAGE,
        help="Docker image tag for the Harbor verifier container.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (used for reproducible shuffles if added later).",
    )
    return p


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from benchmarks.core.registry import load_questions, get_config

    from benchmarks.core.profile import load_benchmark_profile
    selected = load_benchmark_profile(args.benchmark_profile)
    args.benchmark = selected.benchmark
    cfg = get_config(args.benchmark)
    pipeline = args.pipeline or cfg.default_pipeline

    questions_obj = load_questions(
        args.benchmark,
        **selected.configuration["dataset"],
    )

    # Convert BenchmarkQuestion dataclass instances to plain dicts.
    questions = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.ground_truth,
            "answer_type": q.answer_type,
            "category": q.metadata.get("category", ""),
            "image_path": q.image_path,
            "file_path": q.file_path,
            "file_name": q.file_name,
            "metadata": q.metadata,
            "evaluation_metadata": q.metadata,
            "benchmark_profile": selected.snapshot(),
            "judge_variant": selected.configuration["judge"].get("variant"),
            "benchmark": args.benchmark,
        }
        for q in questions_obj
    ]

    logger.info(
        "Loaded %d questions from %s benchmark", len(questions), cfg.name,
    )

    out = Path(args.out)
    generate_task_dirs(
        questions,
        out,
        verifier_image=args.verifier_image,
        pipeline_id=pipeline,
    )

    print(f"Generated {len(questions)} task dirs → {out.resolve()}")


if __name__ == "__main__":
    main()
