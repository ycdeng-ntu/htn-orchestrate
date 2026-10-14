"""Grade completed DRB reports while generation continues, using official scripts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks import grade_drb
from agent_harness.core.atomic_files import atomic_write_text


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def official_aggregate_summary(out: Path, model: str, scored: int) -> dict:
    """Read the official aggregate's avg row; never average criteria ourselves."""
    summary = {"mean_score100": None, "scored_reports": scored,
               "source": str(out / "agg_scores_total.csv"), "dimensions": {}}
    for name in ("total", "inforecall", "analysis", "presentation", "blocked"):
        path = out / f"agg_scores_{name}.csv"
        if not path.is_file() or not scored:
            continue
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            averages = [row for row in rows if row.get("idx") == "avg"]
            report_rows = [row for row in rows if row.get("idx") != "avg"]
            if len(averages) != 1 or len(report_rows) != scored:
                raise ValueError("Official aggregate cohort does not match validated scored reports")
            value = float(averages[0][model])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid official aggregate ratio")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary.setdefault("errors", {})[name] = str(exc)
            continue
        summary["dimensions"][name] = {"mean_score100": value * 100,
                                        "source": str(path)}
        if name == "total":
            summary["mean_score100"] = value * 100
    return summary


def refresh_run_summary(run: Path, *, generation_report=None, grading_status=None) -> dict:
    """Publish a joined DRB view without either producer overwriting the other.

    Generation owns generation_results.json and the watcher owns
    pipeline_status.json. Both refresh the shared view under the same lock and
    read the latest committed inputs inside it, including at runner shutdown.
    Per-trial rewards retain the generation-only deferred-scoring contract.
    """
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if generation_report is not None:
            write_json(run / "generation_results.json", generation_report)
        if grading_status is not None:
            write_json(run / "pipeline_status.json", grading_status)

        def read(name):
            path = run / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        # Older runs have only results.json; retain their run metadata when a
        # watcher first upgrades the summary without a concurrent new runner.
        generation = read("generation_results.json") or read("results.json")
        grading = read("pipeline_status.json")
        generation_source = ("generation_results.json" if (run / "generation_results.json").exists()
                             else "trials/*/result.json")
        generation_stage = generation.get("generation_stage", (
            "completed" if grading.get("generation_running") is False else "running"))
        rows = generation.get("results", [])
        if generation.get("generation_stage") != "completed":
            rows = []
            for path in sorted((run / "trials").glob("*/result.json")):
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(row, dict) and row.get("question_id") == path.parent.name:
                        rows.append(row)
                except (OSError, ValueError):
                    continue
        total = generation.get("total", grading.get("total", len(rows)))
        scored = grading.get("scored", 0)
        failed = sum(bool(row.get("error")) for row in rows)
        ready = sum(not row.get("error") and bool(row.get("predicted_answer")) for row in rows)
        report = {**generation, "benchmark": "deepresearch_bench_ii", "total": total,
                  "results": rows, "attempted": ready, "correct": None, "accuracy": None,
                  "graded": scored, "not_graded": total - scored,
                  "by_method": {"official_drb": scored, "not_graded": total - scored},
                  "official_mean_score100": grading.get("aggregate", {}).get("mean_score100"),
                  "generation": {"stage": generation_stage,
                                 "completed": len(rows), "reports_ready": ready, "failed": failed,
                                 "missing": total - len(rows), "source": generation_source},
                  "grading": grading or {"stage": "awaiting_official_grading", "scored": 0,
                                         "grading": 0, "failed": 0},
                  "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        score = report["official_mean_score100"]
        score_text = f"{score:.4f}" if score is not None else "N/A (no official aggregate available)"
        summary = (
            "DeepResearch-Bench-II Results\n"
            f"Pipeline: {generation.get('pipeline', 'unknown')}\n"
            f"Profile: {generation.get('profile', 'unknown')}\n"
            f"Total: {total}\n"
            f"Generation: {report['generation']['stage']}; {ready} reports ready, "
            f"{failed} failed, {total - len(rows)} missing\n"
            f"Official grading: {report['grading']['stage']}; {scored}/{total} scored, "
            f"{report['grading'].get('grading', 0)} running, "
            f"{report['grading'].get('failed', 0)} failed\n"
            f"Official mean score (0-100): {score_text}; denominator: {scored} scored reports\n"
            "Unscored and failed reports are excluded from the official aggregate.\n"
            f"Generation results: {generation_source} (per-trial rewards remain deferred)\n"
            "Live grading status: pipeline_status.json (refreshed by benchmarks.watch_drb)\n"
        )
        if grading.get("aggregate"):
            summary += f"Official aggregate source: {grading['aggregate']['source']}\n"
        write_json(run / "results.json", report)
        atomic_write_text(run / "summary.txt", summary)
        return report


def process_token(pid):
    """Linux PID plus start time avoids adopting an unrelated reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return f"{pid}:{fields[19]}" if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def validated_result(bundle, question, model):
    rows = [json.loads(line) for line in (bundle / "result.jsonl").read_text().splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected exactly one official report result")
    row = rows[0]
    if row.get("idx") != int(question.metadata["idx"]) or row.get("model") != model:
        raise ValueError("Official result identity mismatch")
    result = row.get("result", {})
    if result.get("error"):
        raise ValueError("Official evaluator reported a scoring failure")
    rubric = json.loads(question.ground_truth)["rubric"]
    # The official evaluator maps repeated criterion text to its last dimension.
    dimensions = {item: dim for dim in ("info_recall", "analysis", "presentation")
                  for item in rubric.get(dim, [])}
    scores = result.get("scores", {})
    if not dimensions:
        raise ValueError("Empty official rubric")
    for dim in ("info_recall", "analysis", "presentation"):
        actual = scores.get(dim, {})
        expected = {item for item, owner in dimensions.items() if owner == dim}
        if not isinstance(actual, dict) or set(actual) != expected:
            raise ValueError("Incomplete official criterion coverage")
        if any(type(v.get("score")) not in (int, float) or v["score"] not in (-1, 0, 1)
               for v in actual.values()):
            raise ValueError("Invalid official criterion score")
    return row


class LiveGrader:
    def __init__(self, run, out, official, profile, model, concurrency, generation_pid):
        self.run, self.out, self.official = run.resolve(), out.resolve(), official.resolve()
        self.model, self.concurrency = model, concurrency
        self.generation_pid = generation_pid
        self.children = {}
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ("run_evaluation.py", "aggregate_scores.py", "gpt_client.py"):
            if not (self.official / name).is_file():
                raise FileNotFoundError(name)
        self.questions = grade_drb.load_run_questions(self.run, "deepresearch_bench_ii")
        manifest_path = self.out / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            if (self.manifest["run_dir"], self.manifest["model_name"], self.manifest["official_dir"]) != (
                    str(self.run), model, str(self.official)):
                raise ValueError("Existing live grading belongs to a different run/evaluator")
        else:
            self.manifest = dict(run_dir=str(self.run), model_name=model,
                                 official_dir=str(self.official), mode="incremental",
                                 judge=grade_drb.resolve_benchmark_judge(profile, run_dir=self.run),
                                 generation_token=process_token(generation_pid),
                                 tasks=len(self.questions))
            write_json(manifest_path, self.manifest)
        self.env = grade_drb.official_environment(self.manifest["judge"])
        self.env.update(MAX_WORKERS="1", PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        grade_drb.build_tasks_file("deepresearch_bench_ii", self.out / "tasks_and_rubrics.jsonl",
                                  questions=self.questions)
        self.records = {}
        for q in self.questions:
            path = self.bundle(q) / "status.json"
            if path.exists():
                self.records[q.id] = json.loads(path.read_text())
        self.last_merged = None

    def bundle(self, q):
        return self.out / "items" / f"idx-{int(q.metadata['idx'])}"

    def record(self, q, **value):
        self.records[q.id] = value
        write_json(self.bundle(q) / "status.json", value)

    def ready(self, q):
        trial = self.run / "trials" / q.id
        try:
            result = json.loads((trial / "result.json").read_text())
        except (OSError, ValueError):
            return False
        return (str(result.get("question_id")) == q.id and not result.get("error")
                and bool(grade_drb._read_answer(trial)))

    def start(self, q):
        bundle = self.bundle(q)
        bundle.mkdir(parents=True, exist_ok=True)
        if (bundle / "result.jsonl").exists():
            self.record(q, status="failed", error="Untracked existing result; manual review required")
            return
        grade_drb.build_tasks_file("deepresearch_bench_ii", bundle / "tasks_and_rubrics.jsonl", [q])
        count = grade_drb.export_run(self.run, bundle / "report", model_name=self.model, questions=[q])
        if count != 1:
            raise ValueError(f"Expected one completed report for {q.id}")
        command = grade_drb.official_commands(self.official, bundle)[0]
        report = bundle / "report" / self.model / f"idx-{q.metadata['idx']}.md"
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        # Save intent before spawning. Ambiguous interrupted launches are not retried automatically.
        self.record(q, status="launching", report_sha256=digest)
        with (bundle / "launcher.log").open("a") as log:
            child = subprocess.Popen(command, cwd=self.official, env=self.env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.children[q.id] = child
        self.record(q, status="running", pid=child.pid, token=process_token(child.pid),
                    report_sha256=digest)

    def step(self):
        for q in self.questions:
            state = self.records.get(q.id, {})
            if state.get("status") not in ("running", "launching"):
                continue
            child = self.children.get(q.id)
            alive = child.poll() is None if child else (
                bool(state.get("token")) and process_token(state.get("pid")) == state["token"])
            if alive:
                continue
            try:
                validated_result(self.bundle(q), q, self.model)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.record(q, **{**state, "status": "failed", "error": str(exc)})
            else:
                self.record(q, **{**state, "status": "scored"})
        active = sum(r["status"] == "running" for r in self.records.values())
        for q in self.questions:
            if active >= self.concurrency:
                break
            if q.id not in self.records and self.ready(q):
                self.start(q)
                active += self.records[q.id]["status"] == "running"
        scored = [q for q in self.questions if self.records.get(q.id, {}).get("status") == "scored"]
        signature = tuple(q.id for q in scored)
        if signature != self.last_merged:
            rows = [validated_result(self.bundle(q), q, self.model) for q in scored]
            temporary = self.out / "result.jsonl.tmp"
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            temporary.replace(self.out / "result.jsonl")
            if rows:
                command = grade_drb.official_commands(self.official, self.out)[1]
                with (self.out / "aggregate.log").open("a") as log:
                    subprocess.run(command, check=True, cwd=self.official, stdout=log,
                                   stderr=subprocess.STDOUT)
            self.last_merged = signature
        producer_alive = (bool(self.manifest["generation_token"]) and
                          process_token(self.generation_pid) == self.manifest["generation_token"])
        counts = Counter(r["status"] for r in self.records.values())
        finished = not producer_alive and not active
        status = dict(mode="incremental", stage="completed" if finished else "watching",
                      pid=os.getpid(), generation_pid=self.generation_pid,
                      generation_running=producer_alive, total=len(self.questions),
                      reports_ready=sum(self.ready(q) for q in self.questions),
                      scored=counts["scored"], grading=counts["running"], failed=counts["failed"],
                      concurrency=self.concurrency, judge=self.manifest["judge"],
                      model_name=self.model, grading_dir=str(self.out),
                      official_results=str(self.out / "result.jsonl"),
                      aggregate=official_aggregate_summary(self.out, self.model, counts["scored"]),
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        if finished and (counts["failed"] or counts["scored"] != len(self.questions)):
            status["stage"] = "completed_with_missing_or_failed_reports"
        write_json(self.out / "status.json", status)
        refresh_run_summary(self.run, grading_status=status)
        return finished


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--benchmark_profile", help="Defaults to saved run configuration")
    parser.add_argument("--generation-pid", type=int, required=True)
    parser.add_argument("--model-name", default="dynamic_htn_informed_research_deepseek_v4_1_flash")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.concurrency < 1 or args.poll_seconds <= 0:
        parser.error("concurrency and poll-seconds must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        grader = LiveGrader(args.run_dir, args.out_dir, args.official_dir,
                            args.benchmark_profile, args.model_name, args.concurrency,
                            args.generation_pid)
        while not grader.step():
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
