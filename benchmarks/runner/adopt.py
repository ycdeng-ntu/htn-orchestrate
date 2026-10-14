"""Extend a PRL pilot without duplicating its completed or active questions."""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

from agent_harness.core.atomic_files import atomic_write_text
from benchmarks.core.contracts import is_graded
from benchmarks.core.registry import load_questions


def process_token(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return f"{pid}:{fields[19]}" if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


@dataclass
class Adoption:
    source: Path
    ids: frozenset[str]
    pid: int
    token: str | None


def prepare(source, questions, profile, *, concurrency, benchmark_profile=None):
    from workflows.dynamic_htn_informed_research.profile import load_profile_selection
    source = Path(source).resolve()
    launch = json.loads((source / "launch.json").read_text())
    cfg = json.loads((source / "config.json").read_text())
    if cfg["benchmark"] != "prl_bench" or cfg["pipeline"] != "dynamic_htn_informed_research":
        raise ValueError("Only a matching PRL pilot can be adopted")
    if benchmark_profile is not None:
        from benchmarks.core.profile import snapshot_config
        previous = snapshot_config(json.loads((source / "benchmark_profile.json").read_text()), "prl_bench")
        current = snapshot_config(benchmark_profile, "prl_bench")
        if any(previous[key] != current[key] for key in ("judge", "web_policy")):
            raise ValueError("Pilot benchmark judge/web policy differs from the full run")
    original = load_questions("prl_bench", subset=cfg["subset"], limit=cfg.get("limit"),
                              offset=cfg.get("offset", 0), category=cfg.get("category"))
    selected = {q.id: q for q in questions}
    if any(selected.get(q.id) != q for q in original):
        raise ValueError("Pilot questions differ from the new full cohort")
    configs = []
    for path in (cfg["profile"], profile):
        _, value = load_profile_selection({"profile": str(path)}).select("report_research", "scientific_report")
        configs.append({k: v for k, v in value.items() if k != "evaluation"})
    if configs[0] != configs[1]:
        raise ValueError("Pilot solving/judging settings differ from the full run")
    token = process_token(launch["pid"])
    missing = sum(not (source / "trials" / q.id / "result.json").exists() for q in original)
    if token and missing > concurrency:
        raise ValueError("Pilot already exceeds the requested total concurrency")
    return Adoption(source, frozenset(q.id for q in original), launch["pid"], token)


async def adopt_result(plan, qid, out, sem, *, poll_seconds=5, timeout=30600):
    """An active external worker occupies a slot until its score is committed."""
    source = plan.source / "trials" / qid
    dest = out / "trials" / qid
    async with sem:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_text(dest / "adopted_from.json", json.dumps({
            "source_trial": str(source), "source_runner_pid": plan.pid,
            "status": "waiting", "diagnostics_remain_at_source": True,
        }, indent=2))
        deadline = time.monotonic() + timeout
        while True:
            result_path = source / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result.get("question_id") == qid and not result.get("error")
                        and not result.get("scoring_error") and is_graded(result)):
                    (dest / "agent").mkdir(exist_ok=True)
                    for name in ("final_content.md", "final_answer.txt"):
                        path = source / "agent" / name
                        if path.exists():
                            shutil.copy2(path, dest / "agent" / name)
                    atomic_write_text(dest / "result.json", json.dumps(result, ensure_ascii=False, indent=2))
                    atomic_write_text(dest / "adopted_from.json", json.dumps({
                        "source_trial": str(source), "status": "reused",
                        "diagnostics_remain_at_source": True,
                    }, indent=2))
                    return result
                return None  # Failed pilot result stays intact; new runner retries locally.
            if not plan.token or process_token(plan.pid) != plan.token or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(poll_seconds)
