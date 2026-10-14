"""Download a pinned official PRL-Bench release and preserve solver/judge boundaries."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "c656d7f98528518065e6820260b9354a164fa7fc"
REPOSITORY = "AdrianMiao/PRL_Bench"
PROTOCOL = "local_equal_criterion_binary_v1"


def standardize(rows, revision=REVISION):
    output, ids = [], set()
    for row in rows:
        qid = str(row["id"])
        if qid in ids or not qid.isdecimal():
            raise ValueError("PRL paper IDs must be unique nonnegative integers")
        ids.add(qid)
        tasks = row["tasks"]
        task_ids = [str(t["id"]) for t in tasks]
        rubric_ids = [str(r["task_id"]) for r in row["rubrics"]]
        if not tasks or len(set(task_ids)) != len(tasks) or len(set(rubric_ids)) != len(rubric_ids):
            raise ValueError(f"Duplicate or empty subtask IDs in {qid}")
        if set(task_ids) != set(rubric_ids):
            raise ValueError(f"Subtask/rubric mapping mismatch in {qid}")
        rubrics = {str(r["task_id"]): r["rubrics"] for r in row["rubrics"]}
        criteria = []
        for task in tasks:
            if not isinstance(task["prompt"], str) or not task["prompt"].strip():
                raise ValueError(f"Empty task prompt in {qid}")
            items = rubrics[str(task["id"])]
            if not isinstance(items, list) or not items or any(not isinstance(s, str) or not s.strip() for s in items):
                raise ValueError(f"Invalid rubric in {qid}")
            criteria.extend({"id": f"{task['id']}:{i}", "task_id": str(task["id"]), "criterion": s}
                            for i, s in enumerate(items, 1))
        question = (f"# {row['title']}\n\n{row['introduction']}\n\n" +
                    "\n\n".join(f"## Task {t['id']}\n\n{t['prompt']}" for t in tasks) +
                    "\n\n## Evaluation environment\n"
                    "Solve all subtasks and submit one complete scientific report with derivations, "
                    "code, numerical outputs and explanations of requested figures. A Python code "
                    "interpreter is available. Web search and external information retrieval are "
                    "disabled; do not retrieve the source paper or benchmark answers. Clearly "
                    "distinguish computations actually executed from unexecuted proposed code.\n")
        output.append(dict(task_id=qid, task_question=question, answer_type="report",
                           category=row["subfield"], subfield=row["subfield"],
                           source_revision=revision, scoring_protocol=PROTOCOL,
                           ground_truth={"protocol": PROTOCOL, "criteria": criteria}))
    return output


def prepare(dest, source=None, revision=REVISION):
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/prl_bench.jsonl"
    raw = Path(source).read_bytes() if source else urllib.request.urlopen(url, timeout=60).read()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    converted = standardize(rows, revision)
    if len(converted) != 100:
        raise ValueError("Expected the complete 100-problem public release")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "official_prl_bench.jsonl").write_bytes(raw)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted)
    (dest / "standardized_data.jsonl").write_text(data)
    manifest = dict(repository=REPOSITORY, revision=revision, source_url=url, license="MIT",
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    standardized_sha256=hashlib.sha256(data.encode()).hexdigest(),
                    problems=len(rows), subtasks=sum(len(r["tasks"]) for r in rows),
                    criteria=sum(len(r["ground_truth"]["criteria"]) for r in converted),
                    subfields=dict(Counter(r["subfield"] for r in rows)),
                    scoring_protocol=PROTOCOL, official_score_reproduction=False,
                    limitation="Public release lacks judge prompt, answer weights and evaluator; local equal-criterion scoring is not the paper leaderboard protocol.")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # A reproducible initial sample: first problem in each released subfield.
    sample = list({r["subfield"]: next(q for q in converted if q["subfield"] == r["subfield"])
                   for r in converted}.values())
    (dest / "smoke6.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "datasets/PRL-Bench")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.source), indent=2))


if __name__ == "__main__":
    main()
