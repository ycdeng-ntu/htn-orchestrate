"""Dataset registry and loader for benchmarks.

Each dataset is a JSONL file with a known schema. Attachments (images, files)
live alongside the JSONL in the same data directory.

Adding a new benchmark:
    1. Put standardized_data.jsonl + attachments under benchmarks/datasets/<KEY>/
    2. Add an entry under benchmarks/families/<family>.py CONFIGS
    3. (Optional) Register a judge in benchmarks/judges/<family>.py
       and benchmarks/judges/__init__.py:JUDGE_REGISTRY
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from benchmarks.core.question import BenchmarkQuestion  # noqa: F401

logger = logging.getLogger(__name__)

# Single root for all eval inputs. Run artefacts land in the per-run
# directory passed via ``--out`` (the bundled scripts default to
# ``./results/<benchmark>``). Each registered benchmark resolves to
# ``_DATA_ROOT / <key>/``.
_DATA_ROOT = Path("benchmarks/datasets")


# ── Registry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Declarative description of one benchmark dataset."""

    name: str               # Human-readable ("HLE", "GAIA")
    key: str                # Directory name under data_root/
    jsonl: str = "standardized_data.jsonl"
    extra_metadata_fields: tuple[str, ...] = ()
    # Which JSONL fields hold attachment paths
    image_field: str = "image_path"
    file_field: str = ""          # e.g. "file_path" for GAIA
    file_name_field: str = ""     # e.g. "file_name" for GAIA
    default_pipeline: str = "react_base"
    # Override the global ``_DATA_ROOT`` (e.g. workflow-local data/).
    data_root: str = ""
    # Field-name overrides for non-standard column names.
    id_field: str = "id"
    question_field: str = "question"
    answer_field: str = "answer"
    default_answer_type: str = "exactMatch"


# REGISTRY is built by per-family modules under benchmarks/families/.
# Imported lazily here to break the circular dependency:
# families/<x>.py needs DatasetConfig from this module, so this module
# must finish defining DatasetConfig before triggering families load.

def _load_registry() -> dict[str, DatasetConfig]:
    from benchmarks.families import REGISTRY as _r
    return _r


REGISTRY: dict[str, DatasetConfig] = _load_registry()


# ── Loader ───────────────────────────────────────────────────────────────


def load_questions(
    dataset: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    answer_type: str | None = None,
    category: str | None = None,
    subset: str | None = None,
) -> list[BenchmarkQuestion]:
    """Load questions from a registered dataset.

    Args:
        dataset: Registry key ("browsecomp", "widesearch", ...).
        limit: Max questions to return (None = all).
        offset: Skip this many questions at the start.
        answer_type: Filter by answer_type field.
        category: Filter by category field.
        subset: Override JSONL path (for custom splits).

    Returns:
        List of BenchmarkQuestion.
    """
    cfg = REGISTRY.get(dataset)
    if cfg is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {available}"
        )

    root = Path(cfg.data_root) if cfg.data_root else _DATA_ROOT
    jsonl_path = Path(subset) if subset else root / cfg.key / cfg.jsonl
    if not jsonl_path.exists():
        hint = f"Place the JSONL at {jsonl_path}."
        if dataset == "widesearch":
            base = jsonl_path.parent / "standardized_data.jsonl"
            if base.exists():
                hint = (
                    f"Found {base.name} but not {jsonl_path.name}. "
                    f"WideSearch needs the gold_table baked into the JSONL."
                )
        raise FileNotFoundError(f"Dataset not found at {jsonl_path}. {hint}")

    questions: list[BenchmarkQuestion] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < offset:
                continue
            row = json.loads(line)

            row_answer_type = row.get("answer_type") or cfg.default_answer_type
            if answer_type and row_answer_type != answer_type:
                continue
            if category and row.get("category", "") != category:
                continue

            metadata = {"category": row.get("category", "")}
            for field_name in cfg.extra_metadata_fields:
                metadata[field_name] = row.get(field_name, "")

            # Attachment paths
            img = row.get(cfg.image_field)
            img = img if img and img != "None" else None

            fp = row.get(cfg.file_field) if cfg.file_field else None
            fp = fp if fp else None

            fn = row.get(cfg.file_name_field) if cfg.file_name_field else None
            fn = fn if fn else None

            questions.append(BenchmarkQuestion(
                id=str(row[cfg.id_field]),
                question=row[cfg.question_field],
                ground_truth=(row[cfg.answer_field] if isinstance(row[cfg.answer_field], str)
                              else json.dumps(row[cfg.answer_field], ensure_ascii=False)),
                answer_type=row_answer_type,
                image_path=img,
                file_path=fp,
                file_name=fn,
                metadata=metadata,
            ))

            if limit is not None and len(questions) >= limit:
                break

    logger.info(
        "Loaded %d %s questions (offset=%d, limit=%s)",
        len(questions), cfg.name, offset, limit,
    )
    return questions


def get_config(dataset: str) -> DatasetConfig:
    """Get dataset config by key. Raises ValueError if not found."""
    cfg = REGISTRY.get(dataset)
    if cfg is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {available}"
        )
    return cfg
