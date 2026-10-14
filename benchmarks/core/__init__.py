"""Benchmark-agnostic infrastructure.

Lightweight re-exports only. ``kernel_adapter`` pulls in the LLM
runtime, so import it from its submodule path directly.
"""

from benchmarks.core.question import BenchmarkQuestion
from benchmarks.core.registry import (
    REGISTRY,
    DatasetConfig,
    get_config,
    load_questions,
)

__all__ = [
    "BenchmarkQuestion",
    "DatasetConfig",
    "REGISTRY",
    "get_config",
    "load_questions",
]
