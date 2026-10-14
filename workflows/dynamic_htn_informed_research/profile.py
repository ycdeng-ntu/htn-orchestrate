"""Declarative profiles for the Dynamic HTN workflow.

Profiles keep model names and runtime budgets in version control while provider
credentials remain in ``.env``.  Profiles are selected per task; importing this module never changes environment
variables, shared clients, or the selected workflow strategy.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def active_profile_name() -> str:
    """Return the explicitly selected profile name, or an empty string."""
    return os.getenv("DYNAMIC_HTN_INFORMED_RESEARCH_PROFILE", "").strip()


def profile_path(name: str) -> Path:
    """Built-in names resolve under profiles/; explicit paths resolve from cwd."""
    if not isinstance(name, str) or not name.strip() or name in {".", ".."}:
        raise ValueError("profile must be a nonempty name or YAML path")
    path = Path(name).expanduser()
    if path.name == name and path.suffix not in {".yaml", ".yml"}:
        path = _PROFILES_DIR / f"{name}.yaml"
    return path.resolve()


def _load_profile_path(path: Path) -> tuple[dict[str, Any], str]:
    import yaml
    from agent_harness.infra.config import _resolve_env_vars

    content = path.read_bytes()
    raw = yaml.safe_load(content.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"profile must be a mapping: {path}")
    _validate_solver_fields(raw)
    return _resolve_env_vars(raw), hashlib.sha256(content).hexdigest()


def _validate_solver_fields(profile: dict[str, Any]) -> None:
    forbidden = set(profile) & {"benchmark", "benchmark_profile", "dataset", "evaluation", "judge", "web_policy", "benchmark_web_policy"}
    web = profile.get("web") or {}
    if isinstance(web, dict) and set(web) & {"blocked_domains", "blocked_terms", "network_enabled"}:
        raise ValueError("Web access restrictions belong in --benchmark_profile web_policy")
    if forbidden:
        raise ValueError(f"Solver profile contains benchmark settings {sorted(forbidden)}; use --benchmark_profile")


def load_profile(name: str) -> dict[str, Any]:
    """Load a built-in profile name or an explicit YAML path."""
    return _load_profile_path(profile_path(name))[0]


def load_active_profile() -> dict[str, Any]:
    """Load the selected profile; preserve legacy env-only runs when unset."""
    name = active_profile_name()
    if not name:
        return {}
    profile = load_profile(name)
    logger.info("dynamic_htn_informed_research profile: %s", name)
    return profile


def resolve_reasoning_efforts(
    profile: dict[str, Any],
    *,
    fallback: str | None = None,
) -> dict[str, str | None]:
    """Resolve per-call reasoning efforts with global-LLM inheritance.

    Missing role entries preserve the historical behaviour by inheriting
    ``llm.extra_body.reasoning_effort``.  Planner routing is epoch-aware because
    a semantic rebuild must remain more deliberative than initial planning.
    """
    llm = profile.get("llm") if isinstance(profile, dict) else None
    llm = llm if isinstance(llm, dict) else {}
    extra_body = llm.get("extra_body")
    extra_body = extra_body if isinstance(extra_body, dict) else {}
    inherited = extra_body.get("reasoning_effort", fallback)

    section = profile.get("reasoning_effort") if isinstance(profile, dict) else None
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("reasoning_effort profile section must be a mapping")
    planner = section.get("planner")
    if planner is None:
        planner = {}
    if not isinstance(planner, dict):
        raise ValueError("reasoning_effort.planner must be a mapping")

    def resolve(raw: Any, path: str) -> str | None:
        value = inherited if raw is None or str(raw).strip().lower() == "inherit" else raw
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip().lower()
        if normalized not in _REASONING_EFFORTS:
            allowed = ", ".join(sorted(_REASONING_EFFORTS))
            raise ValueError(f"{path} must be one of {allowed} or inherit, got {value!r}")
        return normalized

    return {
        "worker": resolve(section.get("worker"), "reasoning_effort.worker"),
        "coordinator": resolve(section.get("coordinator"), "reasoning_effort.coordinator"),
        "instruction_generation": resolve(
            section.get("instruction_generation"),
            "reasoning_effort.instruction_generation",
        ),
        "leaf_synthesis": resolve(section.get("leaf_synthesis"), "reasoning_effort.leaf_synthesis"),
        "tree_synthesis": resolve(section.get("tree_synthesis"), "reasoning_effort.tree_synthesis"),
        "planner_epoch_0": resolve(planner.get("epoch_0"), "reasoning_effort.planner.epoch_0"),
        "planner_rebuild": resolve(planner.get("rebuild"), "reasoning_effort.planner.rebuild"),
        "planner_audit": resolve(planner.get("audit"), "reasoning_effort.planner.audit"),
        "planner_revision": resolve(planner.get("revision"), "reasoning_effort.planner.revision"),
        "recovery": resolve(section.get("recovery"), "reasoning_effort.recovery"),
    }


def sampling_with_reasoning_effort(
    sampling: dict[str, Any],
    effort: str | None,
) -> dict[str, Any]:
    """Return an isolated sampling copy with the requested effort applied."""
    resolved = deepcopy(sampling)
    extra_body = resolved.get("extra_body")
    extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
    if effort is None:
        extra_body.pop("reasoning_effort", None)
    else:
        normalized = str(effort).strip().lower()
        if normalized not in _REASONING_EFFORTS:
            allowed = ", ".join(sorted(_REASONING_EFFORTS))
            raise ValueError(f"reasoning effort must be one of {allowed}, got {effort!r}")
        extra_body["reasoning_effort"] = normalized
    if extra_body:
        resolved["extra_body"] = extra_body
    else:
        resolved.pop("extra_body", None)
    return resolved


def merge_profile(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge without sharing mutable configuration with another task."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_profile(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _selection_name(metadata: dict[str, Any]) -> str:
    return str(metadata.get("profile") or active_profile_name() or "auto")


def _task_overrides(metadata: dict[str, Any]) -> dict[str, Any]:
    override = metadata.get("dynir_config", {})
    if not isinstance(override, dict):
        raise ValueError("metadata.dynir_config must be a mapping")
    return deepcopy(override)


def resolve_task_profile(metadata: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Compatibility loader for a single execution profile, without routing."""
    metadata = metadata or {}
    name = _selection_name(metadata)
    profile = load_profile(name)
    if "routes" in profile or "router" in profile:
        raise ValueError("Meta profiles require load_profile_selection and a task route")
    overrides = _task_overrides(metadata)
    _validate_solver_fields(overrides)
    return name, merge_profile(profile, overrides)


_ROUTE_KEYS = ("research", "science_olympiad", "science_mcq", "report_research")
_SECTIONS = ("llm", "agent", "planner", "reasoning_effort", "web", "summary_llm", "behavior")


def _validate_execution_profile(profile: dict[str, Any], path: Path, route_key: str | None = None) -> None:
    if "router" in profile or "routes" in profile:
        raise ValueError(f"Nested meta profiles are not supported: {path}")
    for section in _SECTIONS:
        if section in profile and not isinstance(profile[section], dict):
            raise ValueError(f"{path}: {section} must be a mapping")
    llm = profile.get("llm") or {}
    for key in ("provider", "model"):
        if not isinstance(llm.get(key), str) or not llm[key].strip():
            raise ValueError(f"{path}: llm.{key} must be a nonempty string")
    if "extra_body" in llm and not isinstance(llm["extra_body"], dict):
        raise ValueError(f"{path}: llm.extra_body must be a mapping")
    for section, keys in {
        "agent": ("max_depth", "max_children", "max_rounds", "max_agents_per_round", "max_iterations",
                  "subagent_max_turns", "max_total_workers", "max_rebuilds", "question_timeout_s",
                  "leaf_timeout_s", "finalization_reserve_s", "fan_in_grace_s", "answer_format_timeout_s",
                  "worker_token_budget", "keep_tool_results"),
        "behavior": ("router_timeout_s", "router_max_tokens"),
        "llm": ("max_tokens", "high_effort_timeout_s"),
    }.items():
        for key in keys:
            if key not in (profile.get(section) or {}):
                continue
            value = profile[section][key]
            if isinstance(value, str) and value.strip().lstrip("+-").isascii() and value.strip().lstrip("+-").isdecimal():
                value = int(value)
            if type(value) is not int:
                raise ValueError(f"{path}: {section}.{key} must be an integer")
            minimum = -1 if key == "keep_tool_results" else 0 if key in {
                "max_depth", "max_total_workers", "max_rebuilds", "finalization_reserve_s",
                "fan_in_grace_s", "worker_token_budget", "high_effort_timeout_s",
            } else 1
            if value < minimum:
                requirement = "positive" if minimum == 1 else f">= {minimum}"
                raise ValueError(f"{path}: {section}.{key} must be {requirement}")
    resolve_reasoning_efforts(profile)
    strategy = (profile.get("behavior") or {}).get("strategy")
    expected = route_key
    if expected is not None and strategy not in (None, "auto", expected):
        raise ValueError(f"{path}: behavior.strategy {strategy!r} conflicts with route {route_key!r}")


@dataclass(frozen=True)
class ProfileSelection:
    """Eagerly loaded profiles; select() always returns a task-owned deep copy."""
    name: str
    path: Path
    initial_profile: dict[str, Any]
    routes: dict[str, dict[str, Any]]
    paths: dict[str, Path]
    hashes: dict[str, str]
    overrides: dict[str, Any]
    router_path: Path | None = None

    @property
    def is_meta(self) -> bool:
        return self.router_path is not None

    def select(self, strategy_name: str) -> tuple[str, dict[str, Any]]:
        if strategy_name not in _ROUTE_KEYS:
            raise ValueError(f"Unmapped task route: {strategy_name!r}")
        if not self.is_meta:
            return self.name, deepcopy(self.initial_profile)
        key = strategy_name
        if key not in self.routes:
            raise ValueError(f"Missing profile for task route: {key}")
        profile = merge_profile(self.routes[key], self.overrides)
        _validate_execution_profile(profile, self.paths[key], key)
        # Route identity is authoritative; execution profiles never reroute.
        profile.setdefault("behavior", {})["strategy"] = strategy_name
        return str(self.paths[key]), profile

    def snapshot(self) -> dict[str, Any]:
        return {
            "selection": self.name, "path": str(self.path),
            "router_profile": str(self.router_path) if self.router_path else None,
            "routes": {key: str(path) for key, path in self.paths.items()},
            "file_sha256": dict(self.hashes),
            "router_configuration": deepcopy(self.initial_profile) if self.is_meta else None,
            "execution_profiles": deepcopy(self.routes), "overrides": deepcopy(self.overrides),
        }


def load_profile_selection(metadata: dict[str, Any] | None = None) -> ProfileSelection:
    """Validate and preload the complete route table before any model call.

    All paths in a meta profile are relative to that file. Single execution
    profiles retain their existing explicit/auto behavior for compatibility.
    Task overrides apply only after choosing an execution profile, never to the
    router. No global cache or mutable process configuration is installed.
    """
    metadata = metadata or {}
    name = _selection_name(metadata)
    path = profile_path(name)
    raw, digest = _load_profile_path(path)
    overrides = _task_overrides(metadata)
    _validate_solver_fields(overrides)
    hashes = {str(path): digest}
    if "routes" not in raw and "router" not in raw:
        return ProfileSelection(name, path, merge_profile(raw, overrides), {}, {}, hashes, overrides)
    if set(raw) != {"router", "routes"}:
        raise ValueError("Meta profile must contain exactly router and routes")
    router = raw["router"]
    if not isinstance(router, dict) or set(router) != {"profile"}:
        raise ValueError("Meta profile router must contain exactly profile")
    table = raw["routes"]
    if not isinstance(table, dict) or set(table) != {"research", "science_olympiad", "science_mcq", "report_research"}:
        raise ValueError("Meta profile routes must map research, science_olympiad, science_mcq, report_research")
    def read(reference: Any) -> tuple[Path, dict[str, Any]]:
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Profile references must be nonempty YAML paths")
        target = (path.parent / Path(reference).expanduser()).resolve()
        profile, sha = _load_profile_path(target)
        hashes[str(target)] = sha
        return target, profile

    router_path, router_profile = read(router["profile"])
    _validate_execution_profile(router_profile, router_path)
    if (router_profile.get("behavior") or {}).get("strategy") not in (None, "auto"):
        raise ValueError("Router profile must not select an execution strategy")
    routes, paths = {}, {}
    for key in _ROUTE_KEYS:
        target, profile = read(table[key])
        _validate_execution_profile(profile, target, key)
        _validate_execution_profile(merge_profile(profile, overrides), target, key)
        routes[key], paths[key] = profile, target
    return ProfileSelection(name, path, router_profile, routes, paths, hashes, overrides, router_path)
