"""Task-owned web access constraints supplied by the evaluation environment."""
from __future__ import annotations
import json
from copy import deepcopy


def validate_web_policy(value: dict | str | None) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"blocked_domains", "blocked_terms", "network_enabled"}:
        raise ValueError("web_policy allows only blocked_domains, blocked_terms and network_enabled")
    result = {"blocked_domains": [], "blocked_terms": [], "network_enabled": True, **deepcopy(value)}
    if not isinstance(result["network_enabled"], bool):
        raise ValueError("web_policy.network_enabled must be boolean")
    for key in ("blocked_domains", "blocked_terms"):
        items = result[key]
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"web_policy.{key} must be a list of nonempty strings")
        result[key] = list(dict.fromkeys(item.strip() for item in items))
    return result


def blocked_sources(policy: dict) -> tuple[str, ...]:
    clean = validate_web_policy(policy)
    return tuple(dict.fromkeys([*clean["blocked_domains"], *clean["blocked_terms"]]))
