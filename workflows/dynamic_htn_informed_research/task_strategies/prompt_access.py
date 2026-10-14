"""Render trusted strategy templates using task-local benchmark capabilities."""
from __future__ import annotations

import re
from dataclasses import fields, is_dataclass, replace
from typing import Any

from .contracts import TaskStrategy

_BLOCK = re.compile(r"\[\[(WEB|OFFLINE)\]\](.*?)\[\[/\1\]\]", re.DOTALL)


def render_web_sections(template: str, *, network_enabled: bool) -> str:
    """Only render trusted templates, never question text or worker evidence."""
    rendered = _BLOCK.sub(
        lambda match: match[2] if (match[1] == "WEB") == network_enabled else "",
        template,
    )
    if any(marker in rendered for marker in ("[[WEB", "[[/WEB", "[[OFFLINE", "[[/OFFLINE")):
        raise ValueError("Malformed capability template")
    return rendered



def render_strategy(
    template: TaskStrategy, *, network_enabled: bool, system_suffix: str = "",
) -> TaskStrategy:
    """Copy the immutable strategy; never mutate shared worker definitions or schemas."""
    def system_prompt(text: str) -> str:
        rendered = render_web_sections(text, network_enabled=network_enabled)
        return rendered.strip() + "\n\n" + system_suffix if system_suffix else rendered

    def render(value: Any, field_name: str = "") -> Any:
        if isinstance(value, str):
            if field_name.endswith("system") or field_name == "system_prompt":
                return system_prompt(value)
            return render_web_sections(value, network_enabled=network_enabled)
        if is_dataclass(value):
            return replace(value, **{f.name: render(getattr(value, f.name), f.name) for f in fields(value)})
        if hasattr(value, "model_copy") and hasattr(value, "system_prompt"):
            return value.model_copy(deep=True, update={
                "system_prompt": system_prompt(value.system_prompt),
                "description": render_web_sections(value.description, network_enabled=network_enabled),
            })
        if isinstance(value, tuple):
            return tuple(render(item) for item in value)
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {key: render(item) for key, item in value.items()}
        return value
    return render(template)
