"""Pure text cleanup shared by native HTN control and worker adapters."""

import re

_LEAKED_TOOL_BLOCK_RE = re.compile(
    r"<\s*(tool_calls?|tool_response|use_mcp_tool)\b[^>]*>[\s\S]*?"
    r"<\s*/\s*\1\s*>",
    re.IGNORECASE,
)
_LEAKED_FUNCTION_BLOCK_RE = re.compile(
    r"<\s*function(?:\s*=|\s+name=)[\s\S]*?<\s*/\s*function\s*>",
    re.IGNORECASE,
)
_LEAKED_DSML_TOOL_CALLS_RE = re.compile(
    r"<｜+DSML｜+tool_calls\b[^>]*>[\s\S]*?</｜+DSML｜+tool_calls\s*>",
    re.IGNORECASE,
)
_LEAKED_DSML_INVOKE_BLOCK_RE = re.compile(
    r"<｜+DSML｜+invoke\b[^>]*>[\s\S]*?</｜+DSML｜+invoke\s*>",
    re.IGNORECASE,
)
_LEAKED_DSML_PARAMETER_BLOCK_RE = re.compile(
    r"<｜+DSML｜+parameter\b[^>]*>[\s\S]*?</｜+DSML｜+parameter\s*>",
    re.IGNORECASE,
)
_LEAKED_TOOL_TAG_FRAGMENT_RE = re.compile(
    r"<\s*/?\s*(?:tool_calls?|tool_response|use_mcp_tool|function|parameter)\b[^>]*>?",
    re.IGNORECASE,
)
_LEAKED_DSML_TAG_FRAGMENT_RE = re.compile(
    r"<\s*/?｜+DSML｜+(?:tool_calls|invoke|parameter)\b[^>]*>?",
    re.IGNORECASE,
)
_DANGLING_DSML_BLOCK_RE = re.compile(
    r"<｜+DSML｜+(?:tool_calls|invoke|parameter)\b[\s\S]*\Z",
    re.IGNORECASE,
)

# Seed reasoning-mode FunctionCall wrapper. Body is a JSON array of
# ``{"name": str, "parameters": dict}`` objects. Emitted by the model
# verbatim when an upstream proxy doesn't repack it into native
# ``response.tool_calls``.
_FC_WRAPPED_RE = re.compile(
    r"<\|FunctionCallBegin\|>\s*(\[.*?\])\s*<\|FunctionCallEnd\|>", re.DOTALL,
)


def strip_leaked_tool_markup(text: str) -> str:
    """Remove text-mode tool calls from user-visible/final model content.

    Covers the generic XML, Qwen/Seed function, MCP, and DeepSeek DSML forms
    understood by this module.  Ordinary report text around a complete leaked
    block is preserved.  An unfinished DSML block is dropped from its opening
    marker onward because the remainder is serialized tool input, not evidence.
    """
    if not text:
        return text
    out = _LEAKED_DSML_TOOL_CALLS_RE.sub("", text)
    out = _LEAKED_DSML_INVOKE_BLOCK_RE.sub("", out)
    out = _LEAKED_DSML_PARAMETER_BLOCK_RE.sub("", out)
    out = _LEAKED_TOOL_BLOCK_RE.sub("", out)
    out = _LEAKED_FUNCTION_BLOCK_RE.sub("", out)
    out = _DANGLING_DSML_BLOCK_RE.sub("", out)
    out = _LEAKED_DSML_TAG_FRAGMENT_RE.sub("", out)
    out = _LEAKED_TOOL_TAG_FRAGMENT_RE.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

