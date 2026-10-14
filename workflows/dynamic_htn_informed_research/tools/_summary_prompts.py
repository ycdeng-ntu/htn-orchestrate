"""Miro extraction prompts and request payload, preserved verbatim."""

from copy import deepcopy
from typing import Any

_EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{focus}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{content}

EXTRACTED INFORMATION:"""

_EXTRACT_INFO_PROMPT_SCIENCE = """You are extracting from scientific content (a \
paper, protocol, or dataset) the information requested below. This is for \
reproducing the source's exact result, so FIDELITY matters more than brevity.

INFORMATION TO EXTRACT:
{focus}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. PRESERVE SPECIFICS VERBATIM (most important): copy every equation, formula,
   derivation step, exact numerical value WITH its units, and named identifier
   (compound / protein / gene / model / method name, physical constant) EXACTLY
   as written — character for character, including subscripts/superscripts and
   factors like √λ vs 1/√λ. NEVER paraphrase a formula, round or drop a number,
   or compress a multi-step derivation into prose. A downstream agent must
   reproduce these from your extraction alone — losing or altering one is a
   failure. Keep ALL relevant equations/values even if the extraction gets long.
3. When the focus targets a derivation, keep the intermediate steps in order, not
   just the final result.
4. Clearly organize the extracted information.
5. Omit only prose that is genuinely unrelated to the focus — specifics
   (equations / values / identifiers) are never "unrelated".

CONTENT TO ANALYZE:
{content}

EXTRACTED INFORMATION:"""

def _build_payload(
    model: str,
    prompt: str,
    *,
    reasoning_effort: str = "",
    thinking_disabled: bool = False,
    provider_routing: dict[str, Any] | None = None,
) -> dict:
    """Build OpenAI-compatible chat-completions payload.

    GPT-5 family uses ``max_completion_tokens`` + reasoning_effort knobs;
    everything else uses the standard ``max_tokens`` + ``temperature``.

    Self-hosted Qwen/MiroThinker/Apodex (SGLang) get
    ``chat_template_kwargs.enable_thinking=false`` — extraction is an
    auxiliary call; with thinking ON (their default) a reasoning model
    burns the whole completion budget on hidden reasoning and returns
    ``content=None`` (same convention as reporter_v2's aux calls, see
    ``report_llm.create_completion``).
    """
    lowered = model.lower()
    if "gpt-5" in lowered or "gpt5" in lowered:
        payload = {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": reasoning_effort or "minimal",
            "service_tier": "flex",
        }
        if provider_routing:
            payload["provider"] = deepcopy(provider_routing)
        return payload
    payload: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    if thinking_disabled and "deepseek" in lowered:
        payload["thinking"] = {"type": "disabled"}
    elif reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if provider_routing:
        payload["provider"] = deepcopy(provider_routing)
    if any(k in lowered for k in ("qwen", "mirothinker", "sglang", "397b", "apodex")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload
