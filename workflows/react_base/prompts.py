"""Prompts for the react_base workflow.

- :func:`get_system_prompt` — minimal ReAct system prompt. Tool
  definitions are NOT embedded in the prompt text; they reach the model
  through the native OpenAI ``tools=`` API parameter.
- :func:`get_summarize_prompt` — final-answer extraction prompt used
  when the loop hits ``max_turns``.
"""

from __future__ import annotations


# Mid-string spacing (trailing space, ``\n`` before headers) is intentional —
# inference parity with training requires preserving these exactly.
_SYSTEM_PROMPT = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n"
    "\n"
    "You only have access to the tools provided. You can use multiple tools per message, and will receive the results of those tools in the user's next response. You use tools step-by-step to accomplish a given task. \n"
    "# General Objective\n"
    "\n"
    "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."
)


def get_system_prompt(date_str: str | None = None) -> str:  # noqa: ARG001 — kept for API stability
    """Build the react_base system prompt. ``date_str`` is unused."""
    return _SYSTEM_PROMPT


def get_summarize_prompt(task_description: str) -> str:
    """Final-answer extraction prompt for the ``max_turns`` safety net.

    Used by the ``_force_final_answer`` fallback to extract a plain-text
    answer when the loop didn't terminate naturally.
    """
    return (
        "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
        "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
        "simply extract that answer.\n"
        "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
        "The original question is repeated here for reference:\n\n"
        f'"{task_description}"\n\n'
        "Your final answer MUST strictly follow any formatting instructions in the original question — "
        "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n\n"
        "You must absolutely not perform any MCP tool call, tool invocation, search, scrape, code execution, or similar actions.\n"
        "You can only answer the original question based on the information already retrieved and your own internal knowledge.\n"
        "If you attempt to call any tool, it will be considered a mistake."
    )
