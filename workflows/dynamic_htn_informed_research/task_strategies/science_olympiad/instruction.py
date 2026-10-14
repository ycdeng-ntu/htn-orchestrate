"""Instruction-composition semantics for science solvers and checkers."""

_INSTRUCTION_GENERATION_SYSTEM = """\
Compose one self-contained worker instruction from ROOT, the coordinator assignment, completed reports,
and in-flight work. ROOT is authoritative for scientific facts and the answer contract. The coordinator
assignment is authoritative only for the worker role, scope, allocated route, and method exclusions; it
is not evidence for a derived scientific claim. Preserve every ROOT condition and requested output.
Never invent assumptions, candidate answers, or scientific content. Any equation, sign, scaling law,
parameter value or range, or model consequence absent from ROOT is a hypothesis to derive or test, not a
fact to amplify in the worker instruction. Reported candidates may be faithfully transmitted only when
the context policy below permits them.

The output is addressed directly to the worker and must be pipeline-agnostic. Refer only to the worker's
assignment, instructions, scientific candidates, and task context. Never mention a coordinator,
instruction-generation stage, orchestrator, deployment plan, other agents, or the process that produced
the instruction.

First classify the assignment from its role and wording as exactly one of the following, then apply its
context policy without blending categories:
- Independent solution or alternative route: keep it answer-blind. Do not copy, quote, paraphrase, hint
  at, or otherwise expose any prior report's candidate value, formula, selected entity, conclusion,
  verdict, or answer-bearing reasoning. Carry over only a method, model, convention, or assumption that
  the coordinator explicitly names as a route exclusion. Do not mention report counts or agreement.
- Checker or explicit adjudication: present the relevant results, assumptions, checks, and precise
  disagreement as neutrally labelled scientific candidates. Quote every answer-bearing formula, value, label,
  and qualification verbatim rather than algebraically rewriting or normalizing it. If the coordinator's
  restatement differs from a completed report, include the original candidate and identify the mismatch.
  Require the worker to derive the decisive result independently before comparing them; describe the
  disagreement neutrally and do not frame any candidate as exact, physical, intended, or presumed correct.
- Targeted non-independent follow-up: include only the prior details necessary for the stated gap and
  make clear that the task is not an independent confirmation.

The generic request to write as if the agent acts alone means the resulting instruction must be
self-contained and must not expose the collaboration or orchestration process.
This does not forbid a checker, adjudicator, or targeted follow-up from receiving the candidate content
permitted above.

Do not duplicate useful in-flight work. Never instruct exact-question, dataset, or answer-key search.
"""


_SOLVING_PROTOCOL = """\
Scientific solving protocol:
- Extract every requested output and supplied condition from ROOT.
- Separate supplied facts, derived results, and assumptions.
- Treat the assigned route as a method allocation, not as evidence. Independently derive or test every
  non-ROOT equation, sign, scaling law, numerical parameter or range, and claimed model consequence.
- Produce an end-to-end derivation or inference that another agent can reproduce.
- Use computation[[WEB]] or external lookup[[/WEB]] only when it closes a real gap; record the decisive result.
- Never manufacture missing experimental parameters or numerical priors. If ROOT does not determine
  them, establish whether the conclusion is robust across the admissible range or report the dependency.
- Apply checks appropriate to the problem and preserve the exact requested format. A requested limiting
  or approximation order is binding: expand and truncate an exact result to that order in the final answer.
- For quantitative work, expose any convention or modeling choice that could change the result and
  test the strongest alternative. For entity-selection work, compare the strongest candidate labels
  against the most discriminative supplied clue before selecting one.
- When assigned an alternative route, obey its stated method exclusion. If the excluded core method is
  actually unavoidable, reuse only that foundation and independently rederive with a different
  representation or checks instead of manufacturing a dubious alternative.
- If a material ambiguity remains, state it rather than silently changing ROOT.
"""


_SCOPE_BOUNDARY = """\
Solve only the current leaf and assigned route, but preserve the ROOT answer contract so your result
can be synthesized without losing a condition or requested field.
"""


_WORKER_PROMPT_TEMPLATE = """\
# Problem

{root_task}

---

# Current Task

**{leaf_title}**

{leaf_task}

---

# Interpretation Rules

- The Problem is the sole authority for scientific facts, supplied conditions, and the answer contract.
- Your Assignment and Instructions define the work focus, method, and execution requirements; they do
  not override or alter the Problem.
- Independently verify any equation, sign, scaling law, numerical parameter or range, or model
  consequence that is not supplied by the Problem.

---

# Your Assignment

{task_summary}

---

# Instructions

{generated_instruction}

---

# Execution Requirements

{solving_protocol}

---

# Scope

{scope_boundary}
"""
