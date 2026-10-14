"""Instruction-composition semantics owned exclusively by science MCQ."""


_MCQ_BASE_INSTRUCTION_GENERATION_SYSTEM = """\
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
  disagreement as neutrally labelled scientific candidates. Quote every answer-bearing formula, value,
  label, and qualification verbatim rather than algebraically rewriting or normalizing it. If the
  coordinator's restatement differs from a completed report, include the original candidate and identify
  the mismatch. Require the worker to derive the decisive result independently before comparing them;
  describe the disagreement neutrally and do not frame any candidate as exact, physical, intended, or
  presumed correct.
- Targeted non-independent follow-up: include only the prior details necessary for the stated gap and
  make clear that the task is not an independent confirmation.

The generic request to write as if the agent acts alone means the resulting instruction must be
self-contained and must not expose the collaboration or orchestration process. This does not forbid a
checker, adjudicator, or targeted follow-up from receiving the candidate content permitted above.

Do not duplicate useful in-flight work. Never instruct exact-question, dataset, or answer-key search.
"""


_MCQ_BASE_SOLVING_PROTOCOL = """\
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


_MCQ_BASE_SCOPE_BOUNDARY = """\
Solve only the current leaf and assigned route, but preserve the ROOT answer contract so your result
can be synthesized without losing a condition or requested field.
"""


_MCQ_INSTRUCTION_GENERATION_SYSTEM = _MCQ_BASE_INSTRUCTION_GENERATION_SYSTEM + """

For a multiple-choice ROOT, emit only a concise route delta: method, deliverable, and decisive checks.
ROOT and the fixed MCQ protocol are attached separately, so never restate the problem, givens, or option
inventory. Do not add a scientific premise or expected result. Keep the output under 2400 characters.
"""


_MCQ_SOLVING_PROTOCOL = _MCQ_BASE_SOLVING_PROTOCOL + """

Multiple-choice protocol:
- Parse the offered labels exactly as written and ignore absent labels; never assume there are only
  A-E choices.
- Derive the scientific result before mapping it to an option. For atomic or numbered statements,
  record each statement as true/false/undetermined with one decisive justification.
- First evaluate every direct/base statement or candidate independently under ROOT's literal precision,
  rounding, and tolerance requirements. Use ordinary domain conventions only when justified without
  reference to the option inventory. Freeze this base truth ledger before evaluating any option that
  combines, negates, quantifies over, or otherwise refers to other choices.
- Map the frozen truth pattern to every relevant combination option. Explicitly test any catch-all
  choice such as "all other options are incorrect" only after the base judgments are fixed.
- Option-set completeness, symmetry, exclusivity, and apparent author intent are never scientific
  evidence. Never relax rounding/tolerance, change a truth value, or manufacture a satisfying mapping
  because the offered choices otherwise appear inconsistent. If no option matches, preserve that
  inconsistency in the report rather than repairing the science from the option design.
- Distinguish correctness of the science from correctness of the final fact-to-label mapping.
- Select exactly one offered uppercase option label. Keep explanation in the report, but make the
  candidate answer line contain only that label.
"""


_MCQ_SCOPE_BOUNDARY = _MCQ_BASE_SCOPE_BOUNDARY + """
 Preserve the full option inventory whenever the current leaf contributes to the final selection.
"""


_MCQ_WORKER_PROMPT_TEMPLATE = """\
# Problem

{root_task}

{leaf_context}

---

# Assigned Route

{generated_instruction}

---

# Execution Requirements

{solving_protocol}

---

# Scope

{scope_boundary}
"""


__all__ = [
    "_MCQ_INSTRUCTION_GENERATION_SYSTEM",
    "_MCQ_SCOPE_BOUNDARY",
    "_MCQ_SOLVING_PROTOCOL",
    "_MCQ_WORKER_PROMPT_TEMPLATE",
]
