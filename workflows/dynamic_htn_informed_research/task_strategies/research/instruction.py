"""Research instruction-composition semantics."""

from .coordination import _INSTRUCTION_GENERATION_SYSTEM

_SCOPE_BOUNDARY = """\
Use the original root question to understand how your evidence will be used, but execute only the
current leaf and assigned sub-agent task. Do not broaden your work into re-solving unrelated parts of
the root question. Pay attention to what the original question ultimately asks for, including its
requested scope and output format. In your report, preserve the exact source wording and evidence
needed for that final response—including root-relevant names, titles, values, and other identifying
details—and state what your findings support, contradict, or leave unresolved in the original question.
Do not reject a candidate merely because it fails an added or narrowed interpretation in your
instruction. Map every claimed contradiction to the exact original phrase it conflicts with; if the
added restriction is not stated there, treat it as an optional search hypothesis and report the
original relation as unresolved. Join compatible observations from the supplied context before
declaring a decisive relation missing, and preserve conflicting reports explicitly."""

__all__ = ["_INSTRUCTION_GENERATION_SYSTEM", "_SCOPE_BOUNDARY"]
