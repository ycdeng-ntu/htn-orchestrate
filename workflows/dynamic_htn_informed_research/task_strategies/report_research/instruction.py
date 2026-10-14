"""Assignment generation prompts for report workers."""

_INSTRUCTION_GENERATION_SYSTEM = """\
Write one self-contained report-research assignment from the public ROOT question, leaf scope, and
untrusted precursor evidence. Preserve every allocated coverage item and require an evidence memo
with exact claims, [[WEB]]source URLs, verbatim support excerpts, [[/WEB]]reasoning, conflicts, and remaining gaps. Never add private
rubric content, expected answers, or unsupported candidates. For quantitative work require
reproducible calculation; [[WEB]]for source claims require direct provenance; [[/WEB]]for enumerations and tables
name every allocated row, field, and table cell; for science name every sub-question, equation,
named method, assumption, unit, and intermediate value needed within the assigned scope.
The worker produces research evidence, not the complete report. Do not turn root-wide requirements
into duplicated work for every leaf or ask a worker to perform final-report writing or final review.
The runtime separately attaches original reports, dependency results and recovery evidence. Explain
what to do with that evidence; do not copy all of it or replace it with a reconstructed account.
Preserve report/check references, existing uncertainty, the specific gap and the expected new check.
Do not expand a targeted repair into another complete investigation or prescribe an unsupported answer.
For verifier assignments, distinguish facts explicitly given by ROOT from upstream hypotheses.
Never freeze an upstream answer as an assignment premise. First verification must independently
check decisive constraints, premises and requested results. Follow-up verification checks the changed
claims and affected dependent conclusions against the original evidence and prior checks. Preserve
this distinction from the coordinator's assignment. If reopening an unchanged checked claim, retain
the concrete new counterexample, evidence, faulty earlier check or dependency change that warrants it.
An unchecked decisive claim still needs independent checking; agreement alone is not prior verification.
Recomputing an arbitrary weighted ranking checks its arithmetic, not its criteria, weights or validity.
Require an explicit premise verdict, actual checks and the dependent conclusions needing repair.
"""

_SCOPE_BOUNDARY = """\
Complete the assigned milestone result or targeted evidence gap, including its necessary self-checks. Return a research memo rather than drafting
a final-report section. Preserve root-wide terminology and scientific requirements; do not claim
global coverage or write the complete report. Verification may reopen
upstream premises that the assigned section depends on; report corrections and their downstream
impact even when the mistaken premise originated in another leaf.
"""


__all__ = ["_INSTRUCTION_GENERATION_SYSTEM", "_SCOPE_BOUNDARY"]
