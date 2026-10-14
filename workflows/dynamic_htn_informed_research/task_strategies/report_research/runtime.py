"""Explicit, task-local mutable state for one report-research question."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMClient
from workflows.dynamic_htn_informed_research.htn.orchestrator import AgentResultRecord
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode, HTNTaskStatus
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree

from .evidence import EvidenceCard, extract_evidence_update
from .policy import ReportPolicy
from .requirements import (
    Requirement,
    RequirementSet,
    RequirementStatus,
    derive_requirement_set,
)


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_OWNERSHIP_STOPWORDS = frozenset(
    "the and for with from this that their which what how should must give find "
    "report requirement requirements criterion criteria task subtask research "
    "answer requested provide explain show complete remaining delivery work".split()
)
_MAX_PRECURSOR_EVIDENCE = 8
_MAX_DESCRIPTION_CHARS = 800
_MAX_CRITERION_CHARS = 400
_MAX_EVIDENCE_TEXT_CHARS = 600
_MAX_DIRECTIVE_CHARS = 2_000


def _bounded(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower())) - _OWNERSHIP_STOPWORDS


def _ownership_score(requirement: Requirement, leaf: HTNTaskNode) -> float:
    requirement_tokens = _tokens(
        " ".join(
            (
                requirement.description,
                *requirement.sub_obligations,
                *requirement.acceptance_criteria,
                requirement.kind,
            )
        )
    )
    leaf_tokens = _tokens(f"{leaf.title} {leaf.task}")
    return len(requirement_tokens & leaf_tokens) / max(1, len(requirement_tokens))


def _requirement_snapshot(requirement: Requirement) -> dict[str, Any]:
    return {
        "id": requirement.id,
        "description": requirement.description,
        "acceptance_criteria": list(requirement.acceptance_criteria),
        "priority": requirement.priority,
        "kind": requirement.kind,
        "owner_leaf_id": requirement.owner_leaf_id,
        "status": requirement.status.value,
        "evidence_ids": list(requirement.evidence_ids),
        "covered_criterion_ids": list(requirement.covered_criterion_ids),
        "sub_obligations": list(requirement.sub_obligations),
        "disputed_evidence_ids": list(requirement.disputed_evidence_ids),
        "reopen_reasons": list(requirement.reopen_reasons),
    }


def _evidence_snapshot(card: EvidenceCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "leaf_id": card.leaf_id,
        "requirement_ids": list(card.requirement_ids),
        "claim": card.claim,
        "source_url": card.source_url,
        "source_title": card.source_title,
        "support_summary": card.support_summary,
        "confidence": card.confidence,
        "acceptance_criterion_ids": list(card.acceptance_criterion_ids),
    }


@dataclass
class ReportRuntime:
    """Mutable requirement coverage owned by exactly one report question."""

    question: str
    policy: ReportPolicy
    requirements: RequirementSet
    evidence_cards: dict[str, EvidenceCard] = field(default_factory=dict)
    recovery_wave_used: bool = False
    verification_repair_used: bool = False
    _diagnostics: DiagnosticRecorder | None = field(default=None, repr=False)

    @classmethod
    async def create(
        cls,
        *,
        question: str,
        policy: ReportPolicy,
        llm: LLMClient,
        diagnostics: DiagnosticRecorder,
        question_deadline: float | None = None,
    ) -> ReportRuntime:
        """Derive the public delivery contract and return fresh task-local state."""
        requirements = await derive_requirement_set(
            question=question,
            policy=policy,
            llm=llm,
            diagnostics=diagnostics,
            question_deadline=question_deadline,
        )
        return cls(
            question=question,
            policy=policy,
            requirements=requirements,
            _diagnostics=diagnostics,
        )

    async def refresh_ownership(self, tree: HTNTree, llm: LLMClient) -> None:
        """Keep ownership across waves, including leaves waiting on prerequisites.

        Unmatched requirements remain unassigned for gap recovery. Completion
        changes readiness, not scope; a requirement moves only when its owner
        disappears, fails, or is decomposed into more specific children.
        """
        del llm
        leaves = sorted(
            (node for node in tree.nodes.values()
             if node.is_leaf and (node.id != tree.root_id or not node.needs_decomposition)
             and node.status is not HTNTaskStatus.FAILED),
            key=lambda node: node.id,
        )
        ownership_load = {leaf.id: 0 for leaf in leaves}
        unassigned: list[Requirement] = []
        for requirement in self.requirements.requirements:
            owner = tree.nodes.get(requirement.owner_leaf_id)
            if requirement.status is RequirementStatus.GROUNDED:
                if owner is None:
                    requirement.owner_leaf_id = None
                continue
            if owner is not None and owner in leaves:
                ownership_load[owner.id] += 1
                continue
            unassigned.append(requirement)

        for requirement in unassigned:
            old_owner = tree.nodes.get(requirement.owner_leaf_id)
            candidates = leaves
            if old_owner is not None and old_owner.children:
                descendants: set[str] = set()
                pending = list(old_owner.children)
                while pending:
                    node_id = pending.pop()
                    descendants.add(node_id)
                    pending.extend(tree.nodes[node_id].children)
                candidates = [leaf for leaf in leaves if leaf.id in descendants]
            # A directly executed root owns the original question in full.
            scores = {leaf.id: (1.0 if leaf.id == tree.root_id else
                               _ownership_score(requirement, leaf)) for leaf in candidates}
            relevant = [leaf for leaf in candidates if scores[leaf.id] > 0]
            if not relevant:
                requirement.owner_leaf_id = old_owner.id if old_owner and old_owner.children else None
                continue
            owner = min(relevant, key=lambda leaf: (
                -scores[leaf.id], ownership_load[leaf.id], leaf.id,
            ))
            requirement.owner_leaf_id = owner.id
            ownership_load[owner.id] += 1
        if self._diagnostics is not None:
            self._diagnostics.emit(
                "report_requirement_ownership",
                candidate_leaf_count=len(leaves),
                requirement_count=len(self.requirements.requirements),
                ownership={
                    requirement.id: requirement.owner_leaf_id
                    for requirement in self.requirements.requirements
                },
            )

    def delivery_context_for(self, leaf: HTNTaskNode) -> str:
        """Render a bounded contract plus grounded facts from direct precursors."""
        owned = sorted(
            (
                requirement
                for requirement in self.requirements.requirements
                if requirement.owner_leaf_id == leaf.id
            ),
            key=lambda requirement: (requirement.priority, requirement.id),
        )
        lines = [
            "REPORT DELIVERY CONTRACT",
            f"Policy: {self.policy.key}",
        ]
        if self.policy.delivery_directive:
            lines.append(
                "Policy directive: "
                f"{_bounded(self.policy.delivery_directive, _MAX_DIRECTIVE_CHARS)}"
            )
        lines.append(
            "Contract precedence: requirements restate the public question, nothing "
            "more. If a requirement's wording asserts a factual convention, value, or "
            "method that verified evidence or the question's own data contradicts, "
            "follow the evidence, deliver the evidence-backed result, and flag the "
            "conflict — never discard a supported result to satisfy contract wording."
        )
        lines.append(f"Assigned: {len(owned)} owned requirements")
        if owned:
            for requirement in owned:
                lines.append(
                    f"- [{requirement.id}] priority={requirement.priority} "
                    f"status={requirement.status.value}: "
                    f"{_bounded(requirement.description, _MAX_DESCRIPTION_CHARS)}"
                )
                for obligation in requirement.sub_obligations:
                    lines.append(
                        f"  Sub-obligation: {_bounded(obligation, _MAX_DESCRIPTION_CHARS)}"
                    )
                for criterion in requirement.acceptance_criteria:
                    lines.append(f"  Acceptance: {_bounded(criterion, _MAX_CRITERION_CHARS)}")
        else:
            lines.append("- (no requirements currently owned by this leaf)")

        requirement_by_id = {
            requirement.id: requirement for requirement in self.requirements.requirements
        }
        precursor_cards = [
            card
            for card in sorted(self.evidence_cards.values(), key=lambda item: item.id)
            if card.leaf_id in leaf.depends_on
            and any(
                (requirement := requirement_by_id.get(requirement_id)) is not None
                and requirement.status is RequirementStatus.GROUNDED
                and card.id in requirement.evidence_ids
                for requirement_id in card.requirement_ids
            )
        ][:_MAX_PRECURSOR_EVIDENCE]
        lines.append("Grounded precursor facts:")
        if not precursor_cards:
            lines.append("- (none)")
        for card in precursor_cards:
            lines.append(f"- [{card.id}] {_bounded(card.claim, _MAX_EVIDENCE_TEXT_CHARS)}")
            lines.append(f"  Support: {_bounded(card.support_summary, _MAX_EVIDENCE_TEXT_CHARS)}")
            if card.source_url or card.source_title:
                source = " — ".join(
                    value for value in (card.source_title, card.source_url) if value
                )
                lines.append(f"  Source: {_bounded(source, _MAX_EVIDENCE_TEXT_CHARS)}")
        return "\n".join(lines)

    def reopen_requirement(self, requirement_id: str, reason: str) -> None:
        """Suspend support for one requirement; keep disputed cards for review."""
        requirement = next(r for r in self.requirements.requirements if r.id == requirement_id)
        # Include retained cards even when only some acceptance criteria were covered.
        for card in self.evidence_cards.values():
            if requirement_id in card.requirement_ids and card.id not in requirement.evidence_ids:
                requirement.evidence_ids.append(card.id)
        requirement.reopen(reason)
        if self._diagnostics is not None:
            self._diagnostics.emit("report_requirement_reopened", requirement_id=requirement_id,
                                   reason=reason)

    async def observe_records(
        self,
        *,
        leaf_id: str,
        leaf_task: str,
        requirement_ids: Sequence[str],
        records: Sequence[AgentResultRecord],
        llm: LLMClient,
        deadline: float | None = None,
        accept_results: Callable[[], bool] | None = None,
    ) -> None:
        """Extract and bind evidence only within the explicit requirement allowlist."""
        requested_ids = set(requirement_ids)
        candidates = [
            requirement
            for requirement in self.requirements.requirements
            if requirement.id in requested_ids
        ]
        update = await extract_evidence_update(
            leaf_id=leaf_id,
            leaf_task=leaf_task,
            candidate_requirements=candidates,
            records=records,
            policy=self.policy,
            llm=llm,
            diagnostics=self._diagnostics,
            deadline=deadline,
            existing_evidence=tuple(
                card for card in self.evidence_cards.values()
                if any(r.id in card.requirement_ids and card.id not in r.disputed_evidence_ids
                       for r in candidates)
            ),
        )
        if accept_results is not None and not accept_results():
            return
        requirement_by_id = {requirement.id: requirement for requirement in candidates}
        challenged = {item.requirement_id for item in update.challenges}
        for item in update.challenges:
            self.reopen_requirement(item.requirement_id, f"{item.record_id}: {item.reason}")
        for card in update.cards:
            self.evidence_cards[card.id] = card
            for requirement_id in card.requirement_ids:
                requirement = requirement_by_id.get(requirement_id)
                if requirement is None:
                    continue
                if requirement_id in challenged:
                    if card.id not in requirement.disputed_evidence_ids:
                        requirement.disputed_evidence_ids.append(card.id)
                if card.id in requirement.disputed_evidence_ids:
                    continue
                if card.id not in requirement.evidence_ids:
                    requirement.evidence_ids.append(card.id)
                expected_criteria = {
                    f"{requirement.id}-criterion-{index:03d}"
                    for index, _criterion in enumerate(
                        requirement.acceptance_criteria,
                        start=1,
                    )
                }
                for criterion_id in card.acceptance_criterion_ids:
                    if (
                        criterion_id in expected_criteria
                        and criterion_id not in requirement.covered_criterion_ids
                    ):
                        requirement.covered_criterion_ids.append(criterion_id)
                if (
                    requirement.status is RequirementStatus.PENDING
                    and expected_criteria.issubset(requirement.covered_criterion_ids)
                ):
                    requirement.transition_to(RequirementStatus.GROUNDED)

    async def observe_leaf(
        self,
        leaf: HTNTaskNode,
        records: Sequence[AgentResultRecord],
        llm: LLMClient,
        deadline: float | None = None,
    ) -> None:
        """Observe a normal leaf using only the requirements it currently owns."""
        await self.observe_records(
            leaf_id=leaf.id,
            leaf_task=leaf.task,
            requirement_ids=tuple(
                requirement.id
                for requirement in self.requirements.requirements
                if requirement.owner_leaf_id == leaf.id
            ),
            records=records,
            llm=llm,
            deadline=deadline,
        )

    def uncovered_requirements(self) -> tuple[Requirement, ...]:
        """Return all non-grounded requirements in stable recovery order."""
        return tuple(
            sorted(
                (
                    requirement
                    for requirement in self.requirements.requirements
                    if requirement.status is not RequirementStatus.GROUNDED
                ),
                key=lambda requirement: (requirement.priority, requirement.id),
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-compatible diagnostic view of report coverage."""
        return {
            "question": self.question,
            "policy": asdict(self.policy),
            "ownership": {
                requirement.id: requirement.owner_leaf_id
                for requirement in self.requirements.requirements
            },
            "requirements": [
                _requirement_snapshot(requirement) for requirement in self.requirements.requirements
            ],
            "evidence": [
                _evidence_snapshot(card)
                for card in sorted(self.evidence_cards.values(), key=lambda item: item.id)
            ],
            "uncovered_requirements": [
                _requirement_snapshot(requirement) for requirement in self.uncovered_requirements()
            ],
            "recovery_wave_used": self.recovery_wave_used,
            "verification_repair_used": self.verification_repair_used,
        }


__all__ = ["ReportRuntime"]
