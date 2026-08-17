"""Cross-agent review: structured findings, computed before they are opined on.

M4.7 asks for review *interfaces*, not an autonomous impact engine. The shape is:

    PM produces a PRD
      -> UX reviews requirement coverage
      -> Architect reviews feasibility
      -> Critic checks contradictions

The important design choice is what a "review" is allowed to be. Every finding here is
produced by a deterministic function over artifact structure: which requirements nothing
covers, which references dangle, which properties conflict, which tasks cannot be ordered.
None of it asks a model whether the document looks good.

That is not because model review is worthless — a model can notice that a requirement is
ambiguously worded, which no rule can. It is because a review that mixes computed facts with
model opinions produces a report where the reader cannot tell which is which, and the
opinions are the ones that get trusted. So the computed findings are the review, and a model
critique is a separate, clearly-labelled supplement.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import Field

from edith.observability.logging import get_logger
from edith.schemas.common import EdithModel, Verdict, new_id, utc_now

from .architecture import ImplementationPlanDocument, SystemArchitectureDocument
from .artifacts import (
    ArtifactDocument,
    ArtifactKind,
    ArtifactRef,
    ValidationIssue,
    register_document,
)
from .contradictions import Contradiction, check_all
from .prd import PRDDocument
from .ux import UXSpecDocument
from .validation import coverage_issues, find_cycle

logger = get_logger(__name__)


class ReviewPerspective(StrEnum):
    """Which lens a review was performed through.

    A UX reviewer and an Architect reviewing the same PRD are answering different questions,
    and a finding is much more useful when the reader knows which was being asked.
    """

    #: Does the interface serve every requirement?
    UX_COVERAGE = "UX_COVERAGE"
    #: Can this be built under the stated constraints?
    ARCHITECTURE_FEASIBILITY = "ARCHITECTURE_FEASIBILITY"
    #: Do these documents contradict each other?
    CONTRADICTION = "CONTRADICTION"
    #: Are the requirements complete, unambiguous, and checkable?
    REQUIREMENT_QUALITY = "REQUIREMENT_QUALITY"
    #: Can the plan actually be executed?
    PLAN_EXECUTABILITY = "PLAN_EXECUTABILITY"


class FindingSeverity(StrEnum):
    """How much a finding matters."""

    #: Must be fixed before approval.
    BLOCKER = "BLOCKER"
    #: Should be fixed; approval is a judgement call.
    MAJOR = "MAJOR"
    #: Worth knowing.
    MINOR = "MINOR"
    #: Not a defect. Recorded so the reader knows it was checked.
    INFO = "INFO"

    @property
    def blocking(self) -> bool:
        """Whether this severity prevents approval."""
        return self is FindingSeverity.BLOCKER


class ReviewFinding(EdithModel):
    """One structured observation about an artifact."""

    finding_id: str = Field(pattern=r"^FND-\d{3,}$")
    perspective: ReviewPerspective
    severity: FindingSeverity = FindingSeverity.MAJOR
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=1000)
    #: Element the finding attaches to, when attributable.
    element_id: str = Field(default="", max_length=40)
    #: What would resolve it. A finding with no remedy is a complaint.
    remedy: str = Field(default="", max_length=500)

    def render(self) -> str:
        """A single readable line."""
        target = f" [{self.element_id}]" if self.element_id else ""
        return f"{self.severity} {self.code}{target}: {self.message}"


@register_document
class ReviewDocument(ArtifactDocument):
    """The result of reviewing one artifact.

    The body of a :attr:`~edith.product.artifacts.ArtifactKind.REVIEW` artifact, so a review
    is itself versioned, attributable, and persistent -- the same treatment as the documents
    it reviews.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.REVIEW

    #: The artifact that was reviewed, at the version actually read.
    subject: ArtifactRef
    reviewer: str = Field(min_length=1, max_length=60)
    perspectives: tuple[ReviewPerspective, ...] = ()
    verdict: Verdict = Verdict.PASS
    findings: tuple[ReviewFinding, ...] = ()
    summary: str = Field(default="", max_length=2000)
    #: A model's critique, when one was requested. Kept separate from the computed findings
    #: on purpose: a reader must be able to tell a checked fact from an opinion.
    model_critique: str = Field(default="", max_length=4000)

    def element_ids(self) -> tuple[str, ...]:
        """Every finding id this review defines."""
        return tuple(finding.finding_id for finding in self.findings)

    @property
    def blockers(self) -> tuple[ReviewFinding, ...]:
        """Findings that must be resolved before approval."""
        return tuple(finding for finding in self.findings if finding.severity.blocking)

    def render(self) -> str:
        """Render for a human reader."""
        lines = [
            f"REVIEW of {self.subject.kind} {self.subject.artifact_id}"
            f"@v{self.subject.version} by {self.reviewer}: {self.verdict}",
        ]
        if self.summary:
            lines.append(self.summary)
        for finding in self.findings:
            lines.append(f"  {finding.render()}")
        if self.model_critique:
            lines.extend(["", "MODEL CRITIQUE (opinion, not a computed finding):",
                          self.model_critique])
        return "\n".join(lines)


def _finding(
    index: int,
    perspective: ReviewPerspective,
    severity: FindingSeverity,
    code: str,
    message: str,
    *,
    element_id: str = "",
    remedy: str = "",
) -> ReviewFinding:
    """Build a numbered finding."""
    return ReviewFinding(
        finding_id=f"FND-{index:03d}",
        perspective=perspective,
        severity=severity,
        code=code,
        message=message,
        element_id=element_id,
        remedy=remedy,
    )


def _from_validation_issues(
    issues: list[ValidationIssue],
    perspective: ReviewPerspective,
    start: int,
) -> list[ReviewFinding]:
    """Convert validation issues into review findings, preserving blocking status."""
    return [
        _finding(
            start + offset,
            perspective,
            FindingSeverity.BLOCKER if issue.blocking else FindingSeverity.MINOR,
            issue.code,
            issue.message,
            element_id=issue.element_id,
        )
        for offset, issue in enumerate(issues)
    ]


def _from_contradictions(
    findings: tuple[Contradiction, ...], start: int
) -> list[ReviewFinding]:
    """Convert contradictions into review findings."""
    return [
        _finding(
            start + offset,
            ReviewPerspective.CONTRADICTION,
            FindingSeverity.BLOCKER if item.blocking else FindingSeverity.MINOR,
            item.code,
            item.render(),
            element_id=item.elements[0] if item.elements else "",
            remedy=(
                "Change one of the two documents so they agree, or record why the conflict "
                "is acceptable."
            ),
        )
        for offset, item in enumerate(findings)
    ]


def review_prd_for_ux_coverage(
    prd: PRDDocument, ux: UXSpecDocument, *, reviewer: str = "ux_designer"
) -> ReviewDocument:
    """The UX agent's review of a PRD: is every requirement served by the interface?"""
    issues = coverage_issues(prd, ux=ux)
    findings = _from_validation_issues(issues, ReviewPerspective.UX_COVERAGE, 1)

    next_index = len(findings) + 1
    for flow_id in ux.flows_without_error_paths():
        findings.append(
            _finding(
                next_index,
                ReviewPerspective.UX_COVERAGE,
                FindingSeverity.MAJOR,
                "FLOW_WITHOUT_ERROR_PATH",
                f"flow {flow_id} has no failure path",
                element_id=flow_id,
                remedy="Add the step the user reaches when the action fails.",
            )
        )
        next_index += 1

    return _assemble(
        subject=prd,
        reviewer=reviewer,
        perspectives=(ReviewPerspective.UX_COVERAGE,),
        findings=findings,
        summary=(
            f"{len(ux.covered_requirements)} of {len(prd.requirements)} requirement(s) are "
            f"served by a flow or screen."
        ),
    )


def review_prd_for_feasibility(
    prd: PRDDocument,
    architecture: SystemArchitectureDocument,
    *,
    reviewer: str = "architect",
) -> ReviewDocument:
    """The Architect's review of a PRD: can this be built, and does the design match it?"""
    issues = coverage_issues(prd, architecture=architecture)
    findings = _from_validation_issues(
        issues, ReviewPerspective.ARCHITECTURE_FEASIBILITY, 1
    )

    contradictions = check_all(prd=prd, architecture=architecture, include_hints=False)
    findings.extend(_from_contradictions(contradictions, len(findings) + 1))

    next_index = len(findings) + 1
    for name in architecture.unjustified_technologies():
        findings.append(
            _finding(
                next_index,
                ReviewPerspective.ARCHITECTURE_FEASIBILITY,
                FindingSeverity.MINOR,
                "TECHNOLOGY_WITHOUT_ALTERNATIVES",
                f"{name} was selected without naming a rejected alternative",
                remedy="Record what else was considered and why it lost.",
            )
        )
        next_index += 1

    return _assemble(
        subject=prd,
        reviewer=reviewer,
        perspectives=(
            ReviewPerspective.ARCHITECTURE_FEASIBILITY,
            ReviewPerspective.CONTRADICTION,
        ),
        findings=findings,
        summary=(
            f"{len(architecture.components)} component(s) address "
            f"{len(architecture.covered_requirements)} of {len(prd.requirements)} "
            f"requirement(s)."
        ),
    )


def review_requirement_quality(
    prd: PRDDocument, *, reviewer: str = "critic"
) -> ReviewDocument:
    """The Critic's review of a PRD: is it complete and checkable?

    Deterministic quality checks only -- a requirement with no acceptance criterion, an open
    question with no owner, a document with no non-goals. Ambiguity detection is left to a
    model critique, and labelled as such.
    """
    findings: list[ReviewFinding] = []
    index = 1

    for requirement_id in prd.unverified_requirements():
        findings.append(
            _finding(
                index,
                ReviewPerspective.REQUIREMENT_QUALITY,
                FindingSeverity.MAJOR,
                "REQUIREMENT_WITHOUT_ACCEPTANCE_CRITERION",
                f"{requirement_id} cannot be demonstrated: no acceptance criterion verifies it",
                element_id=requirement_id,
                remedy="Add an acceptance criterion naming this requirement.",
            )
        )
        index += 1

    if not prd.non_goals:
        findings.append(
            _finding(
                index,
                ReviewPerspective.REQUIREMENT_QUALITY,
                FindingSeverity.MINOR,
                "NO_NON_GOALS",
                "the PRD states no non-goals, so nothing bounds the scope",
                remedy="State what this product deliberately will not do.",
            )
        )
        index += 1

    for question in prd.open_questions:
        findings.append(
            _finding(
                index,
                ReviewPerspective.REQUIREMENT_QUALITY,
                FindingSeverity.INFO,
                "OPEN_QUESTION",
                f"{question.question_id}: {question.question}",
                element_id=question.question_id,
                remedy=f"Answer before implementation; owner is {question.owner}.",
            )
        )
        index += 1

    return _assemble(
        subject=prd,
        reviewer=reviewer,
        perspectives=(ReviewPerspective.REQUIREMENT_QUALITY,),
        findings=findings,
        summary=(
            f"{len(prd.requirements)} requirement(s), "
            f"{len(prd.acceptance_criteria)} acceptance criterion/criteria, "
            f"{len(prd.open_questions)} open question(s)."
        ),
    )


def review_plan(
    plan: ImplementationPlanDocument,
    architecture: SystemArchitectureDocument,
    *,
    prd: PRDDocument | None = None,
    reviewer: str = "critic",
) -> ReviewDocument:
    """Review an implementation plan: can it be ordered, and does it cover the work?"""
    findings: list[ReviewFinding] = []
    index = 1

    cycle = find_cycle(plan)
    if cycle:
        findings.append(
            _finding(
                index,
                ReviewPerspective.PLAN_EXECUTABILITY,
                FindingSeverity.BLOCKER,
                "PLAN_CIRCULAR_DEPENDENCY",
                f"the plan cannot be ordered: {' -> '.join(cycle)}",
                remedy="Break the cycle by removing or reversing one dependency.",
            )
        )
        index += 1

    known_components = architecture.component_ids
    for task in plan.tasks:
        for component in task.components:
            if component not in known_components:
                findings.append(
                    _finding(
                        index,
                        ReviewPerspective.PLAN_EXECUTABILITY,
                        FindingSeverity.BLOCKER,
                        "PLAN_UNKNOWN_COMPONENT",
                        f"{task.task_id} names component {component}, which does not exist",
                        element_id=task.task_id,
                        remedy="Point the task at a component the architecture defines.",
                    )
                )
                index += 1

    if prd is not None:
        for issue in coverage_issues(prd, plan=plan):
            findings.append(
                _finding(
                    index,
                    ReviewPerspective.PLAN_EXECUTABILITY,
                    FindingSeverity.MAJOR,
                    issue.code,
                    issue.message,
                    element_id=issue.element_id,
                    remedy="Add a task that implements it, or record why none is needed.",
                )
            )
            index += 1

    return _assemble(
        subject=plan,
        reviewer=reviewer,
        perspectives=(ReviewPerspective.PLAN_EXECUTABILITY,),
        findings=findings,
        summary=(
            f"{len(plan.tasks)} task(s) covering "
            f"{len(plan.covered_requirements)} requirement(s)."
        ),
    )


def _assemble(
    *,
    subject: ArtifactDocument,
    reviewer: str,
    perspectives: tuple[ReviewPerspective, ...],
    findings: list[ReviewFinding],
    summary: str,
) -> ReviewDocument:
    """Build a review document and derive its verdict from the findings.

    The verdict is computed, never asserted. A review with a blocker is a FAIL whatever its
    author thinks, which is the same principle the M2.1 Critic runs on: "looks good" is not
    a verification result.
    """
    blocking = [finding for finding in findings if finding.severity.blocking]
    verdict = Verdict.FAIL if blocking else Verdict.PASS

    document = ReviewDocument(
        subject=ArtifactRef(
            artifact_id=new_id("art"),
            kind=getattr(type(subject), "kind", ArtifactKind.PRD),
            version=1,
        ),
        reviewer=reviewer,
        perspectives=perspectives,
        verdict=verdict,
        findings=tuple(findings),
        summary=summary,
    )
    logger.info(
        "product.review",
        reviewer=reviewer,
        verdict=str(verdict),
        findings=len(findings),
        blockers=len(blocking),
    )
    return document


def bind_subject(review: ReviewDocument, subject: ArtifactRef) -> ReviewDocument:
    """Attach the real artifact reference to a review.

    The review functions take documents rather than artifacts, so they can be used on an
    unsaved draft. This binds the result to a stored artifact once one exists, keeping the
    reviewed *version* recorded rather than merely the artifact id.
    """
    return review.model_copy(update={"subject": subject})


def merge_findings(reviews: tuple[ReviewDocument, ...]) -> tuple[ReviewFinding, ...]:
    """Every finding across several reviews, renumbered into one stable sequence."""
    merged: list[ReviewFinding] = []
    for review in reviews:
        for finding in review.findings:
            merged.append(
                finding.model_copy(update={"finding_id": f"FND-{len(merged) + 1:03d}"})
            )
    return tuple(merged)


def overall_verdict(reviews: tuple[ReviewDocument, ...]) -> Verdict:
    """The combined verdict: any blocker anywhere fails the set."""
    if any(review.blockers for review in reviews):
        return Verdict.FAIL
    return Verdict.PASS


def review_timestamp() -> str:
    """An ISO timestamp for a review record."""
    return utc_now().isoformat()
