"""Quality artifacts: findings, evidence, verdicts, and who is allowed to decide.

M5.2 got EDITH as far as "the generated code executes". This layer answers the different
question of whether it is *acceptable*, and the structure exists mainly to stop one specific
failure: a model saying PASS louder than the evidence says FAIL.

Three rules are enforced by the types themselves rather than by convention.

**A finding without evidence is not a finding.** ``QualityFinding.evidence`` is required and
non-empty. "This code looks bad" cannot be constructed, so it cannot be reported, so it cannot
consume a repair attempt or block a merge.

**Every finding declares where it came from.** :class:`FindingOrigin` separates a deterministic
result (a test exited non-zero; a scanner matched a pattern) from a model's opinion. The
adjudicator weighs them differently, and it can only do that if the distinction survives into
the report.

**Deterministic evidence outranks model judgement.** :func:`adjudicate` is a pure function over
the collected findings. It takes the Judge's verdict as *input*, never as the answer -- M6
item 8: no model-generated verdict authorizes a merge. This mirrors M2.1's critic rule, where
a failing test suite is a FAIL whatever the critic said, generalised to every gate.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from edith.errors import FailureCategory
from edith.schemas.common import EdithModel, Severity


class QualityVerdict(StrEnum):
    """The adjudicated outcome of the quality pipeline.

    Deliberately richer than M2's :class:`~edith.schemas.common.Verdict`, which stays as it is
    because M2's gates depend on its three states. The distinction that matters here is
    between "not acceptable, and the agent can fix it" and "not acceptable, and it cannot" --
    collapsing those is how an environment fault becomes a false verdict on an agent's work.
    """

    PASS = "PASS"  # noqa: S105 - a verdict, not a credential
    #: Acceptable, with findings worth recording that do not block.
    PASS_WITH_ADVISORIES = "PASS_WITH_ADVISORIES"  # noqa: S105 - a verdict, not a credential
    #: Blocking findings the responsible agent could plausibly fix.
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    #: Blocking findings nobody can fix by rewriting code: security, policy, environment.
    BLOCKED = "BLOCKED"
    #: The pipeline itself could not reach a judgement.
    FAILED = "FAILED"

    @property
    def merges(self) -> bool:
        """Whether this verdict permits a merge."""
        return self in {QualityVerdict.PASS, QualityVerdict.PASS_WITH_ADVISORIES}


class FindingOrigin(StrEnum):
    """Whether a finding is a measurement or an opinion.

    Not cosmetic. A deterministic finding is reproducible and can block on its own; a model
    finding is a hypothesis, and M6 requires that LLM judgement never replace a deterministic
    check that could have answered the same question.
    """

    #: Produced by executing something or matching a rule: a test run, a scanner, a parser.
    DETERMINISTIC = "DETERMINISTIC"
    #: Produced by a model reviewing code. Advisory unless corroborated.
    MODEL = "MODEL"


class ReviewEvidence(EdithModel):
    """What was actually observed, in a form a human can check.

    Kept separate from the finding's prose so that "here is what I claim" and "here is what I
    saw" cannot be conflated. A finding may cite several pieces.
    """

    #: What produced this observation: a command, a scanner name, a check id.
    source: str = Field(min_length=1, max_length=200)
    #: The observation itself: output, a matched line, a diff fragment.
    detail: str = Field(min_length=1, max_length=4000)
    #: Where it was observed, when that is a file.
    file: str = Field(default="", max_length=400)
    #: 1-indexed line, when the observation anchors to one.
    line: int | None = Field(default=None, ge=1)

    @field_validator("source", "detail")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence source and detail must not be blank")
        return value


class QualityFinding(EdithModel):
    """One defect, with everything needed to act on it or dispute it.

    M6 item 7 lists the required fields; the reason each is mandatory is that omitting it
    permits a specific abuse. Without ``origin`` a model opinion can masquerade as a
    measurement. Without ``repairable`` a security finding can be fed to a coder as though
    rewriting the function would help. Without ``evidence`` nothing is falsifiable.
    """

    category: str = Field(min_length=1, max_length=60)
    severity: Severity = Severity.MEDIUM
    summary: str = Field(min_length=1, max_length=1000)
    #: At least one observation. A finding with no evidence cannot be constructed.
    evidence: tuple[ReviewEvidence, ...] = Field(min_length=1, max_length=20)
    affected_files: tuple[str, ...] = Field(default=(), max_length=20)
    #: The requirement or task this bears on, when known.
    requirement_id: str = Field(default="", max_length=120)
    task_id: str = Field(default="", max_length=120)
    origin: FindingOrigin = FindingOrigin.MODEL
    #: The taxonomy entry this belongs to, which decides repairability downstream.
    failure_category: FailureCategory | None = None
    #: Whether rewriting the implementation could plausibly resolve it.
    repairable: bool = False
    #: 0.0-1.0. Deterministic findings are 1.0 by construction; see :meth:`normalise`.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def blocking(self) -> bool:
        """Whether this finding alone prevents a merge.

        HIGH and CRITICAL block. A model's unaided opinion does not block on its own, because
        M6 makes deterministic checks authoritative -- but a *corroborated* model finding
        (one carrying deterministic evidence) does, which is why origin is what decides.
        """
        if self.severity not in {Severity.HIGH, Severity.CRITICAL}:
            return False
        return self.origin is FindingOrigin.DETERMINISTIC

    def normalise(self) -> QualityFinding:
        """Return a copy with confidence consistent with the finding's origin."""
        if self.origin is FindingOrigin.DETERMINISTIC and self.confidence < 1.0:
            return self.model_copy(update={"confidence": 1.0})
        return self


class RepairRecommendation(EdithModel):
    """What the responsible agent should change, and why it is worth an attempt.

    Only produced for findings that are actually repairable. M5.2 established that spending
    budget on an unrepairable failure both wastes the attempt and misattributes the failure to
    the agent's work; this type is where that policy becomes visible to the executor.
    """

    finding_category: str = Field(min_length=1, max_length=60)
    action: str = Field(min_length=1, max_length=1000)
    #: The files the repair is expected to touch. Bounds the scope handed to the agent.
    target_files: tuple[str, ...] = Field(default=(), max_length=20)
    #: The evidence to show the agent. Repairing without the failure output does not work.
    evidence: tuple[ReviewEvidence, ...] = Field(default=(), max_length=10)


class QualityReport(EdithModel):
    """Every finding for one task, plus the verdict deterministically derived from them.

    The verdict is not a field an agent sets. It is computed by :func:`adjudicate` from the
    findings, so a report cannot claim PASS while carrying a blocking finding.
    """

    task_id: str = Field(default="", max_length=120)
    findings: tuple[QualityFinding, ...] = Field(default=(), max_length=200)
    #: The Judge's opinion, recorded as evidence and never as the decision.
    judge_verdict: QualityVerdict | None = None
    judge_rationale: str = Field(default="", max_length=2000)

    @property
    def blocking(self) -> tuple[QualityFinding, ...]:
        """Findings that prevent a merge."""
        return tuple(item for item in self.findings if item.blocking)

    @property
    def advisories(self) -> tuple[QualityFinding, ...]:
        """Findings worth recording that do not block."""
        return tuple(item for item in self.findings if not item.blocking)

    def by_severity(self, severity: Severity) -> tuple[QualityFinding, ...]:
        return tuple(item for item in self.findings if item.severity is severity)

    def recommendations(self) -> tuple[RepairRecommendation, ...]:
        """Repair guidance for the blocking findings that are actually repairable."""
        return tuple(
            RepairRecommendation(
                finding_category=item.category,
                action=item.summary,
                target_files=item.affected_files,
                evidence=item.evidence[:10],
            )
            for item in self.blocking
            if item.repairable
        )

    def verdict(self) -> QualityVerdict:
        """The adjudicated verdict. See :func:`adjudicate`."""
        return adjudicate(self)


def adjudicate(report: QualityReport) -> QualityVerdict:
    """Derive the verdict from evidence, taking the Judge's view only as input.

    The precedence is the whole point of M6:

    1. A CRITICAL deterministic finding is ``BLOCKED``. Nothing overrides it -- not a passing
       suite, not a Judge saying PASS. M6 item 4: green tests are necessary, not sufficient.
    2. Remaining blocking findings are ``REPAIR_REQUIRED`` if *every* one of them is
       repairable, otherwise ``BLOCKED``. A set containing one unrepairable finding cannot be
       handed to a coder as though effort would clear it.
    3. Only with no blocking findings does the Judge's opinion matter, and even then it can
       only *lower* the verdict. A Judge may withhold a pass it doubts; it may not grant one
       the evidence does not support.

    This is deliberately a pure function of the report. It has no model, no I/O, and no
    configuration, so its behaviour under an adversarial agent is testable exhaustively.
    """
    blocking = report.blocking

    if any(item.severity is Severity.CRITICAL for item in blocking):
        return QualityVerdict.BLOCKED

    if blocking:
        if all(item.repairable for item in blocking):
            return QualityVerdict.REPAIR_REQUIRED
        return QualityVerdict.BLOCKED

    # No blocking evidence. The Judge may still decline, but cannot upgrade.
    judged = report.judge_verdict
    if judged is not None and not judged.merges:
        return judged

    if report.advisories:
        return QualityVerdict.PASS_WITH_ADVISORIES
    return QualityVerdict.PASS
