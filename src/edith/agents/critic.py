"""The Critic / Judge Agent.

Independent of the Coding Agent by construction: it is read-only, it is given the *real*
verification evidence rather than the coder's report, and its verdict is overruled by that
evidence.

The last point is the important one. A small model asked "did this work?" will often say
yes. So the Critic's verdict is not the final word: :meth:`CriticAgent.adjudicate` applies
the deterministic rule first -- **if the tests failed, the verdict is FAIL, whatever the
model said**. The model's judgement only decides cases the evidence leaves open.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.integrity import IntegrityReport
from edith.observability.logging import get_logger
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel, Severity, Verdict
from edith.schemas.model import Message, Role
from edith.verification.runner import VerificationReport

from .base import Agent

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the independent reviewer in a software engineering system.

You judge whether a completed task actually satisfies its requirements. You are given the
task, the code diff, and REAL output from tests that were actually executed.

Rules:
- Trust the test evidence over any claim about the code.
- PASS only if the task is genuinely complete and the evidence supports it.
- FAIL if the change is wrong, incomplete, or the tests failed.
- BLOCKED if you cannot judge because something outside the code is broken.
- Every finding must reference concrete evidence. Never write "looks good".
- If existing tests were changed, weakened, deleted or skipped, that is a FAIL unless the
  task explicitly asked for it. Making a test match broken code is not a fix."""

USER_TEMPLATE = """TASK: {title}

{description}

{criteria}CHANGED FILES: {files}

DIFF:
{diff}

VERIFICATION EVIDENCE (actually executed):
{evidence}

TEST INTEGRITY (deterministic comparison against the baseline):
{integrity}

Judge whether this task is complete."""


class Finding(EdithModel):
    """One issue the Critic identified."""

    severity: Severity = Severity.MEDIUM
    category: str = Field(default="correctness", max_length=60)
    description: str = Field(min_length=1, max_length=1000)
    evidence: str = Field(default="", max_length=1000)
    affected_files: list[str] = Field(default_factory=list, max_length=10)
    required_action: str = Field(default="", max_length=500)


class CriticInput(EdithModel):
    """Input contract for :class:`CriticAgent`."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=10)
    changed_files: list[str] = Field(default_factory=list, max_length=20)
    diff: str = Field(default="", max_length=20_000)
    evidence: str = Field(default="", max_length=8000)
    #: Deterministic baseline comparison of the test suite. Given to the Critic so it can
    #: reason about *what changed*, not just about whether the suite is green.
    integrity: str = Field(default="", max_length=4000)


class CriticOutput(EdithModel):
    """Output contract for :class:`CriticAgent`."""

    verdict: Verdict = Verdict.FAIL
    reasoning: str = Field(default="", max_length=2000)
    findings: list[Finding] = Field(default_factory=list, max_length=10)

    @property
    def passed(self) -> bool:
        """Whether the verdict is PASS."""
        return self.verdict is Verdict.PASS

    @property
    def blocking_findings(self) -> list[Finding]:
        """Findings severe enough to require a fix."""
        return [
            finding
            for finding in self.findings
            if finding.severity in {Severity.HIGH, Severity.CRITICAL}
        ]


class CriticAgent(Agent):
    """Independently judges whether a task is genuinely complete.

    Read-only: no write scope and no shell. A judge that could edit the code it is judging
    is not a judge.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="critic",
        description="Independently judges task completion against real verification evidence.",
        capabilities=frozenset({Capability.CODE_REVIEW}),
        permissions=AgentPermissions(
            allowed_tools=frozenset(
                {"filesystem.read", "filesystem.search", "git.diff", "git.status"}
            ),
            allowed_read_paths=("**",),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = CriticInput
    output_schema: ClassVar[type[BaseModel]] = CriticOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, CriticInput)  # noqa: S101 - guaranteed by validate_input
        provider = self.require_provider()

        criteria = ""
        if payload.acceptance_criteria:
            listed = "\n".join(f"- {item}" for item in payload.acceptance_criteria)
            criteria = f"ACCEPTANCE CRITERIA:\n{listed}\n\n"

        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    title=payload.title,
                    description=payload.description,
                    criteria=criteria,
                    files=", ".join(payload.changed_files) or "(none reported)",
                    diff=payload.diff or "(no diff available)",
                    evidence=payload.evidence or "(no verification was run)",
                    integrity=payload.integrity or "(not checked)",
                ),
            ),
        ]
        return provider.structured_generate(messages, CriticOutput, max_repair_attempts=2)


def adjudicate(
    critic: CriticOutput | None,
    report: VerificationReport,
    *,
    changes_made: bool,
    integrity: IntegrityReport | None = None,
) -> tuple[Verdict, str]:
    """Combine deterministic evidence with the Critic's judgement.

    Evidence wins. The model is consulted only where the evidence is silent, because a small
    model's "PASS" is not reliable enough to be the final gate on whether work is done
    (CLAUDE.md: never allow "looks good" to count as verification).

    The integrity gate runs **before** the test results are even considered. That ordering
    is the whole lesson of the M2 false positive: an agent that weakens a test can make the
    suite green, so "the tests passed" is only meaningful once you know the tests are still
    the tests. Passing verification with tampered tests is a *worse* outcome than failing
    it, and is treated as such.

    Args:
        critic: The Critic's verdict, or ``None`` when it could not be obtained.
        report: Real verification results.
        changes_made: Whether the coder actually wrote anything.
        integrity: Baseline comparison of the test suite, when one could be made.

    Returns:
        ``(verdict, reason)``.
    """
    if integrity is not None and integrity.tampered:
        blocking = integrity.blocking_findings
        return (
            Verdict.FAIL,
            f"test integrity violated: {blocking[0].detail} "
            f"({len(blocking)} blocking finding(s)); a green suite proves nothing when the "
            f"tests themselves were weakened",
        )

    unavailable = [outcome for outcome in report.outcomes if not outcome.ran]
    if unavailable and not report.outcomes[0].ran:
        return (
            Verdict.BLOCKED,
            f"verification could not run: {unavailable[0].unavailable_reason}",
        )

    if report.failures:
        failure = report.failures[0]
        return (
            Verdict.FAIL,
            f"{failure.kind} failed with exit code {failure.exit_code}",
        )

    if not changes_made:
        return (Verdict.FAIL, "no files were changed, so the task cannot be complete")

    if not report.outcomes:
        # Nothing was configured to verify this task. Fall back to the model, but say so.
        if critic is None:
            return (Verdict.BLOCKED, "no verification configured and no critic verdict")
        return (
            critic.verdict,
            f"no verification configured; critic said {critic.verdict}",
        )

    if critic is None:
        return (Verdict.PASS, "verification passed; no critic verdict was available")

    if critic.verdict is Verdict.PASS:
        return (Verdict.PASS, "verification passed and the critic agreed")

    # Evidence says the checks passed but the Critic sees a real problem. Only a
    # high-severity finding is allowed to override passing tests -- otherwise a model's
    # vague misgiving would block completed work forever.
    if critic.blocking_findings:
        return (
            Verdict.FAIL,
            f"verification passed but the critic raised "
            f"{len(critic.blocking_findings)} high-severity finding(s)",
        )
    return (
        Verdict.PASS,
        "verification passed; critic concerns were not severe enough to block",
    )
