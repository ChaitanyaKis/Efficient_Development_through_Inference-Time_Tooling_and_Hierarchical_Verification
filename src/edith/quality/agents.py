"""The four model-backed quality agents.

Each is a thin reviewer over the deterministic layer, never a replacement for it. Their output
reaches the pipeline through :meth:`QualityPipeline.model_findings_gate`, which rewrites every
finding's origin to ``MODEL`` -- so nothing here can promote its own opinion into a merge gate,
however confidently it words it.

**Schemas are small on purpose.** M4.1 measured this model producing 0/10 valid documents for a
large schema and 10/10 once the request was decomposed, and M5.1 lost a whole benchmark arm to
a hoisted envelope key. So each agent asks for a short list of flat objects with three or four
required fields, and nothing nested beyond one level. A reviewer that returns nothing usable is
worse than no reviewer, because it costs a model call and produces false confidence.

**Identity and authority stay system-owned.** The model supplies a description and a suggested
severity; the *origin*, the confidence, the task id, and whether a finding blocks are all set
by EDITH. M4 established that rule for product artifacts and it applies with more force here,
where the model is being asked to grade work.

**Evidence is required, and unsupported findings are dropped.** :func:`to_findings` refuses any
model finding whose quoted evidence does not appear in the source it claims to be reviewing.
A model that invents a line to justify a verdict gets that finding discarded rather than
recorded, which is the only defence against hallucinated evidence that does not itself rely on
the model being honest.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.agents.base import Agent
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    QualityVerdict,
    ReviewEvidence,
)
from edith.quality.principals import JUDGE, REVIEWER, SECURITY, TESTER
from edith.schemas.agent import AgentIdentity, AgentRequest, Capability
from edith.schemas.common import EdithModel, Severity
from edith.schemas.model import Message, Role

logger = get_logger(__name__)

#: How much of a file a reviewer is shown. M3.2's lesson: a bigger prompt is not a better one,
#: and this model degrades well before its nominal 4096-token context is full.
SOURCE_BUDGET = 2400

#: How many findings a reviewer may return. A long list from a 3B model is noise, not thoroughness.
MAX_FINDINGS = 5


class ModelFinding(EdithModel):
    """One finding as the *model* states it. Deliberately four flat fields.

    No origin, no confidence, no task id, no repairability: those are EDITH's to assign, and a
    model that could set them could grade its own authority.
    """

    category: str = Field(min_length=1, max_length=60)
    severity: Severity = Severity.MEDIUM
    description: str = Field(min_length=1, max_length=400)
    #: A line quoted verbatim from the source. Checked against the file; see :func:`to_findings`.
    quoted_line: str = Field(default="", max_length=300)


class ReviewOutput(EdithModel):
    """What every reviewing agent returns."""

    findings: list[ModelFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)


class ReviewInput(EdithModel):
    """What a reviewing agent is shown. One file at a time, deliberately."""

    path: str = Field(min_length=1, max_length=400)
    source: str = Field(min_length=1)
    task: str = Field(default="", max_length=600)


class JudgeInput(EdithModel):
    """The Judge sees conclusions, not raw material.

    It is given the deterministic findings as *text* rather than as a structure it could
    contradict field by field, because its job is to add a view, not to re-adjudicate.
    """

    task: str = Field(default="", max_length=600)
    tests_passed: bool = False
    deterministic_findings: str = Field(default="", max_length=3000)
    changed_files: str = Field(default="", max_length=1000)


class JudgeOutput(EdithModel):
    """The Judge's opinion. Recorded as evidence; never the decision."""

    verdict: QualityVerdict = QualityVerdict.FAILED
    rationale: str = Field(default="", min_length=0, max_length=800)


def to_findings(
    output: ReviewOutput,
    *,
    path: str,
    source: str,
    agent: str,
    task_id: str = "",
) -> tuple[QualityFinding, ...]:
    """Convert model output into quality findings, discarding unsupported ones.

    Two system-owned decisions happen here and nowhere else:

    *Evidence is verified.* If the model quotes a line, that line must actually occur in the
    file. A quote that does not is a hallucination, and the finding built on it is dropped with
    a warning rather than recorded at a lower severity -- a fabricated citation does not become
    trustworthy by being called advisory.

    *Origin is assigned, not accepted.* Every finding is ``MODEL``. The pipeline enforces this
    again at the gate; doing it twice is deliberate, because the property is load-bearing.
    """
    findings: list[QualityFinding] = []
    for item in output.findings:
        quote = item.quoted_line.strip()
        if quote and quote not in source:
            logger.warning(
                "quality.unsupported_finding",
                agent=agent,
                path=path,
                category=item.category,
                reason="quoted evidence does not appear in the source",
            )
            continue
        detail = quote or item.description
        findings.append(
            QualityFinding(
                category=item.category,
                severity=item.severity,
                summary=item.description,
                evidence=(
                    ReviewEvidence(source=f"model:{agent}", detail=detail, file=path),
                ),
                affected_files=(path,),
                task_id=task_id,
                origin=FindingOrigin.MODEL,
                repairable=True,
                confidence=0.5,
            )
        )
    return tuple(findings)


def _clip(text: str, budget: int = SOURCE_BUDGET) -> str:
    """Trim a file to the review budget, keeping the head where definitions live."""
    if len(text) <= budget:
        return text
    return text[:budget] + "\n# ... truncated for review ...\n"


class _ReviewBehaviour(Agent):
    """Shared behaviour for the three file reviewers.

    A mixin rather than an abstract Agent: the Agent base rejects a concrete subclass with no
    identity at import time, which is the right check, so the shared ``_run`` lives here and
    each agent supplies its own identity. Same shape as M5's ``_EngineeringBehaviour``.
    """

    prompt: ClassVar[str] = ""
    input_schema: ClassVar[type[BaseModel]] = ReviewInput
    output_schema: ClassVar[type[BaseModel]] = ReviewOutput

    #: Marks this class as abstract for :meth:`Agent.__init_subclass__`, which requires a
    #: concrete agent to declare an identity. This one deliberately has none.
    __abstractmethods__ = frozenset({"identity"})

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, ReviewInput)  # noqa: S101 - guaranteed by validate_input
        provider: ModelProvider = self.require_provider()
        task = f"TASK: {payload.task}\n\n" if payload.task else ""
        messages = [
            Message(role=Role.SYSTEM, content=self.prompt),
            Message(
                role=Role.USER,
                content=(
                    f"{task}FILE: {payload.path}\n\n```python\n{_clip(payload.source)}\n```\n\n"
                    "Report only defects you can point at a specific line for. "
                    "Quote that line verbatim in quoted_line. "
                    "If the file is fine, return an empty findings list."
                ),
            ),
        ]
        return provider.structured_generate(messages, ReviewOutput)


REVIEW_PROMPT = """You are a senior engineer reviewing one file.

Look for: incorrect logic, unhandled errors, duplicated logic, dead code, missing type safety,
inconsistent APIs, leaked resources, unsafe assumptions, and obvious concurrency problems.

Report a defect only if you can quote the exact line it is on. Do not report style preferences.
Do not report anything you cannot point at. An empty list is a valid and common answer."""

SECURITY_PROMPT = """You are a security reviewer examining one file.

Look for: missing authentication, missing authorization checks, unvalidated input, insecure
configuration, mishandled secrets, unsafe data flow into requests or the filesystem, injection
risk, and sensitive values reaching logs.

An automated scanner has already checked for eval, exec, os.system, shell=True, pickle, hardcoded
credentials, and path traversal. Do not repeat those. Report what needs understanding of the
code's intent.

Quote the exact line for every finding. An empty list is a valid answer."""

TESTING_PROMPT = """You are a test engineer reviewing one implementation file.

Identify behaviour that is not covered by tests: error paths, boundary values, empty and null
inputs, and documented behaviour with no corresponding assertion.

Report each gap as a finding whose description names the specific untested behaviour. Quote the
line of the implementation that lacks coverage. Do not propose tests that assert nothing.

An empty list is a valid answer."""


class CodeReviewAgent(_ReviewBehaviour, Agent):
    """Read-only correctness and maintainability review."""

    prompt: ClassVar[str] = REVIEW_PROMPT
    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="code_reviewer",
        description="Reviews one implementation file for correctness and maintainability.",
        capabilities=frozenset({Capability.CODE_REVIEW}),
        permissions=REVIEWER,
    )


class SecurityAgent(_ReviewBehaviour, Agent):
    """Read-only security review, supplementing the deterministic scanner.

    Holds no write scope and no shell: a security reviewer that can execute is a security
    reviewer that can be turned into a payload.
    """

    prompt: ClassVar[str] = SECURITY_PROMPT
    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="security_reviewer",
        description="Reviews one file for security defects a scanner cannot decide.",
        capabilities=frozenset({Capability.SECURITY_ANALYSIS}),
        permissions=SECURITY,
    )


class TestingAgent(_ReviewBehaviour, Agent):
    """Identifies untested behaviour.

    Writes only within the test scope. It cannot edit implementation, so it cannot make a suite
    green by changing the code under test -- the failure mode this separation exists to stop.
    """

    prompt: ClassVar[str] = TESTING_PROMPT
    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="testing",
        description="Identifies untested behaviour and writes tests within the test scope.",
        capabilities=frozenset({Capability.TESTING}),
        permissions=TESTER,
    )


JUDGE_PROMPT = """You are an independent reviewer giving a second opinion on completed work.

You are NOT the final authority. Automated gates have already run and their results are final:
if a test failed or a critical security defect was found, the work is rejected no matter what
you say. Your opinion is recorded alongside theirs.

Answer with one verdict:
- PASS: the evidence supports the work being complete and correct.
- PASS_WITH_ADVISORIES: acceptable, with minor concerns.
- REPAIR_REQUIRED: something is wrong that the author could fix.
- BLOCKED: something is wrong that the author cannot fix.

Give a one-sentence rationale. If you disagree with the automated findings, say so plainly --
your disagreement is kept as evidence."""


class JudgeAgent(Agent):
    """Produces a structured second opinion. Never authoritative.

    Its verdict enters :class:`~edith.quality.artifacts.QualityReport` as an *input* to
    adjudication, which can lower the outcome but never raise it. Holds read access alone: no
    shell, no writes, no git.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="judge",
        description="Gives an independent second opinion on a task's quality evidence.",
        capabilities=frozenset({Capability.CODE_REVIEW}),
        permissions=JUDGE,
    )
    input_schema: ClassVar[type[BaseModel]] = JudgeInput
    output_schema: ClassVar[type[BaseModel]] = JudgeOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, JudgeInput)  # noqa: S101 - guaranteed by validate_input
        provider: ModelProvider = self.require_provider()
        findings = payload.deterministic_findings or "none"
        messages = [
            Message(role=Role.SYSTEM, content=JUDGE_PROMPT),
            Message(
                role=Role.USER,
                content=(
                    f"TASK: {payload.task}\n"
                    f"TESTS PASSED: {payload.tests_passed}\n"
                    f"CHANGED FILES: {payload.changed_files or 'none'}\n\n"
                    f"AUTOMATED FINDINGS:\n{findings}\n"
                ),
            ),
        ]
        return provider.structured_generate(messages, JudgeOutput)


def render_findings(findings: tuple[QualityFinding, ...], limit: int = 8) -> str:
    """Summarise findings for the Judge's prompt.

    Text rather than structure, and capped: the Judge is being asked for a view on the whole,
    not to re-litigate each item, and M3.2 showed this model degrading as its prompt grows.
    """
    if not findings:
        return "none"
    lines = [
        f"- [{item.severity.value}] {item.category}: {item.summary[:160]}"
        for item in findings[:limit]
    ]
    if len(findings) > limit:
        lines.append(f"- (+{len(findings) - limit} more)")
    return "\n".join(lines)
