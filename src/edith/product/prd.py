"""The Product Requirements Document.

The PRD is the root of the traceability chain. Everything downstream -- a UX flow, an
architecture decision, an implementation task, a test -- eventually points back at a
requirement id defined here, so these ids are the most load-bearing strings in the system.

Three design choices carry most of the weight:

**Requirements are records, not sentences.** A requirement has an id, a priority, an
authority level, and structured properties. A PRD written as prose can be read but not
checked; one written as records can be verified for coverage, contradiction, and
traceability without asking a model anything.

**Acceptance criteria reference requirements.** An ``AC`` that names no ``REQ`` is a test
nobody asked for, and a ``REQ`` with no ``AC`` is a requirement nobody can verify. Both are
detectable, and validation reports them.

**Authority travels with the requirement.** A requirement an agent drafted is an
``AGENT_RECOMMENDATION`` until a human approves it, at which point it becomes a
``USER_APPROVED_REQUIREMENT`` and outranks everything the system can produce on its own.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator, model_validator

from edith.authority import AuthorityLevel
from edith.schemas.common import EdithModel

from .artifacts import ArtifactDocument, ArtifactKind, is_element_id, register_document
from .properties import ProductProperty


class RequirementKind(StrEnum):
    """What sort of requirement this is.

    Kept distinct because they are verified differently: a functional requirement is
    demonstrated by a test, a non-functional one by a measurement, and a constraint by an
    inspection of what was built.
    """

    FUNCTIONAL = "FUNCTIONAL"
    NON_FUNCTIONAL = "NON_FUNCTIONAL"
    CONSTRAINT = "CONSTRAINT"


class Priority(StrEnum):
    """MoSCoW priority.

    ``WONT`` is not a rejected requirement -- it is an explicit decision to exclude
    something, which is worth recording precisely because it stops the question being
    reopened every sprint.
    """

    MUST = "MUST"
    SHOULD = "SHOULD"
    COULD = "COULD"
    WONT = "WONT"


class Requirement(EdithModel):
    """One requirement, addressable by a stable id."""

    requirement_id: str = Field(pattern=r"^REQ-\d{3,}$")
    title: str = Field(min_length=1, max_length=200)
    #: The requirement itself, stated so that satisfaction is checkable.
    statement: str = Field(min_length=1, max_length=2000)
    kind: RequirementKind = RequirementKind.FUNCTIONAL
    priority: Priority = Priority.MUST
    rationale: str = Field(default="", max_length=1000)
    #: Structural claims this requirement makes, for deterministic contradiction detection.
    properties: frozenset[ProductProperty] = frozenset()
    #: Authority of this requirement. Drafts are recommendations until a human approves.
    authority: AuthorityLevel = AuthorityLevel.AGENT_RECOMMENDATION
    #: Where it came from: a user message, a research report id, an interview note.
    source: str = Field(default="", max_length=300)
    #: Requirements this one depends on or refines.
    related_to: tuple[str, ...] = ()

    @field_validator("related_to")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"related_to entry {identifier!r} is not a REQ id")
        return value

    @model_validator(mode="after")
    def _no_self_reference(self) -> Requirement:
        if self.requirement_id in self.related_to:
            raise ValueError(f"{self.requirement_id} cannot relate to itself")
        return self

    @property
    def binding(self) -> bool:
        """Whether this requirement must be satisfied for the product to be acceptable."""
        return self.priority is Priority.MUST and self.authority in {
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
            AuthorityLevel.AGENT_RECOMMENDATION,
        }


class AcceptanceCriterion(EdithModel):
    """A checkable condition, tied to the requirements it verifies.

    ``verifies`` is required and non-empty: an acceptance criterion that names no
    requirement is a test for something nobody asked for.
    """

    criterion_id: str = Field(pattern=r"^AC-\d{3,}$")
    statement: str = Field(min_length=1, max_length=1000)
    verifies: tuple[str, ...] = Field(min_length=1)
    #: How it will be checked: a test, a measurement, a manual inspection.
    method: str = Field(default="test", max_length=100)

    @field_validator("verifies")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"verifies entry {identifier!r} is not a REQ id")
        return value


class Persona(EdithModel):
    """A named user archetype."""

    persona_id: str = Field(pattern=r"^PER-\d{3,}$")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    goals: tuple[str, ...] = ()
    frustrations: tuple[str, ...] = ()


class UserStory(EdithModel):
    """A story in the standard form, tied to a persona and its requirements."""

    story_id: str = Field(pattern=r"^US-\d{3,}$")
    #: "As a <role>" -- a PER id when the persona is defined, otherwise a plain role.
    persona_id: str = Field(default="", max_length=40)
    #: "I want <capability>".
    capability: str = Field(min_length=1, max_length=500)
    #: "So that <benefit>".
    benefit: str = Field(default="", max_length=500)
    implements: tuple[str, ...] = ()

    @field_validator("implements")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"implements entry {identifier!r} is not a REQ id")
        return value

    def render(self) -> str:
        """The story in its conventional sentence form."""
        who = self.persona_id or "user"
        tail = f" so that {self.benefit}" if self.benefit else ""
        return f"As {who}, I want {self.capability}{tail}."


class Risk(EdithModel):
    """Something that could stop the product succeeding."""

    risk_id: str = Field(pattern=r"^RISK-\d{3,}$")
    description: str = Field(min_length=1, max_length=1000)
    #: LOW / MEDIUM / HIGH. Free-form rather than an enum because a 3B model produces these
    #: as plain words and the value is advisory, not load-bearing.
    likelihood: str = Field(default="MEDIUM", max_length=20)
    impact: str = Field(default="MEDIUM", max_length=20)
    mitigation: str = Field(default="", max_length=1000)
    affects: tuple[str, ...] = ()


class SuccessMetric(EdithModel):
    """How the product will be judged after it ships."""

    metric_id: str = Field(pattern=r"^KPI-\d{3,}$")
    name: str = Field(min_length=1, max_length=200)
    #: What is actually counted, stated so it could be instrumented.
    measurement: str = Field(min_length=1, max_length=500)
    target: str = Field(default="", max_length=200)
    measures: tuple[str, ...] = ()


class OpenQuestion(EdithModel):
    """Something the PM could not resolve.

    Recorded rather than guessed at. A PRD that answers every question by invention is worse
    than one that admits what it does not know, because the invention is indistinguishable
    from a requirement downstream.
    """

    question_id: str = Field(pattern=r"^Q-\d{3,}$")
    question: str = Field(min_length=1, max_length=1000)
    #: Why it matters: what decision is blocked until it is answered.
    blocks: str = Field(default="", max_length=500)
    owner: str = Field(default="user", max_length=60)


@register_document
class PRDDocument(ArtifactDocument):
    """A Product Requirements Document.

    The body of a :attr:`~edith.product.artifacts.ArtifactKind.PRD` artifact.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.PRD

    product_name: str = Field(min_length=1, max_length=200)
    problem: str = Field(min_length=1, max_length=4000)
    target_users: str = Field(default="", max_length=2000)
    personas: tuple[Persona, ...] = ()
    goals: tuple[str, ...] = ()
    #: What this product deliberately will not do. As important as the goals: a non-goal is
    #: what stops scope arriving later disguised as an obvious omission.
    non_goals: tuple[str, ...] = ()
    user_stories: tuple[UserStory, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    risks: tuple[Risk, ...] = ()
    success_metrics: tuple[SuccessMetric, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> PRDDocument:
        """Every id defined by this document must be unique.

        A duplicate ``REQ-002`` makes every downstream reference ambiguous, and the ambiguity
        would not surface until something built the wrong thing.
        """
        identifiers = list(self.element_ids())
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise ValueError(f"PRD defines duplicate element ids: {', '.join(duplicates)}")
        return self

    def element_ids(self) -> tuple[str, ...]:
        """Every stable id this PRD defines."""
        return (
            tuple(item.requirement_id for item in self.requirements)
            + tuple(item.criterion_id for item in self.acceptance_criteria)
            + tuple(item.persona_id for item in self.personas)
            + tuple(item.story_id for item in self.user_stories)
            + tuple(item.risk_id for item in self.risks)
            + tuple(item.metric_id for item in self.success_metrics)
            + tuple(item.question_id for item in self.open_questions)
        )

    def referenced_ids(self) -> tuple[str, ...]:
        """Every id this PRD points at, including its own."""
        references: list[str] = []
        for requirement in self.requirements:
            references.extend(requirement.related_to)
        for criterion in self.acceptance_criteria:
            references.extend(criterion.verifies)
        for story in self.user_stories:
            references.extend(story.implements)
            if story.persona_id:
                references.append(story.persona_id)
        for risk in self.risks:
            references.extend(risk.affects)
        for metric in self.success_metrics:
            references.extend(metric.measures)
        return tuple(references)

    # -- Convenience -----------------------------------------------------------------

    @property
    def requirement_ids(self) -> frozenset[str]:
        """Ids of every requirement in this PRD."""
        return frozenset(item.requirement_id for item in self.requirements)

    def requirement(self, requirement_id: str) -> Requirement | None:
        """Look up one requirement by id."""
        for item in self.requirements:
            if item.requirement_id == requirement_id:
                return item
        return None

    @property
    def declared_properties(self) -> frozenset[ProductProperty]:
        """Every structural property the requirements claim.

        This is what an architecture is checked against for contradictions.
        """
        result: set[ProductProperty] = set()
        for requirement in self.requirements:
            result |= requirement.properties
        return frozenset(result)

    def unverified_requirements(self) -> tuple[str, ...]:
        """Requirements no acceptance criterion verifies.

        A requirement nobody can check is a requirement nobody will notice missing.
        """
        verified = {
            requirement_id
            for criterion in self.acceptance_criteria
            for requirement_id in criterion.verifies
        }
        return tuple(
            requirement.requirement_id
            for requirement in self.requirements
            if requirement.requirement_id not in verified
        )

    def render(self) -> str:
        """Render for a human reader or a downstream agent's prompt."""
        lines = [f"# {self.product_name}", "", "## Problem", self.problem, ""]
        if self.goals:
            lines.extend(["## Goals", *(f"- {goal}" for goal in self.goals), ""])
        if self.non_goals:
            lines.extend(["## Non-goals", *(f"- {item}" for item in self.non_goals), ""])
        if self.requirements:
            lines.append("## Requirements")
            for requirement in self.requirements:
                lines.append(
                    f"- {requirement.requirement_id} [{requirement.priority}] "
                    f"{requirement.title}: {requirement.statement}"
                )
            lines.append("")
        if self.acceptance_criteria:
            lines.append("## Acceptance criteria")
            for criterion in self.acceptance_criteria:
                lines.append(
                    f"- {criterion.criterion_id} verifies "
                    f"{', '.join(criterion.verifies)}: {criterion.statement}"
                )
            lines.append("")
        if self.open_questions:
            lines.append("## Open questions")
            for question in self.open_questions:
                lines.append(f"- {question.question_id}: {question.question}")
        return "\n".join(lines).strip()
