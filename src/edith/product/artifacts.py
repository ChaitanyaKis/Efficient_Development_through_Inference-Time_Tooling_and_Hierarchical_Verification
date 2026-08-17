"""The artifact envelope: how product-development agents talk to each other.

CLAUDE.md's architectural rule is that agents communicate through **structured artifacts and
project state**, not by chatting. This module is that contract made concrete.

An artifact is metadata plus a document. The metadata is uniform across every kind -- who
authored it, which project it belongs to, what it depends on, what authority it carries,
whether it validated, and which artifact it replaced. The document is kind-specific and
strictly typed, and the envelope refuses to hold a body that does not validate against its
kind's schema.

Two properties matter more than the field list:

**Nothing approved is ever destroyed.** A revision creates a *new* artifact that supersedes
its predecessor; the predecessor moves to ``SUPERSEDED`` and remains readable. An approved
PRD stays recoverable forever, which is what makes "why did we build it this way" answerable
six months later.

**Every element is addressable.** Requirements, flows, decisions and tasks carry stable
prefixed ids (``REQ-001``, ``UX-003``, ``ADR-002``, ``TASK-017``) and reference each other by
id. M4 does not build the full traceability graph -- that is M7's Impact Engine -- but every
edge that graph will need is recorded here from the start, because retrofitting identity
onto documents that were written without it is not possible.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from edith.authority import AuthorityLevel
from edith.schemas.common import EdithModel, new_id, utc_now


class ArtifactKind(StrEnum):
    """What a document is.

    The kind selects the schema its body must satisfy and the id prefix its elements use.
    """

    #: Product Requirements Document, authored by the Product Manager.
    PRD = "PRD"
    #: UX specification: flows, screens, components, states, tokens.
    UX_SPEC = "UX_SPEC"
    #: The Architect's system decomposition.
    SYSTEM_ARCHITECTURE = "SYSTEM_ARCHITECTURE"
    #: How data moves between components.
    DATA_FLOW = "DATA_FLOW"
    #: Externally visible interface contract.
    API_CONTRACT = "API_CONTRACT"
    #: Entities, fields, and relationships.
    DATA_MODEL = "DATA_MODEL"
    #: Assets, threats, and mitigations.
    THREAT_MODEL = "THREAT_MODEL"
    #: Technology choices with their alternatives and rationale.
    TECHNOLOGY_DECISIONS = "TECHNOLOGY_DECISIONS"
    #: Tasks the M2 Planner can eventually consume.
    IMPLEMENTATION_PLAN = "IMPLEMENTATION_PLAN"
    #: One architecture decision record.
    ADR = "ADR"
    #: A structured cross-agent review of another artifact.
    REVIEW = "REVIEW"


class ArtifactStatus(StrEnum):
    """Lifecycle of an artifact.

    ``APPROVED`` is the only status that confers authority. A draft ADR is an agent
    recommendation no matter how confidently it is written; approval is a human act, and the
    status is where that fact is recorded.
    """

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    #: Replaced by a newer artifact, which points back at this one. Never deleted.
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"

    @property
    def terminal(self) -> bool:
        """Whether no further transition is possible."""
        return self in {ArtifactStatus.SUPERSEDED, ArtifactStatus.REJECTED}


#: Allowed status transitions. Anything absent is rejected, so an artifact cannot reach an
#: inconsistent state through a bug in a caller.
ARTIFACT_TRANSITIONS: dict[ArtifactStatus, frozenset[ArtifactStatus]] = {
    ArtifactStatus.DRAFT: frozenset(
        {ArtifactStatus.REVIEW, ArtifactStatus.APPROVED, ArtifactStatus.REJECTED}
    ),
    ArtifactStatus.REVIEW: frozenset(
        {ArtifactStatus.APPROVED, ArtifactStatus.REJECTED, ArtifactStatus.DRAFT}
    ),
    # An approved artifact is never edited. It is superseded by a successor, or retired.
    ArtifactStatus.APPROVED: frozenset(
        {ArtifactStatus.SUPERSEDED, ArtifactStatus.REJECTED}
    ),
    ArtifactStatus.SUPERSEDED: frozenset(),
    ArtifactStatus.REJECTED: frozenset(),
}


def can_transition(current: ArtifactStatus, target: ArtifactStatus) -> bool:
    """Whether ``current -> target`` is a legal artifact transition."""
    return target in ARTIFACT_TRANSITIONS.get(current, frozenset())


class ValidationState(StrEnum):
    """Whether an artifact passed its checks.

    ``UNVALIDATED`` is the honest default. An artifact that has not been checked must not be
    indistinguishable from one that passed.
    """

    UNVALIDATED = "UNVALIDATED"
    VALID = "VALID"
    INVALID = "INVALID"


#: Id prefix per element kind. Stable across the whole traceability chain, so a reference
#: like ``REQ-004`` means the same thing in a PRD, a UX flow, an ADR, and a task.
ID_PREFIXES: dict[str, str] = {
    "requirement": "REQ",
    "acceptance": "AC",
    "persona": "PER",
    "story": "US",
    "risk": "RISK",
    "metric": "KPI",
    "question": "Q",
    "flow": "UX",
    "screen": "SCR",
    "component": "CMP",
    "token": "TOK",
    "architecture": "ARCH",
    "decision": "ADR",
    "entity": "ENT",
    "endpoint": "API",
    "threat": "THR",
    "task": "TASK",
    "finding": "FND",
}


def element_id(prefix: str, number: int) -> str:
    """Build a stable element id such as ``REQ-001``.

    Zero-padded to three digits so ids sort lexicographically in reports and diffs, which is
    the whole reason they are strings rather than integers.
    """
    return f"{prefix}-{number:03d}"


def is_element_id(value: str, prefix: str | None = None) -> bool:
    """Whether ``value`` looks like an element id, optionally of a specific prefix."""
    head, separator, tail = value.partition("-")
    if not separator or not tail.isdigit():
        return False
    if prefix is not None:
        return head == prefix
    return head in set(ID_PREFIXES.values())


class ArtifactRef(EdithModel):
    """A pointer to another artifact, by identity rather than by content.

    Carrying the version matters: "derived from the PRD" is ambiguous once the PRD has been
    revised twice, and a downstream artifact that cannot say *which* PRD it read cannot be
    checked for staleness.
    """

    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    version: int = Field(default=1, ge=1)
    title: str = Field(default="", max_length=200)

    def matches(self, artifact: Artifact) -> bool:
        """Whether this reference points at exactly ``artifact``."""
        return (
            self.artifact_id == artifact.artifact_id and self.version == artifact.version
        )


class ArtifactDocument(EdithModel):
    """Base class for the typed body of an artifact.

    Subclasses declare the kind they serve, which is how the envelope knows which schema to
    validate a body against without a hand-maintained mapping drifting out of date.
    """

    #: The kind this document type is the body of. Set by every concrete subclass.
    kind: ClassVar[ArtifactKind]

    def element_ids(self) -> tuple[str, ...]:
        """Every stable id this document defines.

        Used by validation to resolve cross-artifact references. The base returns nothing;
        each document type reports the ids it owns.
        """
        return ()

    def referenced_ids(self) -> tuple[str, ...]:
        """Every element id this document points at but does not define."""
        return ()


class ValidationIssue(EdithModel):
    """One problem found in an artifact."""

    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=1000)
    #: The element the problem is attached to, when it is attributable to one.
    element_id: str = ""
    #: ``True`` when this issue must prevent approval.
    blocking: bool = True


class ValidationOutcome(EdithModel):
    """The result of validating an artifact."""

    state: ValidationState = ValidationState.UNVALIDATED
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_at: datetime | None = None

    @property
    def valid(self) -> bool:
        """Whether the artifact passed."""
        return self.state is ValidationState.VALID

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        """Issues that must prevent approval."""
        return [issue for issue in self.issues if issue.blocking]

    def summary(self) -> str:
        """A one-line description of the outcome."""
        if self.state is ValidationState.UNVALIDATED:
            return "not yet validated"
        if self.valid:
            advisories = len(self.issues)
            suffix = f" ({advisories} advisory)" if advisories else ""
            return f"valid{suffix}"
        blocking = len(self.blocking_issues)
        return f"invalid: {blocking} blocking issue(s)"


class Artifact(EdithModel):
    """A versioned, attributable product-development document.

    The envelope is uniform; the body is kind-specific and validated against that kind's
    schema by :meth:`document`.
    """

    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    kind: ArtifactKind
    #: Monotonic per artifact_id. A revision increments it and supersedes its predecessor.
    version: int = Field(default=1, ge=1)
    project_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    #: Registered name of the agent that authored this, or ``user`` for a human.
    author: str = Field(min_length=1, max_length=60)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    authority: AuthorityLevel = AuthorityLevel.AGENT_RECOMMENDATION
    #: Artifacts this one was derived from, with the versions actually read.
    depends_on: tuple[ArtifactRef, ...] = ()
    #: Where the content came from: an execution id, a research report id, a user message.
    source_references: tuple[str, ...] = ()
    #: The typed document, as JSON. Validated against the kind's schema by :meth:`document`.
    body: dict[str, Any] = Field(default_factory=dict)
    validation: ValidationOutcome = ValidationOutcome()
    #: The artifact this one replaces.
    supersedes: str | None = None
    #: Set on the older artifact when a successor is approved.
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _approved_artifacts_must_be_valid(self) -> Artifact:
        """An invalid artifact cannot be approved.

        M4.6's rule, enforced by the type rather than by whoever calls the approval path.
        Letting an artifact with unresolved references into approved project state is how a
        dangling requirement id reaches an implementation task months later.
        """
        if self.status is ArtifactStatus.APPROVED and not self.validation.valid:
            raise ValueError(
                f"artifact {self.artifact_id!r} cannot be APPROVED while its validation "
                f"state is {self.validation.state}"
            )
        return self

    @model_validator(mode="after")
    def _authority_follows_approval(self) -> Artifact:
        """Only an approved artifact may claim approved-decision authority.

        Otherwise an agent could mint an authoritative architecture decision simply by
        writing one down, which is exactly the override the hierarchy exists to prevent.
        """
        elevated = {
            AuthorityLevel.APPROVED_ARCHITECTURE_DECISION,
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
        }
        if self.authority in elevated and self.status is not ArtifactStatus.APPROVED:
            raise ValueError(
                f"artifact {self.artifact_id!r} claims {self.authority} authority but its "
                f"status is {self.status}; approval is what confers authority"
            )
        return self

    @property
    def ref(self) -> ArtifactRef:
        """A reference to this exact artifact version."""
        return ArtifactRef(
            artifact_id=self.artifact_id,
            kind=self.kind,
            version=self.version,
            title=self.title,
        )

    @property
    def approved(self) -> bool:
        """Whether this artifact carries approved status."""
        return self.status is ArtifactStatus.APPROVED

    @property
    def is_current(self) -> bool:
        """Whether this artifact still represents current intent."""
        return self.status in {
            ArtifactStatus.DRAFT,
            ArtifactStatus.REVIEW,
            ArtifactStatus.APPROVED,
        }

    def document(self) -> ArtifactDocument:
        """Parse and return the typed body.

        Raises:
            ValueError: The body does not validate against this kind's schema, or the kind
                has no registered document type.
        """
        schema = DOCUMENT_SCHEMAS.get(self.kind)
        if schema is None:
            raise ValueError(f"no document schema is registered for kind {self.kind}")
        return schema.model_validate(self.body)

    def transition_to(self, target: ArtifactStatus) -> Artifact:
        """Return a copy at ``target`` status, rejecting an illegal transition.

        Returns a copy rather than mutating: an artifact that has been read by something
        else must not change under it, and the store writes rows rather than editing them.

        Raises:
            ValueError: The transition is not permitted, or approval was requested for an
                artifact that has not validated.
        """
        if not can_transition(self.status, target):
            raise ValueError(
                f"illegal artifact transition {self.status} -> {target} "
                f"for {self.artifact_id}"
            )
        return self.model_copy(update={"status": target, "updated_at": utc_now()})

    def revise(
        self,
        *,
        body: dict[str, Any],
        author: str,
        title: str | None = None,
        source_references: tuple[str, ...] = (),
    ) -> Artifact:
        """Create the next version of this artifact.

        The successor is a new record carrying the same ``artifact_id`` at ``version + 1``.
        The predecessor is untouched here; the store marks it ``SUPERSEDED`` when the
        successor is approved, so a draft revision cannot retire an approved document.
        """
        return Artifact(
            artifact_id=self.artifact_id,
            kind=self.kind,
            version=self.version + 1,
            project_id=self.project_id,
            title=title or self.title,
            author=author,
            status=ArtifactStatus.DRAFT,
            depends_on=self.depends_on,
            source_references=source_references or self.source_references,
            body=body,
            supersedes=f"{self.artifact_id}@{self.version}",
        )


#: Kind -> document schema. Populated by :func:`register_document` as each document module
#: is imported, so a new artifact kind cannot be added without also declaring its schema.
DOCUMENT_SCHEMAS: dict[ArtifactKind, type[ArtifactDocument]] = {}


def register_document(schema: type[ArtifactDocument]) -> type[ArtifactDocument]:
    """Register a document type against the kind it serves.

    Usable as a decorator. Raises on a duplicate registration rather than overwriting: two
    schemas claiming one kind means one of them will silently never be used.
    """
    kind = schema.kind
    existing = DOCUMENT_SCHEMAS.get(kind)
    if existing is not None and existing is not schema:
        raise ValueError(
            f"kind {kind} is already registered to {existing.__name__}; "
            f"{schema.__name__} cannot claim it"
        )
    DOCUMENT_SCHEMAS[kind] = schema
    return schema


def build_artifact(
    *,
    kind: ArtifactKind,
    project_id: str,
    title: str,
    author: str,
    document: ArtifactDocument,
    depends_on: tuple[ArtifactRef, ...] = (),
    source_references: tuple[str, ...] = (),
) -> Artifact:
    """Build a draft artifact around a typed document.

    Going through here rather than constructing :class:`Artifact` directly means the body is
    always a document that validated, never a hand-assembled dict.
    """
    if not isinstance(document, DOCUMENT_SCHEMAS.get(kind, ArtifactDocument)):
        raise ValueError(
            f"document {type(document).__name__} is not the registered schema for {kind}"
        )
    return Artifact(
        kind=kind,
        project_id=project_id,
        title=title,
        author=author,
        depends_on=depends_on,
        source_references=source_references,
        body=document.model_dump(mode="json"),
    )


def document_schema_for(kind: ArtifactKind) -> type[BaseModel] | None:
    """Return the document schema registered for a kind, if any."""
    return DOCUMENT_SCHEMAS.get(kind)
