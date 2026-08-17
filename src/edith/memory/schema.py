"""Memory records and the trust model.

The invariant this module exists to enforce: **every memory has provenance**. A fact with
no traceable origin never becomes project knowledge, because the question Edith must always
be able to answer is not just "what do I believe" but "why do I believe it".

The second invariant is that sources are not equal. A test that actually ran is stronger
evidence than a model's suggestion, and the schema encodes that difference rather than
leaving it to whoever reads the memory later.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from edith.schemas.common import EdithModel, Timestamped, new_id, utc_now


class MemoryType(StrEnum):
    """What kind of knowledge a record holds."""

    #: Facts about one project: technology choices, conventions, structure, constraints.
    PROJECT = "PROJECT"
    #: Reusable engineering lessons that may apply beyond one project.
    ENGINEERING = "ENGINEERING"
    #: An observed failure, its cause, and what fixed it.
    FAILURE = "FAILURE"
    #: A decision and its rationale.
    DECISION = "DECISION"
    #: Concise record of a past task execution, referencing artifacts rather than inlining them.
    TASK = "TASK"


class MemoryScope(StrEnum):
    """How widely a memory may be retrieved.

    ``GLOBAL`` is deliberately hard to reach: it is the only way a record crosses a project
    boundary, so it must be an explicit classification rather than a default.
    """

    PROJECT = "PROJECT"
    GLOBAL = "GLOBAL"


class MemorySource(StrEnum):
    """Where a memory came from. Ordered loosely from strongest to weakest evidence."""

    #: A human said so.
    USER = "USER"
    #: Derived from a file in the repository.
    PROJECT_ARTIFACT = "PROJECT_ARTIFACT"
    #: Observed by a deterministic tool (a command's exit code, a file's existence).
    TOOL_OBSERVATION = "TOOL_OBSERVATION"
    #: Derived from a test that actually executed.
    TEST_RESULT = "TEST_RESULT"
    #: An agent's reasoning over evidence it had.
    AGENT_INFERENCE = "AGENT_INFERENCE"
    #: A model said it, unverified.
    MODEL_SUGGESTION = "MODEL_SUGGESTION"
    #: Retrieved from outside the machine, with a citation.
    EXTERNAL_RESEARCH = "EXTERNAL_RESEARCH"


class MemoryStatus(StrEnum):
    """Lifecycle state of a record."""

    ACTIVE = "ACTIVE"
    #: Replaced by a newer record, which points back at this one. Never deleted.
    SUPERSEDED = "SUPERSEDED"
    #: No longer relevant, kept for history.
    ARCHIVED = "ARCHIVED"
    #: Failed validation; retained so the rejection itself is auditable.
    REJECTED = "REJECTED"


#: Baseline confidence per source. Deterministic evidence outranks inference, and a model's
#: unverified suggestion sits at the bottom on purpose -- it is a hypothesis, not knowledge.
SOURCE_CONFIDENCE: dict[MemorySource, float] = {
    MemorySource.TEST_RESULT: 0.95,
    MemorySource.TOOL_OBSERVATION: 0.90,
    MemorySource.USER: 0.90,
    MemorySource.PROJECT_ARTIFACT: 0.80,
    MemorySource.EXTERNAL_RESEARCH: 0.60,
    MemorySource.AGENT_INFERENCE: 0.40,
    MemorySource.MODEL_SUGGESTION: 0.25,
}

#: Sources trustworthy enough to be written without a human saying so. Everything else is
#: a *proposal*: it can be stored, but only as REJECTED-by-default pending approval.
AUTO_TRUSTED_SOURCES: frozenset[MemorySource] = frozenset(
    {
        MemorySource.USER,
        MemorySource.TEST_RESULT,
        MemorySource.TOOL_OBSERVATION,
        MemorySource.PROJECT_ARTIFACT,
    }
)

#: Minimum confidence for a memory to be offered to an agent by default.
DEFAULT_MIN_CONFIDENCE = 0.35


class MemoryRecord(Timestamped):
    """One durable fact, lesson, failure, decision, or task summary.

    ``content`` is deliberately short. A memory is a *claim plus a pointer*, not an archive:
    anything large lives in the artifact store and is referenced by
    :attr:`source_reference`, which keeps retrieval cheap and prompts small.
    """

    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    type: MemoryType
    scope: MemoryScope = MemoryScope.PROJECT
    #: ``None`` only for GLOBAL records. Enforced below.
    project_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)
    tags: tuple[str, ...] = ()

    # -- Provenance. Not optional; this is the point of the schema. ------------------
    source: MemorySource
    #: Where the claim can be checked: a file path, a command, an execution id, a URL.
    source_reference: str = Field(min_length=1, max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: int = Field(default=50, ge=0, le=100)

    status: MemoryStatus = MemoryStatus.ACTIVE
    #: The memory this one replaces. History is preserved, never overwritten.
    supersedes: str | None = None
    #: Set on the older record when it is superseded, so the chain is walkable both ways.
    superseded_by: str | None = None

    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    #: How many times this failure/lesson has been observed again.
    recurrence_count: int = Field(default=1, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_scoping(self) -> MemoryRecord:
        """A project-scoped memory must name its project; a global one must not.

        Without this, a record with a null project id would be visible to every project --
        which is exactly the leak the isolation invariant forbids.
        """
        if self.scope is MemoryScope.PROJECT and not self.project_id:
            raise ValueError("a PROJECT-scoped memory must have a project_id")
        if self.scope is MemoryScope.GLOBAL and self.project_id:
            raise ValueError(
                "a GLOBAL memory must not be bound to a project_id; it would be neither "
                "properly shared nor properly isolated"
            )
        return self

    @model_validator(mode="after")
    def _global_scope_is_reusable_only(self) -> MemoryRecord:
        """Only genuinely reusable knowledge may be global.

        Project facts and task records are about one codebase by definition. Letting them
        go global would leak one project's details into another's context.
        """
        if self.scope is MemoryScope.GLOBAL and self.type in {
            MemoryType.PROJECT,
            MemoryType.TASK,
        }:
            raise ValueError(
                f"{self.type} memory is project-specific and cannot be GLOBAL; only "
                "ENGINEERING, FAILURE and DECISION lessons are reusable"
            )
        return self

    @property
    def is_active(self) -> bool:
        """Whether this record still represents current belief."""
        return self.status is MemoryStatus.ACTIVE

    @property
    def provenance(self) -> str:
        """A one-line, human-readable statement of where this came from."""
        return f"{self.source} ({self.source_reference}), confidence {self.confidence:.2f}"

    def touch_access(self) -> None:
        """Record that this memory was retrieved."""
        self.access_count += 1
        self.last_accessed_at = utc_now()


class ScoredMemory(EdithModel):
    """A retrieved memory with the reasoning behind its rank."""

    memory: MemoryRecord
    score: float
    reasons: list[str] = Field(default_factory=list)


class MemoryBundle(EdithModel):
    """What retrieval hands to an agent.

    Mirrors :class:`~edith.context.engine.ContextBundle` on purpose: a rationale accompanies
    the payload, so a bad answer can be diagnosed as a retrieval failure rather than assumed
    to be a reasoning failure.
    """

    query: str = ""
    memories: list[ScoredMemory] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    estimated_context_chars: int = 0
    considered: int = 0
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether anything was retrieved."""
        return not self.memories

    def render(self) -> str:
        """Render as prompt text, with provenance attached to every claim.

        Provenance travels *with* the memory into the prompt. An agent shown a bare
        assertion cannot weigh it; shown "(from a test that ran, confidence 0.95)" it can.
        """
        if not self.memories:
            return "(no relevant prior knowledge)"
        lines = []
        for entry in self.memories:
            record = entry.memory
            lines.append(
                f"- [{record.type}] {record.title}\n"
                f"  {record.content}\n"
                f"  (source: {record.provenance})"
            )
        return "\n".join(lines)


class MemoryProposal(EdithModel):
    """A candidate memory, before validation decides whether it may be stored.

    Kept separate from :class:`MemoryRecord` so that "something wants to be remembered" and
    "this is remembered" are different types. An agent can only construct the former.
    """

    type: MemoryType
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)
    source: MemorySource
    source_reference: str = Field(default="", max_length=500)
    scope: MemoryScope = MemoryScope.PROJECT
    project_id: str | None = None
    tags: tuple[str, ...] = ()
    importance: int = Field(default=50, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supersedes: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
