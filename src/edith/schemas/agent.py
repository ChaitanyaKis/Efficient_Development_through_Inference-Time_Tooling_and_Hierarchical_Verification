"""The agent contract: identity, permissions, request, and response.

Scope note (M0): this file defines *how* an agent is described and invoked. It deliberately
does not define the task DAG, project state machine, or artifact catalogue -- those belong
to M2 and would be speculative here. :class:`TaskRef` is the minimal task linkage the
contract needs so that later milestones can attach a full ``Task`` without changing this
interface.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from edith.errors import FailureCategory

from .common import EdithModel, Timestamped, new_id


class Capability(StrEnum):
    """Coarse capability flags declared by an agent.

    These are advertised intent, checked by the orchestrator when routing work. The
    enforcement of what an agent may actually *do* lives in the M1 permission engine.
    """

    PLANNING = "PLANNING"
    RESEARCH = "RESEARCH"
    CODE_GENERATION = "CODE_GENERATION"
    CODE_REVIEW = "CODE_REVIEW"
    TESTING = "TESTING"
    DEBUGGING = "DEBUGGING"
    ARCHITECTURE = "ARCHITECTURE"
    DOCUMENTATION = "DOCUMENTATION"
    SECURITY_ANALYSIS = "SECURITY_ANALYSIS"
    PERFORMANCE_ANALYSIS = "PERFORMANCE_ANALYSIS"
    SELF_TEST = "SELF_TEST"


class AgentPermissions(EdithModel):
    """Declared, explicit permissions for an agent. No hidden capabilities.

    M0 records and validates these; the Tool Gateway in M1 enforces them. Storing them on
    the identity now means agents written today cannot later claim they were never scoped.

    Path patterns are repo-relative glob patterns. An empty ``allowed_write_paths`` means
    the agent is read-only.
    """

    allowed_tools: frozenset[str] = frozenset()
    allowed_read_paths: tuple[str, ...] = ()
    allowed_write_paths: tuple[str, ...] = ()
    network_access: bool = False

    @field_validator("allowed_read_paths", "allowed_write_paths")
    @classmethod
    def _reject_absolute_and_traversal(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            if pattern.startswith(("/", "\\")) or (len(pattern) > 1 and pattern[1] == ":"):
                raise ValueError(
                    f"path pattern {pattern!r} must be repo-relative, not absolute"
                )
            if ".." in pattern.replace("\\", "/").split("/"):
                raise ValueError(f"path pattern {pattern!r} must not contain '..'")
        return value

    @property
    def read_only(self) -> bool:
        """True when the agent may not write anywhere."""
        return not self.allowed_write_paths


class AgentIdentity(EdithModel):
    """Who an agent is and what it is allowed to do."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "0.1.0"
    description: str = Field(min_length=1)
    capabilities: frozenset[Capability] = frozenset()
    permissions: AgentPermissions = AgentPermissions()
    #: Model profile key from models.yaml; ``None`` uses the agent's configured default.
    model_profile: str | None = None


class TaskRef(EdithModel):
    """Minimal reference to the task an agent invocation belongs to.

    Kept intentionally small in M0. The full ``Task`` schema (dependencies, acceptance
    criteria, allowed paths, attempts) arrives with the orchestrator in M2 and will embed
    this reference rather than replace it.
    """

    task_id: str = Field(default_factory=lambda: new_id("task"))
    project_id: str | None = None
    title: str = ""


class AgentRequest(Timestamped):
    """Structured input to an agent invocation.

    ``payload`` is validated by the concrete agent against its declared ``input_schema``;
    this envelope carries only routing and trace metadata.
    """

    request_id: str = Field(default_factory=lambda: new_id("req"))
    task: TaskRef = Field(default_factory=TaskRef)
    payload: dict[str, Any] = Field(default_factory=dict)
    #: Pre-assembled context from the Context Engine (M2). Empty in M0.
    context: dict[str, Any] = Field(default_factory=dict)
    model_profile: str | None = None


class AgentStatus(StrEnum):
    """Terminal status of an agent invocation."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REJECTED = "REJECTED"


class AgentResponse(Timestamped):
    """Structured output of an agent invocation.

    An agent never returns bare text. Either ``status is SUCCESS`` and ``output`` validated
    against the agent's ``output_schema``, or the failure is explicit and classified --
    there is no third, ambiguous outcome (CLAUDE.md invariant 19: never hide failures).
    """

    request_id: str
    agent: str
    status: AgentStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_category: FailureCategory | None = None
    attempts: int = Field(default=1, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    model: str | None = None
    #: Non-sensitive execution evidence (commands run, tokens used, files touched).
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the invocation succeeded."""
        return self.status is AgentStatus.SUCCESS


class AgentHealth(EdithModel):
    """Health of a single agent and its dependencies."""

    agent: str
    healthy: bool
    detail: str = ""
    provider_state: str | None = None
