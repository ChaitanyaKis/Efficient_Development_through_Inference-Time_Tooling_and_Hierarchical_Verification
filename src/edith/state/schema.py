"""Persisted execution records and the project state machine.

These are the rows the orchestrator writes as it works. They are deliberately *small*:
anything large (a prompt, a diff, captured stdout) is written to the artifact store and
referenced by digest, so the database stays queryable and a runaway test log cannot bloat it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from edith.errors import FailureCategory
from edith.schemas.common import EdithModel, Timestamped, new_id, utc_now


class ProjectState(StrEnum):
    """The project lifecycle from CLAUDE.md.

    M2 exercises RECEIVED -> PLANNING -> IMPLEMENTATION -> VERIFICATION -> REVIEW ->
    (REPAIR) -> RELEASE. The remaining states exist so later milestones slot in without a
    schema migration.
    """

    RECEIVED = "RECEIVED"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    RESEARCHING = "RESEARCHING"
    SPECIFICATION = "SPECIFICATION"
    ARCHITECTURE = "ARCHITECTURE"
    IMPLEMENTATION = "IMPLEMENTATION"
    INTEGRATION = "INTEGRATION"
    VERIFICATION = "VERIFICATION"
    REVIEW = "REVIEW"
    REPAIR = "REPAIR"
    RELEASE = "RELEASE"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        """True when the execution is finished."""
        return self in {ProjectState.RELEASE, ProjectState.FAILED}


#: Legal project transitions. FAILED is reachable from everywhere: any state can go wrong.
PROJECT_TRANSITIONS: dict[ProjectState, frozenset[ProjectState]] = {
    ProjectState.RECEIVED: frozenset({ProjectState.ANALYZING, ProjectState.PLANNING}),
    ProjectState.ANALYZING: frozenset({ProjectState.PLANNING, ProjectState.RESEARCHING}),
    ProjectState.RESEARCHING: frozenset({ProjectState.PLANNING}),
    ProjectState.PLANNING: frozenset(
        {
            ProjectState.SPECIFICATION,
            ProjectState.ARCHITECTURE,
            ProjectState.IMPLEMENTATION,
        }
    ),
    ProjectState.SPECIFICATION: frozenset({ProjectState.ARCHITECTURE, ProjectState.IMPLEMENTATION}),
    ProjectState.ARCHITECTURE: frozenset({ProjectState.IMPLEMENTATION}),
    ProjectState.IMPLEMENTATION: frozenset({ProjectState.INTEGRATION, ProjectState.VERIFICATION}),
    ProjectState.INTEGRATION: frozenset({ProjectState.VERIFICATION}),
    # The implement/verify/judge cycle runs many times per execution, so verification and
    # review must both be able to send work back for another attempt -- with a repair step
    # when the debugger is consulted, and without one when the attempt is simply retried.
    ProjectState.VERIFICATION: frozenset(
        {ProjectState.REVIEW, ProjectState.REPAIR, ProjectState.IMPLEMENTATION}
    ),
    ProjectState.REVIEW: frozenset(
        {
            ProjectState.RELEASE,
            ProjectState.REPAIR,
            ProjectState.IMPLEMENTATION,
            ProjectState.VERIFICATION,
        }
    ),
    # REPAIR returns to IMPLEMENTATION: the debugger's fix is applied, then re-verified.
    ProjectState.REPAIR: frozenset({ProjectState.IMPLEMENTATION, ProjectState.VERIFICATION}),
    ProjectState.RELEASE: frozenset(),
    ProjectState.FAILED: frozenset(),
}


def can_transition(current: ProjectState, target: ProjectState) -> bool:
    """Whether ``current -> target`` is a legal project transition.

    Staying in the same non-terminal state is allowed: verifying twice in a row, or running
    two implementation attempts back to back, is normal in the retry loop and should not be
    an error. A terminal state is still final.
    """
    if target is ProjectState.FAILED and not current.terminal:
        return True
    if target is current and not current.terminal:
        return True
    return target in PROJECT_TRANSITIONS.get(current, frozenset())


class Project(Timestamped):
    """A workspace Edith operates on."""

    project_id: str = Field(default_factory=lambda: new_id("proj"))
    name: str = Field(min_length=1, max_length=120)
    #: Absolute path to the project workspace. Stored so a restart can rebind tools.
    workspace_root: str
    repository: str | None = None
    description: str = ""


class Execution(Timestamped):
    """One end-to-end run against a project."""

    execution_id: str = Field(default_factory=lambda: new_id("exec"))
    project_id: str
    request: str = Field(min_length=1)
    state: ProjectState = ProjectState.RECEIVED
    branch: str | None = None
    attempts: int = Field(default=0, ge=0)
    finished_at: datetime | None = None
    result_summary: str = ""

    def transition_to(self, target: ProjectState) -> None:
        """Move to ``target``, rejecting an illegal transition."""
        if not can_transition(self.state, target):
            raise ValueError(
                f"illegal project transition {self.state} -> {target} "
                f"for execution {self.execution_id}"
            )
        self.state = target
        self.touch()


class StateTransition(EdithModel):
    """An audit record of one state change."""

    transition_id: str = Field(default_factory=lambda: new_id("trans"))
    execution_id: str
    from_state: str
    to_state: str
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class AgentRun(Timestamped):
    """One invocation of one agent."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    execution_id: str
    task_id: str | None = None
    agent: str
    attempt: int = Field(default=1, ge=1)
    status: str = "RUNNING"
    model: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    #: Artifact digest holding the full structured output.
    output_ref: str | None = None
    error: str | None = None
    failure_category: FailureCategory | None = None


class ToolExecution(EdithModel):
    """One tool call made during an agent run."""

    tool_execution_id: str = Field(default_factory=lambda: new_id("tex"))
    execution_id: str
    run_id: str | None = None
    tool: str
    ok: bool
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    failure_category: FailureCategory | None = None
    #: Artifact digest holding arguments and output.
    detail_ref: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class VerificationRecord(EdithModel):
    """Evidence from one executed verification command.

    This is the row that makes "the tests passed" checkable rather than assertable: it holds
    the command, the exit code, and a reference to the real captured output.
    """

    verification_id: str = Field(default_factory=lambda: new_id("ver"))
    execution_id: str
    task_id: str | None = None
    kind: str
    command: str
    exit_code: int
    passed: bool
    duration_seconds: float = Field(default=0.0, ge=0.0)
    tests_passed: int | None = None
    tests_failed: int | None = None
    #: Artifact digest holding full stdout/stderr.
    output_ref: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MemoryInjectionRecord(EdithModel):
    """One memory injection that actually reached a prompt.

    Context accounting for M3.2: what was sent, how relevant it was judged, what it cost,
    and how much of the execution's allowance remained afterwards.

    Ids, not content. The claims themselves live in the memory store, which is where they
    can be inspected and deleted; copying them here would create a second copy with a second
    deletion path, and a memory a user deleted would go on living in the state database.
    """

    injection_id: str = Field(default_factory=lambda: new_id("minj"))
    execution_id: str
    task_id: str | None = None
    agent: str = ""
    #: The retrieval point, e.g. ``coder_repair``.
    point: str = ""
    memory_ids: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    titles: tuple[str, ...] = ()
    chars: int = Field(default=0, ge=0)
    reason: str = ""
    remaining_chars: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryConsumption(EdithModel):
    """What one execution has already spent of its memory allowance.

    Read back when an interrupted execution resumes, so a restart continues the budget
    instead of granting a fresh one.
    """

    chars: int = Field(default=0, ge=0)
    retrievals: int = Field(default=0, ge=0)
    #: ``(memory_id, title, point, agent, score)`` for everything already injected.
    injected: tuple[tuple[str, str, str, str, float], ...] = ()

    @property
    def memory_ids(self) -> frozenset[str]:
        """Every memory already shown in this execution."""
        return frozenset(entry[0] for entry in self.injected)


class FailureRecord(EdithModel):
    """A classified failure, with the action policy chose."""

    failure_id: str = Field(default_factory=lambda: new_id("fail"))
    execution_id: str
    task_id: str | None = None
    category: FailureCategory
    action: str
    message: str
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)

