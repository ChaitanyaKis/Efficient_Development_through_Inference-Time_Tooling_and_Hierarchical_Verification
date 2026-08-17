"""The executable task model and its state machine.

M0 deliberately shipped only :class:`~edith.schemas.agent.TaskRef` -- the minimal linkage the
agent contract needed -- and deferred the full task to M2. This is that task. ``TaskRef``
still identifies a task inside an :class:`~edith.schemas.agent.AgentRequest`; nothing about
the M0 agent contract changes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from edith.errors import FailureCategory
from edith.schemas.common import EdithModel, Timestamped, new_id


class TaskStatus(StrEnum):
    """Lifecycle of a single task.

    ``BLOCKED`` is distinct from ``FAILED``: a blocked task never ran, because something it
    depended on failed. Collapsing the two would make a failure report unreadable.
    """

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        """True when no further transition is possible."""
        return self in _TERMINAL_TASK_STATES


_TERMINAL_TASK_STATES = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
)

#: Allowed task transitions. Anything absent here is rejected, so an inconsistent status
#: cannot be reached by a bug in the orchestrator.
TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
    ),
    TaskStatus.READY: frozenset(
        {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
    ),
    # RUNNING may return to READY: a bounded retry re-queues the same task.
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.READY,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.BLOCKED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """Whether ``current -> target`` is a legal task transition."""
    return target in TASK_TRANSITIONS.get(current, frozenset())


class TaskScope(EdithModel):
    """The filesystem scope a task's agent is permitted to touch.

    Converted into :class:`~edith.schemas.agent.AgentPermissions` by the orchestrator, so a
    planner-proposed scope is enforced by the M1 gateway rather than merely recorded. The
    same repo-relative validation as ``AgentPermissions`` applies.
    """

    read_paths: tuple[str, ...] = ("**",)
    write_paths: tuple[str, ...] = ()

    @field_validator("read_paths", "write_paths")
    @classmethod
    def _reject_absolute_and_traversal(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            if pattern.startswith(("/", "\\")) or (len(pattern) > 1 and pattern[1] == ":"):
                raise ValueError(f"scope pattern {pattern!r} must be repo-relative")
            if ".." in pattern.replace("\\", "/").split("/"):
                raise ValueError(f"scope pattern {pattern!r} must not contain '..'")
        return value


class VerificationRequirement(EdithModel):
    """A check that must pass before a task may be considered done.

    ``kind`` selects a command from the project's verification profile rather than letting a
    task carry a raw command line: a planner-authored command string would be an
    arbitrary-execution hole straight through the M1 shell allowlist.
    """

    kind: str = Field(pattern=r"^(tests|lint|typecheck|build)$")
    #: Optional argument passed to the profile command, e.g. a test path to narrow the run.
    selector: str | None = Field(default=None, max_length=200)
    required: bool = True


class Task(Timestamped):
    """A unit of executable work in the plan DAG."""

    task_id: str = Field(default_factory=lambda: new_id("task"))
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    #: Registered agent name that executes this task.
    agent: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    priority: int = Field(default=100, ge=0, le=1000)
    dependencies: tuple[str, ...] = ()
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    verification: tuple[VerificationRequirement, ...] = ()
    scope: TaskScope = TaskScope()
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    failure_reason: str | None = None
    failure_category: FailureCategory | None = None

    @model_validator(mode="after")
    def _reject_self_dependency(self) -> Task:
        if self.task_id in self.dependencies:
            raise ValueError(f"task {self.task_id!r} cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"task {self.task_id!r} has duplicate dependencies")
        return self

    @property
    def exhausted(self) -> bool:
        """True when the attempt budget is spent."""
        return self.attempts >= self.max_attempts

    def transition_to(self, target: TaskStatus) -> None:
        """Move to ``target``, rejecting an illegal transition.

        Raises:
            ValueError: The transition is not permitted from the current status.
        """
        if not can_transition(self.status, target):
            raise ValueError(
                f"illegal task transition {self.status} -> {target} for {self.task_id}"
            )
        self.status = target
        self.touch()


class Plan(EdithModel):
    """A validated set of tasks forming a DAG."""

    goal: str = Field(min_length=1, max_length=2000)
    tasks: tuple[Task, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _unique_ids(self) -> Plan:
        identifiers = [task.task_id for task in self.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("plan contains duplicate task ids")
        return self
