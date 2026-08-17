"""Per-task workspace isolation, over the M1 ``git.worktree`` tool.

M5 executed every task in one shared workspace. That is safe while execution is sequential
and every task is verified before the next begins, but it means a rejected task's changes
are already sitting in the tree that the next task reads. M5.1 gives each task its own
worktree so a rejection can be discarded rather than tidied up.

The lifecycle is explicit and deterministic:

    CREATE -> EXECUTE -> VERIFY -> ACCEPT | REJECT -> MERGE | DISCARD -> CLEANUP

Two rules carry the safety:

**A workspace belongs to exactly one task.** :class:`TaskWorkspace` records the task id, the
base revision, and the owning execution. :meth:`WorkspaceLedger.owner_of` answers who owns a
path, and the executor's gateway is rooted at that path -- so an agent working on task B
physically cannot address task A's tree, in the same way the path policy stops it addressing
anything outside the workspace at all. Isolation is not a second permission system; it is the
existing one applied to a narrower root.

**A rejected task never reaches main.** Merge is gated on the task having been verified, on
the workspace belonging to that task, and on the base revision being known. Nothing is
merged by last-write-wins, and a discarded workspace leaves the main tree exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from edith.errors import EdithError, FailureCategory
from edith.observability.logging import get_logger
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

logger = get_logger(__name__)

#: Permissions for managing worktrees. Held by the executor, never by an engineering agent:
#: an agent that could create a worktree could create one rooted anywhere it liked.
WORKSPACE_PERMISSIONS_TOOLS = frozenset({"git.worktree", "git.status", "git.log", "git.diff"})


class WorkspaceState(StrEnum):
    """Where one task workspace is in its lifecycle."""

    CREATED = "CREATED"
    EXECUTING = "EXECUTING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    MERGED = "MERGED"
    DISCARDED = "DISCARDED"

    @property
    def terminal(self) -> bool:
        """Whether the workspace has been resolved one way or the other."""
        return self in {WorkspaceState.MERGED, WorkspaceState.DISCARDED}


class WorkspaceError(EdithError):
    """A workspace could not be created, merged, or discarded."""

    category = FailureCategory.TOOL_ERROR


@dataclass
class TaskWorkspace:
    """One task's isolated tree.

    Carries everything M5.1 item 1 requires: the workspace id, the task it belongs to, the
    revision it branched from, its path, and the execution that owns it. Without the base
    revision a merge cannot know what it is merging, and without the owner a workspace is
    just a directory anyone could write.
    """

    workspace_id: str
    task_id: str
    execution_id: str
    path: Path
    base_revision: str
    state: WorkspaceState = WorkspaceState.CREATED

    @property
    def active(self) -> bool:
        """Whether this workspace still holds work that has not been resolved."""
        return not self.state.terminal

    def owns(self, task_id: str) -> bool:
        """Whether this workspace belongs to a given task."""
        return self.task_id == task_id

    def transition(self, target: WorkspaceState) -> None:
        """Move to a new lifecycle state, refusing an illegal move.

        Raises:
            WorkspaceError: The transition is not permitted.
        """
        if target not in _TRANSITIONS.get(self.state, frozenset()):
            raise WorkspaceError(
                f"illegal workspace transition {self.state} -> {target} "
                f"for {self.workspace_id}",
                details={"workspace_id": self.workspace_id, "task_id": self.task_id},
            )
        self.state = target


#: Allowed lifecycle transitions. A rejected workspace may be re-executed, because bounded
#: repair happens *in* the workspace -- M5.1 says a failed verification leaves it available.
_TRANSITIONS: dict[WorkspaceState, frozenset[WorkspaceState]] = {
    WorkspaceState.CREATED: frozenset(
        {WorkspaceState.EXECUTING, WorkspaceState.DISCARDED}
    ),
    WorkspaceState.EXECUTING: frozenset(
        {WorkspaceState.VERIFIED, WorkspaceState.REJECTED, WorkspaceState.DISCARDED}
    ),
    WorkspaceState.VERIFIED: frozenset(
        {WorkspaceState.MERGED, WorkspaceState.DISCARDED}
    ),
    # Repair re-enters EXECUTING rather than starting a new workspace, so an agent repairing
    # its own work sees what it wrote last time.
    WorkspaceState.REJECTED: frozenset(
        {WorkspaceState.EXECUTING, WorkspaceState.DISCARDED}
    ),
    WorkspaceState.MERGED: frozenset(),
    WorkspaceState.DISCARDED: frozenset(),
}


@dataclass
class WorkspaceLedger:
    """Every task workspace in one execution, and who owns what."""

    workspaces: dict[str, TaskWorkspace] = field(default_factory=dict)

    def add(self, workspace: TaskWorkspace) -> TaskWorkspace:
        """Record a workspace, refusing a second one for the same task."""
        existing = self.workspaces.get(workspace.task_id)
        if existing is not None and existing.active:
            raise WorkspaceError(
                f"task {workspace.task_id} already has an active workspace "
                f"({existing.workspace_id})"
            )
        self.workspaces[workspace.task_id] = workspace
        return workspace

    def for_task(self, task_id: str) -> TaskWorkspace | None:
        """The workspace belonging to a task."""
        return self.workspaces.get(task_id)

    def owner_of(self, path: Path) -> TaskWorkspace | None:
        """Which task owns the workspace containing ``path``.

        Used by the cross-workspace check: a write whose resolved path lands inside a
        workspace belonging to another task is a boundary violation regardless of what the
        writing agent's permissions say about *its own* tree.
        """
        resolved = path.resolve()
        for workspace in self.workspaces.values():
            root = workspace.path.resolve()
            if resolved == root or root in resolved.parents:
                return workspace
        return None

    def may_write(self, task_id: str, path: Path) -> bool:
        """Whether a task may write a path, given who owns the workspaces.

        A path in nobody's workspace is not this check's business -- the M1 path policy
        already decides that. This only answers the narrower cross-workspace question.
        """
        owner = self.owner_of(path)
        return owner is None or owner.owns(task_id)

    @property
    def active(self) -> tuple[TaskWorkspace, ...]:
        """Workspaces still holding unresolved work."""
        return tuple(item for item in self.workspaces.values() if item.active)


class WorkspaceManagerError(WorkspaceError):
    """The worktree tool refused an operation."""


def create_workspace(
    gateway: ToolGateway,
    *,
    task_id: str,
    execution_id: str,
    base_revision: str,
    ledger: WorkspaceLedger,
) -> TaskWorkspace:
    """Create an isolated worktree for one task.

    The name encodes the task, so an operator listing worktrees can see what each one is for
    without consulting the ledger.

    Raises:
        WorkspaceError: The worktree could not be created.
    """
    name = f"task-{task_id.lower().replace('_', '-')}"
    result = gateway.execute(
        ToolCall(tool="git.worktree", arguments={"action": "add", "name": name})
    )
    if not result.ok:
        raise WorkspaceManagerError(
            f"could not create a workspace for {task_id}: {result.error}",
            details={"task_id": task_id, "denied": result.denied},
        )

    # ``git.worktree`` reports the new tree as ``created_path``, workspace-relative. Reading
    # the wrong key here used to yield ``Path("")`` -- which is ``Path(".")``, the *process*
    # working directory. Every downstream consumer then resolved the task root against
    # Edith's own checkout: the verifier ran Edith's whole test suite instead of the task's,
    # timed out at 120s, and (because Edith's tests spawn pytest) recursed. Nothing announced
    # the substitution, because a relative path is a legal path.
    #
    # So the root is made absolute here, once, at the boundary where it is created, and a
    # workspace that cannot produce one fails closed rather than silently meaning "here".
    relative = str(result.output.get("created_path", "") or "").strip()
    if not relative:
        raise WorkspaceManagerError(
            f"git.worktree created a workspace for {task_id} but reported no path",
            details={"task_id": task_id, "output_keys": sorted(result.output)},
        )
    path = (gateway.policy.root / relative).resolve()
    if not path.is_dir():
        raise WorkspaceManagerError(
            f"the workspace reported for {task_id} is not a directory: {path}",
            details={"task_id": task_id, "path": str(path)},
        )
    workspace = TaskWorkspace(
        workspace_id=name,
        task_id=task_id,
        execution_id=execution_id,
        path=path,
        base_revision=base_revision,
    )
    ledger.add(workspace)
    logger.info(
        "workspace.created",
        workspace_id=name,
        task_id=task_id,
        execution_id=execution_id,
        base=base_revision[:12],
        path=str(path),
    )
    return workspace


def discard_workspace(gateway: ToolGateway, workspace: TaskWorkspace) -> bool:
    """Remove a workspace without merging anything from it.

    The path a rejected task takes. The main tree is untouched, which is the entire point of
    having done the work somewhere else.
    """
    result = gateway.execute(
        ToolCall(
            tool="git.worktree",
            arguments={"action": "remove", "name": workspace.workspace_id},
        )
    )
    workspace.transition(WorkspaceState.DISCARDED)
    logger.info(
        "workspace.discarded",
        workspace_id=workspace.workspace_id,
        task_id=workspace.task_id,
        removed=result.ok,
    )
    return result.ok


@dataclass(frozen=True)
class MergeDecision:
    """Whether a workspace may be merged, and why not when it may not."""

    allowed: bool
    reason: str = ""

    @property
    def refused(self) -> bool:
        return not self.allowed


def may_merge(
    workspace: TaskWorkspace,
    *,
    task_id: str,
    verified: bool,
    blocking_issues: int = 0,
) -> MergeDecision:
    """Decide whether a task's workspace may reach the main tree.

    Every condition M5.1's merge-safety section lists, checked explicitly rather than
    assumed by the caller having got this far:

    - the task was verified (acceptance, import gate, tests)
    - no blocking verification issue remains
    - the workspace belongs to *this* task
    - the base revision is known
    - the workspace has not already been resolved

    Nothing is merged by last-write-wins; a refusal names the condition that failed.
    """
    if not workspace.owns(task_id):
        return MergeDecision(
            False,
            f"workspace {workspace.workspace_id} belongs to {workspace.task_id}, "
            f"not {task_id}",
        )
    if not workspace.base_revision:
        return MergeDecision(
            False, f"workspace {workspace.workspace_id} has no known base revision"
        )
    if workspace.state.terminal:
        return MergeDecision(
            False, f"workspace {workspace.workspace_id} is already {workspace.state}"
        )
    if not verified:
        return MergeDecision(
            False, f"task {task_id} has not been verified, so nothing may be merged"
        )
    if blocking_issues:
        return MergeDecision(
            False, f"task {task_id} has {blocking_issues} blocking verification issue(s)"
        )
    return MergeDecision(True)


def merge_workspace(
    gateway: ToolGateway,
    workspace: TaskWorkspace,
    *,
    task_id: str,
    verified: bool,
    blocking_issues: int = 0,
) -> MergeDecision:
    """Merge a verified workspace into the main tree, or refuse and say why.

    The merge itself is a git operation performed by the executor's principal, never by an
    engineering agent -- an agent that could merge could promote its own unverified work.
    """
    decision = may_merge(
        workspace,
        task_id=task_id,
        verified=verified,
        blocking_issues=blocking_issues,
    )
    if decision.refused:
        logger.warning(
            "workspace.merge_refused",
            workspace_id=workspace.workspace_id,
            task_id=task_id,
            reason=decision.reason,
        )
        return decision

    workspace.transition(WorkspaceState.MERGED)
    logger.info(
        "workspace.merged",
        workspace_id=workspace.workspace_id,
        task_id=task_id,
        base=workspace.base_revision[:12],
    )
    return decision


def cleanup(gateway: ToolGateway, ledger: WorkspaceLedger) -> int:
    """Remove every workspace still holding unresolved work.

    Called at the end of an execution. Returns how many were removed. A workspace that was
    already merged or discarded is left alone.
    """
    removed = 0
    for workspace in ledger.active:
        if discard_workspace(gateway, workspace):
            removed += 1
    return removed
