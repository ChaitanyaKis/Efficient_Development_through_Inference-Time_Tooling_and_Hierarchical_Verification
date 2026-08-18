"""Executing an implementation plan: isolated workspaces, DAG order, verification gates.

This is where M4's plan becomes code. The shape is deliberately unglamorous:

    assign -> detect conflicts -> serialise -> for each ready task:
        isolated workspace -> scoped gateway -> agent -> verify -> accept or reject

Four rules do most of the work, and each exists because of a specific way this can go wrong:

**A task is not complete because code was generated.** M5 item 12: completion requires the
expected files to have changed, the verification to pass, and the diff to stay inside the
task's scope. An agent that writes nothing and reports success fails here, and so does one
that writes something unrelated.

**Prerequisites must have been verified, not merely attempted.** A task whose dependency
failed is ``BLOCKED``, never run. Building on unverified work is how one bad task poisons
everything downstream of it.

**Scope is checked after the fact as well as before.** The gateway refuses writes outside the
task's paths, but a task may legitimately hold a broad scope; comparing the actual diff
against the *declared* files catches an agent that stayed inside its permissions and still
did something nobody asked for.

**Sequential, on purpose.** CLAUDE.md caps this machine at one heavy inference at a time, and
M5 item 9 says sequential is acceptable. Workspaces are isolated anyway, so the design does
not have to change when that constraint lifts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from edith.config.schema import EdithConfig
from edith.errors import EdithError, FailureCategory
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.product.architecture import ImplementationPlanDocument
from edith.product.prd import PRDDocument
from edith.product.ux import UXSpecDocument
from edith.quality.artifacts import (
    FindingOrigin,
    QualityReport,
    QualityVerdict,
)
from edith.schemas.agent import AgentPermissions, AgentRequest, TaskRef
from edith.tools.gateway import ToolGateway
from edith.tools.registry import ToolRegistry
from edith.tools.schemas import ToolCall
from edith.verification.runner import VerificationReport, VerificationRunner
from edith.workspaces import ProjectWorkspace

from .agents import EngineeringInput, agent_for
from .conflicts import TaskConflict, detect_conflicts, serialise
from .isolation import (
    TaskWorkspace,
    WorkspaceError,
    WorkspaceLedger,
    WorkspaceState,
    cleanup,
    create_workspace,
    discard_workspace,
    merge_workspace,
)
from .ownership import Assignment, assign_plan

logger = get_logger(__name__)

#: Failures a coding agent can plausibly repair by rewriting its own output.
#:
#: Everything outside this set -- a timeout where the tests never executed, a missing
#: dependency, an unusable interpreter, a denied write -- describes the *environment* or the
#: *policy*, not the code. Feeding those back as repair evidence wastes the bounded budget
#: and, worse, ends in ``REPAIR_EXHAUSTED``, which reads as "the agent could not write
#: working code" when nothing was ever run against it.
REPAIRABLE_FAILURES = frozenset(
    {
        FailureCategory.CODE_FAILURE,
        FailureCategory.TEST_FAILURE,
        FailureCategory.BUILD_ERROR,
        FailureCategory.VALIDATION_FAILURE,
    }
)

#: Permissions for the verifier. Separate principal from every engineering agent: the thing
#: that runs the tests must not be the thing that wrote the code, which is the M2.1
#: separation applied to specialised agents.
VERIFIER_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset({"shell.run", "filesystem.read"}),
    allowed_read_paths=("**",),
)

#: Permissions for reading the diff a task produced. Read-only and git-only.
INSPECTOR_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset({"git.diff", "git.status", "filesystem.read"}),
    allowed_read_paths=("**",),
)


class TaskOutcome(StrEnum):
    """How one engineering task ended."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    #: A prerequisite failed, so this never ran.
    BLOCKED = "BLOCKED"
    #: Could not be assigned to a role, or its scope was refused.
    UNASSIGNED = "UNASSIGNED"
    #: Ran, produced changes, but verification rejected them.
    REJECTED = "REJECTED"
    #: Rejected, repaired to the limit, and still rejected. Distinct from REJECTED because
    #: it says the budget is spent -- M5.1 item 9: an exhausted repair is never COMPLETE.
    REPAIR_EXHAUSTED = "REPAIR_EXHAUSTED"


class QualityState(StrEnum):
    """How far one task actually got.

    M5.1 item 6 forbids collapsing these. M5 measured 6/6 tasks complete and 1/3
    applications runnable, and the whole value of that finding is that the two numbers are
    different. A single boolean would have hidden it.
    """

    #: The model produced edits. Says nothing about whether they are usable.
    GENERATED = "GENERATED"
    #: Edits reached disk inside the task's scope.
    TASK_COMPLETE = "TASK_COMPLETE"
    #: The generated modules import and the configured checks pass.
    VERIFIED = "VERIFIED"
    #: The verified work reached the main workspace.
    INTEGRATED = "INTEGRATED"

    @property
    def rank(self) -> int:
        """How far along this state is, for comparison."""
        return _QUALITY_RANK[self]


_QUALITY_RANK: dict[QualityState, int] = {
    QualityState.GENERATED: 0,
    QualityState.TASK_COMPLETE: 1,
    QualityState.VERIFIED: 2,
    QualityState.INTEGRATED: 3,
}


@dataclass
class TaskExecution:
    """What happened to one task."""

    task_id: str
    role: str
    outcome: TaskOutcome
    #: How far the task actually got. ``None`` when nothing was generated at all.
    quality: QualityState | None = None
    #: The isolated workspace this task ran in, when isolation was used.
    workspace_id: str = ""
    #: The revision the workspace branched from.
    base_revision: str = ""
    changed_files: tuple[str, ...] = ()
    rejected_files: tuple[str, ...] = ()
    #: Files the agent changed that its task never named.
    out_of_scope: tuple[str, ...] = ()
    verification: VerificationReport | None = None
    model_calls: int = 0
    #: How many times this task was re-attempted after a rejection.
    repair_attempts: int = 0
    #: The M6 quality report, when the quality pipeline ran.
    quality_report: QualityReport | None = None
    duration_seconds: float = 0.0
    detail: str = ""
    failure_category: FailureCategory | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is TaskOutcome.COMPLETED

    def summary(self) -> str:
        if self.ok:
            return (
                f"{self.task_id} [{self.role}]: COMPLETED "
                f"({len(self.changed_files)} file(s))"
            )
        return f"{self.task_id} [{self.role}]: {self.outcome} - {self.detail[:160]}"


@dataclass
class ExecutionReport:
    """The result of executing a whole plan."""

    executions: list[TaskExecution] = field(default_factory=list)
    conflicts: tuple[TaskConflict, ...] = ()
    started_at: float = 0.0
    duration_seconds: float = 0.0

    @property
    def completed(self) -> int:
        return sum(1 for item in self.executions if item.ok)

    @property
    def failed(self) -> int:
        return sum(
            1
            for item in self.executions
            if item.outcome
            in {
                TaskOutcome.FAILED,
                TaskOutcome.REJECTED,
                TaskOutcome.REPAIR_EXHAUSTED,
            }
        )

    @property
    def blocked(self) -> int:
        return sum(1 for item in self.executions if item.outcome is TaskOutcome.BLOCKED)

    @property
    def model_calls(self) -> int:
        return sum(item.model_calls for item in self.executions)

    @property
    def changed_files(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in self.executions:
            for path in item.changed_files:
                seen.setdefault(path, None)
        return tuple(seen)

    @property
    def scope_violations(self) -> int:
        return sum(len(item.out_of_scope) for item in self.executions)

    def summary(self) -> str:
        lines = [item.summary() for item in self.executions]
        if self.conflicts:
            lines.append("")
            lines.extend(f"  {conflict.render()}" for conflict in self.conflicts)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable metrics, for the M5 benchmark."""
        by_role: dict[str, int] = {}
        for item in self.executions:
            by_role[item.role] = by_role.get(item.role, 0) + item.model_calls
        return {
            "tasks": len(self.executions),
            "completed": self.completed,
            "failed": self.failed,
            "blocked": self.blocked,
            "conflicts": len(self.conflicts),
            "model_calls": self.model_calls,
            "repair_attempts": sum(item.repair_attempts for item in self.executions),
            "repair_exhausted": sum(
                1
                for item in self.executions
                if item.outcome is TaskOutcome.REPAIR_EXHAUSTED
            ),
            "quality": {
                state.value: sum(
                    1 for item in self.executions if item.quality is state
                )
                for state in QualityState
            },
            "calls_by_role": by_role,
            "changed_files": list(self.changed_files),
            "scope_violations": self.scope_violations,
            "duration_seconds": round(self.duration_seconds, 1),
            "outcomes": {item.task_id: item.outcome.value for item in self.executions},
        }


class EngineeringExecutor:
    """Runs an implementation plan through the specialised agents."""

    def __init__(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        *,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
        isolate: bool = True,
    ) -> None:
        """
        Args:
            config: Resolved configuration, re-rooted at the workspace.
            workspace: Where the code is written. Never the Edith repository.
            provider: The model provider. One inference at a time.
            registry: Tool registry; the M1 default when omitted.
        """
        self.config = workspace.config_for(config)
        self.workspace = workspace
        self._provider = provider
        self._registry = registry
        self._model_calls = 0
        #: Whether each task gets its own worktree. On by default; a caller that has already
        #: isolated the tree itself (the benchmark harness gives every trial a fresh
        #: directory) can turn it off rather than nesting one isolation inside another.
        self.isolate = isolate

    def _gateway(
        self, permissions: AgentPermissions, agent: str, *, root: Path | None = None
    ) -> ToolGateway:
        """A gateway scoped to one principal, rooted at one workspace.

        ``root`` is the isolation boundary. A gateway rooted at a task's worktree cannot
        address another task's tree or the main tree at all -- not because a rule forbids it
        but because the M1 path policy resolves every path against this root and refuses
        anything outside it. That is why isolation needs no second permission system: it is
        the existing one, pointed somewhere narrower.
        """
        from edith.tools.paths import PathPolicy  # noqa: PLC0415 - avoids a cycle
        from edith.tools.registry import build_default_registry  # noqa: PLC0415

        if root is not None and not root.is_absolute():
            # Fail closed. A relative task root resolves against the *process* working
            # directory, which is Edith's own checkout -- so this is how an agent's work
            # would come to be verified against the platform's repository instead of its own
            # workspace. There is no safe interpretation of a relative root here.
            raise WorkspaceError(
                f"task root must be absolute, got {root!r}",
                details={"agent": agent, "root": str(root)},
            )
        target = root or self.workspace.root
        scoped = self.config.model_copy(
            update={
                "tools": self.config.tools.model_copy(
                    update={"workspace_root": target}
                )
            }
        )
        return ToolGateway(
            scoped,
            permissions,
            registry=self._registry or build_default_registry(),
            agent=agent,
            policy=PathPolicy.create(target, self.config.tools.paths),
        )

    def execute(
        self,
        plan: ImplementationPlanDocument,
        *,
        prd: PRDDocument | None = None,
        ux: UXSpecDocument | None = None,
        architecture_text: str = "",
        verify: bool = True,
    ) -> ExecutionReport:
        """Execute every task in a plan, in dependency order.

        Never raises for a task-level failure: the report carries the outcome of each task,
        and a plan where three tasks succeeded and one failed is a real state an operator
        needs to see rather than an exception.
        """
        started = time.monotonic()
        report = ExecutionReport(started_at=started)

        assignments = assign_plan(plan)
        report.conflicts = detect_conflicts(assignments)
        ordered = self._dependency_order(serialise(assignments, report.conflicts))

        succeeded: set[str] = set()
        failed: set[str] = set()
        ledger = WorkspaceLedger()

        for assignment in ordered:
            task = assignment.task

            if not assignment.assigned:
                report.executions.append(
                    TaskExecution(
                        task_id=task.task_id,
                        role=assignment.role.value,
                        outcome=TaskOutcome.UNASSIGNED,
                        detail=assignment.rejection,
                        failure_category=FailureCategory.CONFIGURATION_ERROR,
                    )
                )
                failed.add(task.task_id)
                continue

            unmet = [item for item in task.depends_on if item not in succeeded]
            if unmet:
                # A prerequisite that failed, or one that never ran. Either way this task
                # would be building on unverified work.
                report.executions.append(
                    TaskExecution(
                        task_id=task.task_id,
                        role=assignment.role.value,
                        outcome=TaskOutcome.BLOCKED,
                        detail=f"prerequisite(s) not verified: {', '.join(unmet)}",
                    )
                )
                failed.add(task.task_id)
                continue

            try:
                workspace = self._create_workspace(task.task_id, ledger)
            except WorkspaceError as exc:
                # Isolation was required and could not be provided. The task does not run.
                report.executions.append(
                    TaskExecution(
                        task_id=task.task_id,
                        role=assignment.role.value,
                        outcome=TaskOutcome.FAILED,
                        detail=f"workspace unavailable: {exc.message}",
                        failure_category=exc.category,
                    )
                )
                failed.add(task.task_id)
                continue

            execution = self._run_task(
                assignment,
                prd=prd,
                ux=ux,
                architecture_text=architecture_text,
                verify=verify,
                root=workspace.path if workspace else None,
            )
            if workspace is not None:
                execution.workspace_id = workspace.workspace_id
                execution.base_revision = workspace.base_revision
                self._resolve_workspace(workspace, execution)
            report.executions.append(execution)
            (succeeded if execution.ok else failed).add(task.task_id)

        # Anything still unresolved is discarded rather than left behind. An orphaned
        # worktree is a tree nobody owns that the next run would inherit.
        if self.isolate:
            cleanup(self._workspace_gateway(), ledger)

        report.duration_seconds = time.monotonic() - started
        logger.info(
            "engineering.executed",
            tasks=len(report.executions),
            completed=report.completed,
            failed=report.failed,
            blocked=report.blocked,
            conflicts=len(report.conflicts),
            model_calls=report.model_calls,
            duration_seconds=round(report.duration_seconds, 1),
        )
        return report

    @staticmethod
    def _dependency_order(assignments: tuple[Assignment, ...]) -> tuple[Assignment, ...]:
        """Order tasks so every prerequisite precedes its dependents.

        Kahn's algorithm, the same one the M2 task graph and the M4 plan validator use. A
        cycle leaves tasks unordered; they are appended at the end and will be reported as
        BLOCKED when their prerequisites turn out never to have succeeded.
        """
        by_id = {item.task.task_id: item for item in assignments}
        pending = {
            item.task.task_id: set(item.task.depends_on) & set(by_id)
            for item in assignments
        }
        ordered: list[Assignment] = []

        progressed = True
        while progressed:
            progressed = False
            for task_id in sorted(item for item, deps in pending.items() if not deps):
                ordered.append(by_id[task_id])
                del pending[task_id]
                progressed = True
                for remaining in pending.values():
                    remaining.discard(task_id)

        ordered.extend(by_id[task_id] for task_id in sorted(pending))
        return tuple(ordered)

    def _workspace_gateway(self) -> ToolGateway:
        """A gateway that may manage worktrees. Held by the executor, never by an agent."""
        return self._gateway(
            AgentPermissions(
                allowed_tools=frozenset({"git.worktree", "git.status", "git.log"}),
                allowed_read_paths=("**",),
                # Creating a worktree writes under the configured worktree directory. This
                # is the executor's own principal, held by nothing else: no engineering
                # agent has `git.worktree`, so none can mint a workspace of its own.
                allowed_write_paths=(".edith/**",),
            ),
            "workspace",
        )

    def _base_revision(self) -> str:
        """The revision a task workspace branches from."""
        # Uses the workspace principal, which holds `git.log`. The inspector does not --
        # and a denied call would silently yield an empty revision, which the merge gate
        # would then correctly refuse for a reason that looks like a git problem.
        result = self._workspace_gateway().execute(
            ToolCall(tool="git.log", arguments={"max_entries": 1})
        )
        if not result.ok:
            return ""
        commits = result.output.get("commits") or []
        if not commits:
            return ""
        head = commits[0]
        return str(head.get("sha") or head.get("commit") or head.get("hash") or "")

    def _create_workspace(
        self, task_id: str, ledger: WorkspaceLedger
    ) -> TaskWorkspace | None:
        """Create this task's isolated worktree, or ``None`` when isolation is off.

        Raises when isolation is required and the worktree cannot be created. Falling back
        to the shared tree would be the exact failure M5.2 forbids: an agent silently
        executing against main when isolation was asked for. A guarantee that quietly turns
        itself off under load is not a guarantee, so the task fails instead.
        """
        if not self.isolate:
            return None
        return create_workspace(
            self._workspace_gateway(),
            task_id=task_id,
            execution_id=self.workspace.project_id,
            base_revision=self._base_revision(),
            ledger=ledger,
        )

    def _resolve_workspace(
        self, workspace: TaskWorkspace, execution: TaskExecution
    ) -> None:
        """Merge a verified task's workspace, or discard a rejected one.

        The only authorised path by which a task's work reaches the main tree. A refusal is
        recorded on the execution rather than worked around -- there is deliberately no
        fallback that copies files across.
        """
        workspace.transition(WorkspaceState.EXECUTING)
        if not execution.ok:
            workspace.transition(WorkspaceState.REJECTED)
            discard_workspace(self._workspace_gateway(), workspace)
            return

        workspace.transition(WorkspaceState.VERIFIED)
        decision = merge_workspace(
            self._workspace_gateway(),
            workspace,
            task_id=execution.task_id,
            verified=True,
            blocking_issues=0,
            destination=self.workspace.root,
            changed_files=execution.changed_files,
        )
        if decision.refused:
            execution.outcome = TaskOutcome.FAILED
            execution.detail = f"MERGE_FAILURE: {decision.reason}"
            execution.failure_category = FailureCategory.TOOL_ERROR
            return
        execution.quality = QualityState.INTEGRATED

    def _run_task(
        self,
        assignment: Assignment,
        *,
        prd: PRDDocument | None,
        ux: UXSpecDocument | None,
        architecture_text: str,
        verify: bool,
        root: Path | None = None,
    ) -> TaskExecution:
        """Run one task, repairing a rejected attempt within a bounded budget.

        The repair is the M2 shape: the same agent, the same scope, plus the real evidence of
        what went wrong. Bounded because an agent that cannot fix its own output in two
        attempts is not going to in five, and CLAUDE.md requires every autonomous loop to
        terminate.
        """
        attempts = max(1, self.config.orchestration.max_repair_attempts)
        evidence = ""
        execution = TaskExecution(
            task_id=assignment.task.task_id,
            role=assignment.role.value,
            outcome=TaskOutcome.FAILED,
        )

        for attempt in range(attempts):
            execution = self._attempt_task(
                assignment,
                prd=prd,
                ux=ux,
                architecture_text=architecture_text,
                verify=verify,
                evidence=evidence,
                root=root,
            )
            execution.repair_attempts = attempt
            if execution.ok or execution.outcome is not TaskOutcome.REJECTED:
                # Only a rejection is worth repairing: a task the agent could not produce
                # output for at all will not be helped by being shown its own absence.
                break
            category = execution.failure_category
            if category is None or category not in REPAIRABLE_FAILURES:
                # The rejection is not the agent's to fix. A timeout where the tests never
                # ran, a missing dependency, a broken interpreter -- rewriting the code
                # cannot resolve any of them, and spending the budget trying converts an
                # environment fault into a false verdict on the agent's work. Escalate
                # with the category intact so the failure is attributed where it belongs.
                execution.outcome = TaskOutcome.FAILED
                label = category.value if category else "unclassified"
                execution.detail = (
                    f"not repairable by the coding agent "
                    f"({label}): {execution.detail[:400]}"
                )
                logger.warning(
                    "engineering.escalated",
                    task_id=assignment.task.task_id,
                    category=label,
                    attempt=attempt + 1,
                )
                break
            evidence = execution.detail
            logger.info(
                "engineering.repairing",
                task_id=assignment.task.task_id,
                attempt=attempt + 1,
                reason=execution.detail[:200],
            )
        else:
            # The loop ran to its limit without breaking, so the last attempt was still a
            # rejection. M5.1 item 9: an exhausted budget fails closed and says so.
            if execution.outcome is TaskOutcome.REJECTED:
                execution.outcome = TaskOutcome.REPAIR_EXHAUSTED
                execution.detail = (
                    f"repair budget of {attempts} attempt(s) exhausted; "
                    f"last failure: {execution.detail[:400]}"
                )

        return execution

    def _attempt_task(
        self,
        assignment: Assignment,
        *,
        prd: PRDDocument | None,
        ux: UXSpecDocument | None,
        architecture_text: str,
        verify: bool,
        evidence: str = "",
        root: Path | None = None,
    ) -> TaskExecution:
        """One attempt at a task: generate, apply, check imports, verify, and judge.

        Everything happens under ``root``: the agent writes there, the import gate loads
        from there, and verification runs there. A repair passes the same root, so it sees
        what the previous attempt wrote instead of starting from a clean tree.
        """
        task = assignment.task
        started = time.monotonic()
        execution = TaskExecution(
            task_id=task.task_id,
            role=assignment.role.value,
            outcome=TaskOutcome.FAILED,
        )

        try:
            gateway = self._gateway(
                assignment.permissions(), assignment.role.value, root=root
            )
            agent_cls = agent_for(assignment.role)
            agent = agent_cls(
                provider=self._provider,
                settings=self.config.agents.for_agent(agent_cls.identity.name),
                tools=gateway,
            )

            payload = EngineeringInput(
                title=task.title,
                description=task.description,
                acceptance_criteria=list(task.acceptance_criteria),
                requirements=_requirements_for(prd, task.implements),
                architecture=architecture_text[:4000],
                ux=_ux_for(ux, assignment.role.value),
                context=self._read_context(gateway, task.paths),
                scope="\n".join(assignment.write_paths),
                failure_evidence=evidence[:8000],
            )

            response = agent.execute(
                AgentRequest(
                    payload=payload.model_dump(),
                    task=TaskRef(
                        task_id=task.task_id,
                        project_id=self.workspace.project_id,
                        title=task.title,
                    ),
                )
            )
            execution.model_calls = response.attempts
            self._model_calls += response.attempts

            if not response.ok:
                execution.detail = response.error or "the agent failed"
                execution.failure_category = response.failure_category
                execution.duration_seconds = time.monotonic() - started
                return execution

            changed = tuple(response.output.get("changed_files", []))
            rejected = tuple(response.output.get("rejected_files", []))
            execution.changed_files = changed
            execution.rejected_files = rejected
            execution.quality = QualityState.GENERATED

            if not changed:
                # M5 item 12: a task is not complete because the agent said so.
                execution.detail = (
                    "the agent changed no files"
                    + (f"; rejected: {', '.join(rejected)}" if rejected else "")
                )
                execution.duration_seconds = time.monotonic() - started
                return execution

            execution.quality = QualityState.TASK_COMPLETE
            execution.out_of_scope = _out_of_scope(changed, task.paths)

            # Importability first. M5 item 12 requires the build to succeed, and for Python
            # the build is the import: a module that parses but raises NameError on import is
            # broken in a way the syntax gate cannot see. Without this the task suite can pass
            # while never loading the generated code, which proves nothing about it -- the
            # same vacuous-check failure M2.1 found when a missing runner was reported as a
            # failing test.
            importable, import_error = self._check_importable(changed, root=root)
            if not importable:
                execution.outcome = TaskOutcome.REJECTED
                execution.detail = f"the generated code does not import: {import_error}"
                execution.failure_category = FailureCategory.CODE_FAILURE
                execution.duration_seconds = time.monotonic() - started
                return execution

            if verify:
                report = self._verify(task.verification, root=root)
                execution.verification = report
                if not report.passed:
                    execution.outcome = TaskOutcome.REJECTED
                    execution.detail = report.evidence(600)
                    execution.failure_category = report.failure_category
                    execution.duration_seconds = time.monotonic() - started
                    return execution

            quality = self._quality(task.task_id, changed, report=execution.verification, root=root)
            execution.quality_report = quality
            if quality is not None and not quality.verdict().merges:
                # The adjudicator decided, not the reviewer. A model finding reaches this point
                # as advisory evidence; only the deterministic verdict rejects the task, and
                # only a repairable verdict re-enters the coder's budget.
                execution.outcome = TaskOutcome.REJECTED
                execution.detail = _render_quality(quality)
                execution.failure_category = (
                    FailureCategory.CODE_FAILURE
                    if quality.verdict() is QualityVerdict.REPAIR_REQUIRED
                    else FailureCategory.SECURITY_FAILURE
                )
                execution.duration_seconds = time.monotonic() - started
                return execution

            execution.outcome = TaskOutcome.COMPLETED
            execution.quality = QualityState.VERIFIED
            execution.detail = f"changed {len(changed)} file(s)"

        except EdithError as exc:
            execution.detail = exc.message
            execution.failure_category = exc.category
        except Exception as exc:  # noqa: BLE001 - one task must not abort the plan
            execution.detail = f"{type(exc).__name__}: {exc}"
            execution.failure_category = FailureCategory.UNKNOWN

        execution.duration_seconds = time.monotonic() - started
        return execution

    def _quality(
        self,
        task_id: str,
        changed: tuple[str, ...],
        *,
        report: VerificationReport | None,
        root: Path | None,
    ) -> QualityReport | None:
        """Run the M6 quality gates over what this task changed.

        Always runs the deterministic gates; runs the model reviewers only when
        ``orchestration.model_quality_review`` is on. The pipeline itself decides to skip the
        model once deterministic evidence has blocked, so nothing here re-implements that.

        Returns ``None`` when there is nothing to review. A quality pipeline that cannot read
        the changed files reports nothing rather than guessing -- the verification gate has
        already had its say, and inventing a finding here would be the opposite of the
        evidence rule.
        """
        from edith.quality.pipeline import (  # noqa: PLC0415 - avoids a cycle
            QualityPipeline,
            read_sources,
            run_model_review,
        )

        base = root or self.workspace.root
        sources = read_sources(base, changed)
        if not sources:
            return None

        pipeline = QualityPipeline(task_id=task_id)
        pipeline.security_gate(sources)
        pipeline.review_gate(sources)

        if not self.config.orchestration.model_quality_review:
            return pipeline.report()

        from edith.quality.agents import (  # noqa: PLC0415 - optional path, model only
            CodeReviewAgent,
            JudgeAgent,
            JudgeInput,
            JudgeOutput,
            SecurityAgent,
            render_findings,
        )

        for agent_class, name in (
            (SecurityAgent, "model-security"),
            (CodeReviewAgent, "model-review"),
        ):
            run_model_review(
                pipeline,
                agent_class(provider=self._provider),
                name=name,
                sources=sources,
                task=task_id,
            )

        judge_verdict = None
        rationale = ""
        if not pipeline.blocked:
            try:
                response = JudgeAgent(provider=self._provider).execute(
                    AgentRequest(
                        payload=JudgeInput(
                            task=task_id,
                            tests_passed=bool(report and report.passed),
                            deterministic_findings=render_findings(
                                pipeline.report().findings
                            ),
                            changed_files=", ".join(changed[:20]),
                        ).model_dump()
                    )
                )
                if not response.ok:
                    # The agent envelope reports failure by returning an empty payload, which
                    # would validate as JudgeOutput's default of FAILED and silently block
                    # good code. An unreachable Judge contributes nothing instead.
                    raise RuntimeError(response.error or "the judge did not answer")
                judged = JudgeOutput.model_validate(response.output)
                judge_verdict = judged.verdict
                rationale = judged.rationale
            except Exception as exc:  # noqa: BLE001 - the Judge is advisory; it may fail
                # A Judge that cannot be reached is infrastructure, not a coder defect. It
                # contributes nothing rather than blocking, and the deterministic gates stand.
                logger.warning(
                    "quality.judge_failed",
                    task_id=task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )

        return pipeline.report(
            judge_verdict=judge_verdict, judge_rationale=rationale
        )

    def _check_importable(
        self, changed: tuple[str, ...], *, root: Path | None = None
    ) -> tuple[bool, str]:
        """Whether every Python module the task wrote can actually be imported.

        Runs through the verifier's principal, so the check has ``shell.run`` and the
        engineering agent still does not. Non-Python files and packages without an
        ``__init__`` are skipped rather than guessed at: a false rejection here would block
        a task that is fine.
        """
        modules = [
            path[:-3].replace("\\", "/").replace("/", ".")
            for path in changed
            if path.endswith(".py") and not path.endswith("__init__.py")
        ]
        if not modules:
            return (True, "")

        gateway = self._gateway(VERIFIER_PERMISSIONS, "verifier", root=root)
        statement = "; ".join(f"import {module}" for module in modules)
        result = gateway.execute(
            ToolCall(
                tool="shell.run",
                arguments={"argv": ["python", "-c", statement]},
            )
        )
        if not result.ok:
            # The check could not be run -- a denied command or a missing interpreter. That
            # is an environment problem, not evidence the code is broken, so it does not
            # reject the task.
            logger.warning("engineering.import_check_unavailable", error=result.error)
            return (True, "")

        if int(result.output["exit_code"]) == 0:
            return (True, "")
        detail = str(result.output.get("stderr", "")) or str(result.output.get("stdout", ""))
        return (False, detail.strip()[-400:])

    def _verify(
        self, kinds: tuple[str, ...], *, root: Path | None = None
    ) -> VerificationReport:
        """Run this task's configured checks in the task's own workspace.

        Rooted at ``root`` so a passing suite in the *main* tree can never be mistaken for
        verification of a task's changes -- the checks run where the changes are.
        """
        runner = VerificationRunner(
            self._gateway(VERIFIER_PERMISSIONS, "verifier", root=root),
            self.config.orchestration.profile(),
        )
        return runner.run_all(tuple((kind, None) for kind in kinds or ("tests",)))

    def _read_context(self, gateway: ToolGateway, paths: tuple[str, ...]) -> str:
        """Read the files a task will change, so the agent edits rather than invents.

        Only the task's own files. M5 item 15 forbids dumping the project into every prompt,
        and a task that is creating a new file legitimately gets nothing.
        """
        parts: list[str] = []
        for path in paths[:6]:
            result = gateway.execute(
                ToolCall(tool="filesystem.read", arguments={"path": path})
            )
            if result.ok:
                content = str(result.output.get("content", ""))[:4000]
                parts.append(f"--- {path} ---\n{content}")
        return "\n\n".join(parts)


def _requirements_for(prd: PRDDocument | None, identifiers: tuple[str, ...]) -> str:
    """Only the requirements this task implements."""
    if prd is None:
        return ""
    wanted = set(identifiers)
    selected = [item for item in prd.requirements if item.requirement_id in wanted]
    return "\n".join(
        f"{item.requirement_id} {item.title}: {item.statement}" for item in selected
    )


def _ux_for(ux: UXSpecDocument | None, role: str) -> str:
    """The UX specification, for the role that has to respect it.

    Only the frontend gets this. A backend task that receives screen definitions spends
    context on material it cannot act on.
    """
    if ux is None or role != "frontend":
        return ""
    return ux.render()[:4000]


def _out_of_scope(changed: tuple[str, ...], declared: tuple[str, ...]) -> tuple[str, ...]:
    """Files an agent changed that its task never named.

    Not a security check -- the gateway already refused anything outside the task's
    permissions. This catches the narrower case of an agent staying inside its permissions
    and still editing something nobody asked it to, which is the "large speculative rewrite"
    M5 item 17 warns against.
    """
    if not declared:
        return ()
    allowed = {path.replace("\\", "/") for path in declared}
    return tuple(
        path for path in changed if path.replace("\\", "/") not in allowed
    )


def _render_quality(report: QualityReport) -> str:
    """Repair evidence from a quality report.

    Blocking findings only, and the deterministic ones first: an agent given a model's stylistic
    opinion alongside a real defect tends to address the opinion. M2.1 established that repair
    works when it is shown the actual failure.
    """
    ordered = sorted(
        report.blocking, key=lambda item: item.origin is not FindingOrigin.DETERMINISTIC
    )
    lines = [
        f"[{item.severity.value}] {item.category}: {item.summary}"
        + (f"\n    {item.evidence[0].detail[:200]}" if item.evidence else "")
        for item in ordered[:5]
    ]
    return "quality review rejected the change:\n" + "\n".join(lines)
