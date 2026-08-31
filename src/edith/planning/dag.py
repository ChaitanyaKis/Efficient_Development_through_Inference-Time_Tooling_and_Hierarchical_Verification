"""The task dependency graph.

Deterministic by construction: ready tasks come back in a stable order (priority, then id),
so the same plan executes in the same sequence every run. That matters more than it sounds
-- a reproducible execution order is what makes a failed autonomous run debuggable.

Execution is sequential. Parallel LLM calls are explicitly out of scope: on a 6 GB GPU a
second concurrent inference evicts the first model's KV cache and both slow down.
"""

from __future__ import annotations

from collections import deque

from edith.errors import EdithError, FailureCategory

from .task import Task, TaskStatus


class PlanValidationError(EdithError):
    """The proposed plan is not a valid DAG."""

    category = FailureCategory.REQUIREMENT_FAILURE


class TaskGraph:
    """A dependency-aware view over a set of tasks.

    The graph owns status transitions that depend on *other* tasks -- unlocking a task whose
    dependencies succeeded, blocking one whose dependency failed -- so that logic lives in
    one place rather than being re-derived by the orchestrator.
    """

    def __init__(self, tasks: list[Task]) -> None:
        """
        Args:
            tasks: Tasks forming the graph.

        Raises:
            PlanValidationError: Duplicate ids, unknown dependencies, or a cycle.
        """
        self._tasks: dict[str, Task] = {}
        for task in tasks:
            if task.task_id in self._tasks:
                raise PlanValidationError(
                    f"duplicate task id {task.task_id!r}",
                    details={"task_id": task.task_id},
                )
            self._tasks[task.task_id] = task

        self._validate_dependencies()
        self._assert_acyclic()

    # -- Construction checks --------------------------------------------------------

    def _validate_dependencies(self) -> None:
        for task in self._tasks.values():
            for dependency in task.dependencies:
                if dependency not in self._tasks:
                    raise PlanValidationError(
                        f"task {task.task_id!r} depends on unknown task {dependency!r}",
                        details={"task_id": task.task_id, "missing": dependency},
                    )

    def _assert_acyclic(self) -> None:
        """Reject cycles using Kahn's algorithm.

        A cycle means the plan can never complete; catching it at construction turns a
        hang into an immediate, explainable rejection.
        """
        indegree = {task_id: len(task.dependencies) for task_id, task in self._tasks.items()}
        dependents = self._dependents()
        queue = deque(sorted(t for t, degree in indegree.items() if degree == 0))
        visited = 0

        while queue:
            current = queue.popleft()
            visited += 1
            for dependent in sorted(dependents.get(current, ())):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(self._tasks):
            unresolved = sorted(t for t, degree in indegree.items() if degree > 0)
            raise PlanValidationError(
                f"plan contains a dependency cycle involving: {unresolved}",
                details={"tasks": unresolved},
            )

    def _dependents(self) -> dict[str, list[str]]:
        """Map task id -> ids of tasks that depend on it."""
        mapping: dict[str, list[str]] = {task_id: [] for task_id in self._tasks}
        for task in self._tasks.values():
            for dependency in task.dependencies:
                mapping[dependency].append(task.task_id)
        return mapping

    # -- Queries --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._tasks

    def get(self, task_id: str) -> Task:
        """Return a task by id."""
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise PlanValidationError(
                f"unknown task {task_id!r}", details={"task_id": task_id}
            ) from exc

    def tasks(self) -> tuple[Task, ...]:
        """Every task, ordered by priority then id for determinism."""
        return tuple(sorted(self._tasks.values(), key=lambda t: (t.priority, t.task_id)))

    def topological_order(self) -> tuple[str, ...]:
        """A deterministic topological ordering of task ids."""
        indegree = {task_id: len(task.dependencies) for task_id, task in self._tasks.items()}
        dependents = self._dependents()
        available = sorted(
            (t for t, degree in indegree.items() if degree == 0),
            key=lambda t: (self._tasks[t].priority, t),
        )
        queue = deque(available)
        ordered: list[str] = []

        while queue:
            current = queue.popleft()
            ordered.append(current)
            unlocked = []
            for dependent in dependents.get(current, ()):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    unlocked.append(dependent)
            for task_id in sorted(
                unlocked, key=lambda t: (self._tasks[t].priority, t)
            ):
                queue.append(task_id)
        return tuple(ordered)

    #: Statuses that let a dependent task proceed. ``DEFERRED`` is included deliberately: a
    #: deferred task wrote its files and is waiting on the final gate, so blocking its
    #: dependents would strand the very work that might turn the suite green.
    _UNLOCKING = frozenset({TaskStatus.SUCCEEDED, TaskStatus.DEFERRED})

    def dependencies_satisfied(self, task: Task) -> bool:
        """Whether every dependency of ``task`` has finished without failing."""
        return all(
            self._tasks[dependency].status in self._UNLOCKING
            for dependency in task.dependencies
        )

    def has_failed_dependency(self, task: Task) -> bool:
        """Whether any dependency has failed, been blocked, or been cancelled."""
        return any(
            self._tasks[dependency].status
            in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
            for dependency in task.dependencies
        )

    # -- Scheduling -----------------------------------------------------------------

    def refresh(self) -> None:
        """Recompute derived statuses.

        Promotes PENDING tasks whose dependencies all succeeded to READY, and blocks any
        task whose dependency failed. Idempotent, so it is safe to call before every step
        and after loading persisted state.
        """
        # Repeat until stable: blocking one task can block its dependents in turn.
        changed = True
        while changed:
            changed = False
            for task in self.tasks():
                if task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                    if self.has_failed_dependency(task):
                        task.transition_to(TaskStatus.BLOCKED)
                        changed = True
                    elif task.status is TaskStatus.PENDING and self.dependencies_satisfied(
                        task
                    ):
                        task.transition_to(TaskStatus.READY)
                        changed = True

    def ready_tasks(self) -> tuple[Task, ...]:
        """Tasks eligible to run now, in deterministic order."""
        self.refresh()
        return tuple(task for task in self.tasks() if task.status is TaskStatus.READY)

    def next_task(self) -> Task | None:
        """The single highest-priority ready task, or ``None``.

        Sequential by design: one heavy inference at a time (CLAUDE.md resource rules).
        """
        ready = self.ready_tasks()
        return ready[0] if ready else None

    # -- Outcomes -------------------------------------------------------------------

    def mark_succeeded(self, task_id: str) -> None:
        """Record success and unlock dependents."""
        self.get(task_id).transition_to(TaskStatus.SUCCEEDED)
        self.refresh()

    def mark_deferred(self, task_id: str, reason: str) -> None:
        """Record that a task changed files but did not verify, and unlock dependents.

        Not a success and not a failure: the final gate decides. The reason is kept on the
        task so a report can say why it is neither, rather than leaving a status the reader
        has to guess at.
        """
        task = self.get(task_id)
        task.transition_to(TaskStatus.DEFERRED)
        task.failure_reason = reason
        self.refresh()

    def mark_failed(
        self,
        task_id: str,
        reason: str,
        category: FailureCategory | None = None,
    ) -> None:
        """Record a terminal failure and block everything downstream."""
        task = self.get(task_id)
        task.transition_to(TaskStatus.FAILED)
        task.failure_reason = reason
        task.failure_category = category
        self.refresh()

    def requeue(self, task_id: str) -> bool:
        """Return a RUNNING task to READY for another attempt.

        Returns:
            ``True`` when the task was re-queued, ``False`` when its attempt budget is
            spent -- the caller should then fail it. Retries are always bounded.
        """
        task = self.get(task_id)
        if task.exhausted:
            return False
        task.transition_to(TaskStatus.READY)
        return True

    # -- Aggregate state ------------------------------------------------------------

    def is_complete(self) -> bool:
        """True when no task can make further progress."""
        self.refresh()
        return all(task.status.terminal for task in self._tasks.values())

    def succeeded(self) -> bool:
        """True when every task actually verified. Deferred tasks do not count."""
        return all(task.status is TaskStatus.SUCCEEDED for task in self._tasks.values())

    def settled_without_failure(self) -> bool:
        """True when every task finished and none failed, deferrals included.

        This, not :meth:`succeeded`, is the precondition for running the final gate: a run
        whose tasks all deferred has produced exactly the code the gate exists to judge, and
        skipping the gate would report a failure without ever checking the work.
        """
        return all(task.status in self._UNLOCKING for task in self._tasks.values())

    def summary(self) -> dict[str, int]:
        """Count of tasks by status, for reporting."""
        counts: dict[str, int] = {}
        for task in self._tasks.values():
            counts[str(task.status)] = counts.get(str(task.status), 0) + 1
        return counts
