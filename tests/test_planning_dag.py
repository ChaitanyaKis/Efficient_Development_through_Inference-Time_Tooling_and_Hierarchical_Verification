"""Task model, state machine, and dependency graph.

The DAG is the one part of M2 that must be exactly right regardless of what the model does,
so it is tested exhaustively rather than sampled.
"""

from __future__ import annotations

import pytest

from edith.errors import FailureCategory
from edith.planning.dag import PlanValidationError, TaskGraph
from edith.planning.task import (
    Plan,
    Task,
    TaskScope,
    TaskStatus,
    VerificationRequirement,
    can_transition,
)


def make_task(task_id: str, *, depends: tuple[str, ...] = (), priority: int = 100) -> Task:
    """A minimal valid task."""
    return Task(
        task_id=task_id,
        title=f"task {task_id}",
        description=f"do {task_id}",
        agent="coder",
        dependencies=depends,
        priority=priority,
    )


def start(graph: TaskGraph, task_id: str) -> Task:
    """Drive a task to RUNNING from wherever it currently is.

    ``refresh()`` promotes eligible tasks to READY on its own, so a test must not assume a
    task is still PENDING after an earlier task succeeded.
    """
    task = graph.get(task_id)
    if task.status is TaskStatus.PENDING:
        task.transition_to(TaskStatus.READY)
    task.transition_to(TaskStatus.RUNNING)
    return task


def succeed(graph: TaskGraph, task_id: str) -> None:
    """Run a task to success."""
    start(graph, task_id)
    graph.mark_succeeded(task_id)


class TestTaskSchema:
    def test_defaults(self) -> None:
        task = make_task("t1")
        assert task.status is TaskStatus.PENDING
        assert task.attempts == 0
        assert not task.exhausted

    def test_self_dependency_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot depend on itself"):
            Task(
                task_id="t1", title="t", description="d", agent="coder", dependencies=("t1",)
            )

    def test_duplicate_dependencies_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate dependencies"):
            Task(
                task_id="t1",
                title="t",
                description="d",
                agent="coder",
                dependencies=("a", "a"),
            )

    def test_agent_name_must_be_a_registered_style_identifier(self) -> None:
        with pytest.raises(ValueError):
            Task(task_id="t1", title="t", description="d", agent="Coder Agent")

    def test_exhausted_after_max_attempts(self) -> None:
        task = make_task("t1")
        task.attempts = task.max_attempts
        assert task.exhausted

    def test_scope_rejects_absolute_paths(self) -> None:
        with pytest.raises(ValueError, match="repo-relative"):
            TaskScope(write_paths=("C:/Windows/**",))

    def test_scope_rejects_traversal(self) -> None:
        with pytest.raises(ValueError, match="'\\.\\.'"):
            TaskScope(write_paths=("../escape/**",))

    def test_verification_requirement_kind_is_constrained(self) -> None:
        """A planner cannot smuggle a raw command through the verification requirement."""
        VerificationRequirement(kind="tests")
        with pytest.raises(ValueError):
            VerificationRequirement(kind="rm -rf /")

    def test_plan_rejects_duplicate_task_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate task ids"):
            Plan(goal="g", tasks=(make_task("t1"), make_task("t1")))


class TestTaskTransitions:
    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (TaskStatus.PENDING, TaskStatus.READY),
            (TaskStatus.READY, TaskStatus.RUNNING),
            (TaskStatus.RUNNING, TaskStatus.SUCCEEDED),
            (TaskStatus.RUNNING, TaskStatus.FAILED),
            (TaskStatus.RUNNING, TaskStatus.READY),
        ],
    )
    def test_legal_transitions(self, start: TaskStatus, target: TaskStatus) -> None:
        assert can_transition(start, target)

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (TaskStatus.PENDING, TaskStatus.RUNNING),
            (TaskStatus.PENDING, TaskStatus.SUCCEEDED),
            (TaskStatus.SUCCEEDED, TaskStatus.RUNNING),
            (TaskStatus.FAILED, TaskStatus.READY),
            (TaskStatus.BLOCKED, TaskStatus.RUNNING),
            (TaskStatus.CANCELLED, TaskStatus.READY),
        ],
    )
    def test_illegal_transitions(self, start: TaskStatus, target: TaskStatus) -> None:
        assert not can_transition(start, target)

    def test_illegal_transition_raises(self) -> None:
        task = make_task("t1")
        with pytest.raises(ValueError, match="illegal task transition"):
            task.transition_to(TaskStatus.SUCCEEDED)

    def test_terminal_states(self) -> None:
        assert TaskStatus.SUCCEEDED.terminal
        assert TaskStatus.FAILED.terminal
        assert TaskStatus.BLOCKED.terminal
        assert not TaskStatus.RUNNING.terminal


class TestGraphConstruction:
    def test_accepts_a_valid_chain(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        assert len(graph) == 2 and "a" in graph

    def test_rejects_unknown_dependency(self) -> None:
        with pytest.raises(PlanValidationError, match="unknown task"):
            TaskGraph([make_task("a", depends=("ghost",))])

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(PlanValidationError, match="duplicate task id"):
            TaskGraph([make_task("a"), make_task("a")])

    def test_rejects_two_node_cycle(self) -> None:
        with pytest.raises(PlanValidationError, match="cycle"):
            TaskGraph([make_task("a", depends=("b",)), make_task("b", depends=("a",))])

    def test_rejects_long_cycle(self) -> None:
        with pytest.raises(PlanValidationError, match="cycle"):
            TaskGraph(
                [
                    make_task("a", depends=("c",)),
                    make_task("b", depends=("a",)),
                    make_task("c", depends=("b",)),
                ]
            )

    def test_accepts_a_diamond(self) -> None:
        graph = TaskGraph(
            [
                make_task("a"),
                make_task("b", depends=("a",)),
                make_task("c", depends=("a",)),
                make_task("d", depends=("b", "c")),
            ]
        )
        assert len(graph) == 4

    def test_unknown_task_lookup_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="unknown task"):
            TaskGraph([make_task("a")]).get("ghost")


class TestOrdering:
    def test_topological_order_respects_dependencies(self) -> None:
        graph = TaskGraph(
            [
                make_task("a"),
                make_task("b", depends=("a",)),
                make_task("c", depends=("b",)),
            ]
        )
        assert graph.topological_order() == ("a", "b", "c")

    def test_ordering_is_deterministic(self) -> None:
        """The same plan must execute in the same sequence every run."""
        def build() -> TaskGraph:
            return TaskGraph(
                [
                    make_task("a"),
                    make_task("b", depends=("a",)),
                    make_task("c", depends=("a",)),
                    make_task("d", depends=("b", "c")),
                ]
            )

        assert build().topological_order() == build().topological_order()

    def test_priority_breaks_ties(self) -> None:
        graph = TaskGraph(
            [make_task("z", priority=1), make_task("a", priority=500)]
        )
        assert graph.topological_order()[0] == "z"


class TestScheduling:
    def test_root_tasks_become_ready(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        ready = graph.ready_tasks()
        assert [task.task_id for task in ready] == ["a"]

    def test_dependent_unlocks_on_success(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        succeed(graph, "a")
        assert [task.task_id for task in graph.ready_tasks()] == ["b"]

    def test_dependent_blocks_on_failure(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        start(graph, "a")
        graph.mark_failed("a", "boom", FailureCategory.TEST_FAILURE)
        assert graph.get("b").status is TaskStatus.BLOCKED
        assert graph.ready_tasks() == ()

    def test_blocking_cascades_transitively(self) -> None:
        """A failure two levels up must block the whole downstream chain."""
        graph = TaskGraph(
            [
                make_task("a"),
                make_task("b", depends=("a",)),
                make_task("c", depends=("b",)),
            ]
        )
        start(graph, "a")
        graph.mark_failed("a", "boom")
        assert graph.get("b").status is TaskStatus.BLOCKED
        assert graph.get("c").status is TaskStatus.BLOCKED

    def test_diamond_waits_for_both_branches(self) -> None:
        graph = TaskGraph(
            [
                make_task("a"),
                make_task("b", depends=("a",)),
                make_task("c", depends=("a",)),
                make_task("d", depends=("b", "c")),
            ]
        )
        for task_id in ("a", "b"):
            succeed(graph, task_id)
        assert graph.get("d").status is TaskStatus.PENDING
        assert [t.task_id for t in graph.ready_tasks()] == ["c"]

    def test_next_task_is_singular(self) -> None:
        """Sequential execution: one heavy inference at a time."""
        graph = TaskGraph([make_task("a"), make_task("b")])
        assert graph.next_task() is not None
        assert len(graph.ready_tasks()) == 2

    def test_next_task_is_none_when_nothing_ready(self) -> None:
        graph = TaskGraph([make_task("a")])
        start(graph, "a")
        assert graph.next_task() is None

    def test_refresh_is_idempotent(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        for _ in range(3):
            graph.refresh()
        assert graph.get("a").status is TaskStatus.READY


class TestRetryBounds:
    def test_requeue_within_budget(self) -> None:
        graph = TaskGraph([make_task("a")])
        task = start(graph, "a")
        task.attempts = 1
        assert graph.requeue("a") is True
        assert task.status is TaskStatus.READY

    def test_requeue_refused_when_exhausted(self) -> None:
        """Bounded retries: the graph refuses rather than looping forever."""
        graph = TaskGraph([make_task("a")])
        task = start(graph, "a")
        task.attempts = task.max_attempts
        assert graph.requeue("a") is False
        assert task.status is TaskStatus.RUNNING


class TestAggregateState:
    def test_incomplete_while_work_remains(self) -> None:
        assert not TaskGraph([make_task("a")]).is_complete()

    def test_complete_when_all_succeeded(self) -> None:
        graph = TaskGraph([make_task("a")])
        succeed(graph, "a")
        assert graph.is_complete() and graph.succeeded()

    def test_complete_but_not_succeeded_after_failure(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        start(graph, "a")
        graph.mark_failed("a", "boom")
        assert graph.is_complete()
        assert not graph.succeeded()

    def test_summary_counts_by_status(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b")])
        summary = graph.summary()
        assert summary["PENDING"] == 2
