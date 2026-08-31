"""A task that never verified must not be reported as a success.

Found by running the fan-out calculator end to end and reading the log against the screen.
Every implementation task had been adjudicated FAIL on all three attempts, and all four were
reported SUCCEEDED with no failure reason. The run's own verdict was correct -- the final
gate failed it -- but the task list said the four functions were done and only assembly had
broken, when in truth one test file had no import, one asserted ``multiply(1, 2) == 3``, and
one compared a call to ``ValueError(...)``.

The deferral itself is right, and this does not remove it: a task that wrote its files but
left the suite red may well be waiting on a later task, and failing it there would strand the
remaining work and guarantee the run fails. What was wrong is that deferral borrowed
``SUCCEEDED`` to say so, which is the one status that means the opposite.

So ``DEFERRED`` is its own terminal state: it unlocks dependents exactly as success did, it
does not count as success anywhere, and the final gate still runs -- because a run whose
tasks all deferred has produced precisely the code the gate exists to judge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.planning.dag import TaskGraph
from edith.planning.task import Task, TaskStatus, can_transition


def task(task_id: str, *, depends_on: tuple[str, ...] = ()) -> Task:
    return Task(
        task_id=task_id,
        title=f"task {task_id}",
        description="d",
        agent="coder",
        dependencies=list(depends_on),
    )


def graph_of(*tasks: Task) -> TaskGraph:
    graph = TaskGraph(list(tasks))
    graph.refresh()
    return graph


def start(graph: TaskGraph, task_id: str) -> None:
    """Move a task to RUNNING the way the orchestrator does, via READY."""
    graph.refresh()
    graph.get(task_id).transition_to(TaskStatus.RUNNING)


class TestDeferredIsItsOwnState:
    def test_it_is_not_succeeded(self) -> None:
        assert TaskStatus.DEFERRED is not TaskStatus.SUCCEEDED

    def test_it_is_terminal(self) -> None:
        """The task is finished; only the final gate can still rule on the work."""
        assert TaskStatus.DEFERRED.terminal

    def test_a_running_task_may_defer(self) -> None:
        assert can_transition(TaskStatus.RUNNING, TaskStatus.DEFERRED)

    def test_a_deferred_task_cannot_be_promoted_afterwards(self) -> None:
        """Otherwise a later step could quietly launder it into a success."""
        for target in TaskStatus:
            assert not can_transition(TaskStatus.DEFERRED, target)


class TestDeferralStillUnlocksDependents:
    """The reason the old code used SUCCEEDED, and the behaviour that must not regress."""

    def test_a_dependent_of_a_deferred_task_becomes_ready(self) -> None:
        graph = graph_of(task("a"), task("b", depends_on=("a",)))
        start(graph, "a")
        graph.mark_deferred("a", "suite not green yet")
        assert graph.get("b").status is TaskStatus.READY

    def test_the_dependent_is_actually_scheduled(self) -> None:
        graph = graph_of(task("a"), task("b", depends_on=("a",)))
        start(graph, "a")
        graph.mark_deferred("a", "suite not green yet")
        nxt = graph.next_task()
        assert nxt is not None and nxt.task_id == "b"

    def test_a_failed_dependency_still_blocks(self) -> None:
        """Deferral must not have widened what counts as an acceptable dependency."""
        graph = graph_of(task("a"), task("b", depends_on=("a",)))
        start(graph, "a")
        graph.mark_failed("a", "broken")
        assert graph.get("b").status is TaskStatus.BLOCKED


class TestDeferralIsNotCountedAsSuccess:
    def test_succeeded_is_false_when_a_task_deferred(self) -> None:
        graph = graph_of(task("a"))
        start(graph, "a")
        graph.mark_deferred("a", "suite not green yet")
        assert not graph.succeeded()

    def test_the_summary_does_not_report_it_as_succeeded(self) -> None:
        """The number the CLI and benchmarks print as `tasks succeeded`."""
        graph = graph_of(task("a"))
        start(graph, "a")
        graph.mark_deferred("a", "suite not green yet")
        assert graph.summary().get("SUCCEEDED", 0) == 0
        assert graph.summary()["DEFERRED"] == 1

    def test_the_reason_is_recorded(self) -> None:
        """A status the reader has to guess at is barely better than the wrong one."""
        graph = graph_of(task("a"))
        start(graph, "a")
        graph.mark_deferred("a", "changed files but the suite was not green yet")
        assert "not green" in (graph.get("a").failure_reason or "")


class TestTheFinalGateStillRuns:
    """Skipping it would report a failure without ever judging the work."""

    def test_a_fully_deferred_run_is_settled_without_failure(self) -> None:
        graph = graph_of(task("a"), task("b", depends_on=("a",)))
        for task_id in ("a", "b"):
            start(graph, task_id)
            graph.mark_deferred(task_id, "suite not green yet")
        assert graph.settled_without_failure()
        assert not graph.succeeded()

    def test_a_failed_run_is_not(self) -> None:
        graph = graph_of(task("a"))
        start(graph, "a")
        graph.mark_failed("a", "broken")
        assert not graph.settled_without_failure()

    def test_the_orchestrator_gates_on_settlement_not_success(self) -> None:
        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "graph.settled_without_failure()" in source
        assert "graph.succeeded()" not in source

    def test_the_orchestrator_defers_rather_than_marking_success(self) -> None:
        """The exact line that produced four green tasks over unverified code."""
        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "graph.mark_deferred(" in source


class TestTheScreenDoesNotShowItGreen:
    @pytest.fixture
    def page(self) -> str:
        return Path("src/edith/ui/static/index.html").read_text(encoding="utf-8")

    def test_deferred_is_not_matched_by_the_success_pattern(self, page: str) -> None:
        assert "COMPLETE|PASS|VERIFIED|SUCCEED|MERGED|DONE" in page

    def test_deferred_is_rendered_as_a_warning(self, page: str) -> None:
        assert "PENDING|DEFER" in page
