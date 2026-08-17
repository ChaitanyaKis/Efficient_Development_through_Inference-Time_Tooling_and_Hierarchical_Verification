"""Persistence, artifacts, and restart recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.errors import ConfigurationError, FailureCategory
from edith.planning.dag import TaskGraph
from edith.planning.task import Task, TaskStatus
from edith.state.schema import (
    AgentRun,
    Execution,
    FailureRecord,
    Project,
    ProjectState,
    ToolExecution,
    VerificationRecord,
    can_transition,
)
from edith.state.store import ArtifactStore, StateStore, open_store


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    with open_store(tmp_path / "state") as opened:
        yield opened


@pytest.fixture
def project(store: StateStore, tmp_path: Path) -> Project:
    return store.save_project(
        Project(name="demo", workspace_root=str(tmp_path / "ws"))
    )


@pytest.fixture
def execution(store: StateStore, project: Project) -> Execution:
    return store.save_execution(
        Execution(project_id=project.project_id, request="add a multiply function")
    )


def make_task(task_id: str, *, depends: tuple[str, ...] = ()) -> Task:
    return Task(
        task_id=task_id,
        title=f"task {task_id}",
        description="do the thing",
        agent="coder",
        dependencies=depends,
    )


class TestArtifactStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        artifacts = ArtifactStore(tmp_path / "a")
        digest = artifacts.put("hello world")
        assert artifacts.get(digest) == "hello world"

    def test_identical_content_stored_once(self, tmp_path: Path) -> None:
        artifacts = ArtifactStore(tmp_path / "a")
        assert artifacts.put("same") == artifacts.put("same")

    def test_unknown_digest_returns_none(self, tmp_path: Path) -> None:
        assert ArtifactStore(tmp_path / "a").get("0" * 64) is None

    def test_json_payload(self, tmp_path: Path) -> None:
        artifacts = ArtifactStore(tmp_path / "a")
        digest = artifacts.put_json({"exit_code": 1, "stdout": "boom"})
        assert '"exit_code": 1' in (artifacts.get(digest) or "")

    def test_large_output_does_not_live_in_the_database(
        self, store: StateStore, execution: Execution
    ) -> None:
        """Rows stay small; the payload lives on disk and is referenced by digest."""
        huge = "x" * 500_000
        digest = store.artifacts.put(huge)
        store.save_verification(
            VerificationRecord(
                execution_id=execution.execution_id,
                kind="tests",
                command="pytest",
                exit_code=1,
                passed=False,
                output_ref=digest,
            )
        )
        assert store.database_path.stat().st_size < 200_000
        assert store.artifacts.get(digest) == huge


class TestProjects:
    def test_save_and_load(self, store: StateStore, tmp_path: Path) -> None:
        saved = store.save_project(Project(name="demo", workspace_root=str(tmp_path)))
        loaded = store.get_project(saved.project_id)
        assert loaded is not None and loaded.name == "demo"

    def test_lookup_by_name(self, store: StateStore, project: Project) -> None:
        assert store.find_project_by_name("demo") is not None

    def test_unknown_returns_none(self, store: StateStore) -> None:
        assert store.get_project("proj_missing") is None

    def test_update_is_idempotent(self, store: StateStore, project: Project) -> None:
        project.description = "updated"
        store.save_project(project)
        assert len(store.list_projects()) == 1
        assert store.get_project(project.project_id).description == "updated"


class TestExecutions:
    def test_save_and_load(self, store: StateStore, execution: Execution) -> None:
        loaded = store.get_execution(execution.execution_id)
        assert loaded is not None
        assert loaded.state is ProjectState.RECEIVED
        assert loaded.request == "add a multiply function"

    def test_transition_is_persisted_and_audited(
        self, store: StateStore, execution: Execution
    ) -> None:
        store.record_transition(execution, ProjectState.PLANNING, "planner starting")
        reloaded = store.get_execution(execution.execution_id)
        assert reloaded.state is ProjectState.PLANNING

        history = store.transitions(execution.execution_id)
        assert len(history) == 1
        assert history[0].from_state == "RECEIVED" and history[0].to_state == "PLANNING"

    def test_illegal_transition_rejected(
        self, store: StateStore, execution: Execution
    ) -> None:
        with pytest.raises(ValueError, match="illegal project transition"):
            store.record_transition(execution, ProjectState.RELEASE)

    def test_terminal_state_records_finished_at(
        self, store: StateStore, execution: Execution
    ) -> None:
        store.record_transition(execution, ProjectState.FAILED, "aborted")
        assert store.get_execution(execution.execution_id).finished_at is not None

    def test_failed_reachable_from_anywhere(self) -> None:
        for state in ProjectState:
            if not state.terminal:
                assert can_transition(state, ProjectState.FAILED)

    def test_terminal_states_are_final(self) -> None:
        assert not can_transition(ProjectState.RELEASE, ProjectState.REPAIR)
        assert not can_transition(ProjectState.FAILED, ProjectState.PLANNING)

    def test_repair_returns_to_implementation(self) -> None:
        """The repair loop must be expressible in the state machine."""
        assert can_transition(ProjectState.REVIEW, ProjectState.REPAIR)
        assert can_transition(ProjectState.REPAIR, ProjectState.IMPLEMENTATION)
        assert can_transition(ProjectState.IMPLEMENTATION, ProjectState.VERIFICATION)


class TestTaskPersistence:
    def test_round_trip_preserves_everything(
        self, store: StateStore, execution: Execution
    ) -> None:
        task = make_task("t1")
        task.attempts = 2
        store.save_task(execution.execution_id, task)

        loaded = store.load_tasks(execution.execution_id)
        assert len(loaded) == 1
        assert loaded[0].task_id == "t1"
        assert loaded[0].attempts == 2
        assert loaded[0].agent == "coder"

    def test_dependencies_are_stored_as_edges(
        self, store: StateStore, execution: Execution
    ) -> None:
        store.save_tasks(
            execution.execution_id, [make_task("a"), make_task("b", depends=("a",))]
        )
        assert store.task_dependencies(execution.execution_id) == {"b": ("a",)}

    def test_resaving_replaces_edges(self, store: StateStore, execution: Execution) -> None:
        store.save_task(execution.execution_id, make_task("b", depends=("a",)))
        store.save_task(execution.execution_id, make_task("b"))
        assert store.task_dependencies(execution.execution_id) == {}

    def test_status_updates_persist(self, store: StateStore, execution: Execution) -> None:
        task = make_task("t1")
        store.save_task(execution.execution_id, task)
        task.transition_to(TaskStatus.READY)
        task.transition_to(TaskStatus.RUNNING)
        store.save_task(execution.execution_id, task)
        assert store.load_tasks(execution.execution_id)[0].status is TaskStatus.RUNNING


class TestRestartRecovery:
    def test_state_survives_process_restart(self, tmp_path: Path) -> None:
        """The acceptance criterion: kill the process, reopen, resume where it stopped."""
        state_dir = tmp_path / "state"

        # --- First "process": plan and partially execute.
        with open_store(state_dir) as first:
            project = first.save_project(
                Project(name="demo", workspace_root=str(tmp_path / "ws"))
            )
            execution = first.save_execution(
                Execution(project_id=project.project_id, request="do the work")
            )
            first.record_transition(execution, ProjectState.PLANNING)
            first.record_transition(execution, ProjectState.IMPLEMENTATION)

            graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
            graph.refresh()
            task = graph.get("a")
            task.transition_to(TaskStatus.RUNNING)
            task.attempts = 1
            first.save_tasks(execution.execution_id, list(graph.tasks()))
            execution_id = execution.execution_id

        # --- Second "process": nothing in memory is carried over.
        with open_store(state_dir) as second:
            reloaded = second.get_execution(execution_id)
            assert reloaded is not None
            assert reloaded.state is ProjectState.IMPLEMENTATION

            tasks = second.load_tasks(execution_id)
            resumed = TaskGraph(tasks)
            assert resumed.get("a").status is TaskStatus.RUNNING
            assert resumed.get("a").attempts == 1
            assert resumed.get("b").status is TaskStatus.PENDING

            # And the resumed graph still schedules correctly.
            resumed.mark_succeeded("a")
            assert [t.task_id for t in resumed.ready_tasks()] == ["b"]

    def test_evidence_survives_restart(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        with open_store(state_dir) as first:
            project = first.save_project(Project(name="d", workspace_root=str(tmp_path)))
            execution = first.save_execution(
                Execution(project_id=project.project_id, request="r")
            )
            digest = first.artifacts.put("FAILED tests/test_x.py::test_y")
            first.save_verification(
                VerificationRecord(
                    execution_id=execution.execution_id,
                    kind="tests",
                    command="pytest -q",
                    exit_code=1,
                    passed=False,
                    tests_failed=1,
                    output_ref=digest,
                )
            )
            execution_id = execution.execution_id

        with open_store(state_dir) as second:
            records = second.verifications(execution_id)
            assert len(records) == 1 and not records[0].passed
            assert "FAILED" in (second.artifacts.get(records[0].output_ref) or "")

    def test_schema_version_mismatch_is_refused(self, tmp_path: Path) -> None:
        """A database from an incompatible build must not be silently reinterpreted."""
        state_dir = tmp_path / "state"
        with open_store(state_dir) as first:
            first._connection.execute(
                "UPDATE meta SET value = '99' WHERE key = 'schema_version'"
            )
        with pytest.raises(ConfigurationError, match="schema v99"):
            open_store(state_dir)


class TestEvidenceRecords:
    def test_agent_run(self, store: StateStore, execution: Execution) -> None:
        run = store.save_agent_run(
            AgentRun(
                execution_id=execution.execution_id,
                agent="coder",
                model="qwen",
                prompt_tokens=100,
                completion_tokens=50,
            )
        )
        run.status = "SUCCESS"
        store.save_agent_run(run)
        runs = store.agent_runs(execution.execution_id)
        assert len(runs) == 1 and runs[0].status == "SUCCESS"
        assert runs[0].prompt_tokens == 100

    def test_tool_execution(self, store: StateStore, execution: Execution) -> None:
        store.save_tool_execution(
            ToolExecution(
                execution_id=execution.execution_id, tool="filesystem.write", ok=True
            )
        )
        records = store.tool_executions(execution.execution_id)
        assert len(records) == 1 and records[0].ok is True

    def test_denied_tool_execution_is_recorded(
        self, store: StateStore, execution: Execution
    ) -> None:
        """A refused operation must be visible in the persisted audit trail."""
        store.save_tool_execution(
            ToolExecution(
                execution_id=execution.execution_id,
                tool="filesystem.read",
                ok=False,
                error="path targets a protected location",
                failure_category=FailureCategory.SECURITY_FAILURE,
            )
        )
        record = store.tool_executions(execution.execution_id)[0]
        assert record.failure_category is FailureCategory.SECURITY_FAILURE

    def test_verification_record(self, store: StateStore, execution: Execution) -> None:
        store.save_verification(
            VerificationRecord(
                execution_id=execution.execution_id,
                kind="tests",
                command="pytest",
                exit_code=0,
                passed=True,
                tests_passed=12,
            )
        )
        record = store.verifications(execution.execution_id)[0]
        assert record.passed is True and record.tests_passed == 12

    def test_failure_record(self, store: StateStore, execution: Execution) -> None:
        store.save_failure(
            FailureRecord(
                execution_id=execution.execution_id,
                category=FailureCategory.TEST_FAILURE,
                action="REPAIR",
                message="2 tests failed",
            )
        )
        record = store.failures(execution.execution_id)[0]
        assert record.category is FailureCategory.TEST_FAILURE
        assert record.action == "REPAIR"

    def test_records_are_scoped_to_their_execution(
        self, store: StateStore, project: Project, execution: Execution
    ) -> None:
        other = store.save_execution(
            Execution(project_id=project.project_id, request="another")
        )
        store.save_failure(
            FailureRecord(
                execution_id=execution.execution_id,
                category=FailureCategory.UNKNOWN,
                action="ESCALATE",
                message="x",
            )
        )
        assert len(store.failures(other.execution_id)) == 0
