"""M5: executing a plan — DAG order, scope enforcement, and the completion gate.

The claim under test is M5 item 12: a task is not complete because code was generated. These
tests drive the executor with a scripted model so every gate can be exercised deterministically
— an agent that writes nothing, one that writes outside its task, one whose prerequisite
failed, and one whose verification rejects the result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from edith.config.schema import (
    EdithConfig,
    ModelParams,
    ModelsConfig,
    OrchestrationConfig,
    ShellPolicyConfig,
    ToolsConfig,
    VerificationProfile,
)
from edith.engineering.executor import (
    EngineeringExecutor,
    ExecutionReport,
    TaskOutcome,
)
from edith.engineering.ownership import EngineeringRole
from edith.product.architecture import (
    Complexity,
    ImplementationPlanDocument,
    PlannedTask,
)
from edith.product.prd import AcceptanceCriterion, PRDDocument, Requirement
from edith.workspaces import ProjectWorkspace

from .fakes import FakeProvider

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PARAMS = ModelParams(model_name="test-model:q4")


def edits(*files: tuple[str, str], summary: str = "done") -> str:
    """A scripted ModelEdits response."""
    return json.dumps(
        {
            "edits": [
                {"path": path, "mode": "replace_file", "content": content}
                for path, content in files
            ],
            "summary": summary,
            "notes": "",
        }
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A workspace with a passing test suite, so verification means something."""
    root = tmp_path / "project"
    (root / "src" / "backend").mkdir(parents=True)
    (root / "src" / "frontend").mkdir(parents=True)
    (root / "migrations").mkdir()
    (root / "tests").mkdir()
    (root / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n", encoding="utf-8"
    )
    (root / "src" / "backend" / "api.py").write_text(
        "def existing():\n    return 1\n", encoding="utf-8"
    )

    # A real repository. Worktree isolation needs one, and a fixture without a commit would
    # let an isolation test fail for the wrong reason -- which is exactly what happened the
    # first time this was written.
    #
    # `core.hooksPath` points at an empty directory *inside this temp repo*: the user's
    # global config installs a commit hook requiring a review attestation, which blocks the
    # initial commit and leaves the repo with no HEAD, and a repo with no HEAD cannot have
    # worktrees. Same local-only workaround M1 established -- global config is never
    # touched, and `--skip`/`--vouch` are never used.
    hooks = root / ".githooks-empty"
    hooks.mkdir()
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Edith Test"],
        ["git", "config", "core.hooksPath", str(hooks)],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "base"],
    ):
        completed = subprocess.run(argv, cwd=str(root), capture_output=True, text=True)
        # Never swallow a setup failure: it surfaces later as a confusing assertion about
        # the thing under test rather than about the fixture.
        assert completed.returncode == 0, (
            f"fixture setup failed: {' '.join(argv)} -> {completed.stderr.strip()}"
        )
    return root


@pytest.fixture
def config(repo: Path) -> EdithConfig:
    import sys

    return EdithConfig(
        models=ModelsConfig(profiles={"default": PARAMS}),
        tools=ToolsConfig(
            workspace_root=repo,
            shell=ShellPolicyConfig(
                allowed_executables=(Path(sys.executable).stem, "python")
            ),
        ),
        orchestration=OrchestrationConfig(
            workspaces_root=repo.parent,
            verification_profiles={
                "python": VerificationProfile(tests=("python", "-m", "pytest", "-q"))
            },
        ),
    )


@pytest.fixture
def workspace(repo: Path) -> ProjectWorkspace:
    return ProjectWorkspace(project_id="proj_test", name="test", root=repo)


def task(
    task_id: str,
    agent: str,
    *,
    paths: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
    verification: tuple[str, ...] = ("tests",),
) -> PlannedTask:
    return PlannedTask(
        task_id=task_id,
        title=f"Task {task_id}",
        description="Implement the thing.",
        agent=agent,
        paths=paths,
        depends_on=depends_on,
        verification=verification,
        complexity=Complexity.SMALL,
    )


def plan(*tasks: PlannedTask) -> ImplementationPlanDocument:
    return ImplementationPlanDocument(
        product_name="Demo", goal="Build the demo.", tasks=tasks
    )


def build(
    config: EdithConfig, workspace: ProjectWorkspace, responses: list[str]
) -> tuple[EngineeringExecutor, FakeProvider]:
    provider = FakeProvider(PARAMS, responses)
    # isolate=False: these tests exercise the *gates* -- completion, scope, DAG order,
    # context -- and assert against files in the shared tree. Isolation has its own suite
    # below, which runs with it on.
    return (
        EngineeringExecutor(config, workspace, provider=provider, isolate=False),
        provider,
    )


def run(
    config: EdithConfig,
    workspace: ProjectWorkspace,
    responses: list[str],
    *tasks: PlannedTask,
    verify: bool = True,
) -> ExecutionReport:
    executor, _ = build(config, workspace, responses)
    return executor.execute(plan(*tasks), verify=verify)


class TestCompletionGate:
    """M5 item 12: generating code is not the same as finishing a task."""

    def test_a_task_that_changes_files_and_verifies_completes(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [edits(("src/backend/service.py", "def serve():\n    return 2\n"))],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        assert report.completed == 1
        assert report.executions[0].changed_files == ("src/backend/service.py",)

    def test_an_agent_whose_every_edit_is_refused_does_not_complete(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """The failure M5 item 12 names first: nothing changed, so nothing is done.

        The agent produced edits and the gateway refused all of them, which is the realistic
        shape of "changed no files" -- ``ModelEdits`` will not validate an empty edit list,
        so an agent cannot report doing nothing.
        """
        report = run(
            config,
            workspace,
            [edits(("src/frontend/app.py", "x = 1\n"))],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        assert report.completed == 0
        execution = report.executions[0]
        assert execution.outcome is TaskOutcome.FAILED
        assert "changed no files" in execution.detail
        assert execution.rejected_files

    def test_failing_verification_exhausts_the_repair_budget(
        self, config: EdithConfig, workspace: ProjectWorkspace, repo: Path
    ) -> None:
        """Code that exists but does not work is not a completed task.

        The scripted model repeats the same broken edit, so every repair attempt fails and
        the task ends REPAIR_EXHAUSTED -- M5.1 item 9's fail-closed terminal state, which is
        deliberately distinct from a single REJECTED attempt.
        """
        (repo / "tests" / "test_smoke.py").write_text(
            "def test_smoke():\n    assert False\n", encoding="utf-8"
        )
        report = run(
            config,
            workspace,
            [edits(("src/backend/service.py", "def serve():\n    return 2\n"))],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        execution = report.executions[0]
        assert execution.outcome is TaskOutcome.REPAIR_EXHAUSTED
        assert execution.verification is not None
        assert not execution.verification.passed
        assert execution.repair_attempts >= 1
        assert "exhausted" in execution.detail

    def test_a_malformed_model_response_fails_the_task(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            ['{"nonsense": true}'],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        assert report.executions[0].outcome is TaskOutcome.FAILED
        assert report.executions[0].failure_category is not None


class TestScopeEnforcement:
    """The boundary is the gateway's, and the executor reports what it refused."""

    def test_a_write_outside_the_role_is_refused(
        self, config: EdithConfig, workspace: ProjectWorkspace, repo: Path
    ) -> None:
        """A frontend agent cannot write a migration, whatever it produces."""
        report = run(
            config,
            workspace,
            [edits(("migrations/0001.sql", "CREATE TABLE t (id INT);\n"))],
            task("TASK-001", "frontend", paths=("src/frontend/app.py",)),
        )
        execution = report.executions[0]
        assert execution.outcome is not TaskOutcome.COMPLETED
        assert not (repo / "migrations" / "0001.sql").exists()

    def test_a_task_naming_a_forbidden_path_is_never_assigned(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [edits(("migrations/0001.sql", "-- x\n"))],
            task("TASK-001", "frontend", paths=("migrations/0001.sql",)),
        )
        assert report.executions[0].outcome is TaskOutcome.UNASSIGNED
        assert "outside that role's remit" in report.executions[0].detail

    def test_an_unknown_agent_is_unassigned_not_executed(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [edits(("src/backend/x.py", "x = 1\n"))],
            task("TASK-001", "security"),
        )
        assert report.executions[0].outcome is TaskOutcome.UNASSIGNED

    def test_changing_a_file_the_task_never_named_is_recorded(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """Inside its permissions, outside its brief. M5 item 17's speculative rewrite."""
        report = run(
            config,
            workspace,
            [
                edits(
                    ("src/backend/service.py", "def serve():\n    return 2\n"),
                    ("src/backend/api.py", "def existing():\n    return 99\n"),
                )
            ],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        execution = report.executions[0]
        assert "src/backend/api.py" in execution.out_of_scope
        assert report.scope_violations == 1


class TestDependencyOrder:
    """M5 item 9: nothing runs before its prerequisites are verified."""

    def test_tasks_run_in_dependency_order(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [
                edits(("migrations/0001.sql", "-- schema\n")),
                edits(("src/backend/service.py", "def serve():\n    return 2\n")),
            ],
            task(
                "TASK-002",
                "backend",
                paths=("src/backend/service.py",),
                depends_on=("TASK-001",),
            ),
            task("TASK-001", "database", paths=("migrations/0001.sql",)),
        )
        order = [item.task_id for item in report.executions]
        assert order == ["TASK-001", "TASK-002"]
        assert report.completed == 2

    def test_a_dependent_task_is_blocked_when_its_prerequisite_fails(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """Building on unverified work is how one bad task poisons the rest."""
        report = run(
            config,
            workspace,
            [
                json.dumps({"edits": [], "summary": "nothing", "notes": ""}),
                edits(("src/backend/service.py", "def serve():\n    return 2\n")),
            ],
            task("TASK-001", "database", paths=("migrations/0001.sql",)),
            task(
                "TASK-002",
                "backend",
                paths=("src/backend/service.py",),
                depends_on=("TASK-001",),
            ),
        )
        outcomes = {item.task_id: item.outcome for item in report.executions}
        assert outcomes["TASK-001"] is TaskOutcome.FAILED
        assert outcomes["TASK-002"] is TaskOutcome.BLOCKED
        assert report.blocked == 1

    def test_a_blocked_task_consumes_no_model_calls(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [json.dumps({"edits": [], "summary": "nothing", "notes": ""})],
            task("TASK-001", "database", paths=("migrations/0001.sql",)),
            task("TASK-002", "backend", paths=("src/backend/x.py",), depends_on=("TASK-001",)),
        )
        blocked = next(
            item for item in report.executions if item.outcome is TaskOutcome.BLOCKED
        )
        assert blocked.model_calls == 0


class TestConflictReporting:
    def test_conflicting_tasks_are_reported_on_the_execution(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [
                edits(("src/backend/api.py", "def existing():\n    return 2\n")),
                edits(("src/backend/api.py", "def existing():\n    return 3\n")),
            ],
            task("TASK-001", "backend", paths=("src/backend/api.py",)),
            task("TASK-002", "backend", paths=("src/backend/api.py",)),
        )
        assert report.conflicts
        assert report.conflicts[0].code == "TASK_CONFLICT"

    def test_execution_is_deterministic_despite_a_conflict(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """Serialised rather than raced, so two runs of one plan agree."""
        first = run(
            config,
            workspace,
            [edits(("src/backend/api.py", "def existing():\n    return 2\n"))],
            task("TASK-002", "backend", paths=("src/backend/api.py",)),
            task("TASK-001", "backend", paths=("src/backend/api.py",)),
        )
        assert [item.task_id for item in first.executions] == ["TASK-001", "TASK-002"]


class TestContextScoping:
    """M5 item 15: an agent receives its task's material, not the project."""

    def test_only_the_tasks_own_files_are_read_into_context(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        executor, provider = build(
            config,
            workspace,
            [edits(("src/backend/api.py", "def existing():\n    return 2\n"))],
        )
        executor.execute(
            plan(task("TASK-001", "backend", paths=("src/backend/api.py",))),
            verify=False,
        )
        prompt = "\n".join(
            content for _, content in provider.calls[0]["messages"]
        )
        assert "src/backend/api.py" in prompt
        assert "test_smoke" not in prompt, "unrelated files must not be in the prompt"

    def test_only_the_requirements_the_task_implements_are_supplied(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        document = PRDDocument(
            product_name="Demo",
            problem="p",
            requirements=(
                Requirement(requirement_id="REQ-001", title="Wanted", statement="do this"),
                Requirement(requirement_id="REQ-002", title="Unrelated", statement="not this"),
            ),
            acceptance_criteria=(
                AcceptanceCriterion(
                    criterion_id="AC-001", statement="c", verifies=("REQ-001",)
                ),
            ),
        )
        executor, provider = build(
            config, workspace, [edits(("src/backend/x.py", "x = 1\n"))]
        )
        assigned = task("TASK-001", "backend", paths=("src/backend/x.py",))
        assigned = assigned.model_copy(update={"implements": ("REQ-001",)})
        executor.execute(plan(assigned), prd=document, verify=False)

        prompt = "\n".join(content for _, content in provider.calls[0]["messages"])
        assert "REQ-001" in prompt
        assert "REQ-002" not in prompt

    def test_the_backend_is_not_given_the_ux_specification(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """Context an agent cannot act on is context it pays for and wastes."""
        from edith.product.ux import Flow, FlowStep, StepKind, UXSpecDocument

        spec = UXSpecDocument(
            product_name="Demo",
            flows=(
                Flow(
                    flow_id="UX-001",
                    name="DistinctiveFlowName",
                    entry_step="S1",
                    steps=(FlowStep(step_id="S1", name="Only", kind=StepKind.TERMINAL),),
                ),
            ),
        )
        executor, provider = build(
            config, workspace, [edits(("src/backend/x.py", "x = 1\n"))]
        )
        executor.execute(
            plan(task("TASK-001", "backend", paths=("src/backend/x.py",))),
            ux=spec,
            verify=False,
        )
        prompt = "\n".join(content for _, content in provider.calls[0]["messages"])
        assert "DistinctiveFlowName" not in prompt


class TestMetrics:
    def test_the_report_records_the_benchmark_metrics(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        report = run(
            config,
            workspace,
            [edits(("src/backend/service.py", "def serve():\n    return 2\n"))],
            task("TASK-001", "backend", paths=("src/backend/service.py",)),
        )
        metrics: dict[str, Any] = report.as_dict()  # type: ignore[assignment]
        assert metrics["tasks"] == 1
        assert metrics["completed"] == 1
        assert metrics["model_calls"] >= 1
        assert EngineeringRole.BACKEND.value in metrics["calls_by_role"]
        assert "src/backend/service.py" in metrics["changed_files"]
        json.dumps(metrics)


class TestRealExecutorIsolation:
    """M5.2: isolation asserted against the *real* executor, not the module alone.

    The critical invariant is that the main workspace does not change until an authorised
    merge. Testing the isolation class on its own cannot show that -- only running a task
    through the executor can.
    """

    def isolating(
        self, config: EdithConfig, workspace: ProjectWorkspace, responses: list[str]
    ) -> EngineeringExecutor:
        return EngineeringExecutor(
            config, workspace, provider=FakeProvider(PARAMS, responses), isolate=True
        )

    def test_a_task_runs_in_its_own_workspace(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        executor = self.isolating(
            config,
            workspace,
            [edits(("src/backend/service.py", "def serve():\n    return 2\n"))],
        )
        report = executor.execute(
            plan(task("TASK-001", "backend", paths=("src/backend/service.py",))),
            verify=False,
        )
        execution = report.executions[0]
        assert execution.workspace_id, "the task must record which workspace it used"
        assert execution.base_revision, "and the revision it branched from"

    def test_the_main_workspace_is_untouched_by_a_rejected_task(
        self, config: EdithConfig, workspace: ProjectWorkspace, repo: Path
    ) -> None:
        """The invariant M5.2 calls critical: a failure never reaches main."""
        # Committed, not merely written: the worktree branches from HEAD, so an uncommitted
        # failing test is invisible to it -- correctly. Before the root-propagation fix this
        # assertion passed for the wrong reason, because verification was running in a tree
        # that *did* see the uncommitted file. Committing makes the rejection genuine.
        (repo / "tests" / "test_smoke.py").write_text(
            "def test_smoke():\n    assert False\n", encoding="utf-8"
        )
        for argv in (["git", "add", "-A"], ["git", "commit", "-qm", "failing smoke test"]):
            completed = subprocess.run(argv, cwd=str(repo), capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
        before = sorted(p.name for p in (repo / "src" / "backend").iterdir())

        executor = self.isolating(
            config,
            workspace,
            [edits(("src/backend/service.py", "def serve():\n    return 2\n"))],
        )
        report = executor.execute(
            plan(task("TASK-001", "backend", paths=("src/backend/service.py",))),
        )
        assert report.completed == 0
        after = sorted(p.name for p in (repo / "src" / "backend").iterdir())
        assert after == before, "a rejected task must leave main exactly as it was"

    def test_isolation_can_be_disabled_without_changing_the_gates(
        self, config: EdithConfig, workspace: ProjectWorkspace
    ) -> None:
        """The benchmark harness already gives each trial a fresh tree."""
        executor = EngineeringExecutor(
            config,
            workspace,
            provider=FakeProvider(
                PARAMS, [edits(("src/backend/service.py", "def serve():\n    return 2\n"))]
            ),
            isolate=False,
        )
        report = executor.execute(
            plan(task("TASK-001", "backend", paths=("src/backend/service.py",)))
        )
        assert report.completed == 1
        assert report.executions[0].workspace_id == ""
