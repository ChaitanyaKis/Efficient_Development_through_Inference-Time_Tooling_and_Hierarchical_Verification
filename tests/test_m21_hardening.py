"""M2.1 hardening: context robustness, environment classification, staged verification.

Deterministic platform tests only. Nothing here calls a real model -- live-model evaluation
lives in the benchmark suite, and conflating the two produces a test suite whose failures
cannot be attributed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from edith.config.schema import ContextConfig, ShellPolicyConfig, VerificationProfile
from edith.context.engine import ContextEngine
from edith.errors import FailureCategory
from edith.integrity import FileKind, classify_path
from edith.planning.dag import TaskGraph
from edith.planning.task import Task, TaskStatus
from edith.schemas.agent import AgentPermissions
from edith.tools.process import resolve_executable
from edith.verification.runner import VerificationRunner

from .tool_fixtures import build_config, build_gateway, build_workspace

PYTHON = "python"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return build_workspace(tmp_path / "ws")


def make_task(task_id: str, *, depends: tuple[str, ...] = ()) -> Task:
    return Task(
        task_id=task_id,
        title=f"task {task_id}",
        description="do the thing",
        agent="coder",
        dependencies=depends,
    )


class TestContextPathHandling:
    """Regressions for the `**/*` bug and every neighbouring path assumption."""

    def test_top_level_files_are_retrieved(self, workspace: Path) -> None:
        """The exact M2 bug: '**/*' required a literal '/', so nothing top-level matched."""
        bundle = ContextEngine(build_gateway(workspace)).build("readme project")
        assert bundle.files_considered > 0
        assert any("/" not in path for path in bundle.file_paths)

    def test_nested_files_are_retrieved(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build("backend api routes handler")
        assert "src/backend/api.py" in bundle.file_paths

    def test_deeply_nested_files_are_retrieved(self, workspace: Path) -> None:
        deep = workspace / "src" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deepmodule.py").write_text("def deepfunc():\n    pass\n", encoding="utf-8")
        bundle = ContextEngine(build_gateway(workspace)).build("deepmodule deepfunc")
        assert "src/a/b/c/deepmodule.py" in bundle.file_paths

    def test_protected_files_are_never_included(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build("env api key secret")
        assert not any(".env" in path for path in bundle.file_paths)
        assert "super-secret-value" not in bundle.render()

    def test_hidden_directories_are_not_traversed(self, workspace: Path) -> None:
        hidden = workspace / ".git"
        hidden.mkdir(exist_ok=True)
        (hidden / "config").write_text("[core]\nbare = false\n", encoding="utf-8")
        bundle = ContextEngine(build_gateway(workspace)).build("core bare config")
        assert not any(path.startswith(".git") for path in bundle.file_paths)

    def test_windows_separators_in_hints_are_normalized(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build(
            "anything", hint_paths=("src\\backend\\api.py",)
        )
        assert "src/backend/api.py" in bundle.file_paths

    def test_leading_dot_slash_in_hints_is_normalized(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build(
            "anything", hint_paths=("./src/app.py",)
        )
        assert "src/app.py" in bundle.file_paths

    def test_directories_are_never_returned_as_files(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build("src backend tests docs")
        for path in bundle.file_paths:
            assert (workspace / path).is_file()

    def test_a_large_repository_stays_within_budget(self, workspace: Path) -> None:
        for index in range(300):
            (workspace / "src" / f"mod_{index}.py").write_text(
                f"def handler_{index}():\n    return {index}\n", encoding="utf-8"
            )
        engine = ContextEngine(
            build_gateway(workspace), ContextConfig(max_files=6, max_total_chars=4000)
        )
        bundle = engine.build("handler module")
        assert len(bundle.relevant_files) <= 6
        assert bundle.estimated_context_chars <= 4000

    @pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
    def test_a_junction_does_not_leak_outside_the_workspace(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret_notes.py").write_text("TOKEN = 'leak'\n", encoding="utf-8")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(workspace / "linked"), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip("could not create a junction in this environment")

        bundle = ContextEngine(build_gateway(workspace)).build("secret notes token")
        assert "leak" not in bundle.render()
        assert not any("secret_notes" in path for path in bundle.file_paths)


class TestContextFailsClosed:
    """An empty bundle must be visible, not silent."""

    def test_empty_workspace_is_flagged_as_degraded(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        bundle = ContextEngine(build_gateway(empty)).build("anything at all")
        assert bundle.degraded
        assert not bundle.usable
        assert "no files were indexed" in bundle.degraded_reason

    def test_no_matching_files_is_flagged(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build(
            "zzzqqq_nonexistent_symbol_xyz"
        )
        assert bundle.degraded
        assert "none matched" in bundle.degraded_reason

    def test_a_healthy_bundle_is_not_degraded(self, workspace: Path) -> None:
        bundle = ContextEngine(build_gateway(workspace)).build("app main helper")
        assert not bundle.degraded
        assert bundle.usable

    def test_degradation_survives_serialisation(self, tmp_path: Path) -> None:
        """The condition must reach anything inspecting the bundle later."""
        from edith.context.engine import ContextBundle

        empty = tmp_path / "empty"
        empty.mkdir()
        bundle = ContextEngine(build_gateway(empty)).build("anything")
        restored = ContextBundle.model_validate_json(bundle.model_dump_json())
        assert restored.degraded


class TestEnvironmentClassification:
    """A missing toolchain is an environment failure, never a test failure."""

    def _runner(self, workspace: Path, argv: tuple[str, ...]) -> VerificationRunner:
        config = build_config(
            workspace, shell=ShellPolicyConfig(allowed_executables=(PYTHON,))
        )
        gateway = build_gateway(
            workspace,
            AgentPermissions(
                allowed_tools=frozenset({"shell.run"}), allowed_read_paths=("**",)
            ),
            config=config,
        )
        return VerificationRunner(gateway, VerificationProfile(tests=argv))

    def test_python_resolves_to_the_running_interpreter(self) -> None:
        """The M2 bug: PATH's `python` had no pytest, and every run reported TEST_FAILURE."""
        assert resolve_executable("python", ("python",)) == sys.executable

    def test_python3_alias_too(self) -> None:
        assert resolve_executable("python3", ("python3",)) == sys.executable

    def test_other_executables_still_resolve_from_path(self) -> None:
        resolved = resolve_executable("git", ("git",))
        assert resolved != sys.executable
        assert Path(resolved).exists()

    def test_missing_runner_is_environment_not_test_failure(self, workspace: Path) -> None:
        runner = self._runner(
            workspace,
            (
                PYTHON,
                "-c",
                "import sys; sys.stderr.write(\"No module named 'pytest'\"); sys.exit(1)",
            ),
        )
        outcome = runner.run("tests")
        assert outcome.failure_category is FailureCategory.ENVIRONMENT_FAILURE
        assert not outcome.ran, "a runner that never started did not test anything"

    def test_a_genuine_test_failure_is_still_a_test_failure(self, workspace: Path) -> None:
        """The classification must not over-trigger and mask real defects."""
        runner = self._runner(
            workspace,
            (PYTHON, "-c", "print('1 failed, 2 passed'); raise SystemExit(1)"),
        )
        outcome = runner.run("tests")
        assert outcome.ran
        assert outcome.failure_category is FailureCategory.TEST_FAILURE
        assert outcome.tests_failed == 1

    def test_environment_failure_escalates_rather_than_repairs(self) -> None:
        """Sending the debugger after a missing dependency wastes the whole budget."""
        from edith.policy import FailureAction, decide

        assert decide(
            FailureCategory.ENVIRONMENT_FAILURE, attempts=1, max_attempts=3
        ) is FailureAction.ESCALATE

    def test_the_pytest_runner_is_actually_importable_here(self) -> None:
        """Guards the environment the rest of the suite depends on."""
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0


class TestStagedVerification:
    """Task-level and project-level verification are different questions."""

    def test_an_intermediate_task_is_not_held_to_final_conditions(self) -> None:
        """The M2 flaw: with a whole-suite check, only the last task could ever pass."""
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        graph.refresh()
        first = graph.get("a")
        first.transition_to(TaskStatus.RUNNING)

        from edith.orchestrator import Orchestrator

        assert Orchestrator._work_remains(graph, first), (
            "work remaining is what makes deferral correct"
        )

    def test_no_work_remains_for_the_last_task(self) -> None:
        graph = TaskGraph([make_task("a")])
        graph.refresh()
        only = graph.get("a")
        only.transition_to(TaskStatus.RUNNING)

        from edith.orchestrator import Orchestrator

        assert not Orchestrator._work_remains(graph, only)

    def test_a_deferred_task_still_unlocks_its_dependents(self) -> None:
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        graph.refresh()
        graph.get("a").transition_to(TaskStatus.RUNNING)
        graph.mark_succeeded("a")
        assert [task.task_id for task in graph.ready_tasks()] == ["b"]

    def test_a_blocked_task_still_stops_the_chain(self) -> None:
        """Deferral must not become a way for genuine failures to pass unnoticed."""
        graph = TaskGraph([make_task("a"), make_task("b", depends=("a",))])
        graph.refresh()
        graph.get("a").transition_to(TaskStatus.RUNNING)
        graph.mark_failed("a", "genuinely broken", FailureCategory.SECURITY_FAILURE)
        assert graph.get("b").status is TaskStatus.BLOCKED


class TestFileClassification:
    """The Judge must distinguish what kind of file changed."""

    def test_source_and_test_are_distinguished(self) -> None:
        assert classify_path("calculator.py") is FileKind.SOURCE
        assert classify_path("test_calculator.py") is FileKind.TEST

    def test_requirements_and_config_are_distinguished(self) -> None:
        assert classify_path("pyproject.toml") is FileKind.CONFIG
        assert classify_path("docs/adr/0001-x.md") is FileKind.DOC

    def test_fixtures_are_not_treated_as_tests(self) -> None:
        assert classify_path("tests/fixtures/payload.json") is FileKind.FIXTURE
