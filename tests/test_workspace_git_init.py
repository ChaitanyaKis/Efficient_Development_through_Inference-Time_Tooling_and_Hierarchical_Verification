"""A created workspace is a git repository, and a failure is filed as what it was.

Both defects came out of the first run started from the UI rather than from a benchmark, and
both were invisible to every benchmark for the same reason: the harnesses ran ``git init``
themselves and nobody read the failure records.

**Git is not optional decoration.** ``git.diff`` reports what a task changed, M5.1 worktrees
give each task an isolated tree, and M5.2's merge copies verified files out of one. A workspace
without a repository loses all three, and the symptom surfaces far from the cause -- as
"workspace is not a git repository" from a tool call, several stages downstream.

**A record that says REPAIR when the policy escalated is a lie in the audit trail.** M5.2 exists
to stop an environment fault being charged to the coder; filing it as a repair reintroduces
exactly that misattribution in the one place a human would go to check.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from edith.config.loader import load_config
from edith.errors import FailureCategory
from edith.policy import FailureAction, decide
from edith.workspaces import WorkspaceManager


@pytest.fixture
def config(tmp_path: Path) -> Any:
    base = load_config(None)
    return base.model_copy(
        update={
            "orchestration": base.orchestration.model_copy(
                update={"workspaces_root": tmp_path / "workspaces"}
            )
        }
    )


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=False
    )


class TestACreatedWorkspaceIsUsable:
    def test_it_is_a_git_repository(self, config: Any) -> None:
        workspace = WorkspaceManager(config).create("proj", "proj_1")
        assert (workspace.root / ".git").is_dir()

    def test_it_has_a_base_revision_to_branch_from(self, config: Any) -> None:
        """``git worktree add`` needs a HEAD; without one isolation cannot begin."""
        workspace = WorkspaceManager(config).create("proj", "proj_1")
        result = git(workspace.root, "rev-parse", "HEAD")
        assert result.returncode == 0, result.stderr

    def test_a_worktree_can_actually_be_created(self, config: Any) -> None:
        """The end the whole thing exists for: M5.1 per-task isolation."""
        workspace = WorkspaceManager(config).create("proj", "proj_1")
        result = git(workspace.root, "worktree", "add", "-b", "t1", ".edith/wt/t1")
        assert result.returncode == 0, result.stderr

    def test_git_diff_works_in_it(self, config: Any) -> None:
        """The tool call that failed in the first UI run."""
        workspace = WorkspaceManager(config).create("proj", "proj_1")
        assert git(workspace.root, "diff", "--stat").returncode == 0

    def test_the_identity_is_local_and_edith_s_own(self, config: Any) -> None:
        """Edith must not write to the user's global config, or commit under their name."""
        workspace = WorkspaceManager(config).create("proj", "proj_1")
        email = git(workspace.root, "config", "--local", "user.email").stdout.strip()
        assert email == "edith@localhost"

    def test_creating_twice_does_not_reinitialise(self, config: Any) -> None:
        manager = WorkspaceManager(config)
        first = manager.create("proj", "proj_1")
        git(first.root, "commit", "--allow-empty", "-qm", "user work")
        before = git(first.root, "rev-list", "--count", "HEAD").stdout.strip()
        manager.create("proj", "proj_1")
        after = git(first.root, "rev-list", "--count", "HEAD").stdout.strip()
        assert before == after, "an existing repository must be left alone"

    def test_adopting_a_directory_does_not_force_git_on_it(
        self, config: Any, tmp_path: Path
    ) -> None:
        """A directory the user pointed Edith at is theirs; initialising it is not our call."""
        existing = tmp_path / "someone-elses"
        existing.mkdir()
        WorkspaceManager(config).adopt(existing, "proj_2")
        assert not (existing / ".git").exists()


class TestAFailureIsFiledAsWhatItWas:
    """The record must agree with the policy, or the audit trail misleads."""

    def test_an_environment_failure_escalates(self) -> None:
        action = decide(FailureCategory.ENVIRONMENT_FAILURE, attempts=1, max_attempts=3)
        assert action is FailureAction.ESCALATE

    def test_a_code_failure_repairs(self) -> None:
        action = decide(FailureCategory.CODE_FAILURE, attempts=1, max_attempts=3)
        assert action is FailureAction.REPAIR

    def test_the_orchestrator_records_the_decided_action(self) -> None:
        """It previously wrote "REPAIR" before consulting the policy at all."""
        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "str(action.value), reason, task.attempts" in source
        assert 'last_category, "REPAIR", reason' not in source

    def test_a_security_failure_is_filed_as_blocked(self) -> None:
        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert 'last_category, "BLOCKED", reason' in source


class TestGreenTestsMustExerciseTheChange:
    """A suite that passes without importing the change proves the suite works, not the code.

    Found by reading a real run that reported PASS: the coder wrote
    ``tests/test_mathops.py`` containing ``assert 1 + 1 == 2`` and never imported
    ``mathops``. Four tests passed, verification was green, and nothing had been verified --
    the same vacuous verification M5 caught in its own benchmark and M8 caught in generated
    tests, arriving by a third route.
    """

    def write(self, root: Path, relative: str, body: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def test_a_suite_that_never_imports_the_change_is_refused(self, tmp_path: Path) -> None:
        from edith.integrity import tests_exercise_changes

        self.write(tmp_path, "src/backend/mathops.py", "def add(a, b):\n    return a + b\n")
        self.write(
            tmp_path,
            "tests/test_mathops.py",
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(1 + 1, 2)\n",
        )
        reason = tests_exercise_changes(tmp_path, ("src/backend/mathops.py",))
        assert reason is not None
        assert "never import" in reason
        assert "mathops" in reason

    def test_a_suite_that_imports_the_change_is_accepted(self, tmp_path: Path) -> None:
        from edith.integrity import tests_exercise_changes

        self.write(tmp_path, "src/backend/mathops.py", "def add(a, b):\n    return a + b\n")
        self.write(
            tmp_path,
            "tests/test_mathops.py",
            "from src.backend.mathops import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        assert tests_exercise_changes(tmp_path, ("src/backend/mathops.py",)) is None

    def test_a_bare_module_import_counts(self, tmp_path: Path) -> None:
        """Layout varies; any import path that reaches the module is evidence."""
        from edith.integrity import tests_exercise_changes

        self.write(tmp_path, "calculator.py", "def add(a, b):\n    return a + b\n")
        self.write(
            tmp_path,
            "tests/test_calculator.py",
            "import calculator\n\n\ndef test_add():\n    assert calculator.add(1, 1) == 2\n",
        )
        assert tests_exercise_changes(tmp_path, ("calculator.py",)) is None

    def test_a_project_with_no_tests_is_not_penalised_here(self, tmp_path: Path) -> None:
        """That is the verifier's job, and it already reports it."""
        from edith.integrity import tests_exercise_changes

        self.write(tmp_path, "src/backend/m.py", "x = 1\n")
        assert tests_exercise_changes(tmp_path, ("src/backend/m.py",)) is None

    def test_changing_only_tests_is_not_checked(self, tmp_path: Path) -> None:
        """A task that edits tests has no implementation change to exercise."""
        from edith.integrity import tests_exercise_changes

        self.write(tmp_path, "tests/test_a.py", "def test_x():\n    assert True\n")
        assert tests_exercise_changes(tmp_path, ("tests/test_a.py",)) is None

    def test_the_orchestrator_cannot_pass_an_unexercised_change(self) -> None:
        source = Path("src/edith/orchestrator.py").read_text(encoding="utf-8")
        assert "tests_exercise_changes(self.workspace.root, tuple(changed))" in source
        assert "if unexercised is None:" in source
