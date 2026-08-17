"""M5.2: the task root must be absolute, and verification must run inside it.

This is the regression suite for a bug that took five falsified hypotheses to find, so it is
worth stating the mechanism plainly.

``git.worktree`` reports the tree it created as ``created_path``. ``create_workspace`` read
``result.output.get("path", "")``, which missed, defaulted to ``""``, and became ``Path(".")``
-- a *legal relative path* meaning "the process working directory". That directory is Edith's
own checkout. So:

    workspace.path == Path(".")
        -> _gateway(root=Path("."))
        -> PathPolicy rooted at Edith's repo
        -> pytest cwd == Edith's repo
        -> collects Edith's ~1,450 tests instead of the task's
        -> 120s timeout, and recursion, because Edith's own tests spawn pytest

Nothing announced the substitution. A relative path is valid, `Path("")` silently equals
`Path(".")`, and the misread key had a default. Three separate silences composed into one
vacuous-verification vulnerability: had Edith's suite been small enough to finish green, the
task would have been marked VERIFIED on tests the agent never touched.

Two invariants are defended here, because either alone is insufficient:

1. the path is made absolute at the boundary where it is created, and
2. a relative root is *refused* downstream rather than resolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.config.schema import ModelParams
from edith.engineering.executor import EngineeringExecutor
from edith.engineering.isolation import (
    WorkspaceError,
    WorkspaceLedger,
    create_workspace,
)
from edith.schemas.agent import AgentPermissions
from edith.workspaces import ProjectWorkspace

from .fakes import FakeProvider


class _FakeResult:
    """A ``git.worktree`` result whose payload the caller controls."""

    def __init__(self, output: dict[str, object], *, ok: bool = True) -> None:
        self.ok = ok
        self.output = output
        self.error = None
        self.denied = False


class _FakeGateway:
    def __init__(self, root: Path, output: dict[str, object]) -> None:
        from edith.tools.paths import PathPolicy

        from .tool_fixtures import build_gateway

        self._output = output
        real = build_gateway(root, AgentPermissions(allowed_tools=frozenset()))
        self.policy: PathPolicy = real.policy

    def execute(self, call: object) -> _FakeResult:
        return _FakeResult(self._output)


def executor_for(root: Path) -> EngineeringExecutor:
    from edith.config.loader import load_config

    return EngineeringExecutor(
        load_config(None),
        ProjectWorkspace(project_id="p", name="n", root=root),
        provider=FakeProvider(ModelParams(model_name="t"), []),
    )


class TestTheWorkspacePathIsAbsolute:
    def test_a_reported_relative_path_is_resolved_against_the_project(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / ".edith" / "worktrees" / "task-task-001"
        target.mkdir(parents=True)
        workspace = create_workspace(
            _FakeGateway(tmp_path, {"created_path": ".edith/worktrees/task-task-001"}),  # type: ignore[arg-type]
            task_id="TASK-001",
            execution_id="e",
            base_revision="abc123",
            ledger=WorkspaceLedger(),
        )
        assert workspace.path.is_absolute()
        assert workspace.path == target.resolve()

    def test_the_bug_itself_a_missing_path_key_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """The exact defect: the old code read ``"path"`` and got ``Path(".")``.

        A workspace that cannot say where it is must not default to "here".
        """
        with pytest.raises(WorkspaceError, match="reported no path"):
            create_workspace(
                _FakeGateway(tmp_path, {"path": "ignored-wrong-key"}),  # type: ignore[arg-type]
                task_id="TASK-001",
                execution_id="e",
                base_revision="abc123",
                ledger=WorkspaceLedger(),
            )

    def test_an_empty_reported_path_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceError, match="reported no path"):
            create_workspace(
                _FakeGateway(tmp_path, {"created_path": "   "}),  # type: ignore[arg-type]
                task_id="TASK-001",
                execution_id="e",
                base_revision="abc123",
                ledger=WorkspaceLedger(),
            )

    def test_a_path_that_is_not_a_directory_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceError, match="not a directory"):
            create_workspace(
                _FakeGateway(tmp_path, {"created_path": "never-created"}),  # type: ignore[arg-type]
                task_id="TASK-001",
                execution_id="e",
                base_revision="abc123",
                ledger=WorkspaceLedger(),
            )

    def test_the_workspace_path_is_never_the_process_directory(
        self, tmp_path: Path
    ) -> None:
        """The specific outcome that produced the hang."""
        target = tmp_path / "wt"
        target.mkdir()
        workspace = create_workspace(
            _FakeGateway(tmp_path, {"created_path": "wt"}),  # type: ignore[arg-type]
            task_id="TASK-001",
            execution_id="e",
            base_revision="abc123",
            ledger=WorkspaceLedger(),
        )
        assert workspace.path != Path().resolve()
        assert workspace.path != Path(".").resolve()


class TestARelativeRootIsRefused:
    """Defence in depth: even if a relative root were produced, it must not be used."""

    def test_a_relative_task_root_is_refused(self, tmp_path: Path) -> None:
        executor = executor_for(tmp_path)
        with pytest.raises(WorkspaceError, match="must be absolute"):
            executor._gateway(
                AgentPermissions(allowed_tools=frozenset()), "verifier", root=Path(".")
            )

    def test_the_dot_path_specifically_is_refused(self, tmp_path: Path) -> None:
        """``Path("")`` equals ``Path(".")``, which is how the bug expressed itself."""
        executor = executor_for(tmp_path)
        with pytest.raises(WorkspaceError, match="must be absolute"):
            executor._gateway(
                AgentPermissions(allowed_tools=frozenset()), "verifier", root=Path("")
            )

    def test_a_relative_subdirectory_root_is_refused(self, tmp_path: Path) -> None:
        executor = executor_for(tmp_path)
        with pytest.raises(WorkspaceError, match="must be absolute"):
            executor._gateway(
                AgentPermissions(allowed_tools=frozenset()),
                "verifier",
                root=Path(".edith/worktrees/task-001"),
            )

    def test_an_absolute_root_is_accepted(self, tmp_path: Path) -> None:
        executor = executor_for(tmp_path)
        gateway = executor._gateway(
            AgentPermissions(allowed_tools=frozenset()), "verifier", root=tmp_path
        )
        assert gateway.policy.root == tmp_path.resolve()

    def test_omitting_the_root_still_uses_the_project(self, tmp_path: Path) -> None:
        """Not every call is a task: the executor's own principals legitimately pass none."""
        executor = executor_for(tmp_path)
        gateway = executor._gateway(
            AgentPermissions(allowed_tools=frozenset()), "workspace"
        )
        assert gateway.policy.root == tmp_path.resolve()
