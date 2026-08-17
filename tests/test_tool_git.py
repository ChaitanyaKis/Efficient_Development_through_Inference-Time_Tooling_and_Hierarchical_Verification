"""Git tools against real repositories.

These use an actual ``git`` binary in a temporary repo rather than mocking subprocess: the
value of these tools is that they drive real git correctly, which a mock cannot show.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from edith.errors import ToolExecutionError
from edith.schemas.agent import AgentPermissions
from edith.tools.gateway import ToolGateway
from edith.tools.git import _validate_ref
from edith.tools.schemas import ToolCall

from .tool_fixtures import build_gateway, build_workspace

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)


def git(repo: Path, *arguments: str) -> str:
    """Run a git command directly, for arranging test state."""
    completed = subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, text=True, check=True
    )
    return completed.stdout


def init_repo(workspace: Path, *, commit: bool = True) -> Path:
    """Initialize a throwaway repository with a local identity and no inherited hooks.

    ``core.hooksPath`` is pointed at an empty directory *for this repo only*. A developer
    machine may have a global hooks path installed (a commit gate, a linter); inheriting it
    would make these tests depend on unrelated tooling. The developer's global config is
    never modified.
    """
    git(workspace, "init", "-b", "main")
    hooks = workspace.parent / "empty-hooks"
    hooks.mkdir(exist_ok=True)
    git(workspace, "config", "core.hooksPath", str(hooks))
    git(workspace, "config", "user.email", "test@localhost")
    git(workspace, "config", "user.name", "Test User")
    git(workspace, "config", "commit.gpgsign", "false")
    if commit:
        git(workspace, "add", "-A")
        git(workspace, "commit", "-m", "initial commit")
    return workspace


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An initialized repository with one commit."""
    return init_repo(build_workspace(tmp_path / "ws"))


@pytest.fixture
def gateway(repo: Path) -> ToolGateway:
    return build_gateway(repo)


def call(gateway: ToolGateway, tool: str, **arguments: object) -> object:
    return gateway.execute(ToolCall(tool=tool, arguments=arguments))


class TestRefValidation:
    @pytest.mark.parametrize(
        "value",
        ["--upload-pack=evil", "-x", "a..b", "branch.lock", "a//b", "", "  ", "bad name"],
    )
    def test_dangerous_refs_rejected(self, value: str) -> None:
        """A ref starting with '-' would be read by git as an option."""
        with pytest.raises(ToolExecutionError):
            _validate_ref(value)

    @pytest.mark.parametrize("value", ["main", "agent/task-1", "v1.2.3", "HEAD", "origin/main"])
    def test_legitimate_refs_accepted(self, value: str) -> None:
        assert _validate_ref(value) == value


class TestStatus:
    def test_clean_repository(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.status")
        assert result.ok
        assert result.output["branch"] == "main"
        assert result.output["clean"] is True

    def test_detects_modification(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text("changed\n", encoding="utf-8")
        result = call(gateway, "git.status")
        assert result.output["clean"] is False
        assert any(f["path"] == "src/app.py" for f in result.output["files"])

    def test_detects_untracked(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "brand_new.py").write_text("x\n", encoding="utf-8")
        result = call(gateway, "git.status")
        assert any("brand_new" in f["path"] for f in result.output["files"])

    def test_non_repository_reports_clearly(self, tmp_path: Path) -> None:
        plain = build_workspace(tmp_path / "plain")
        result = call(build_gateway(plain), "git.status")
        assert not result.ok and "not a git repository" in result.error


class TestDiff:
    def test_empty_when_clean(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.diff")
        assert result.ok and result.output["empty"] is True

    def test_shows_working_tree_changes(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text(
            "def main():\n    return 'changed'\n", encoding="utf-8"
        )
        result = call(gateway, "git.diff")
        assert result.ok
        assert "changed" in result.output["diff"]
        assert result.output["files_changed"] == 1

    def test_staged_only(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text("staged change\n", encoding="utf-8")
        git(repo, "add", "src/app.py")
        (repo / "tests" / "test_app.py").write_text("unstaged\n", encoding="utf-8")
        result = call(gateway, "git.diff", staged=True)
        assert "app.py" in result.output["diff"]
        assert "test_app.py" not in result.output["diff"]

    def test_path_filter(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text("a\n", encoding="utf-8")
        (repo / "tests" / "test_app.py").write_text("b\n", encoding="utf-8")
        result = call(gateway, "git.diff", paths=["src/app.py"])
        assert "app.py" in result.output["diff"]
        assert "test_app.py" not in result.output["diff"]

    def test_stat_only(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text("a\n", encoding="utf-8")
        result = call(gateway, "git.diff", stat_only=True)
        assert result.ok and "|" in result.output["diff"]

    def test_option_lookalike_ref_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.diff", ref="--output=/tmp/pwned")
        assert not result.ok

    def test_path_outside_workspace_denied(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.diff", paths=["../outside.txt"])
        assert not result.ok and result.denied


class TestLog:
    def test_lists_commits(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.log")
        assert result.ok
        assert result.output["commits"][0]["subject"] == "initial commit"
        assert result.output["commits"][0]["sha"]

    def test_respects_max_entries(self, gateway: ToolGateway, repo: Path) -> None:
        for index in range(5):
            (repo / "src" / f"f{index}.py").write_text("x\n", encoding="utf-8")
            git(repo, "add", "-A")
            git(repo, "commit", "-m", f"commit {index}")
        result = call(gateway, "git.log", max_entries=3)
        assert len(result.output["commits"]) == 3

    def test_empty_repository_is_not_an_error(self, tmp_path: Path) -> None:
        """A repo with no commits is a normal state, not a failure."""
        fresh = init_repo(build_workspace(tmp_path / "fresh"), commit=False)
        result = call(build_gateway(fresh), "git.log")
        assert result.ok and result.output["commits"] == []

    def test_path_filter(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "app.py").write_text("x\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-m", "touch app")
        result = call(gateway, "git.log", paths=["src/app.py"])
        assert result.ok and result.output["commits"]


class TestBranch:
    def test_lists_branches(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch")
        assert result.ok
        assert "main" in result.output["branches"]
        assert result.output["current_branch"] == "main"

    def test_creates_a_prefixed_branch(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch", action="create", name="agent/task-1")
        assert result.ok
        assert result.output["created"] == "agent/task-1"
        assert result.output["current_branch"] == "agent/task-1"

    def test_unprefixed_branch_denied(self, gateway: ToolGateway) -> None:
        """Agent work must stay off shared branch names."""
        result = call(gateway, "git.branch", action="create", name="feature-x")
        assert not result.ok and result.denied
        assert "must start with" in result.error

    def test_protected_branch_cannot_be_deleted(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch", action="delete", name="main")
        assert not result.ok and result.denied
        assert "protected" in result.error

    def test_checkout(self, gateway: ToolGateway) -> None:
        call(gateway, "git.branch", action="create", name="agent/task-2")
        result = call(gateway, "git.branch", action="checkout", name="main")
        assert result.ok and result.output["current_branch"] == "main"

    def test_delete_merged_agent_branch(self, gateway: ToolGateway) -> None:
        call(gateway, "git.branch", action="create", name="agent/temp")
        call(gateway, "git.branch", action="checkout", name="main")
        result = call(gateway, "git.branch", action="delete", name="agent/temp")
        assert result.ok and result.output["deleted"] == "agent/temp"

    def test_action_requires_a_name(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch", action="create")
        assert not result.ok and "requires a branch name" in result.error

    def test_option_lookalike_name_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch", action="create", name="-D")
        assert not result.ok

    def test_unknown_action_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.branch", action="force-push")
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"


class TestCommit:
    def test_stages_and_commits(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "src" / "feature.py").write_text("x = 1\n", encoding="utf-8")
        result = call(gateway, "git.commit", message="add feature", paths=["src/feature.py"])
        assert result.ok
        assert result.output["sha"]
        assert result.output["files_changed"] == 1

    def test_records_agent_attribution(self, gateway: ToolGateway, repo: Path) -> None:
        """Every modification must be attributable (CLAUDE.md)."""
        (repo / "src" / "feature.py").write_text("x = 1\n", encoding="utf-8")
        result = call(gateway, "git.commit", message="add feature", paths=["src/feature.py"])
        assert "Edith-Agent: test_agent" in result.output["message"]
        assert "Edith-Call: call_" in result.output["message"]
        assert "Edith-Agent" in git(repo, "log", "-1", "--format=%B")

    def test_refuses_empty_commit(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.commit", message="nothing here")
        assert not result.ok and "nothing staged" in result.error

    def test_allow_empty_when_requested(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.commit", message="checkpoint", allow_empty=True)
        assert result.ok

    def test_path_outside_write_scope_denied(self, gateway: ToolGateway, repo: Path) -> None:
        (repo / "docs" / "notes.md").write_text("x\n", encoding="utf-8")
        result = call(gateway, "git.commit", message="docs", paths=["docs/notes.md"])
        assert not result.ok and result.denied

    def test_commits_without_a_configured_global_identity(
        self, gateway: ToolGateway, repo: Path
    ) -> None:
        """A fresh worktree may have no user.name; the tool supplies one explicitly."""
        git(repo, "config", "--unset", "user.email")
        git(repo, "config", "--unset", "user.name")
        (repo / "src" / "x.py").write_text("x\n", encoding="utf-8")
        result = call(gateway, "git.commit", message="works anyway", paths=["src/x.py"])
        assert result.ok

    def test_empty_message_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.commit", message="")
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"


class TestWorktree:
    def test_lists_the_main_worktree(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.worktree")
        assert result.ok and result.output["worktrees"]

    def test_creates_an_isolated_worktree(self, gateway: ToolGateway, repo: Path) -> None:
        result = call(gateway, "git.worktree", action="add", name="task-1")
        assert result.ok
        assert result.output["created_path"] == ".edith/worktrees/task-1"
        assert (repo / ".edith" / "worktrees" / "task-1" / "src" / "app.py").is_file()

    def test_worktree_gets_its_own_branch(self, gateway: ToolGateway, repo: Path) -> None:
        call(gateway, "git.worktree", action="add", name="task-2")
        branches = git(repo, "branch", "--list", "--format=%(refname:short)")
        assert "agent/task-2" in branches

    def test_duplicate_worktree_refused(self, gateway: ToolGateway) -> None:
        call(gateway, "git.worktree", action="add", name="task-3")
        result = call(gateway, "git.worktree", action="add", name="task-3")
        assert not result.ok and "already exists" in result.error

    def test_removal(self, gateway: ToolGateway, repo: Path) -> None:
        call(gateway, "git.worktree", action="add", name="task-4")
        result = call(gateway, "git.worktree", action="remove", name="task-4")
        assert result.ok
        assert not (repo / ".edith" / "worktrees" / "task-4").exists()

    def test_unprefixed_branch_denied(self, gateway: ToolGateway) -> None:
        result = call(gateway, "git.worktree", action="add", name="t5", branch="main-copy")
        assert not result.ok and result.denied

    @pytest.mark.parametrize("name", ["../escape", "a/b", "name with space", "-x"])
    def test_unsafe_worktree_names_rejected(self, gateway: ToolGateway, name: str) -> None:
        result = call(gateway, "git.worktree", action="add", name=name)
        assert not result.ok

    def test_isolation_between_worktrees(self, gateway: ToolGateway, repo: Path) -> None:
        """Two agents working in parallel must not see each other's edits."""
        call(gateway, "git.worktree", action="add", name="alpha")
        call(gateway, "git.worktree", action="add", name="beta")
        alpha = repo / ".edith" / "worktrees" / "alpha" / "src" / "app.py"
        beta = repo / ".edith" / "worktrees" / "beta" / "src" / "app.py"
        alpha.write_text("alpha edit\n", encoding="utf-8")
        assert "alpha edit" not in beta.read_text(encoding="utf-8")


class TestGitPermissions:
    def test_agent_without_git_grant_denied(self, repo: Path) -> None:
        limited = AgentPermissions(
            allowed_tools=frozenset({"filesystem.read"}), allowed_read_paths=("**",)
        )
        result = call(build_gateway(repo, limited), "git.status")
        assert not result.ok and result.denied

    def test_readonly_agent_can_inspect_but_not_commit(self, repo: Path) -> None:
        inspector = AgentPermissions(
            allowed_tools=frozenset({"git.status", "git.log"}), allowed_read_paths=("**",)
        )
        gateway = build_gateway(repo, inspector)
        assert call(gateway, "git.status").ok
        assert not call(gateway, "git.commit", message="x", allow_empty=True).ok
