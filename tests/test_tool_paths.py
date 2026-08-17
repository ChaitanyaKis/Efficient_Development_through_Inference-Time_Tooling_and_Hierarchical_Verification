"""Path policy: normalization, containment, and the protected deny list.

These are the tests that matter most in M1. A defect here is a sandbox escape, so the
adversarial cases are enumerated explicitly rather than sampled.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from edith.config.schema import PathPolicyConfig
from edith.errors import PathPolicyError
from edith.tools.paths import PathPolicy


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace containing a small tree plus a sibling file that must stay unreachable."""
    root = tmp_path / "workspace"
    (root / "src" / "backend").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "src" / "backend" / "api.py").write_text("# api\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=hunter2\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("do not read me\n", encoding="utf-8")
    return root


@pytest.fixture
def policy(workspace: Path) -> PathPolicy:
    return PathPolicy.create(workspace, PathPolicyConfig())


class TestCreation:
    def test_root_is_resolved(self, workspace: Path) -> None:
        assert PathPolicy.create(workspace, PathPolicyConfig()).root == workspace.resolve()

    def test_missing_root_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathPolicyError, match="not an existing directory"):
            PathPolicy.create(tmp_path / "absent", PathPolicyConfig())

    def test_file_as_root_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "a-file"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(PathPolicyError, match="not an existing directory"):
            PathPolicy.create(target, PathPolicyConfig())


class TestLegitimatePaths:
    @pytest.mark.parametrize(
        "raw", ["src/app.py", "src/backend/api.py", "./src/app.py", "tests"]
    )
    def test_accepted(self, policy: PathPolicy, raw: str) -> None:
        assert policy.resolve(raw).is_relative_to(policy.root)

    def test_backslashes_are_normalized(self, policy: PathPolicy) -> None:
        """Agents on Windows will produce backslashes; they must work."""
        assert policy.resolve("src\\backend\\api.py").name == "api.py"

    def test_nonexistent_path_is_allowed(self, policy: PathPolicy) -> None:
        """Resolution answers 'where', not 'does it exist' -- writes need new paths."""
        assert policy.resolve("src/new_file.py").name == "new_file.py"

    def test_relative_of_round_trips(self, policy: PathPolicy) -> None:
        assert policy.relative_of(policy.resolve("src/app.py")) == "src/app.py"

    def test_dot_resolves_to_root(self, policy: PathPolicy) -> None:
        assert policy.resolve(".") == policy.root


class TestTraversalRejection:
    @pytest.mark.parametrize(
        "raw",
        [
            "../outside.txt",
            "../../etc/passwd",
            "src/../../outside.txt",
            "src/../../../Windows/System32/config/SAM",
            "./../../outside.txt",
            "src\\..\\..\\outside.txt",
            "a/b/c/../../../../outside.txt",
        ],
    )
    def test_parent_traversal_denied(self, policy: PathPolicy, raw: str) -> None:
        with pytest.raises(PathPolicyError):
            policy.resolve(raw)

    def test_traversal_that_lands_inside_is_still_denied(self, policy: PathPolicy) -> None:
        """`src/../src/app.py` resolves inside, but '..' is refused on principle: it is
        never necessary, and allowing it widens the surface for no benefit."""
        with pytest.raises(PathPolicyError, match="traversal"):
            policy.resolve("src/../src/app.py")


class TestAbsoluteAndDeviceRejection:
    @pytest.mark.parametrize(
        "raw",
        [
            "/etc/passwd",
            "//server/share/file",
            "\\\\server\\share\\file",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "C:/Windows/win.ini",
            "C:relative",
            "\\\\?\\C:\\Windows\\win.ini",
            "\\\\.\\PhysicalDrive0",
            "~/.ssh/id_rsa",
            "~",
        ],
    )
    def test_denied(self, policy: PathPolicy, raw: str) -> None:
        with pytest.raises(PathPolicyError):
            policy.resolve(raw)


class TestWindowsSpecificRejection:
    @pytest.mark.parametrize(
        "raw", ["CON", "nul", "src/CON", "COM1", "LPT9.txt", "src/aux.log", "PRN"]
    )
    def test_reserved_device_names_denied(self, policy: PathPolicy, raw: str) -> None:
        """Opening a reserved device can block on hardware rather than fail."""
        with pytest.raises(PathPolicyError, match="reserved Windows device name"):
            policy.resolve(raw)

    @pytest.mark.parametrize("raw", ["notes.txt:hidden", "src/app.py:$DATA"])
    def test_alternate_data_streams_denied(self, policy: PathPolicy, raw: str) -> None:
        with pytest.raises(PathPolicyError, match="alternate data stream"):
            policy.resolve(raw)

    def test_console_like_name_with_extension_denied(self, policy: PathPolicy) -> None:
        with pytest.raises(PathPolicyError):
            policy.resolve("con.txt")

    def test_similar_but_legitimate_names_allowed(self, policy: PathPolicy) -> None:
        """CONFIG and CONTEXT are not device names; over-blocking would be a bug."""
        for raw in ("config.py", "context.md", "console.js", "aux_helper.py"):
            assert policy.resolve(raw).name


class TestMalformedInput:
    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_denied(self, policy: PathPolicy, raw: str) -> None:
        with pytest.raises(PathPolicyError, match="must not be empty"):
            policy.resolve(raw)

    def test_nul_byte_denied(self, policy: PathPolicy) -> None:
        with pytest.raises(PathPolicyError, match="NUL byte"):
            policy.resolve("src/app.py\x00.txt")


class TestProtectedPaths:
    @pytest.mark.parametrize(
        "raw",
        [
            ".env",
            ".git",
            ".git/config",
            "secrets/token.txt",
            "credentials/aws.json",
            "server.pem",
            "certs/server.pem",
            "deploy.key",
            ".ssh/id_rsa",
            ".aws/credentials",
            "nested/.env",
            ".env.production",
        ],
    )
    def test_protected_locations_denied(self, policy: PathPolicy, raw: str) -> None:
        with pytest.raises(PathPolicyError, match="protected"):
            policy.resolve(raw)

    def test_protection_is_case_insensitive(self, policy: PathPolicy) -> None:
        """Treating .ENV as distinct from .env would be a trivial bypass."""
        for raw in (".ENV", ".Git/config", "SECRETS/x", "Server.PEM"):
            with pytest.raises(PathPolicyError, match="protected"):
                policy.resolve(raw)

    def test_nested_git_is_protected(self, policy: PathPolicy) -> None:
        with pytest.raises(PathPolicyError, match="protected"):
            policy.resolve("vendor/lib/.git/config")

    def test_ordinary_files_are_not_protected(self, policy: PathPolicy) -> None:
        for raw in ("src/app.py", "README.md", "environment.py", "keys.py", "gitignore.md"):
            assert policy.resolve(raw)

    def test_is_protected_is_directly_queryable(self, policy: PathPolicy) -> None:
        assert policy.is_protected(".env")
        assert not policy.is_protected("src/app.py")


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks on Windows needs elevation"
)
class TestSymlinkEscape:
    def test_symlink_pointing_outside_is_denied(
        self, workspace: Path, policy: PathPolicy, tmp_path: Path
    ) -> None:
        (workspace / "escape").symlink_to(tmp_path / "outside.txt")
        with pytest.raises(PathPolicyError):
            policy.resolve("escape")

    def test_symlink_inside_is_denied_by_default(
        self, workspace: Path, policy: PathPolicy
    ) -> None:
        """Even a link resolving back inside is refused: in an agent-managed tree it is
        far more likely an escape attempt than a legitimate need."""
        (workspace / "alias.py").symlink_to(workspace / "src" / "app.py")
        with pytest.raises(PathPolicyError, match="symlink"):
            policy.resolve("alias.py")

    def test_symlink_allowed_when_configured(self, workspace: Path) -> None:
        (workspace / "alias.py").symlink_to(workspace / "src" / "app.py")
        permissive = PathPolicy.create(workspace, PathPolicyConfig(allow_symlinks=True))
        assert permissive.resolve("alias.py").name == "app.py"

    def test_symlinked_directory_escape_is_denied(
        self, workspace: Path, policy: PathPolicy, tmp_path: Path
    ) -> None:
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "loot.txt").write_text("x", encoding="utf-8")
        (workspace / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        with pytest.raises(PathPolicyError):
            policy.resolve("link/loot.txt")


def _make_junction(link: Path, target: Path) -> bool:
    """Create a Windows directory junction. Returns False when unsupported."""
    if sys.platform != "win32":
        return False
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
class TestWindowsJunctionEscape:
    """Junctions are the escape vector a POSIX-shaped implementation misses.

    Unlike symlinks they need no elevation, and ``Path.is_symlink()`` returns **False**
    for them -- yet ``Path.resolve()`` follows them straight out of the workspace. These
    tests run for real rather than being skipped, because this is precisely the case that
    a symlink-only check would let through.
    """

    def test_junction_is_not_reported_as_a_symlink(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """Documents the trap: the symlink check alone cannot be the defense."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        link = workspace / "junction"
        if not _make_junction(link, outside):
            pytest.skip("could not create a junction in this environment")
        assert link.is_symlink() is False
        assert link.resolve() == outside.resolve()

    def test_junction_escape_is_denied_by_containment(
        self, workspace: Path, policy: PathPolicy, tmp_path: Path
    ) -> None:
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "loot.txt").write_text("secret", encoding="utf-8")
        link = workspace / "junction"
        if not _make_junction(link, outside):
            pytest.skip("could not create a junction in this environment")

        with pytest.raises(PathPolicyError, match="escapes the workspace"):
            policy.resolve("junction/loot.txt")
        with pytest.raises(PathPolicyError, match="escapes the workspace"):
            policy.resolve("junction")

    def test_junction_escape_denied_even_when_symlinks_are_allowed(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """Enabling symlinks must not disable containment."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "loot.txt").write_text("secret", encoding="utf-8")
        link = workspace / "junction"
        if not _make_junction(link, outside):
            pytest.skip("could not create a junction in this environment")

        permissive = PathPolicy.create(workspace, PathPolicyConfig(allow_symlinks=True))
        with pytest.raises(PathPolicyError, match="escapes the workspace"):
            permissive.resolve("junction/loot.txt")


class TestFileSizeLimit:
    def test_oversize_file_denied(self, workspace: Path) -> None:
        policy = PathPolicy.create(workspace, PathPolicyConfig(max_file_bytes=1024))
        big = workspace / "big.txt"
        big.write_text("x" * 5000, encoding="utf-8")
        with pytest.raises(PathPolicyError, match="exceeding"):
            policy.check_readable_size(big, "big.txt")

    def test_within_limit_returns_size(self, workspace: Path, policy: PathPolicy) -> None:
        target = policy.resolve("src/app.py")
        assert policy.check_readable_size(target, "src/app.py") > 0


class TestErrorHygiene:
    def test_error_does_not_leak_the_resolved_host_path(self, policy: PathPolicy) -> None:
        """A probing agent must not be able to map the host filesystem from error text."""
        with pytest.raises(PathPolicyError) as excinfo:
            policy.resolve("../../outside.txt")
        assert str(policy.root) not in excinfo.value.message
        assert str(policy.root) not in str(excinfo.value.details)
