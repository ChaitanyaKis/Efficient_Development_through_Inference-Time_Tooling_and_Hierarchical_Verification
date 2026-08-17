"""Per-agent authorization: tool grants and path scope."""

from __future__ import annotations

import pytest

from edith.errors import PermissionDeniedError
from edith.schemas.agent import AgentPermissions
from edith.tools.permissions import UNRESTRICTED, PermissionEngine
from edith.tools.schemas import AccessMode, ToolSpec

READ_SPEC = ToolSpec(name="filesystem.read", description="read a file")
NET_SPEC = ToolSpec(name="browser.search", description="search", uses_network=True)


def engine(**kwargs: object) -> PermissionEngine:
    return PermissionEngine(AgentPermissions(**kwargs))  # type: ignore[arg-type]


class TestToolGrants:
    def test_granted_tool_allowed(self) -> None:
        assert engine(allowed_tools=frozenset({"filesystem.read"})).may_use_tool(
            "filesystem.read"
        )

    def test_ungranted_tool_denied(self) -> None:
        assert not engine(allowed_tools=frozenset({"filesystem.read"})).may_use_tool(
            "filesystem.write"
        )

    def test_no_grants_denies_everything(self) -> None:
        """The default agent has no tools at all -- least privilege by construction."""
        assert not engine().may_use_tool("filesystem.read")

    def test_namespace_wildcard(self) -> None:
        subject = engine(allowed_tools=frozenset({"filesystem.*"}))
        assert subject.may_use_tool("filesystem.read")
        assert subject.may_use_tool("filesystem.write")
        assert not subject.may_use_tool("shell.run")

    def test_global_wildcard(self) -> None:
        assert engine(allowed_tools=frozenset({"*"})).may_use_tool("shell.run")

    def test_authorize_raises_with_context(self) -> None:
        with pytest.raises(PermissionDeniedError, match="not permitted to use tool") as info:
            engine().authorize_tool(READ_SPEC, agent="planner")
        assert info.value.details["tool"] == "filesystem.read"
        assert info.value.details["agent"] == "planner"

    def test_denial_is_classified_as_a_security_event(self) -> None:
        """A denial must surface as SECURITY_FAILURE so audit review can find it."""
        with pytest.raises(PermissionDeniedError) as info:
            engine().authorize_tool(READ_SPEC)
        assert str(info.value.category) == "SECURITY_FAILURE"
        assert info.value.retryable is False

    def test_network_tool_requires_network_permission(self) -> None:
        grants = frozenset({"browser.search"})
        with pytest.raises(PermissionDeniedError, match="requires network access"):
            engine(allowed_tools=grants).authorize_tool(NET_SPEC)
        engine(allowed_tools=grants, network_access=True).authorize_tool(NET_SPEC)


class TestPathScope:
    def test_exact_grant(self) -> None:
        subject = engine(allowed_read_paths=("src/app.py",))
        subject.authorize_path("src/app.py", AccessMode.READ, "src/app.py")

    def test_directory_grant_covers_descendants(self) -> None:
        subject = engine(allowed_read_paths=("src/backend",))
        subject.authorize_path("src/backend/api/routes.py", AccessMode.READ, "x")

    def test_recursive_glob_grant(self) -> None:
        subject = engine(allowed_read_paths=("src/**",))
        subject.authorize_path("src/deeply/nested/file.py", AccessMode.READ, "x")

    def test_outside_scope_denied(self) -> None:
        subject = engine(allowed_read_paths=("src/**",))
        with pytest.raises(PermissionDeniedError, match="outside the agent's read scope"):
            subject.authorize_path("tests/test_app.py", AccessMode.READ, "tests/test_app.py")

    def test_no_scope_denies_with_a_clear_reason(self) -> None:
        with pytest.raises(PermissionDeniedError, match="no write scope"):
            engine().authorize_path("src/app.py", AccessMode.WRITE, "src/app.py")

    def test_read_grant_does_not_imply_write(self) -> None:
        """The commonest privilege-escalation bug: conflating the two scopes."""
        subject = engine(allowed_read_paths=("src/**",))
        subject.authorize_path("src/app.py", AccessMode.READ, "src/app.py")
        with pytest.raises(PermissionDeniedError):
            subject.authorize_path("src/app.py", AccessMode.WRITE, "src/app.py")

    def test_write_grant_does_not_imply_read(self) -> None:
        subject = engine(allowed_write_paths=("src/**",))
        with pytest.raises(PermissionDeniedError):
            subject.authorize_path("src/app.py", AccessMode.READ, "src/app.py")

    def test_scope_matching_is_case_insensitive(self) -> None:
        """The filesystem is case-insensitive; the scope check must agree with it."""
        subject = engine(allowed_read_paths=("src/**",))
        subject.authorize_path("SRC/App.py", AccessMode.READ, "SRC/App.py")

    def test_dotfile_grant_is_not_mangled(self) -> None:
        """Regression: a naive lstrip('./') turned '.config/x' into 'config/x'."""
        subject = engine(allowed_read_paths=(".config/**",))
        subject.authorize_path(".config/settings.toml", AccessMode.READ, "x")
        with pytest.raises(PermissionDeniedError):
            subject.authorize_path("config/settings.toml", AccessMode.READ, "x")

    def test_sibling_prefix_is_not_granted(self) -> None:
        """A grant on 'src' must not leak into 'src_secrets'."""
        subject = engine(allowed_read_paths=("src",))
        with pytest.raises(PermissionDeniedError):
            subject.authorize_path("src_secrets/keys.txt", AccessMode.READ, "x")

    def test_extension_glob(self) -> None:
        subject = engine(allowed_read_paths=("*.md",))
        subject.authorize_path("README.md", AccessMode.READ, "README.md")

    def test_full_tree_grant(self) -> None:
        subject = engine(allowed_write_paths=("**",))
        subject.authorize_path("anything/at/all.py", AccessMode.WRITE, "x")


class TestUnrestricted:
    def test_grants_everything(self) -> None:
        subject = PermissionEngine(UNRESTRICTED)
        assert subject.may_use_tool("shell.run")
        subject.authorize_path("any/path.py", AccessMode.WRITE, "x")
        subject.authorize_tool(NET_SPEC)

    def test_is_not_the_default_for_agents(self) -> None:
        """A freshly declared agent must start with nothing, never with UNRESTRICTED."""
        assert AgentPermissions() != UNRESTRICTED
        assert not AgentPermissions().allowed_tools
