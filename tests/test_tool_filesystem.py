"""Filesystem tools: read, search, write, patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.config.schema import PathPolicyConfig
from edith.schemas.agent import AgentPermissions
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

from .tool_fixtures import (
    READONLY_PERMISSIONS,
    build_config,
    build_gateway,
    build_workspace,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return build_workspace(tmp_path / "ws")


@pytest.fixture
def gateway(workspace: Path) -> ToolGateway:
    return build_gateway(workspace)


def call(gateway: ToolGateway, tool: str, **arguments: object) -> object:
    """Execute a tool and return the result."""
    return gateway.execute(ToolCall(tool=tool, arguments=arguments))


class TestRead:
    def test_reads_a_file(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py")
        assert result.ok
        assert "def main()" in result.output["content"]
        assert result.output["total_lines"] == 6
        assert result.output["path"] == "src/app.py"

    def test_line_range(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py", start_line=4, max_lines=2)
        assert result.ok
        assert result.output["start_line"] == 4
        assert result.output["line_count"] == 2
        assert "helper" in result.output["content"]

    def test_truncation_is_reported(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py", max_lines=2)
        assert result.output["truncated"] is True

    def test_missing_file(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/absent.py")
        assert not result.ok and "file not found" in result.error

    def test_directory_is_not_a_file(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src")
        assert not result.ok and "not a regular file" in result.error

    def test_start_beyond_end(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py", start_line=999)
        assert not result.ok and "beyond the end" in result.error

    def test_traversal_denied(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="../outside.txt")
        assert not result.ok and result.denied

    def test_protected_file_denied(self, gateway: ToolGateway) -> None:
        """The .env file exists and is readable on disk; policy must still refuse it."""
        result = call(gateway, "filesystem.read", path=".env")
        assert not result.ok and result.denied
        assert "super-secret-value" not in str(result.model_dump())

    def test_oversize_file_denied(self, tmp_path: Path) -> None:
        workspace = build_workspace(tmp_path / "ws")
        (workspace / "src" / "big.py").write_text("x" * 5000, encoding="utf-8")
        gateway = build_gateway(
            workspace,
            config=build_config(workspace, paths=PathPolicyConfig(max_file_bytes=1024)),
        )
        result = call(gateway, "filesystem.read", path="src/big.py")
        assert not result.ok and result.denied

    def test_invalid_arguments_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py", start_line=0)
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"

    def test_unknown_argument_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.read", path="src/app.py", encoding="utf-16")
        assert not result.ok


class TestSearch:
    def test_by_name_glob(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", name_pattern="*.py")
        assert result.ok
        paths = {m["path"] for m in result.output["matches"]}
        assert "src/app.py" in paths and "tests/test_app.py" in paths
        assert "README.md" not in paths

    def test_recursive_glob_matches_top_level_files(self, gateway: ToolGateway) -> None:
        """Regression: '**/*' requires a literal '/' under fnmatch, so top-level files
        were invisible -- which silently emptied every context bundle."""
        result = call(gateway, "filesystem.search", name_pattern="**/*")
        paths = {m["path"] for m in result.output["matches"]}
        assert "README.md" in paths, "top-level files must match '**/*'"
        assert "src/app.py" in paths

    def test_recursive_glob_with_extension(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", name_pattern="**/*.py")
        paths = {m["path"] for m in result.output["matches"]}
        assert "src/app.py" in paths and "src/backend/api.py" in paths

    def test_by_nested_glob(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", name_pattern="src/**/*.py")
        paths = {m["path"] for m in result.output["matches"]}
        assert "src/backend/api.py" in paths

    def test_by_content_regex(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", content_pattern=r"def\s+helper")
        assert result.ok
        assert result.output["matches"][0]["path"] == "src/app.py"
        assert result.output["matches"][0]["line_number"] == 5

    def test_content_and_name_combined(self, gateway: ToolGateway) -> None:
        result = call(
            gateway, "filesystem.search", name_pattern="src/**", content_pattern="ROUTES"
        )
        paths = {m["path"] for m in result.output["matches"]}
        assert paths == {"src/backend/api.py"}

    def test_case_insensitive_by_default(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", content_pattern="DEF MAIN")
        assert result.output["matches"]

    def test_case_sensitive_when_requested(self, gateway: ToolGateway) -> None:
        result = call(
            gateway, "filesystem.search", content_pattern="DEF MAIN", case_sensitive=True
        )
        assert not result.output["matches"]

    def test_protected_files_are_omitted_not_announced(self, gateway: ToolGateway) -> None:
        """Reporting .env as 'forbidden' would leak its existence; it is simply absent."""
        result = call(gateway, "filesystem.search", content_pattern="API_KEY")
        assert result.ok
        assert not result.output["matches"]

    def test_out_of_scope_files_are_omitted(self, workspace: Path) -> None:
        narrow = AgentPermissions(
            allowed_tools=frozenset({"filesystem.search"}), allowed_read_paths=("src/**",)
        )
        result = call(build_gateway(workspace, narrow), "filesystem.search", name_pattern="*.py")
        paths = {m["path"] for m in result.output["matches"]}
        assert paths and all(p.startswith("src/") for p in paths)

    def test_requires_a_pattern(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search")
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"

    def test_invalid_regex_is_reported(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.search", content_pattern="(unclosed")
        assert not result.ok and "invalid content_pattern" in result.error

    def test_results_are_capped(self, workspace: Path) -> None:
        for index in range(40):
            (workspace / "src" / f"gen_{index}.py").write_text("marker\n", encoding="utf-8")
        gateway = build_gateway(workspace)
        result = call(gateway, "filesystem.search", content_pattern="marker", max_results=5)
        assert len(result.output["matches"]) <= 5
        assert result.output["truncated"] is True

    def test_noisy_directories_are_skipped(self, workspace: Path) -> None:
        vendored = workspace / "node_modules" / "pkg"
        vendored.mkdir(parents=True)
        (vendored / "index.js").write_text("marker\n", encoding="utf-8")
        gateway = build_gateway(workspace)
        result = call(gateway, "filesystem.search", content_pattern="marker")
        assert not any("node_modules" in m["path"] for m in result.output["matches"])

    def test_binary_files_are_skipped(self, workspace: Path) -> None:
        (workspace / "src" / "blob.bin").write_bytes(b"marker\x00\x01\x02")
        gateway = build_gateway(workspace)
        result = call(gateway, "filesystem.search", content_pattern="marker")
        assert not any(m["path"].endswith(".bin") for m in result.output["matches"])


class TestWrite:
    def test_creates_a_new_file(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(gateway, "filesystem.write", path="src/new.py", content="x = 1\n")
        assert result.ok and result.output["created"] is True
        assert (workspace / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_refuses_to_overwrite_by_default(self, gateway: ToolGateway) -> None:
        """An agent must state its intent before destroying existing work."""
        result = call(gateway, "filesystem.write", path="src/app.py", content="wiped")
        assert not result.ok and "already exists" in result.error

    def test_overwrites_when_requested(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(
            gateway, "filesystem.write", path="src/app.py", content="new\n", overwrite=True
        )
        assert result.ok and result.output["created"] is False
        assert (workspace / "src" / "app.py").read_text(encoding="utf-8") == "new\n"

    def test_creates_parent_directories(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(gateway, "filesystem.write", path="src/deep/nested/x.py", content="y\n")
        assert result.ok
        assert (workspace / "src" / "deep" / "nested" / "x.py").is_file()

    def test_write_outside_scope_denied(self, gateway: ToolGateway, workspace: Path) -> None:
        """The agent may READ docs/ but not write there."""
        result = call(gateway, "filesystem.write", path="docs/hacked.md", content="x")
        assert not result.ok and result.denied
        assert not (workspace / "docs" / "hacked.md").exists()

    def test_readonly_agent_cannot_write(self, workspace: Path) -> None:
        gateway = build_gateway(workspace, READONLY_PERMISSIONS)
        result = call(gateway, "filesystem.write", path="src/x.py", content="x")
        assert not result.ok and result.denied

    def test_protected_file_cannot_be_written(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(gateway, "filesystem.write", path=".env", content="x", overwrite=True)
        assert not result.ok and result.denied
        assert "API_KEY" in (workspace / ".env").read_text(encoding="utf-8")

    def test_traversal_write_denied(self, gateway: ToolGateway, tmp_path: Path) -> None:
        result = call(gateway, "filesystem.write", path="../escaped.py", content="x")
        assert not result.ok and result.denied
        assert not (tmp_path / "escaped.py").exists()

    def test_line_endings_are_preserved_exactly(
        self, gateway: ToolGateway, workspace: Path
    ) -> None:
        """Python must not silently rewrite LF to CRLF and corrupt a diff."""
        call(gateway, "filesystem.write", path="src/eol.py", content="a\nb\nc\n")
        assert (workspace / "src" / "eol.py").read_bytes() == b"a\nb\nc\n"

    def test_oversize_content_rejected(self, workspace: Path) -> None:
        gateway = build_gateway(
            workspace,
            config=build_config(workspace, paths=PathPolicyConfig(max_file_bytes=1024)),
        )
        result = call(gateway, "filesystem.write", path="src/big.py", content="x" * 5000)
        assert not result.ok and "exceeding" in result.error


class TestPatch:
    def test_replaces_unique_text(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(
            gateway,
            "filesystem.patch",
            path="src/app.py",
            old_text="return 'hello'",
            new_text="return 'goodbye'",
        )
        assert result.ok and result.output["replacements"] == 1
        assert "goodbye" in (workspace / "src" / "app.py").read_text(encoding="utf-8")

    def test_missing_text_is_an_error(self, gateway: ToolGateway) -> None:
        result = call(
            gateway, "filesystem.patch", path="src/app.py", old_text="absent", new_text="x"
        )
        assert not result.ok and "was not found" in result.error

    def test_ambiguous_match_refused(self, gateway: ToolGateway, workspace: Path) -> None:
        """Refusing an ambiguous patch stops an agent rewriting code it never inspected."""
        (workspace / "src" / "dup.py").write_text("a = 1\na = 1\n", encoding="utf-8")
        result = call(
            gateway, "filesystem.patch", path="src/dup.py", old_text="a = 1", new_text="a = 2"
        )
        assert not result.ok and "appears 2 times" in result.error
        assert (workspace / "src" / "dup.py").read_text(encoding="utf-8") == "a = 1\na = 1\n"

    def test_replace_all_when_requested(self, gateway: ToolGateway, workspace: Path) -> None:
        (workspace / "src" / "dup.py").write_text("a = 1\na = 1\n", encoding="utf-8")
        result = call(
            gateway,
            "filesystem.patch",
            path="src/dup.py",
            old_text="a = 1",
            new_text="a = 2",
            replace_all=True,
        )
        assert result.ok and result.output["replacements"] == 2
        assert (workspace / "src" / "dup.py").read_text(encoding="utf-8") == "a = 2\na = 2\n"

    def test_patch_requires_write_scope(self, gateway: ToolGateway, workspace: Path) -> None:
        result = call(
            gateway, "filesystem.patch", path="docs/guide.md", old_text="# Guide", new_text="# X"
        )
        assert not result.ok and result.denied
        assert "# Guide" in (workspace / "docs" / "guide.md").read_text(encoding="utf-8")

    def test_protected_file_cannot_be_patched(self, gateway: ToolGateway) -> None:
        result = call(
            gateway, "filesystem.patch", path=".env", old_text="API_KEY", new_text="X"
        )
        assert not result.ok and result.denied

    def test_missing_file(self, gateway: ToolGateway) -> None:
        result = call(
            gateway, "filesystem.patch", path="src/absent.py", old_text="a", new_text="b"
        )
        assert not result.ok and "file not found" in result.error

    def test_empty_old_text_rejected(self, gateway: ToolGateway) -> None:
        result = call(gateway, "filesystem.patch", path="src/app.py", old_text="", new_text="x")
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"
