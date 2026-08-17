"""Regression: directory search must not escape the workspace through a reparse point.

``filesystem.search`` walks the tree itself rather than resolving a single caller-supplied
path, so it needs its own containment check. On Windows a junction is not reported by
``is_symlink()``, so a symlink-only guard would let the walk descend outside the workspace
and read files the agent must never see.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from edith.tools.schemas import ToolCall

from .tool_fixtures import build_gateway, build_workspace

#: Canary content placed outside the workspace; it must never appear in a search result.
CANARY = "TOP_SECRET_CANARY_VALUE"


def _make_junction(link: Path, target: Path) -> bool:
    if sys.platform != "win32":
        return False
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


@pytest.fixture
def escape_setup(tmp_path: Path) -> tuple[Path, Path]:
    """A workspace plus an outside directory holding a canary file."""
    workspace = build_workspace(tmp_path / "ws")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.txt").write_text(f"{CANARY}\n", encoding="utf-8")
    return workspace, outside


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_search_does_not_follow_a_junction_out_of_the_workspace(
    escape_setup: tuple[Path, Path],
) -> None:
    workspace, outside = escape_setup
    if not _make_junction(workspace / "linked", outside):
        pytest.skip("could not create a junction in this environment")

    gateway = build_gateway(workspace)
    result = gateway.execute(
        ToolCall(tool="filesystem.search", arguments={"content_pattern": "TOP_SECRET"})
    )

    assert result.ok
    assert result.output["matches"] == [], "search escaped the workspace via a junction"
    assert CANARY not in str(result.output)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_name_search_does_not_list_files_outside_the_workspace(
    escape_setup: tuple[Path, Path],
) -> None:
    workspace, outside = escape_setup
    if not _make_junction(workspace / "linked", outside):
        pytest.skip("could not create a junction in this environment")

    gateway = build_gateway(workspace)
    result = gateway.execute(
        ToolCall(tool="filesystem.search", arguments={"name_pattern": "*.txt"})
    )

    assert result.ok
    assert not any("secrets.txt" in match["path"] for match in result.output["matches"])


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks on Windows needs elevation"
)
def test_search_does_not_follow_a_symlink_out_of_the_workspace(
    escape_setup: tuple[Path, Path],
) -> None:
    workspace, outside = escape_setup
    (workspace / "linked").symlink_to(outside, target_is_directory=True)

    gateway = build_gateway(workspace)
    result = gateway.execute(
        ToolCall(tool="filesystem.search", arguments={"content_pattern": "TOP_SECRET"})
    )
    assert result.ok
    assert result.output["matches"] == []
