"""Shared helpers for tool-layer tests.

Kept out of ``conftest.py`` so the workspace builders can be imported explicitly by the
modules that use them.
"""

from __future__ import annotations

from pathlib import Path

from edith.config.schema import EdithConfig, ModelParams, ModelsConfig, ToolsConfig
from edith.schemas.agent import AgentPermissions
from edith.tools.gateway import ToolGateway
from edith.tools.paths import PathPolicy
from edith.tools.registry import build_default_registry

#: A typical implementation agent: reads broadly, writes only its own area.
SCOPED_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset({"filesystem.*", "shell.run", "git.*"}),
    allowed_read_paths=("**",),
    allowed_write_paths=("src/**", "tests/**", ".edith/**"),
)

#: Read-only reviewer: can inspect everything, change nothing.
READONLY_PERMISSIONS = AgentPermissions(
    allowed_tools=frozenset({"filesystem.read", "filesystem.search", "git.status"}),
    allowed_read_paths=("**",),
)


def build_workspace(root: Path) -> Path:
    """Create a small, realistic project tree under ``root``."""
    (root / "src" / "backend").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "src" / "app.py").write_text(
        "def main():\n    return 'hello'\n\n\ndef helper():\n    return 42\n",
        encoding="utf-8",
    )
    (root / "src" / "backend" / "api.py").write_text(
        "ROUTES = ['/health', '/status']\n\n\ndef handler():\n    return ROUTES\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_app.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )
    (root / "docs" / "guide.md").write_text("# Guide\n\nSome documentation.\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")
    (root / ".env").write_text("API_KEY=super-secret-value\n", encoding="utf-8")
    return root


def build_config(workspace: Path, **tool_overrides: object) -> EdithConfig:
    """Build an :class:`EdithConfig` whose tool layer points at ``workspace``."""
    return EdithConfig(
        models=ModelsConfig(profiles={"default": ModelParams(model_name="test-model:q4")}),
        tools=ToolsConfig(workspace_root=workspace, **tool_overrides),  # type: ignore[arg-type]
    )


def build_gateway(
    workspace: Path,
    permissions: AgentPermissions | None = None,
    *,
    config: EdithConfig | None = None,
    agent: str = "test_agent",
) -> ToolGateway:
    """Build a gateway rooted at ``workspace`` with the given permissions."""
    resolved_config = config or build_config(workspace)
    return ToolGateway(
        resolved_config,
        permissions or SCOPED_PERMISSIONS,
        registry=build_default_registry(),
        agent=agent,
        policy=PathPolicy.create(workspace, resolved_config.tools.paths),
    )
