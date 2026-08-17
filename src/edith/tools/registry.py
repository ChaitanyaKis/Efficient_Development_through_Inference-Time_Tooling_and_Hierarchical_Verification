"""Tool registry.

Mirrors the M0 agent registry so the two layers behave the same way: explicit registration,
duplicate detection, and no implicit discovery.
"""

from __future__ import annotations

from collections.abc import Iterator

from edith.errors import ToolNotFoundError, ToolRegistrationError
from edith.observability.logging import get_logger

from .base import Tool
from .schemas import ToolSpec

logger = get_logger(__name__)


class ToolRegistry:
    """A collection of tool instances addressable by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a tool instance under its declared spec name."""
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ToolRegistrationError(
                f"{type(tool).__name__} has no valid class-level `spec`",
                details={"class": type(tool).__name__},
            )
        if spec.name in self._tools and not replace:
            raise ToolRegistrationError(
                f"tool {spec.name!r} is already registered by "
                f"{type(self._tools[spec.name]).__name__}; pass replace=True to override",
                details={"tool": spec.name},
            )
        self._tools[spec.name] = tool
        logger.debug("tool.registered", tool=spec.name, cls=type(tool).__name__)

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"tool {name!r} is not registered; available: {list(self.names())}",
                details={"tool": name},
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered tool names, sorted."""
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return every registered tool's spec, sorted by name."""
        return tuple(self._tools[name].spec for name in self.names())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


def build_default_registry() -> ToolRegistry:
    """Return a registry populated with the tools shipped in M1."""
    from .filesystem import PatchTool, ReadTool, SearchTool, WriteTool  # noqa: PLC0415
    from .git import (  # noqa: PLC0415
        GitBranchTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitShowTool,
        GitStatusTool,
        GitWorktreeTool,
    )
    from .shell import ShellRunTool  # noqa: PLC0415

    registry = ToolRegistry()
    for tool in (
        ReadTool(),
        SearchTool(),
        WriteTool(),
        PatchTool(),
        ShellRunTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitShowTool(),
        GitLogTool(),
        GitBranchTool(),
        GitCommitTool(),
        GitWorktreeTool(),
    ):
        registry.register(tool)
    return registry
