"""Per-agent authorization.

:class:`~edith.tools.paths.PathPolicy` enforces the rules that apply to everyone;
this module enforces the scope granted to *one* agent via the
:class:`~edith.schemas.agent.AgentPermissions` that M0 already declares and validates.

There is deliberately no second permission model. An agent's identity is the single source
of truth for what it may do, so a permission cannot be granted anywhere except on the
identity, where it is visible to ``edith agents``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from edith.errors import PermissionDeniedError
from edith.schemas.agent import AgentPermissions

from .paths import normalize_relative
from .schemas import AccessMode, ToolSpec

#: Grants every tool and the whole workspace. Used by the CLI when a human operator drives
#: a tool directly -- the human already has shell access, so restricting them is theatre.
#: Never hand this to an agent.
UNRESTRICTED = AgentPermissions(
    allowed_tools=frozenset({"*"}),
    allowed_read_paths=("**",),
    allowed_write_paths=("**",),
    network_access=True,
)


def _matches_any(relative_posix: str, patterns: tuple[str, ...]) -> bool:
    """Whether a workspace-relative path matches any granted glob pattern.

    ``**`` grants the whole tree. A pattern naming a directory (``src/backend``) grants
    everything beneath it, which is what an author writing that pattern means.

    Deliberately *not* shared with :meth:`~edith.tools.paths.PathPolicy.is_protected`.
    That one is a deny list and errs toward matching more; this one is an allow list and
    must err toward matching less. Merging them would make one of the two wrong, and the
    wrong one would be a privilege escalation.
    """
    candidate = normalize_relative(relative_posix)
    for pattern in patterns:
        lowered = normalize_relative(pattern)
        if lowered in {"**", "**/*"}:
            return True
        if lowered.endswith("/**"):
            base = lowered[:-3]
            if candidate == base or candidate.startswith(base + "/"):
                return True
            continue
        if fnmatch.fnmatchcase(candidate, lowered):
            return True
        if candidate.startswith(lowered.rstrip("/") + "/"):
            return True
    return False


@dataclass(frozen=True)
class PermissionEngine:
    """Decides whether a given agent may run a tool or touch a path."""

    permissions: AgentPermissions

    def may_use_tool(self, name: str) -> bool:
        """Whether the agent is granted the named tool."""
        allowed = self.permissions.allowed_tools
        if "*" in allowed:
            return True
        if name in allowed:
            return True
        # A namespace grant such as "filesystem.*" covers every tool in that namespace.
        namespace = name.split(".", 1)[0]
        return f"{namespace}.*" in allowed

    def authorize_tool(self, spec: ToolSpec, agent: str | None = None) -> None:
        """Raise unless the agent may invoke this tool.

        Raises:
            PermissionDeniedError: The tool is not granted, or it needs network access the
                agent does not have.
        """
        if not self.may_use_tool(spec.name):
            raise PermissionDeniedError(
                f"agent is not permitted to use tool {spec.name!r}",
                details={
                    "tool": spec.name,
                    "agent": agent,
                    "granted_tools": sorted(self.permissions.allowed_tools),
                },
            )
        if spec.uses_network and not self.permissions.network_access:
            raise PermissionDeniedError(
                f"tool {spec.name!r} requires network access, which this agent lacks",
                details={"tool": spec.name, "agent": agent},
            )

    def authorize_path(self, relative_posix: str, mode: AccessMode, raw: str) -> None:
        """Raise unless the agent may access a workspace-relative path in ``mode``.

        Write access does not imply read access and vice versa; each is granted explicitly.

        Raises:
            PermissionDeniedError: The path lies outside the agent's granted scope.
        """
        patterns = (
            self.permissions.allowed_write_paths
            if mode is AccessMode.WRITE
            else self.permissions.allowed_read_paths
        )
        if not patterns:
            raise PermissionDeniedError(
                f"agent has no {mode.value.lower()} scope; it cannot "
                f"{mode.value.lower()} {raw!r}",
                details={"path": raw, "mode": str(mode)},
            )
        if not _matches_any(relative_posix, patterns):
            raise PermissionDeniedError(
                f"path {raw!r} is outside the agent's {mode.value.lower()} scope",
                details={"path": raw, "mode": str(mode), "granted": list(patterns)},
            )
