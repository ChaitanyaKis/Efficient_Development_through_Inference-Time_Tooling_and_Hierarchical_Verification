"""The :class:`Workspace` capability handed to a running tool.

This is the load-bearing security decision of M1.

A tool receives no configuration, no workspace root, and no permission object -- only a
``Workspace`` already bound to the calling agent. The *only* way for a tool to obtain an
absolute path is :meth:`Workspace.resolve_read` or :meth:`Workspace.resolve_write`, each of
which normalizes, resolves, containment-checks, and authorizes before returning.

The alternative design -- having the gateway scan a tool's arguments for path-shaped fields
and authorize those -- was rejected. It fails silently for tools with a variable number of
paths, and it depends on every future tool author remembering to mark their fields. Here a
tool author who forgets to call ``resolve_*`` simply has no path to open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edith.errors import PermissionDeniedError, ToolExecutionError

from .paths import PathPolicy
from .permissions import PermissionEngine
from .schemas import AccessMode


@dataclass(frozen=True)
class Workspace:
    """A permission-bound view of the workspace tree."""

    policy: PathPolicy
    engine: PermissionEngine

    @property
    def root(self) -> Path:
        """The absolute workspace root."""
        return self.policy.root

    def _resolve(self, raw: str, mode: AccessMode) -> Path:
        """Normalize and resolve, then authorize. Order is deliberate and load-bearing."""
        resolved = self.policy.resolve(raw)
        self.engine.authorize_path(self.policy.relative_of(resolved), mode, raw)
        return resolved

    def resolve_read(self, raw: str) -> Path:
        """Return an absolute path the agent is permitted to read.

        Raises:
            PathPolicyError: The path is unsafe or escapes the workspace.
            PermissionDeniedError: The path is outside the agent's read scope.
        """
        return self._resolve(raw, AccessMode.READ)

    def resolve_write(self, raw: str) -> Path:
        """Return an absolute path the agent is permitted to write.

        Raises:
            PathPolicyError: The path is unsafe or escapes the workspace.
            PermissionDeniedError: The path is outside the agent's write scope.
        """
        return self._resolve(raw, AccessMode.WRITE)

    def resolve_existing_file(self, raw: str, mode: AccessMode = AccessMode.READ) -> Path:
        """Resolve a path that must already exist and be a regular file."""
        resolved = self._resolve(raw, mode)
        if not resolved.exists():
            raise ToolExecutionError(f"file not found: {raw}", details={"path": raw})
        if not resolved.is_file():
            raise ToolExecutionError(f"not a regular file: {raw}", details={"path": raw})
        return resolved

    def resolve_directory(self, raw: str, mode: AccessMode = AccessMode.READ) -> Path:
        """Resolve a path that must already exist and be a directory."""
        resolved = self._resolve(raw, mode)
        if not resolved.exists():
            raise ToolExecutionError(f"directory not found: {raw}", details={"path": raw})
        if not resolved.is_dir():
            raise ToolExecutionError(f"not a directory: {raw}", details={"path": raw})
        return resolved

    def resolve_traversal_root(self, raw: str) -> Path:
        """Resolve a directory to *walk*, without requiring read scope over it.

        Search needs a starting point, and an agent scoped to ``src/**`` would otherwise be
        unable to search at all because the default base ``.`` is outside its scope.

        This is safe because the base is used only to enumerate candidates: every file that
        reaches the caller is filtered through :meth:`is_visible`, which applies the full
        scope check. Directory names are never returned, so walking a directory the agent
        cannot read leaks nothing about its contents.

        The full path policy -- containment, traversal, protected locations -- still
        applies; only the per-agent scope check is skipped.
        """
        resolved = self.policy.resolve(raw)
        if not resolved.exists():
            raise ToolExecutionError(f"directory not found: {raw}", details={"path": raw})
        if not resolved.is_dir():
            raise ToolExecutionError(f"not a directory: {raw}", details={"path": raw})
        return resolved

    def relative(self, resolved: Path) -> str:
        """Return the workspace-relative POSIX form of a resolved path.

        Tool output reports relative paths so that results are portable and do not leak the
        host's directory layout into model context.
        """
        return self.policy.relative_of(resolved)

    def is_visible(self, candidate: Path, mode: AccessMode = AccessMode.READ) -> bool:
        """Whether a discovered path is contained, unprotected, and in the agent's scope.

        Used by directory and content search to filter candidates without raising: a file
        the agent may not see is simply absent from results rather than announced as
        forbidden, which would leak its existence.

        Containment is re-checked against the *resolved* location. Search walks the tree
        itself rather than resolving a caller-supplied path, so a reparse point encountered
        mid-walk would otherwise yield a candidate whose literal path looks contained while
        its real location is outside the workspace.
        """
        if not self.policy.contains(candidate):
            return False
        try:
            relative = self.policy.relative_of(candidate.resolve())
        except ValueError:
            return False
        if self.policy.is_protected(relative):
            return False
        try:
            self.engine.authorize_path(relative, mode, relative)
        except PermissionDeniedError:
            return False
        return True
