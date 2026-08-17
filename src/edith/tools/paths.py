"""Path normalization and safety policy.

This module answers one question: *given a path string an LLM produced, what absolute path
does it denote, and is touching it allowed?*

Order matters and is not negotiable: **normalize and resolve first, authorize second.**
Authorizing a raw string lets ``src/../../../etc/passwd`` pass a prefix check and then
resolve somewhere else entirely. Every check below therefore runs against the fully
resolved path.

Windows-specific surface that a POSIX-shaped implementation would miss, all rejected here:
UNC shares, the ``\\\\?\\`` / ``\\\\.\\`` device namespaces, drive-relative paths
(``C:foo``), reserved device names (``CON``, ``NUL``, ``COM1`` ...), and NTFS alternate data
streams (``file.txt:hidden``).
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from edith.config.schema import PathPolicyConfig
from edith.errors import PathPolicyError

#: Windows reserved device names. Opening one can block on hardware rather than fail.
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(10)}
    | {f"lpt{i}" for i in range(10)}
)

#: Prefixes that reach outside the local filesystem namespace.
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./")


def normalize_relative(relative_posix: str) -> str:
    """Lower-case a workspace-relative path and strip a leading ``./``.

    Deliberately not ``lstrip("./")``: that strips a *character set*, so ``.env`` would
    become ``env`` and silently escape every dotfile rule. Matching is lower-cased because
    the target filesystem is case-insensitive and ``.ENV`` must not bypass ``.env``.
    """
    candidate = relative_posix.replace("\\", "/").lower()
    while candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate.lstrip("/")


def _deny(reason: str, raw: str, **details: object) -> PathPolicyError:
    """Build a policy error that names the offending input without leaking resolved paths.

    The *raw* argument is echoed because the agent supplied it; the resolved location is
    deliberately not included, so a probing agent cannot use error messages to map the host
    filesystem.
    """
    return PathPolicyError(reason, details={"path": raw, **details})


@dataclass(frozen=True)
class PathPolicy:
    """Resolves agent-supplied paths against a workspace root and enforces safety rules.

    This class knows nothing about *which agent* is asking -- it enforces the rules that
    apply to everyone. Per-agent scoping is layered on top by
    :class:`~edith.tools.permissions.PermissionEngine`.
    """

    root: Path
    config: PathPolicyConfig

    @classmethod
    def create(cls, root: Path, config: PathPolicyConfig) -> PathPolicy:
        """Build a policy with a fully resolved root.

        Raises:
            PathPolicyError: The workspace root does not exist or is not a directory.
        """
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise PathPolicyError(
                f"workspace root is not an existing directory: {resolved}",
                details={"workspace_root": str(resolved)},
            )
        return cls(root=resolved, config=config)

    # -- Syntactic rejection -------------------------------------------------------

    def _reject_unsafe_syntax(self, raw: str) -> None:
        """Reject path strings that are dangerous before they are even resolved."""
        if not raw or not raw.strip():
            raise _deny("path must not be empty", raw)
        if "\x00" in raw:
            raise _deny("path must not contain a NUL byte", raw)

        normalized = raw.replace("\\", "/")

        if raw.startswith(_DEVICE_PREFIXES) or normalized.startswith("//"):
            raise _deny(
                "path must not use a UNC share or the Windows device namespace", raw
            )
        if normalized.startswith("/"):
            raise _deny("path must be relative to the workspace, not absolute", raw)
        if len(raw) >= 2 and raw[1] == ":":
            # Covers both "C:\\abs" and the drive-relative "C:rel" form.
            raise _deny("path must be relative to the workspace, not drive-qualified", raw)
        if raw.startswith("~"):
            raise _deny("path must not reference a home directory", raw)

        for segment in normalized.split("/"):
            if not segment or segment == ".":
                continue
            if segment == "..":
                raise _deny("path must not contain a '..' traversal segment", raw)
            if ":" in segment:
                # NTFS alternate data stream, e.g. "notes.txt:hidden".
                raise _deny("path must not contain ':' (alternate data stream)", raw)
            if segment.split(".")[0].lower() in _RESERVED_STEMS:
                raise _deny(
                    f"path segment {segment!r} is a reserved Windows device name", raw
                )

    # -- Containment ---------------------------------------------------------------

    def contains(self, path: Path) -> bool:
        """Whether an on-disk path *resolves* to a location inside the workspace.

        The path is resolved first. Comparing the literal string would be defeated by any
        reparse point: on Windows a junction is not reported by ``is_symlink()``, so
        ``<workspace>/link/secret.txt`` looks contained while actually living elsewhere.

        Comparison goes through ``os.path.normcase`` so that a case-differing path on
        Windows (``C:\\Projects`` vs ``c:\\projects``) is recognised as the same location.
        """
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - unreadable or malformed path
            return False
        root_key = os.path.normcase(str(self.root))
        target_key = os.path.normcase(str(resolved))
        return target_key == root_key or target_key.startswith(
            root_key.rstrip(os.sep) + os.sep
        )

    def _assert_contained(self, resolved: Path, raw: str) -> None:
        """Verify ``resolved`` lies inside the workspace root."""
        if not self.contains(resolved):
            raise _deny("path escapes the workspace root", raw)

    def _assert_no_symlink(self, resolved: Path, raw: str) -> None:
        """Reject a path whose existing prefix traverses a symlink or junction.

        ``Path.resolve`` already follows links, so containment alone would accept a symlink
        that points back inside the workspace. That is still refused by default: a link
        inside an agent-managed tree is far more likely to be an escape attempt than a
        legitimate need.
        """
        if self.config.allow_symlinks:
            return
        current = self.root
        try:
            relative = resolved.relative_to(self.root)
        except ValueError:  # pragma: no cover - containment is checked first
            raise _deny("path escapes the workspace root", raw) from None
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise _deny(
                    f"path traverses a symlink at {part!r}; symlinks are disabled", raw
                )
            if not current.exists():
                # Nothing further can exist below a missing component.
                break

    # -- Protected files -----------------------------------------------------------

    def is_protected(self, relative_posix: str) -> bool:
        """Whether a workspace-relative path matches the protected deny list.

        Matching is case-insensitive because the target filesystem is case-insensitive;
        treating ``.ENV`` as distinct from ``.env`` would be a trivial bypass.

        This is a **deny** list, so it deliberately errs toward matching more: ``*.pem``
        also catches ``certs/server.pem``. Do not merge it with the allow-list matcher in
        :mod:`edith.tools.permissions`, which must err the opposite way.
        """
        candidate = normalize_relative(relative_posix)
        for pattern in self.config.protected_patterns:
            lowered = normalize_relative(pattern)
            if lowered.endswith("/**"):
                base = lowered[:-3]
                if candidate == base or candidate.startswith(base + "/"):
                    return True
                # "secrets/**" should also catch "nested/secrets/key".
                if f"/{base}/" in f"/{candidate}":
                    return True
                continue
            if fnmatch.fnmatchcase(candidate, lowered):
                return True
            # A bare pattern such as ".git" must also protect everything beneath it.
            if candidate.startswith(lowered + "/"):
                return True
        return False

    def _assert_not_protected(self, relative_posix: str, raw: str) -> None:
        if self.is_protected(relative_posix):
            raise _deny(
                "path targets a protected location (VCS metadata, secrets, or keys)",
                raw,
                relative_path=relative_posix,
            )

    # -- Public API ----------------------------------------------------------------

    def relative_of(self, resolved: Path) -> str:
        """Return the workspace-relative POSIX form of an already-resolved path."""
        return PurePosixPath(resolved.relative_to(self.root).as_posix()).as_posix()

    def resolve(self, raw: str) -> Path:
        """Normalize, resolve, and safety-check an agent-supplied relative path.

        The returned path is absolute, symlink-resolved, inside the workspace, and not
        protected. It may or may not exist -- existence is the caller's concern.

        Raises:
            PathPolicyError: The path is syntactically unsafe, escapes the workspace,
                traverses a symlink, or targets a protected location.
        """
        self._reject_unsafe_syntax(raw)

        candidate = (self.root / raw.replace("\\", "/")).resolve()

        self._assert_contained(candidate, raw)
        self._assert_no_symlink(candidate, raw)
        self._assert_not_protected(self.relative_of(candidate), raw)
        return candidate

    def check_readable_size(self, resolved: Path, raw: str) -> int:
        """Verify a file is within the size limit and return its size in bytes."""
        size = resolved.stat().st_size
        if size > self.config.max_file_bytes:
            raise _deny(
                f"file is {size} bytes, exceeding the {self.config.max_file_bytes} byte limit",
                raw,
            )
        return size
