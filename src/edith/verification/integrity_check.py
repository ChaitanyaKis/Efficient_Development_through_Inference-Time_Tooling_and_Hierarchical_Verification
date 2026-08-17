"""Gathering baseline evidence for the integrity check.

Everything here goes through the M1 tool gateway -- ``git.diff --name-only`` to learn what
changed and ``git.show`` to read what those files used to be. Nothing touches the working
tree, so establishing the baseline can never disturb the work being judged.
"""

from __future__ import annotations

from edith.integrity import FileKind, IntegrityReport, build_report, classify_path
from edith.observability.logging import get_logger
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

logger = get_logger(__name__)

#: Files larger than this are not compared; a baseline read that big is not a unit test.
MAX_BASELINE_CHARS = 200_000


class IntegrityChecker:
    """Compares the current workspace against a git baseline."""

    def __init__(self, gateway: ToolGateway, baseline_ref: str = "HEAD") -> None:
        """
        Args:
            gateway: Permission-scoped gateway. Needs ``git.diff``, ``git.show``, and
                ``filesystem.read``.
            baseline_ref: The ref representing "before the agent started".
        """
        self.gateway = gateway
        self.baseline_ref = baseline_ref

    def available(self) -> bool:
        """Whether a baseline comparison is possible at all."""
        return self.gateway.can_use("git.diff") and self.gateway.can_use("git.show")

    def changed_paths(self) -> list[str]:
        """Paths that differ from the baseline, including untracked files."""
        paths: list[str] = []

        tracked = self.gateway.execute(
            ToolCall(
                tool="git.diff",
                arguments={"name_only": True, "ref": self.baseline_ref},
            )
        )
        if tracked.ok:
            paths.extend(str(item) for item in tracked.output.get("changed_paths", []))
        else:
            logger.warning("integrity.diff_failed", error=tracked.error)

        # Untracked files are changes too: a new test file is legitimate, but a *replaced*
        # one that was deleted and rewritten would otherwise be invisible.
        status = self.gateway.execute(ToolCall(tool="git.status"))
        if status.ok:
            for entry in status.output.get("files", []):
                path = str(entry.get("path", ""))
                if path and path not in paths:
                    paths.append(path)

        return sorted(set(paths))

    def _read_baseline(self, path: str) -> str | None:
        """Content of ``path`` at the baseline ref, or ``None`` when it did not exist."""
        result = self.gateway.execute(
            ToolCall(
                tool="git.show", arguments={"path": path, "ref": self.baseline_ref}
            )
        )
        if not result.ok or not result.output.get("exists"):
            return None
        content = str(result.output.get("content", ""))
        return content if len(content) <= MAX_BASELINE_CHARS else None

    def _read_current(self, path: str) -> str | None:
        """Current content of ``path``, or ``None`` when it no longer exists."""
        result = self.gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": path})
        )
        if not result.ok:
            return None
        return str(result.output.get("content", ""))

    def check(self, *, justification: str = "") -> IntegrityReport:
        """Build an integrity report for the current workspace state.

        Never raises: an unavailable baseline is reported as such rather than being treated
        as "nothing changed", because silently passing is the failure mode this whole module
        exists to prevent.
        """
        if not self.available():
            return build_report({}, {}, [], baseline_unavailable=True)

        paths = self.changed_paths()
        if not paths:
            return build_report({}, {}, [], justification=justification)

        test_paths = [path for path in paths if classify_path(path) is FileKind.TEST]
        baselines: dict[str, str] = {}
        current: dict[str, str] = {}
        for path in test_paths:
            baseline = self._read_baseline(path)
            if baseline is not None:
                baselines[path] = baseline
            content = self._read_current(path)
            if content is not None:
                current[path] = content

        return build_report(
            baselines, current, paths, justification=justification
        )
