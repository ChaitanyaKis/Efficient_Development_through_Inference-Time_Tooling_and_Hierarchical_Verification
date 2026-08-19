"""Project workspaces: where autonomous work is allowed to happen.

M1 shipped ``workspace_root: .``, which is correct for an operator running a tool by hand
and wrong for an autonomous loop -- it points at Edith's own source. A coding agent given a
plausible-but-wrong plan would edit the kernel currently executing it.

M2 therefore separates the two::

    C:\\Projects\\Project_Edith\\      the kernel (never a workspace)
    C:\\Projects\\Edith_Workspaces\\   project-a, project-b, ...

The root is configurable and nothing here hard-codes a path. The one rule that is *not*
configurable is the last one: a workspace may never be the Edith repository itself, or
contain it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from edith.config.schema import EdithConfig, ToolsConfig
from edith.errors import ConfigurationError
from edith.observability.logging import get_logger

logger = get_logger(__name__)


def edith_repository_root() -> Path:
    """Return the root of the Edith installation itself.

    Derived from this module's location: ``src/edith/workspaces.py`` -> two parents up from
    ``edith`` is the repository root.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectWorkspace:
    """A resolved, validated place for an execution to work.

    Everything the orchestrator needs to rebind tools after a restart: which project, where
    it lives, and which branch the work is on.
    """

    project_id: str
    name: str
    root: Path
    branch: str | None = None

    def tools_config(self, base: ToolsConfig) -> ToolsConfig:
        """Return a :class:`ToolsConfig` rooted at this workspace.

        Only ``workspace_root`` changes; every M1 policy -- protected paths, shell
        allowlist, git rules -- is inherited unchanged. A workspace narrows *where* tools
        operate, never *what they may do*.
        """
        return base.model_copy(update={"workspace_root": self.root})

    def config_for(self, config: EdithConfig) -> EdithConfig:
        """Return a config whose tool layer is rooted at this workspace."""
        return config.model_copy(update={"tools": self.tools_config(config.tools)})


class WorkspaceManager:
    """Creates and resolves project workspaces under a configured root."""

    def __init__(self, config: EdithConfig) -> None:
        self.config = config
        self._root = self._resolve_root(config)

    @staticmethod
    def _resolve_root(config: EdithConfig) -> Path:
        """Resolve the workspaces root, relative to the config directory when relative.

        Anchoring a relative root to the config directory rather than the process working
        directory means ``edith run`` behaves the same regardless of where it is invoked.
        """
        configured = config.orchestration.workspaces_root
        if configured.is_absolute():
            return configured.resolve()
        anchor = config.config_dir.parent if config.config_dir else Path.cwd()
        return (anchor / configured).resolve()

    @property
    def root(self) -> Path:
        """The absolute workspaces root."""
        return self._root

    def _assert_not_the_kernel(self, candidate: Path) -> None:
        """Refuse a workspace that is, contains, or lives inside the Edith repository.

        This is the guard that keeps an autonomous run from editing the code executing it.
        It is deliberately not configurable.
        """
        kernel = edith_repository_root()
        candidate_key = os.path.normcase(str(candidate))
        kernel_key = os.path.normcase(str(kernel))

        if candidate_key == kernel_key:
            raise ConfigurationError(
                "a project workspace may not be the Edith repository itself",
                details={"workspace": str(candidate)},
            )
        if candidate_key.startswith(kernel_key.rstrip(os.sep) + os.sep):
            raise ConfigurationError(
                "a project workspace may not live inside the Edith repository",
                details={"workspace": str(candidate), "kernel": str(kernel)},
            )
        if kernel_key.startswith(candidate_key.rstrip(os.sep) + os.sep):
            raise ConfigurationError(
                "a project workspace may not contain the Edith repository",
                details={"workspace": str(candidate), "kernel": str(kernel)},
            )

    @staticmethod
    def _validate_name(name: str) -> str:
        """Reject a project name that could escape the workspaces root."""
        cleaned = name.strip()
        if not cleaned:
            raise ConfigurationError("project name must not be empty")
        if any(char in cleaned for char in '/\\:*?"<>|') or cleaned in {".", ".."}:
            raise ConfigurationError(
                f"project name {name!r} contains characters that are not permitted",
                details={"name": name},
            )
        return cleaned

    def path_for(self, name: str) -> Path:
        """Return the workspace path for a project name, without creating it."""
        return self._root / self._validate_name(name)

    def create(self, name: str, project_id: str, *, exist_ok: bool = True) -> ProjectWorkspace:
        """Create (or adopt) a workspace directory for a project."""
        target = self.path_for(name)
        self._assert_not_the_kernel(target)

        if target.exists() and not exist_ok:
            raise ConfigurationError(
                f"workspace already exists: {target}", details={"workspace": str(target)}
            )
        target.mkdir(parents=True, exist_ok=True)
        _scaffold_project(target)
        _initialise_repository(target)
        logger.info("workspace.ready", project=name, path=str(target))
        return ProjectWorkspace(project_id=project_id, name=name, root=target)

    def adopt(self, path: Path, project_id: str, name: str | None = None) -> ProjectWorkspace:
        """Adopt an existing directory as a workspace.

        Used when a caller points Edith at a repository that already exists rather than
        letting it create one under the managed root.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ConfigurationError(
                f"workspace path is not an existing directory: {resolved}",
                details={"workspace": str(resolved)},
            )
        self._assert_not_the_kernel(resolved)
        return ProjectWorkspace(
            project_id=project_id, name=name or resolved.name, root=resolved
        )

    def list_workspaces(self) -> tuple[str, ...]:
        """Names of workspaces present under the managed root."""
        if not self._root.is_dir():
            return ()
        return tuple(sorted(entry.name for entry in self._root.iterdir() if entry.is_dir()))


def _initialise_repository(target: Path) -> None:
    """Make a newly created workspace a git repository.

    Git is not optional decoration here. CLAUDE.md requires every implementation task to be
    recoverable, and three mechanisms depend on a real repository: ``git.diff`` reports what a
    task changed, M5.1 worktrees give each task an isolated tree, and M5.2's merge copies only
    verified files out of one. A workspace without git silently loses all three -- the tools
    fail, isolation cannot start, and the failure surfaces far from its cause.

    Every benchmark harness ran ``git init`` itself, which is exactly why this gap survived to
    the first run started from the UI.

    Only for workspaces Edith creates. :meth:`WorkspaceManager.adopt` deliberately does not do
    this: a directory the user pointed Edith at is theirs, and initialising a repository inside
    it is not Edith's decision to make.

    A failure here is logged and tolerated rather than raised. Git may be absent on the
    machine, and the tools that need it already report their own refusals clearly; refusing to
    create the workspace at all would be a worse trade.
    """
    if (target / ".git").exists():
        return
    try:
        from edith.tools.process import resolve_executable, run_process  # noqa: PLC0415

        executable = resolve_executable("git", ("git",))
        for argv in (
            [executable, "init", "-q"],
            # A local identity, because the process runs with a sanitised environment and so
            # cannot see the user's global git config. Set on this repository only: Edith has
            # no business writing to anyone's global configuration, and a commit attributed to
            # Edith is more honest than one wearing the user's name.
            [executable, "config", "user.email", "edith@localhost"],
            [executable, "config", "user.name", "Edith"],
            # A first commit gives worktrees a base revision to branch from. Without one,
            # `git worktree add` has no HEAD and isolation cannot begin.
            [executable, "commit", "--allow-empty", "-qm", "edith: workspace created"],
        ):
            result = run_process(
                argv,
                cwd=target,
                timeout_seconds=30.0,
                max_output_bytes=64_000,
                env_passthrough=(),
            )
            if result.exit_code != 0:
                logger.warning(
                    "workspace.git_init_failed",
                    path=str(target),
                    command=" ".join(argv[1:]),
                    error=str(result.stderr)[:200],
                )
                return
    except Exception as exc:  # noqa: BLE001 - a workspace without git is degraded, not broken
        logger.warning("workspace.git_unavailable", path=str(target), error=str(exc)[:200])


#: Written at the workspace root so generated tests can import generated code.
#:
#: pytest puts the directory containing the root ``conftest.py`` on ``sys.path``, which makes
#: ``from src.backend.thing import x`` resolve. Without it a generated test cannot import the
#: module it was written to cover, verification fails on a collection error, and the failure
#: looks like a missing dependency rather than a missing path entry.
_CONFTEST = "\n".join(
    (
        '"""Make this project importable from its own tests.',
        "",
        "Written by Edith when the workspace was created. pytest adds the directory holding",
        "the root conftest to sys.path, so `from src.package.module import thing` resolves",
        "from tests/.",
        '"""',
        "",
    )
)


def _scaffold_project(target: Path) -> None:
    """Give a new workspace the minimum structure a Python project needs to be verifiable.

    Two directories and one conftest -- deliberately close to nothing. Edith does not know
    what the project will become, so it does not impose a layout; it only ensures that code
    written under ``src/`` can be imported by tests written under ``tests/``, which is the
    precondition for verification meaning anything at all.

    Every benchmark harness created this structure itself, which is why its absence only
    surfaced on the first project created through the real workspace path: generated tests
    could not import the generated module, and the collection error was then misread as a
    missing third-party dependency.
    """
    for relative in ("src", "tests"):
        (target / relative).mkdir(parents=True, exist_ok=True)
    conftest = target / "conftest.py"
    if not conftest.exists():
        conftest.write_text(_CONFTEST, encoding="utf-8")
