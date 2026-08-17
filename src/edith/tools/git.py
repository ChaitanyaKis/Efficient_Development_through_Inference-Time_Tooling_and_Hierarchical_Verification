"""Git tools: status, diff, log, branch, commit, worktree.

Git is core architecture, not a convenience: it is what makes agent work recoverable and
attributable (CLAUDE.md). These tools therefore expose a *curated* surface rather than a
generic ``git`` passthrough -- an agent cannot reach ``git reset --hard``, ``push --force``,
or ``clean -fdx`` because no tool constructs those arguments.

Two safety rules are applied everywhere:

1. **No agent string is ever passed where git expects a flag.** Refs, branch names, and
   paths are validated against a conservative pattern and rejected if they could be read as
   an option, and every path list is preceded by ``--``.
2. **Protected branches cannot be deleted or force-updated,** and new branches must carry a
   configured prefix, keeping agent work off shared history.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from edith.config.schema import GitPolicyConfig
from edith.errors import PermissionDeniedError, ToolExecutionError
from edith.schemas.common import EdithModel

from .base import Tool, ToolContext
from .process import ProcessResult, resolve_executable, run_process
from .schemas import AccessMode, ToolSpec

#: Conservative character set for refs, branches, and remotes.
_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@\-]*$")

#: Field separator for machine-readable `git log` output. Chosen because it cannot appear
#: in a commit subject or author name.
_LOG_SEPARATOR = "\x1f"
_LOG_FORMAT = _LOG_SEPARATOR.join(["%H", "%h", "%an", "%aI", "%s"])


def _validate_ref(value: str, *, kind: str = "ref") -> str:
    """Reject a ref/branch name that git could interpret as an option or that is malformed."""
    candidate = value.strip()
    if not candidate:
        raise ToolExecutionError(f"{kind} must not be empty")
    if candidate.startswith("-"):
        raise ToolExecutionError(
            f"{kind} {value!r} must not start with '-' (it would be read as an option)",
            details={kind: value},
        )
    if not _REF_PATTERN.match(candidate):
        raise ToolExecutionError(
            f"{kind} {value!r} contains characters that are not permitted",
            details={kind: value},
        )
    if ".." in candidate or candidate.endswith(".lock") or "//" in candidate:
        raise ToolExecutionError(f"{kind} {value!r} is not a valid name", details={kind: value})
    return candidate


class GitRunner:
    """Builds and executes git commands for a workspace."""

    def __init__(self, ctx: ToolContext, policy: GitPolicyConfig) -> None:
        self.ctx = ctx
        self.policy = policy
        # git is resolved through the same allowlist machinery as any other executable.
        self.executable = resolve_executable("git", ("git",))

    def run(
        self,
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        identity: bool = False,
    ) -> ProcessResult:
        """Execute a git command inside the workspace.

        Args:
            arguments: Git arguments, excluding the executable itself.
            cwd: Directory to run in; defaults to the workspace root.
            check: Raise :class:`ToolExecutionError` on a non-zero exit.
            identity: Prepend ``-c user.*`` so the command can create commits even in a
                fresh worktree with no configured global identity.
        """
        prefix: list[str] = []
        if identity:
            prefix = [
                "-c",
                f"user.name={self.policy.committer_name}",
                "-c",
                f"user.email={self.policy.committer_email}",
            ]
        result = run_process(
            [self.executable, *prefix, *arguments],
            cwd=cwd or self.ctx.workspace.root,
            timeout_seconds=self.ctx.timeout(self.policy.timeout_seconds),
            max_output_bytes=self.policy.max_output_bytes,
            # Git needs no inherited configuration beyond the base allowlist; credential
            # helpers and tokens are deliberately not passed through.
            env_passthrough=self.ctx.config.shell.env_passthrough,
        )
        if check and not result.ok:
            raise ToolExecutionError(
                f"git {arguments[0] if arguments else ''} failed with exit code "
                f"{result.exit_code}: {result.stderr.text.strip()[:400]}",
                details={"exit_code": result.exit_code, "command": arguments[:2]},
            )
        return result

    def assert_repository(self) -> None:
        """Verify the workspace is inside a git work tree."""
        result = self.run(["rev-parse", "--is-inside-work-tree"], check=False)
        if not result.ok or result.stdout.text.strip() != "true":
            raise ToolExecutionError(
                "workspace is not a git repository; run `git init` first",
                details={"workspace": str(self.ctx.workspace.root)},
            )

    def relative_paths(self, paths: list[str], mode: AccessMode) -> list[str]:
        """Authorize each path and return workspace-relative POSIX forms for git."""
        resolved: list[str] = []
        for raw in paths:
            target = (
                self.ctx.workspace.resolve_write(raw)
                if mode is AccessMode.WRITE
                else self.ctx.workspace.resolve_read(raw)
            )
            resolved.append(self.ctx.workspace.relative(target))
        return resolved


# --------------------------------------------------------------------------------------
# git.status
# --------------------------------------------------------------------------------------


class GitStatusInput(EdithModel):
    """Arguments for ``git.status``."""

    untracked: bool = True


class GitFileStatus(EdithModel):
    """One changed path reported by git."""

    path: str
    index_status: str
    worktree_status: str


class GitStatusOutput(EdithModel):
    """Result of ``git.status``."""

    branch: str
    clean: bool
    files: list[GitFileStatus] = Field(default_factory=list)


class GitStatusTool(Tool):
    """Report the working tree status."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.status",
        description="Report the current branch and any modified, staged, or untracked files.",
        access=frozenset({AccessMode.READ}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitStatusInput
    output_schema: ClassVar[type[BaseModel]] = GitStatusOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitStatusInput)  # noqa: S101 - guaranteed by validate_arguments
        runner = GitRunner(ctx, ctx.config.git)
        runner.assert_repository()

        arguments = ["status", "--porcelain=v1", "--branch"]
        arguments.append("--untracked-files=normal" if args.untracked else "--untracked-files=no")
        result = runner.run(arguments)

        branch = "unknown"
        files: list[GitFileStatus] = []
        for line in result.stdout.text.splitlines():
            if line.startswith("## "):
                branch = line[3:].split("...", 1)[0].strip()
                continue
            if len(line) < 4:
                continue
            files.append(
                GitFileStatus(
                    path=line[3:].strip().strip('"'),
                    index_status=line[0].strip(),
                    worktree_status=line[1].strip(),
                )
            )
        return GitStatusOutput(branch=branch, clean=not files, files=files)


# --------------------------------------------------------------------------------------
# git.diff
# --------------------------------------------------------------------------------------


class GitDiffInput(EdithModel):
    """Arguments for ``git.diff``."""

    staged: bool = False
    paths: list[str] = Field(default_factory=list)
    #: Compare against this ref instead of the working tree.
    ref: str | None = None
    stat_only: bool = False
    #: Return only the list of changed paths. Cheaper than parsing a full diff when the
    #: caller just needs to know *what* changed.
    name_only: bool = False
    context_lines: int = Field(default=3, ge=0, le=25)


class GitDiffOutput(EdithModel):
    """Result of ``git.diff``."""

    diff: str
    truncated: bool
    files_changed: int
    empty: bool
    #: Populated when ``name_only`` was requested.
    changed_paths: list[str] = Field(default_factory=list)


class GitDiffTool(Tool):
    """Show changes in the working tree or index."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.diff",
        description="Show the diff of the working tree, the index, or against a ref.",
        access=frozenset({AccessMode.READ}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitDiffInput
    output_schema: ClassVar[type[BaseModel]] = GitDiffOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitDiffInput)  # noqa: S101 - guaranteed by validate_arguments
        runner = GitRunner(ctx, ctx.config.git)
        runner.assert_repository()

        arguments = ["diff", f"--unified={args.context_lines}", "--no-color"]
        if args.staged:
            arguments.append("--cached")
        if args.stat_only:
            arguments.append("--stat")
        if args.name_only:
            arguments.append("--name-only")
        if args.ref:
            arguments.append(_validate_ref(args.ref))
        # `--` terminates options, so a path can never be reinterpreted as a flag.
        arguments.append("--")
        arguments.extend(runner.relative_paths(args.paths, AccessMode.READ))

        result = runner.run(arguments)
        diff_text = result.stdout.text

        if args.name_only:
            paths = [line.strip() for line in diff_text.splitlines() if line.strip()]
            return GitDiffOutput(
                diff=diff_text,
                truncated=result.stdout.truncated,
                files_changed=len(paths),
                empty=not paths,
                changed_paths=paths,
            )

        changed = sum(
            1 for line in diff_text.splitlines() if line.startswith("diff --git ")
        )
        return GitDiffOutput(
            diff=diff_text,
            truncated=result.stdout.truncated,
            files_changed=changed,
            empty=not diff_text.strip(),
        )


# --------------------------------------------------------------------------------------
# git.show
# --------------------------------------------------------------------------------------


class GitShowInput(EdithModel):
    """Arguments for ``git.show``."""

    path: str = Field(min_length=1)
    ref: str = Field(default="HEAD", min_length=1)


class GitShowOutput(EdithModel):
    """Result of ``git.show``."""

    path: str
    ref: str
    content: str
    exists: bool
    truncated: bool = False


class GitShowTool(Tool):
    """Read a file's content as it was at a given ref.

    The baseline half of any tamper check: comparing what a file *is* against what it *was*
    requires reading history without disturbing the working tree.
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.show",
        description="Read a file's content at a specific ref, without touching the worktree.",
        access=frozenset({AccessMode.READ}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitShowInput
    output_schema: ClassVar[type[BaseModel]] = GitShowOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitShowInput)  # noqa: S101 - guaranteed by validate_arguments
        runner = GitRunner(ctx, ctx.config.git)
        runner.assert_repository()

        ref = _validate_ref(args.ref)
        # Authorized like any other read, so history cannot be used to reach a path the
        # agent is not allowed to see in the working tree.
        target = ctx.workspace.resolve_read(args.path)
        relative = ctx.workspace.relative(target)

        result = runner.run(["show", f"{ref}:{relative}"], check=False)
        if not result.ok:
            return GitShowOutput(path=relative, ref=ref, content="", exists=False)
        return GitShowOutput(
            path=relative,
            ref=ref,
            content=result.stdout.text,
            exists=True,
            truncated=result.stdout.truncated,
        )


# --------------------------------------------------------------------------------------
# git.log
# --------------------------------------------------------------------------------------


class GitLogInput(EdithModel):
    """Arguments for ``git.log``."""

    max_entries: int = Field(default=20, ge=1)
    ref: str | None = None
    paths: list[str] = Field(default_factory=list)


class GitCommitInfo(EdithModel):
    """One commit."""

    sha: str
    short_sha: str
    author: str
    date: str
    subject: str


class GitLogOutput(EdithModel):
    """Result of ``git.log``."""

    commits: list[GitCommitInfo] = Field(default_factory=list)
    truncated: bool = False


class GitLogTool(Tool):
    """List recent commits."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.log",
        description="List recent commits, optionally limited to a ref or set of paths.",
        access=frozenset({AccessMode.READ}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitLogInput
    output_schema: ClassVar[type[BaseModel]] = GitLogOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitLogInput)  # noqa: S101 - guaranteed by validate_arguments
        policy = ctx.config.git
        runner = GitRunner(ctx, policy)
        runner.assert_repository()

        limit = min(args.max_entries, policy.max_log_entries)
        arguments = ["log", f"--max-count={limit}", f"--format={_LOG_FORMAT}", "--no-color"]
        if args.ref:
            arguments.append(_validate_ref(args.ref))
        arguments.append("--")
        arguments.extend(runner.relative_paths(args.paths, AccessMode.READ))

        result = runner.run(arguments, check=False)
        if not result.ok:
            # A repository with no commits is a normal state, not a failure.
            if "does not have any commits" in result.stderr.text:
                return GitLogOutput(commits=[], truncated=False)
            raise ToolExecutionError(
                f"git log failed: {result.stderr.text.strip()[:400]}",
                details={"exit_code": result.exit_code},
            )

        commits: list[GitCommitInfo] = []
        for line in result.stdout.text.splitlines():
            parts = line.split(_LOG_SEPARATOR)
            if len(parts) != 5:
                continue
            commits.append(
                GitCommitInfo(
                    sha=parts[0], short_sha=parts[1], author=parts[2],
                    date=parts[3], subject=parts[4],
                )
            )
        return GitLogOutput(commits=commits, truncated=len(commits) >= limit)


# --------------------------------------------------------------------------------------
# git.branch
# --------------------------------------------------------------------------------------


class GitBranchInput(EdithModel):
    """Arguments for ``git.branch``."""

    action: str = Field(default="list", pattern=r"^(list|create|checkout|delete)$")
    name: str | None = None
    #: For ``create``: the ref to branch from. Defaults to the current HEAD.
    start_point: str | None = None
    #: For ``create``: check the branch out after creating it.
    checkout: bool = True

    @field_validator("name", "start_point")
    @classmethod
    def _reject_option_lookalikes(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_ref(value, kind="branch")
        return value


class GitBranchOutput(EdithModel):
    """Result of ``git.branch``."""

    action: str
    current_branch: str
    branches: list[str] = Field(default_factory=list)
    created: str | None = None
    deleted: str | None = None


class GitBranchTool(Tool):
    """List, create, check out, or delete branches.

    New branches must carry a configured prefix (``agent/`` by default) and protected
    branches can never be deleted, so agent work stays off shared history.
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.branch",
        description="List, create, check out, or delete branches within policy limits.",
        access=frozenset({AccessMode.WRITE}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitBranchInput
    output_schema: ClassVar[type[BaseModel]] = GitBranchOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitBranchInput)  # noqa: S101 - guaranteed by validate_arguments
        policy = ctx.config.git
        runner = GitRunner(ctx, policy)
        runner.assert_repository()

        if args.action != "list" and not args.name:
            raise ToolExecutionError(f"action {args.action!r} requires a branch name")

        created: str | None = None
        deleted: str | None = None

        if args.action == "create":
            name = _validate_ref(str(args.name), kind="branch")
            self._assert_prefix(name, policy)
            start = (
                [_validate_ref(args.start_point, kind="start_point")]
                if args.start_point
                else []
            )
            if args.checkout:
                runner.run(["checkout", "-b", name, *start])
            else:
                runner.run(["branch", name, *start])
            created = name

        elif args.action == "checkout":
            name = _validate_ref(str(args.name), kind="branch")
            runner.run(["checkout", name])

        elif args.action == "delete":
            name = _validate_ref(str(args.name), kind="branch")
            self._assert_not_protected(name, policy)
            # Deliberately -d, never -D: an unmerged branch must not be silently discarded.
            runner.run(["branch", "-d", name])
            deleted = name

        listing = runner.run(["branch", "--list", "--format=%(refname:short)"])
        branches = [line.strip() for line in listing.stdout.text.splitlines() if line.strip()]
        current = runner.run(["rev-parse", "--abbrev-ref", "HEAD"], check=False)

        return GitBranchOutput(
            action=args.action,
            current_branch=current.stdout.text.strip() or "unknown",
            branches=branches,
            created=created,
            deleted=deleted,
        )

    @staticmethod
    def _assert_prefix(name: str, policy: GitPolicyConfig) -> None:
        if policy.branch_prefixes and not name.startswith(tuple(policy.branch_prefixes)):
            raise PermissionDeniedError(
                f"branch {name!r} must start with one of {list(policy.branch_prefixes)}",
                details={"branch": name},
            )

    @staticmethod
    def _assert_not_protected(name: str, policy: GitPolicyConfig) -> None:
        if name in policy.protected_branches:
            raise PermissionDeniedError(
                f"branch {name!r} is protected and cannot be deleted",
                details={"branch": name},
            )


# --------------------------------------------------------------------------------------
# git.commit
# --------------------------------------------------------------------------------------


class GitCommitInput(EdithModel):
    """Arguments for ``git.commit``."""

    message: str = Field(min_length=1, max_length=4000)
    #: Paths to stage before committing. Empty commits whatever is already staged.
    paths: list[str] = Field(default_factory=list)
    allow_empty: bool = False


class GitCommitOutput(EdithModel):
    """Result of ``git.commit``."""

    sha: str
    short_sha: str
    branch: str
    files_changed: int
    message: str


class GitCommitTool(Tool):
    """Stage paths and create a commit attributed to the calling agent.

    Every commit records the agent and task in trailers, satisfying the requirement that
    each modification be attributable.
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.commit",
        description="Stage paths and commit, recording agent and task attribution.",
        access=frozenset({AccessMode.WRITE}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitCommitInput
    output_schema: ClassVar[type[BaseModel]] = GitCommitOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitCommitInput)  # noqa: S101 - guaranteed by validate_arguments
        runner = GitRunner(ctx, ctx.config.git)
        runner.assert_repository()

        if args.paths:
            staged = runner.relative_paths(args.paths, AccessMode.WRITE)
            runner.run(["add", "--", *staged])

        if not args.allow_empty:
            pending = runner.run(["diff", "--cached", "--name-only"], check=False)
            if not pending.stdout.text.strip():
                raise ToolExecutionError(
                    "nothing staged to commit; stage paths or pass allow_empty=true"
                )

        message = self._with_attribution(args.message, ctx)
        arguments = ["commit", "-m", message]
        if args.allow_empty:
            arguments.append("--allow-empty")
        runner.run(arguments, identity=True)

        sha = runner.run(["rev-parse", "HEAD"]).stdout.text.strip()
        short = runner.run(["rev-parse", "--short", "HEAD"]).stdout.text.strip()
        branch = runner.run(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        changed = runner.run(
            ["show", "--stat", "--format=", "--name-only", "HEAD"], check=False
        )
        files = [line for line in changed.stdout.text.splitlines() if line.strip()]

        return GitCommitOutput(
            sha=sha,
            short_sha=short,
            branch=branch.stdout.text.strip() or "unknown",
            files_changed=len(files),
            message=message,
        )

    @staticmethod
    def _with_attribution(message: str, ctx: ToolContext) -> str:
        """Append agent/task trailers so every commit is attributable."""
        trailers = []
        if ctx.agent:
            trailers.append(f"Edith-Agent: {ctx.agent}")
        trailers.append(f"Edith-Call: {ctx.call_id}")
        return message.rstrip() + "\n\n" + "\n".join(trailers) + "\n"


# --------------------------------------------------------------------------------------
# git.worktree
# --------------------------------------------------------------------------------------


class GitWorktreeInput(EdithModel):
    """Arguments for ``git.worktree``."""

    action: str = Field(default="list", pattern=r"^(list|add|remove)$")
    #: Worktree identifier. Created under the configured worktree directory.
    name: str | None = None
    branch: str | None = None
    force: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$", value):
            raise ValueError(
                "worktree name may contain only letters, digits, '.', '_' and '-'"
            )
        return value


class GitWorktreeInfo(EdithModel):
    """One registered worktree."""

    path: str
    branch: str = ""
    sha: str = ""


class GitWorktreeOutput(EdithModel):
    """Result of ``git.worktree``."""

    action: str
    worktrees: list[GitWorktreeInfo] = Field(default_factory=list)
    created_path: str | None = None
    removed_path: str | None = None


class GitWorktreeTool(Tool):
    """Create, list, or remove isolated worktrees.

    Worktrees are how independent agent tasks avoid trampling each other: each gets its own
    checkout on its own branch. They live under a configured directory inside the workspace
    so the path policy still applies to anything written there.
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git.worktree",
        description="Create, list, or remove isolated git worktrees for parallel agent tasks.",
        access=frozenset({AccessMode.WRITE}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = GitWorktreeInput
    output_schema: ClassVar[type[BaseModel]] = GitWorktreeOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, GitWorktreeInput)  # noqa: S101 - guaranteed by validate_arguments
        policy = ctx.config.git
        runner = GitRunner(ctx, policy)
        runner.assert_repository()

        created: str | None = None
        removed: str | None = None

        if args.action in {"add", "remove"} and not args.name:
            raise ToolExecutionError(f"action {args.action!r} requires a worktree name")

        if args.action == "add":
            relative = f"{policy.worktree_dir}/{args.name}"
            target = ctx.workspace.resolve_write(relative)
            if target.exists():
                raise ToolExecutionError(
                    f"worktree path already exists: {relative}",
                    details={"path": relative},
                )
            branch = args.branch or f"agent/{args.name}"
            _validate_ref(branch, kind="branch")
            GitBranchTool._assert_prefix(branch, policy)
            target.parent.mkdir(parents=True, exist_ok=True)
            runner.run(["worktree", "add", "-b", branch, str(target)], identity=True)
            created = ctx.workspace.relative(target)

        elif args.action == "remove":
            relative = f"{policy.worktree_dir}/{args.name}"
            target = ctx.workspace.resolve_write(relative)
            arguments = ["worktree", "remove", str(target)]
            if args.force:
                arguments.insert(2, "--force")
            runner.run(arguments)
            removed = relative

        listing = runner.run(["worktree", "list", "--porcelain"])
        return GitWorktreeOutput(
            action=args.action,
            worktrees=self._parse_worktrees(listing.stdout.text, ctx),
            created_path=created,
            removed_path=removed,
        )

    @staticmethod
    def _parse_worktrees(raw: str, ctx: ToolContext) -> list[GitWorktreeInfo]:
        """Parse `git worktree list --porcelain` into structured entries."""
        entries: list[GitWorktreeInfo] = []
        current: dict[str, str] = {}
        for line in [*raw.splitlines(), ""]:
            if not line.strip():
                if current:
                    path = Path(current.get("worktree", ""))
                    try:
                        display = ctx.workspace.relative(path.resolve())
                    except ValueError:
                        display = str(path)
                    entries.append(
                        GitWorktreeInfo(
                            path=display,
                            branch=current.get("branch", "").replace("refs/heads/", ""),
                            sha=current.get("HEAD", ""),
                        )
                    )
                    current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        return entries
