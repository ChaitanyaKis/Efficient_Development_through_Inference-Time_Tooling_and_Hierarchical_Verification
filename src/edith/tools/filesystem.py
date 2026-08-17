"""Filesystem tools: read, search, write, patch.

Every path these tools touch comes from ``ctx.workspace.resolve_read``/``resolve_write``.
None of them constructs a path any other way, which is what makes the permission boundary
enforceable rather than aspirational.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from edith.errors import ToolExecutionError
from edith.schemas.common import EdithModel

from .base import Tool, ToolContext
from .schemas import AccessMode, ToolSpec

#: Directories never worth walking during a search; skipping them keeps results relevant
#: and search cheap on a constrained machine.
_SKIP_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox",
        ".idea", ".vscode", "target", ".next", ".cache", ".edith",
    }
)

#: Heuristic binary sniff: a NUL byte in the first block means "not text".
_BINARY_SNIFF_BYTES = 8192


def _is_probably_binary(path: Path) -> bool:
    """Whether a file looks binary, so search can skip it without decoding cost."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True


def _read_text(path: Path, raw: str) -> str:
    """Read a file as UTF-8, reporting an undecodable file as a tool error."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(
            f"file is not valid UTF-8 text: {raw}", details={"path": raw}
        ) from exc
    except OSError as exc:
        raise ToolExecutionError(
            f"could not read {raw}: {exc.strerror or exc}", details={"path": raw}
        ) from exc


# --------------------------------------------------------------------------------------
# filesystem.read
# --------------------------------------------------------------------------------------


class ReadInput(EdithModel):
    """Arguments for ``filesystem.read``."""

    path: str = Field(min_length=1)
    #: 1-indexed first line to return. ``None`` starts at the beginning.
    start_line: int | None = Field(default=None, ge=1)
    #: Maximum number of lines to return.
    max_lines: int | None = Field(default=None, ge=1)


class ReadOutput(EdithModel):
    """Result of ``filesystem.read``."""

    path: str
    content: str
    start_line: int
    line_count: int
    total_lines: int
    truncated: bool
    size_bytes: int


class ReadTool(Tool):
    """Read a UTF-8 text file, optionally a line range."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="filesystem.read",
        description="Read a UTF-8 text file from the workspace, optionally a line range.",
        access=frozenset({AccessMode.READ}),
    )
    input_schema: ClassVar[type[BaseModel]] = ReadInput
    output_schema: ClassVar[type[BaseModel]] = ReadOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, ReadInput)  # noqa: S101 - guaranteed by validate_arguments
        resolved = ctx.workspace.resolve_existing_file(args.path)
        size = ctx.workspace.policy.check_readable_size(resolved, args.path)

        lines = _read_text(resolved, args.path).splitlines()
        total = len(lines)
        start = (args.start_line or 1) - 1
        if start >= total and total > 0:
            raise ToolExecutionError(
                f"start_line {args.start_line} is beyond the end of the file "
                f"({total} lines)",
                details={"path": args.path, "total_lines": total},
            )
        end = total if args.max_lines is None else min(total, start + args.max_lines)
        selected = lines[start:end]

        return ReadOutput(
            path=ctx.workspace.relative(resolved),
            content="\n".join(selected),
            start_line=start + 1,
            line_count=len(selected),
            total_lines=total,
            truncated=end < total,
            size_bytes=size,
        )


# --------------------------------------------------------------------------------------
# filesystem.search
# --------------------------------------------------------------------------------------


class SearchInput(EdithModel):
    """Arguments for ``filesystem.search``.

    At least one of ``name_pattern`` or ``content_pattern`` must be given; a search with
    neither would walk the whole tree and return everything.
    """

    #: Glob matched against the workspace-relative path, e.g. ``src/**/*.py``.
    name_pattern: str | None = None
    #: Regular expression matched against file contents.
    content_pattern: str | None = None
    #: Directory to search under. Defaults to the workspace root.
    path: str = "."
    case_sensitive: bool = False
    max_results: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_a_pattern(self) -> SearchInput:
        if not self.name_pattern and not self.content_pattern:
            raise ValueError("provide name_pattern, content_pattern, or both")
        return self


class SearchMatch(EdithModel):
    """One search hit."""

    path: str
    line_number: int | None = None
    line: str | None = None


class SearchOutput(EdithModel):
    """Result of ``filesystem.search``."""

    matches: list[SearchMatch] = Field(default_factory=list)
    files_scanned: int = 0
    truncated: bool = False


class SearchTool(Tool):
    """Find files by path glob and/or content regex.

    Files the agent may not read are silently omitted rather than reported as forbidden:
    announcing them would leak the existence of paths outside the agent's scope.
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="filesystem.search",
        description="Find files by path glob and/or content regex within the workspace.",
        access=frozenset({AccessMode.READ}),
    )
    input_schema: ClassVar[type[BaseModel]] = SearchInput
    output_schema: ClassVar[type[BaseModel]] = SearchOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, SearchInput)  # noqa: S101 - guaranteed by validate_arguments
        base = ctx.workspace.resolve_traversal_root(args.path)
        policy = ctx.workspace.policy
        limit = min(
            args.max_results or policy.config.max_search_results,
            policy.config.max_search_results,
        )

        expression: re.Pattern[str] | None = None
        if args.content_pattern:
            try:
                expression = re.compile(
                    args.content_pattern, 0 if args.case_sensitive else re.IGNORECASE
                )
            except re.error as exc:
                raise ToolExecutionError(
                    f"invalid content_pattern regular expression: {exc}",
                    details={"pattern": args.content_pattern},
                ) from exc

        matches: list[SearchMatch] = []
        scanned = 0
        truncated = False

        for candidate in self._walk(base, ctx):
            if not ctx.workspace.is_visible(candidate):
                continue
            relative = ctx.workspace.relative(candidate)

            if args.name_pattern and not _glob_matches(
                relative, args.name_pattern, args.case_sensitive
            ):
                continue

            if expression is None:
                matches.append(SearchMatch(path=relative))
                scanned += 1
            else:
                if candidate.stat().st_size > policy.config.max_file_bytes:
                    continue
                if _is_probably_binary(candidate):
                    continue
                scanned += 1
                try:
                    content = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for number, line in enumerate(content.splitlines(), start=1):
                    if expression.search(line):
                        matches.append(
                            SearchMatch(
                                path=relative,
                                line_number=number,
                                line=line.strip()[:500],
                            )
                        )
                        if len(matches) >= limit:
                            break

            if len(matches) >= limit:
                truncated = True
                break

        return SearchOutput(
            matches=matches[:limit], files_scanned=scanned, truncated=truncated
        )

    @staticmethod
    def _walk(base: Path, ctx: ToolContext) -> list[Path]:
        """Collect regular files under ``base``, pruning noisy and out-of-tree directories.

        A directory is only descended into if it *resolves* inside the workspace. Checking
        ``is_symlink()`` alone is not sufficient on Windows, where a junction reports False
        yet still redirects the walk outside the tree.
        """
        policy = ctx.workspace.policy
        collected: list[Path] = []
        stack = [base]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    # Following a link could also loop forever.
                    continue
                if entry.is_dir():
                    if entry.name in _SKIP_DIRECTORIES:
                        continue
                    if not policy.contains(entry):
                        continue
                    stack.append(entry)
                elif entry.is_file():
                    collected.append(entry)
        return sorted(collected)


def _glob_matches(relative: str, pattern: str, case_sensitive: bool) -> bool:
    """Match a workspace-relative POSIX path against a caller's search glob.

    Separate from the permission and protection matchers by design: this one is a *search
    filter* with no security role, honours ``case_sensitive``, and treats a bare ``*.py``
    as matching at any depth. The other two are security decisions and are always
    case-insensitive.
    """
    subject = relative if case_sensitive else relative.lower()
    candidate_pattern = pattern.replace("\\", "/")
    if not case_sensitive:
        candidate_pattern = candidate_pattern.lower()

    if fnmatchcase(subject, candidate_pattern):
        return True
    # A leading "**/" means "at any depth, including none". fnmatch has no concept of it and
    # would require a literal "/", so "**/*" would miss every top-level file.
    if candidate_pattern.startswith("**/") and fnmatchcase(subject, candidate_pattern[3:]):
        return True
    # "*.py" should also match nested files, which is what a caller means by it.
    if not candidate_pattern.startswith("**/") and "/" not in candidate_pattern:
        return fnmatchcase(subject, f"**/{candidate_pattern}")
    return False


# --------------------------------------------------------------------------------------
# filesystem.write
# --------------------------------------------------------------------------------------


class WriteInput(EdithModel):
    """Arguments for ``filesystem.write``."""

    path: str = Field(min_length=1)
    content: str
    #: Refuse to overwrite an existing file unless set. Defaults to refusing, so an agent
    #: must state its intent to destroy existing work.
    overwrite: bool = False
    create_parents: bool = True


class WriteOutput(EdithModel):
    """Result of ``filesystem.write``."""

    path: str
    bytes_written: int
    created: bool


class WriteTool(Tool):
    """Create or overwrite a text file within the agent's write scope."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="filesystem.write",
        description="Create or overwrite a UTF-8 text file within the agent's write scope.",
        access=frozenset({AccessMode.WRITE}),
    )
    input_schema: ClassVar[type[BaseModel]] = WriteInput
    output_schema: ClassVar[type[BaseModel]] = WriteOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, WriteInput)  # noqa: S101 - guaranteed by validate_arguments
        resolved = ctx.workspace.resolve_write(args.path)
        limit = ctx.workspace.policy.config.max_file_bytes
        payload = args.content.encode("utf-8")
        if len(payload) > limit:
            raise ToolExecutionError(
                f"content is {len(payload)} bytes, exceeding the {limit} byte limit",
                details={"path": args.path},
            )

        existed = resolved.exists()
        if existed:
            if not resolved.is_file():
                raise ToolExecutionError(
                    f"refusing to overwrite a non-file: {args.path}",
                    details={"path": args.path},
                )
            if not args.overwrite:
                raise ToolExecutionError(
                    f"{args.path} already exists; pass overwrite=true to replace it",
                    details={"path": args.path},
                )

        if args.create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.is_dir():
            raise ToolExecutionError(
                f"parent directory does not exist for {args.path}",
                details={"path": args.path},
            )

        try:
            # newline="" keeps the content byte-exact: Python must not silently rewrite
            # LF to CRLF on Windows and corrupt a diff the agent is about to verify.
            with resolved.open("w", encoding="utf-8", newline="") as handle:
                handle.write(args.content)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not write {args.path}: {exc.strerror or exc}",
                details={"path": args.path},
            ) from exc

        return WriteOutput(
            path=ctx.workspace.relative(resolved),
            bytes_written=len(payload),
            created=not existed,
        )


# --------------------------------------------------------------------------------------
# filesystem.patch
# --------------------------------------------------------------------------------------


class PatchInput(EdithModel):
    """Arguments for ``filesystem.patch``.

    Exact string replacement, not a diff format. A unified diff produced by a small local
    model is frequently malformed in ways that are hard to detect; an exact-match
    replacement either applies unambiguously or fails loudly.
    """

    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str
    #: Replace every occurrence. Default requires the match to be unique, so an agent
    #: cannot accidentally rewrite code it never inspected.
    replace_all: bool = False


class PatchOutput(EdithModel):
    """Result of ``filesystem.patch``."""

    path: str
    replacements: int
    bytes_written: int


class PatchTool(Tool):
    """Apply an exact-match text replacement to a file."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="filesystem.patch",
        description="Replace exact text in a file; requires a unique match by default.",
        access=frozenset({AccessMode.READ, AccessMode.WRITE}),
    )
    input_schema: ClassVar[type[BaseModel]] = PatchInput
    output_schema: ClassVar[type[BaseModel]] = PatchOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, PatchInput)  # noqa: S101 - guaranteed by validate_arguments
        resolved = ctx.workspace.resolve_existing_file(args.path, AccessMode.WRITE)
        ctx.workspace.policy.check_readable_size(resolved, args.path)

        original = _read_text(resolved, args.path)
        occurrences = original.count(args.old_text)
        if occurrences == 0:
            raise ToolExecutionError(
                f"old_text was not found in {args.path}",
                details={"path": args.path},
            )
        if occurrences > 1 and not args.replace_all:
            raise ToolExecutionError(
                f"old_text appears {occurrences} times in {args.path}; "
                "make it unique or pass replace_all=true",
                details={"path": args.path, "occurrences": occurrences},
            )

        updated = original.replace(
            args.old_text, args.new_text, -1 if args.replace_all else 1
        )
        payload = updated.encode("utf-8")
        limit = ctx.workspace.policy.config.max_file_bytes
        if len(payload) > limit:
            raise ToolExecutionError(
                f"patched content is {len(payload)} bytes, exceeding the {limit} byte limit",
                details={"path": args.path},
            )

        try:
            with resolved.open("w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
        except OSError as exc:
            raise ToolExecutionError(
                f"could not write {args.path}: {exc.strerror or exc}",
                details={"path": args.path},
            ) from exc

        return PatchOutput(
            path=ctx.workspace.relative(resolved),
            replacements=occurrences if args.replace_all else 1,
            bytes_written=len(payload),
        )
