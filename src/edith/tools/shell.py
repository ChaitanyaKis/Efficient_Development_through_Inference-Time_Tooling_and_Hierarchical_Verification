"""``shell.run`` -- allowlisted command execution.

The tool takes an **argv list**, never a command string, and never invokes a system shell.
That is not a mitigation for injection; it removes the category. There is no string for
``;``, ``&&``, backticks, or ``$(...)`` to be interpreted in, because nothing ever parses
one.

The executable must additionally appear in the configured allowlist, so an agent that
somehow constructs a plausible argv still cannot run an arbitrary program.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from edith.schemas.common import EdithModel

from .base import Tool, ToolContext
from .process import resolve_executable, run_process
from .schemas import AccessMode, ToolSpec


class ShellRunInput(EdithModel):
    """Arguments for ``shell.run``."""

    #: Full command line as a list. ``argv[0]`` must be an allowlisted bare executable name.
    argv: list[str] = Field(min_length=1)
    #: Working directory relative to the workspace root.
    cwd: str = "."
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)

    @field_validator("argv")
    @classmethod
    def _reject_empty_arguments(cls, value: list[str]) -> list[str]:
        for index, item in enumerate(value):
            if not isinstance(item, str):  # pragma: no cover - pydantic enforces the type
                raise ValueError("argv entries must be strings")
            if "\x00" in item:
                raise ValueError(f"argv[{index}] must not contain a NUL byte")
        if not value[0].strip():
            raise ValueError("argv[0] must name an executable")
        return value


class ShellRunOutput(EdithModel):
    """Result of ``shell.run``.

    A non-zero exit code is reported here, not raised: "the command ran and failed" is
    evidence the Testing and Debugging agents need, not an infrastructure error.
    """

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    cwd: str

    @property
    def ok(self) -> bool:
        """True when the command exited zero."""
        return self.exit_code == 0


class ShellRunTool(Tool):
    """Run an allowlisted executable with an argv list, under a timeout."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="shell.run",
        description=(
            "Run an allowlisted executable with an argv list (never a shell string), "
            "under a timeout with bounded output."
        ),
        access=frozenset({AccessMode.READ}),
        spawns_process=True,
    )
    input_schema: ClassVar[type[BaseModel]] = ShellRunInput
    output_schema: ClassVar[type[BaseModel]] = ShellRunOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        assert isinstance(args, ShellRunInput)  # noqa: S101 - guaranteed by validate_arguments
        policy = ctx.config.shell

        # The working directory is authorized like any other path, so a command cannot be
        # launched from somewhere the agent may not read.
        working_dir = ctx.workspace.resolve_directory(args.cwd)
        executable = resolve_executable(args.argv[0], policy.allowed_executables)

        timeout = args.timeout_seconds or ctx.timeout(policy.timeout_seconds)
        result = run_process(
            [executable, *args.argv[1:]],
            cwd=working_dir,
            timeout_seconds=timeout,
            max_output_bytes=policy.max_output_bytes,
            env_passthrough=policy.env_passthrough,
        )

        return ShellRunOutput(
            # Echo the requested argv, not the resolved absolute path: the agent should
            # reason about what it asked for, and the host's layout is not its business.
            argv=list(args.argv),
            exit_code=result.exit_code,
            stdout=result.stdout.text,
            stderr=result.stderr.text,
            stdout_truncated=result.stdout.truncated,
            stderr_truncated=result.stderr.truncated,
            duration_seconds=result.duration_seconds,
            cwd=ctx.workspace.relative(working_dir),
        )
