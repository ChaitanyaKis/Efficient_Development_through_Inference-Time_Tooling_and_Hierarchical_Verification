"""Controlled subprocess execution.

Shared by ``shell.run`` and every ``git.*`` tool so that timeouts, output limits,
environment sanitization, and process-tree termination are implemented once.

Three properties are structural rather than best-effort:

1. **No shell.** ``shell=False`` always, and the command is an argv list. There is no
   string for a metacharacter to hide in, so command injection is not mitigated -- it is
   absent.
2. **Sanitized environment.** The child's environment is built from scratch from an
   allowlist. An ``ANTHROPIC_API_KEY`` or ``AWS_SECRET_ACCESS_KEY`` in the parent process
   cannot reach an agent-invoked program.
3. **Bounded output.** stdout and stderr are truncated to a configured size, so a runaway
   process cannot exhaust memory or flood a model's context window.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from edith.errors import ToolExecutionError, ToolTimeoutError
from edith.observability.logging import get_logger

from .schemas import TruncatedText, truncate

logger = get_logger(__name__)

#: Grace period for a terminated process tree to exit before it is killed outright.
_TERMINATE_GRACE_SECONDS = 3.0

#: Names that mean "the interpreter Edith is running under", not a PATH lookup.
_INTERPRETER_ALIASES = frozenset({"python", "python3"})


@dataclass(frozen=True)
class ProcessResult:
    """The outcome of a completed child process."""

    argv: tuple[str, ...]
    exit_code: int
    stdout: TruncatedText
    stderr: TruncatedText
    duration_seconds: float

    @property
    def ok(self) -> bool:
        """True when the process exited zero."""
        return self.exit_code == 0


def build_environment(passthrough: tuple[str, ...]) -> dict[str, str]:
    """Construct a minimal child environment from an allowlist.

    ``PATH`` is always included -- without it nothing is executable -- but every other
    variable must be named explicitly. Matching is case-insensitive because Windows
    environment variables are.
    """
    wanted = {name.upper() for name in passthrough} | {"PATH"}
    env = {key: value for key, value in os.environ.items() if key.upper() in wanted}
    # Keep child output parseable and unlocalized regardless of the host configuration.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def resolve_executable(name: str, allowed: tuple[str, ...]) -> str:
    """Resolve an allowlisted executable name to an absolute path.

    The allowlist is checked against the *bare name the caller supplied*, before resolution,
    and the name may not contain a path separator. That prevents ``./git`` or
    ``C:\\evil\\git.exe`` from satisfying an allowlist entry of ``git``.

    ``python``/``python3`` resolve to :data:`sys.executable` -- the interpreter Edith is
    running under -- rather than whatever appears first on ``PATH``. Edith is normally
    installed in a virtualenv, and the tools a project needs (pytest, ruff, mypy) live
    there. Taking the PATH entry instead silently runs a different interpreter that has
    none of them, and a check that cannot even import its runner reports as a *failing
    test suite*, which is a deeply misleading thing to hand a debugging agent.

    Raises:
        ToolExecutionError: The name is not allowlisted or is not on PATH.
    """
    if not name or not name.strip():
        raise ToolExecutionError("executable name must not be empty")
    if "/" in name or "\\" in name:
        raise ToolExecutionError(
            f"executable must be a bare name, not a path: {name!r}",
            details={"executable": name},
        )

    bare = Path(name).stem.lower()
    permitted = {entry.lower() for entry in allowed}
    if bare not in permitted and name.lower() not in permitted:
        raise ToolExecutionError(
            f"executable {name!r} is not allowlisted; permitted: {sorted(permitted)}",
            details={"executable": name, "allowed": sorted(permitted)},
        )

    if bare in _INTERPRETER_ALIASES and sys.executable:
        return sys.executable

    resolved = shutil.which(name)
    if resolved is None:
        raise ToolExecutionError(
            f"executable {name!r} was not found on PATH",
            details={"executable": name},
        )
    return resolved


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a process and all of its descendants.

    ``Popen.kill`` reaches only the direct child; a test runner that spawned workers would
    leave them running and holding the workspace open. psutil gives us the whole tree.
    """
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    for victim in (*children, parent):
        try:
            victim.terminate()
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs([*children, parent], timeout=_TERMINATE_GRACE_SECONDS)
    for survivor in alive:
        try:
            survivor.kill()
        except psutil.Error:
            continue


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int,
    env_passthrough: tuple[str, ...],
) -> ProcessResult:
    """Run a child process under a timeout with bounded, captured output.

    Args:
        argv: Full command line. ``argv[0]`` must already be an absolute, resolved path.
        cwd: Working directory; must be inside the workspace.
        timeout_seconds: Wall-clock budget before the process tree is terminated.
        max_output_bytes: Per-stream truncation limit.
        env_passthrough: Environment variable allowlist.

    Raises:
        ToolTimeoutError: The process exceeded its budget and was terminated.
        ToolExecutionError: The process could not be started.
    """
    if not argv:
        raise ToolExecutionError("argv must not be empty")

    started = time.monotonic()
    try:
        process = subprocess.Popen(  # noqa: S603 - argv list, shell=False, allowlisted argv[0]
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=build_environment(env_passthrough),
            shell=False,
        )
    except OSError as exc:
        raise ToolExecutionError(
            f"could not start {argv[0]!r}: {exc}",
            details={"executable": argv[0]},
        ) from exc

    try:
        raw_out, raw_err = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as timeout_exc:
        _kill_tree(process)
        # Drain whatever the process produced before it was killed; without this the pipes
        # can keep the handle open on Windows.
        try:
            raw_out, raw_err = process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, ValueError):  # pragma: no cover - defensive
            raw_out, raw_err = b"", b""
        duration = time.monotonic() - started
        logger.warning(
            "process.timeout",
            executable=Path(argv[0]).name,
            timeout_seconds=timeout_seconds,
            duration_seconds=round(duration, 3),
        )
        raise ToolTimeoutError(
            f"{Path(argv[0]).name!r} exceeded its {timeout_seconds}s budget and was terminated",
            details={
                "executable": Path(argv[0]).name,
                "timeout_seconds": timeout_seconds,
                "stdout": truncate(
                    raw_out.decode("utf-8", errors="replace"), max_output_bytes
                ).text,
                "stderr": truncate(
                    raw_err.decode("utf-8", errors="replace"), max_output_bytes
                ).text,
            },
        ) from timeout_exc

    duration = time.monotonic() - started
    result = ProcessResult(
        argv=tuple(argv),
        exit_code=process.returncode,
        stdout=truncate(raw_out.decode("utf-8", errors="replace"), max_output_bytes),
        stderr=truncate(raw_err.decode("utf-8", errors="replace"), max_output_bytes),
        duration_seconds=duration,
    )
    logger.debug(
        "process.completed",
        executable=Path(argv[0]).name,
        exit_code=result.exit_code,
        duration_seconds=round(duration, 3),
        stdout_truncated=result.stdout.truncated,
    )
    return result
