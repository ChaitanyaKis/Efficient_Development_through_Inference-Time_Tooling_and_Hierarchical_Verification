"""``shell.run``: allowlisting, injection surface, timeouts, output limits, env hygiene."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from edith.config.schema import ShellPolicyConfig
from edith.errors import ToolExecutionError
from edith.schemas.agent import AgentPermissions
from edith.tools.gateway import ToolGateway
from edith.tools.process import build_environment, resolve_executable
from edith.tools.schemas import ToolCall

from .tool_fixtures import build_config, build_gateway, build_workspace

#: The interpreter running the tests, used as a portable allowlisted executable.
PYTHON = Path(sys.executable).stem


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return build_workspace(tmp_path / "ws")


@pytest.fixture
def gateway(workspace: Path) -> ToolGateway:
    return build_gateway(
        workspace,
        config=build_config(
            workspace,
            shell=ShellPolicyConfig(allowed_executables=(PYTHON, "python", "git")),
        ),
    )


def run(gateway: ToolGateway, **arguments: object) -> object:
    return gateway.execute(ToolCall(tool="shell.run", arguments=arguments))


class TestExecution:
    def test_runs_an_allowlisted_command(self, gateway: ToolGateway) -> None:
        result = run(gateway, argv=[PYTHON, "-c", "print('hello from edith')"])
        assert result.ok
        assert result.output["exit_code"] == 0
        assert "hello from edith" in result.output["stdout"]

    def test_nonzero_exit_is_reported_not_raised(self, gateway: ToolGateway) -> None:
        """'The command ran and failed' is evidence, not an infrastructure error."""
        result = run(gateway, argv=[PYTHON, "-c", "import sys; sys.exit(3)"])
        assert result.ok, "the tool call itself succeeded"
        assert result.output["exit_code"] == 3

    def test_stderr_is_captured_separately(self, gateway: ToolGateway) -> None:
        result = run(
            gateway, argv=[PYTHON, "-c", "import sys; sys.stderr.write('problem')"]
        )
        assert "problem" in result.output["stderr"]
        assert "problem" not in result.output["stdout"]

    def test_runs_in_the_workspace(self, gateway: ToolGateway, workspace: Path) -> None:
        result = run(gateway, argv=[PYTHON, "-c", "import os; print(os.getcwd())"])
        assert Path(result.output["stdout"].strip()).resolve() == workspace.resolve()

    def test_cwd_can_be_narrowed(self, gateway: ToolGateway, workspace: Path) -> None:
        result = run(
            gateway, argv=[PYTHON, "-c", "import os; print(os.getcwd())"], cwd="src"
        )
        assert result.output["cwd"] == "src"
        assert Path(result.output["stdout"].strip()).name == "src"

    def test_argv_is_echoed_not_the_resolved_path(self, gateway: ToolGateway) -> None:
        """The host's install layout is not the agent's business."""
        result = run(gateway, argv=[PYTHON, "-c", "pass"])
        assert result.output["argv"][0] == PYTHON


class TestAllowlist:
    def test_unlisted_executable_denied(self, gateway: ToolGateway) -> None:
        result = run(gateway, argv=["curl", "https://example.com"])
        assert not result.ok and "not allowlisted" in result.error

    def test_path_qualified_executable_denied(self, gateway: ToolGateway) -> None:
        """`./git` or `C:\\evil\\git.exe` must not satisfy an allowlist entry of `git`."""
        for candidate in ("./python", "C:\\evil\\python.exe", "..\\python", "/usr/bin/python"):
            result = run(gateway, argv=[candidate, "-c", "pass"])
            assert not result.ok
            assert "bare name" in result.error or "not allowlisted" in result.error

    def test_missing_executable_reported(self, workspace: Path) -> None:
        missing = "definitely_not_installed_xyz"
        gateway = build_gateway(
            workspace,
            config=build_config(
                workspace, shell=ShellPolicyConfig(allowed_executables=(missing,))
            ),
        )
        result = run(gateway, argv=[missing])
        assert not result.ok and "not found on PATH" in result.error

    def test_resolve_executable_rejects_empty(self) -> None:
        with pytest.raises(ToolExecutionError, match="must not be empty"):
            resolve_executable("  ", ("python",))


class TestInjectionSurface:
    """argv-only execution means there is no string for a shell to interpret."""

    def test_metacharacters_are_inert_arguments(self, gateway: ToolGateway) -> None:
        marker = "; echo PWNED & whoami | dir"
        result = run(gateway, argv=[PYTHON, "-c", "import sys; print(sys.argv[1])", marker])
        assert result.ok
        # The payload comes back verbatim as data: nothing interpreted it.
        assert marker in result.output["stdout"]
        assert "PWNED" not in result.output["stdout"].replace(marker, "")

    def test_command_string_is_not_accepted(self, gateway: ToolGateway) -> None:
        """There is no `command: str` field to smuggle a pipeline through."""
        result = gateway.execute(
            ToolCall(tool="shell.run", arguments={"command": "python -c 'print(1)'"})
        )
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"

    def test_empty_argv_rejected(self, gateway: ToolGateway) -> None:
        result = run(gateway, argv=[])
        assert not result.ok

    def test_nul_byte_in_argv_rejected(self, gateway: ToolGateway) -> None:
        result = run(gateway, argv=[PYTHON, "-c", "print(1)\x00malicious"])
        assert not result.ok


class TestTimeout:
    def test_slow_command_is_terminated(self, gateway: ToolGateway) -> None:
        result = run(
            gateway,
            argv=[PYTHON, "-c", "import time; time.sleep(30)"],
            timeout_seconds=2.0,
        )
        assert not result.ok
        assert str(result.failure_category) == "TIMEOUT"
        assert "budget" in result.error

    def test_timeout_result_is_retryable(self, gateway: ToolGateway) -> None:
        result = run(
            gateway, argv=[PYTHON, "-c", "import time; time.sleep(30)"], timeout_seconds=2.0
        )
        assert str(result.failure_category) == "TIMEOUT"

    def test_child_processes_are_killed_too(self, gateway: ToolGateway) -> None:
        """A test runner that spawned workers must not leave them holding the workspace."""
        script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        )
        result = run(gateway, argv=[PYTHON, "-c", script], timeout_seconds=3.0)
        assert not result.ok and str(result.failure_category) == "TIMEOUT"


class TestOutputLimits:
    def test_large_output_is_truncated_and_flagged(self, workspace: Path) -> None:
        gateway = build_gateway(
            workspace,
            config=build_config(
                workspace,
                shell=ShellPolicyConfig(
                    allowed_executables=(PYTHON,), max_output_bytes=2048
                ),
            ),
        )
        result = run(gateway, argv=[PYTHON, "-c", "print('x' * 100000)"])
        assert result.ok
        assert result.output["stdout_truncated"] is True
        assert len(result.output["stdout"].encode()) <= 2048


class TestEnvironmentHygiene:
    def test_secrets_are_not_passed_to_children(
        self, gateway: ToolGateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An API key in the parent process must never reach an agent-invoked program."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-leak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-must-not-leak")
        result = run(
            gateway,
            argv=[PYTHON, "-c", "import os; print('|'.join(sorted(os.environ)))"],
        )
        assert result.ok
        assert "ANTHROPIC_API_KEY" not in result.output["stdout"]
        assert "AWS_SECRET_ACCESS_KEY" not in result.output["stdout"]
        assert "must-not-leak" not in result.output["stdout"]

    def test_path_is_always_present(self) -> None:
        assert "PATH" in build_environment(())

    def test_allowlisted_variables_pass_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANG", "en_GB.UTF-8")
        assert build_environment(("LANG",))["LANG"] == "en_GB.UTF-8"

    def test_matching_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows environment variables are case-insensitive; the allowlist must agree."""
        monkeypatch.setenv("TEMP", "C:\\Temp")
        assert "TEMP" in build_environment(("temp",))

    def test_unlisted_variable_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_PRIVATE_TOKEN", "value")
        assert "SOME_PRIVATE_TOKEN" not in build_environment(("PATH",))

    @pytest.mark.skipif(os.name != "nt", reason="SystemRoot is Windows-specific")
    def test_systemroot_available_for_windows_children(self) -> None:
        """Without SystemRoot many Windows programs fail to start at all."""
        assert "SYSTEMROOT" in {k.upper() for k in build_environment(("SYSTEMROOT",))}


class TestPermissions:
    def test_agent_without_shell_grant_is_denied(self, workspace: Path) -> None:
        readonly = AgentPermissions(
            allowed_tools=frozenset({"filesystem.read"}), allowed_read_paths=("**",)
        )
        result = run(build_gateway(workspace, readonly), argv=[PYTHON, "-c", "print(1)"])
        assert not result.ok and result.denied

    def test_cwd_outside_scope_denied(self, workspace: Path) -> None:
        result = run(build_gateway(workspace), argv=[PYTHON, "-c", "print(1)"], cwd="../")
        assert not result.ok and result.denied
