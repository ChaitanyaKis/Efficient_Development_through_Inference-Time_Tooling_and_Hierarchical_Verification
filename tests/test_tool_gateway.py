"""The Tool Gateway: registration, authorization, audit logging, and structured results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from edith.config.schema import LoggingConfig
from edith.errors import ToolRegistrationError
from edith.observability.logging import configure_logging
from edith.schemas.agent import AgentPermissions
from edith.schemas.common import EdithModel
from edith.tools.base import Tool, ToolContext
from edith.tools.gateway import ToolGateway
from edith.tools.registry import ToolRegistry, build_default_registry
from edith.tools.schemas import AccessMode, ToolCall, ToolSpec, truncate

from .tool_fixtures import READONLY_PERMISSIONS, build_gateway, build_workspace

#: Every tool the gateway ships. `git.show` was added in M2.1 so the integrity check can
#: read a file's baseline content without disturbing the working tree.
M1_TOOLS = {
    "filesystem.read", "filesystem.search", "filesystem.write", "filesystem.patch",
    "shell.run", "git.status", "git.diff", "git.show", "git.log", "git.branch",
    "git.commit", "git.worktree",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return build_workspace(tmp_path / "ws")


@pytest.fixture
def gateway(workspace: Path) -> ToolGateway:
    return build_gateway(workspace)


class ExplodingInput(EdithModel):
    value: int = 1


class ExplodingOutput(EdithModel):
    value: int = 1


class ExplodingTool(Tool):
    """Raises a bare Python exception, to prove the gateway contains it."""

    spec: ClassVar[ToolSpec] = ToolSpec(name="test.explode", description="raises")
    input_schema: ClassVar[type[BaseModel]] = ExplodingInput
    output_schema: ClassVar[type[BaseModel]] = ExplodingOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        raise RuntimeError("catastrophic internal failure")


class BadOutputTool(Tool):
    """Returns something that violates its own declared output schema."""

    spec: ClassVar[ToolSpec] = ToolSpec(name="test.bad_output", description="wrong shape")
    input_schema: ClassVar[type[BaseModel]] = ExplodingInput
    output_schema: ClassVar[type[BaseModel]] = ExplodingOutput

    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        return {"unexpected": "shape"}  # type: ignore[return-value]


#: Grants everything, so a test can reach a synthetic tool that no real agent would hold.
WIDE_OPEN = AgentPermissions(
    allowed_tools=frozenset({"*"}),
    allowed_read_paths=("**",),
    allowed_write_paths=("**",),
)


def _gateway_with(workspace: Path, *extra: Tool) -> ToolGateway:
    """A wide-open gateway whose registry also contains the given synthetic tools."""
    registry = build_default_registry()
    for tool in extra:
        registry.register(tool)
    gateway = build_gateway(workspace, WIDE_OPEN)
    gateway.registry = registry
    return gateway


class TestRegistry:
    def test_ships_every_m1_tool(self) -> None:
        assert set(build_default_registry().names()) == M1_TOOLS

    def test_duplicate_registration_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(ExplodingTool())
        with pytest.raises(ToolRegistrationError, match="already registered"):
            registry.register(ExplodingTool())

    def test_replace_is_explicit(self) -> None:
        registry = ToolRegistry()
        registry.register(ExplodingTool())
        registry.register(ExplodingTool(), replace=True)
        assert len(registry) == 1

    def test_unknown_tool_raises(self) -> None:
        from edith.errors import ToolNotFoundError

        with pytest.raises(ToolNotFoundError, match="not registered"):
            ToolRegistry().get("nope")

    def test_object_without_spec_rejected(self) -> None:
        class NotATool:
            pass

        with pytest.raises(ToolRegistrationError, match="no valid class-level"):
            ToolRegistry().register(NotATool())  # type: ignore[arg-type]

    def test_incomplete_tool_rejected_at_definition(self) -> None:
        with pytest.raises(TypeError, match="must define a class-level"):

            class Incomplete(Tool):  # type: ignore[misc]
                spec = ToolSpec(name="test.incomplete", description="d")

                def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
                    return ExplodingOutput()

    def test_names_are_sorted(self) -> None:
        names = build_default_registry().names()
        assert list(names) == sorted(names)


class TestAuthorization:
    def test_unknown_tool_is_a_configuration_error(self, gateway: ToolGateway) -> None:
        result = gateway.execute(ToolCall(tool="filesystem.teleport"))
        assert not result.ok
        assert str(result.failure_category) == "CONFIGURATION_ERROR"

    def test_ungranted_tool_denied(self, workspace: Path) -> None:
        result = build_gateway(workspace, READONLY_PERMISSIONS).execute(
            ToolCall(tool="filesystem.write", arguments={"path": "src/x.py", "content": "x"})
        )
        assert not result.ok and result.denied

    def test_available_specs_reflect_grants(self, workspace: Path) -> None:
        specs = build_gateway(workspace, READONLY_PERMISSIONS).available_specs()
        assert {s.name for s in specs} == {
            "filesystem.read", "filesystem.search", "git.status"
        }

    def test_can_use_does_not_execute(self, workspace: Path) -> None:
        gateway = build_gateway(workspace, READONLY_PERMISSIONS)
        assert gateway.can_use("filesystem.read")
        assert not gateway.can_use("filesystem.write")
        assert not gateway.can_use("does.not_exist")

    def test_for_agent_rebinds_permissions(self, gateway: ToolGateway) -> None:
        narrowed = gateway.for_agent(READONLY_PERMISSIONS, "reviewer")
        assert narrowed.agent == "reviewer"
        assert not narrowed.can_use("filesystem.write")
        # The original gateway is unchanged.
        assert gateway.can_use("filesystem.write")

    def test_rebound_gateway_shares_registry_and_policy(self, gateway: ToolGateway) -> None:
        narrowed = gateway.for_agent(READONLY_PERMISSIONS, "reviewer")
        assert narrowed.registry is gateway.registry
        assert narrowed.policy is gateway.policy


class TestResultContract:
    def test_execute_never_raises(self, gateway: ToolGateway) -> None:
        """Every failure mode must come back as a structured result."""
        wide = _gateway_with(gateway.policy.root, ExplodingTool(), BadOutputTool())
        for call in (
            ToolCall(tool="test.explode"),
            ToolCall(tool="test.bad_output"),
            ToolCall(tool="does.not_exist"),
            ToolCall(tool="filesystem.read", arguments={"path": "../escape"}),
            ToolCall(tool="filesystem.read", arguments={}),
        ):
            result = wide.execute(call)
            assert result.ok is False
            assert result.error is not None
            assert result.failure_category is not None

    def test_unexpected_exception_is_contained(self, workspace: Path) -> None:
        gateway = _gateway_with(workspace, ExplodingTool())
        result = gateway.execute(ToolCall(tool="test.explode"))
        assert not result.ok
        assert "catastrophic internal failure" in result.error
        assert str(result.failure_category) == "TOOL_ERROR"

    def test_bad_tool_output_is_caught(self, workspace: Path) -> None:
        """A tool cannot emit unvalidated output; the gate is not its own code."""
        gateway = _gateway_with(workspace, BadOutputTool())
        result = gateway.execute(ToolCall(tool="test.bad_output"))
        assert not result.ok
        assert str(result.failure_category) == "VALIDATION_FAILURE"

    def test_call_id_is_echoed(self, gateway: ToolGateway) -> None:
        call = ToolCall(tool="filesystem.read", arguments={"path": "README.md"})
        assert gateway.execute(call).call_id == call.call_id

    def test_success_records_duration_and_agent(self, gateway: ToolGateway) -> None:
        result = gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": "README.md"})
        )
        assert result.ok
        assert result.duration_seconds >= 0.0
        assert result.evidence["agent"] == "test_agent"

    def test_result_round_trips_through_json(self, gateway: ToolGateway) -> None:
        from edith.tools.schemas import ToolResult

        original = gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": "README.md"})
        )
        assert ToolResult.model_validate_json(original.model_dump_json()).output == original.output

    def test_denied_property_distinguishes_policy_from_error(
        self, gateway: ToolGateway
    ) -> None:
        denied = gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": ".env"})
        )
        failed = gateway.execute(
            ToolCall(tool="filesystem.read", arguments={"path": "src/absent.py"})
        )
        assert denied.denied and not failed.denied


class TestAuditLogging:
    def _events(self, log_file: Path) -> list[dict]:
        logging.getLogger().handlers[-1].flush()
        return [
            json.loads(line)
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_successful_call_is_audited(self, gateway: ToolGateway, tmp_path: Path) -> None:
        log_file = tmp_path / "audit.jsonl"
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=log_file),
            base_dir=tmp_path,
        )
        gateway.execute(ToolCall(tool="filesystem.read", arguments={"path": "README.md"}))

        events = self._events(log_file)
        start = next(e for e in events if e["event"] == "tool.start")
        assert start["tool"] == "filesystem.read"
        assert start["agent"] == "test_agent"
        assert start["call_id"].startswith("call_")
        assert any(e["event"] == "tool.success" for e in events)

    def test_denial_is_audited_with_its_reason(
        self, gateway: ToolGateway, tmp_path: Path
    ) -> None:
        """A misconfigured or probing agent must be visible in the audit trail."""
        log_file = tmp_path / "audit.jsonl"
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=log_file),
            base_dir=tmp_path,
        )
        gateway.execute(ToolCall(tool="filesystem.read", arguments={"path": "../secret"}))

        denied = next(e for e in self._events(log_file) if e["event"] == "tool.denied")
        assert denied["category"] == "SECURITY_FAILURE"
        assert denied["level"] == "warning"
        assert denied["tool"] == "filesystem.read"

    def test_audit_records_risk_surface(self, gateway: ToolGateway, tmp_path: Path) -> None:
        log_file = tmp_path / "audit.jsonl"
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=log_file),
            base_dir=tmp_path,
        )
        gateway.execute(
            ToolCall(tool="filesystem.write", arguments={"path": "src/n.py", "content": "x"})
        )
        start = next(e for e in self._events(log_file) if e["event"] == "tool.start")
        assert start["writes"] is True
        assert start["spawns_process"] is False

    def test_secrets_in_arguments_are_redacted(
        self, gateway: ToolGateway, tmp_path: Path
    ) -> None:
        """The M0 redaction pipeline must still hold for tool events."""
        log_file = tmp_path / "audit.jsonl"
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=log_file),
            base_dir=tmp_path,
        )
        from edith.observability.logging import get_logger

        get_logger("edith.tools").info(
            "tool.start", tool="shell.run", api_key="sk-must-not-appear"
        )
        assert "sk-must-not-appear" not in log_file.read_text(encoding="utf-8")

    def test_context_is_cleared_between_calls(self, gateway: ToolGateway, tmp_path: Path) -> None:
        log_file = tmp_path / "audit.jsonl"
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=log_file),
            base_dir=tmp_path,
        )
        gateway.execute(ToolCall(tool="filesystem.read", arguments={"path": "README.md"}))
        from edith.observability.logging import get_logger

        get_logger("unrelated").info("after_tool_call")
        after = next(e for e in self._events(log_file) if e["event"] == "after_tool_call")
        assert "tool" not in after and "call_id" not in after


class TestTruncate:
    def test_short_text_untouched(self) -> None:
        result = truncate("hello", 1024)
        assert result.text == "hello" and not result.truncated

    def test_long_text_is_cut_and_flagged(self) -> None:
        result = truncate("x" * 5000, 100)
        assert result.truncated
        assert len(result.text.encode()) <= 100
        assert result.original_bytes == 5000

    def test_multibyte_characters_are_not_split(self) -> None:
        """Cutting mid-character would produce invalid UTF-8 in the audit log."""
        result = truncate("Ã©" * 500, 101)
        assert result.truncated
        result.text.encode("utf-8").decode("utf-8")


class TestToolSpec:
    def test_writes_property(self) -> None:
        assert ToolSpec(
            name="a.b", description="d", access=frozenset({AccessMode.WRITE})
        ).writes
        assert not ToolSpec(name="a.b", description="d").writes

    @pytest.mark.parametrize("name", ["nodot", "Bad.Name", "1.bad", "a..b", ""])
    def test_malformed_names_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            ToolSpec(name=name, description="d")
