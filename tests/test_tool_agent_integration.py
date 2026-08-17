"""M1 acceptance: an agent safely inspects and modifies a repository and produces a diff.

This is the milestone criterion executed end to end. It uses a real repository, the real
gateway, and the real permission engine -- only the model is faked, because M1 is about the
tool kernel rather than about inference.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from edith.agents.base import Agent
from edith.agents.registry import AgentRegistry
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.tools.schemas import ToolCall

from .fakes import FakeProvider
from .test_tool_git import init_repo
from .tool_fixtures import build_config, build_gateway, build_workspace

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


class EditInput(EdithModel):
    path: str
    old_text: str
    new_text: str


class EditOutput(EdithModel):
    changed_file: str
    diff_lines: int
    committed_sha: str
    tests_exit_code: int


class WorkerAgent(Agent):
    """A minimal implementation agent exercising the full M1 tool surface.

    Declares exactly the scope it needs: read anywhere, write only under ``src/``. Its
    permissions are the same ``AgentPermissions`` M0 defined -- M1 only enforces them.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="worker",
        description="Edits a file, verifies the diff, and commits.",
        capabilities=frozenset({Capability.CODE_GENERATION}),
        permissions=AgentPermissions(
            allowed_tools=frozenset({"filesystem.*", "git.status", "git.diff", "git.commit"}),
            allowed_read_paths=("**",),
            allowed_write_paths=("src/**",),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = EditInput
    output_schema: ClassVar[type[BaseModel]] = EditOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, EditInput)
        tools = self.require_tools()

        patched = tools.execute(
            ToolCall(
                tool="filesystem.patch",
                arguments={
                    "path": payload.path,
                    "old_text": payload.old_text,
                    "new_text": payload.new_text,
                },
            )
        )
        if not patched.ok:
            raise AssertionError(f"patch failed: {patched.error}")

        diff = tools.execute(ToolCall(tool="git.diff"))
        if not diff.ok:
            raise AssertionError(f"diff failed: {diff.error}")

        commit = tools.execute(
            ToolCall(
                tool="git.commit",
                arguments={"message": "apply requested edit", "paths": [payload.path]},
            )
        )
        if not commit.ok:
            raise AssertionError(f"commit failed: {commit.error}")

        return EditOutput(
            changed_file=patched.output["path"],
            diff_lines=len(diff.output["diff"].splitlines()),
            committed_sha=commit.output["short_sha"],
            tests_exit_code=0,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(build_workspace(tmp_path / "ws"))


class TestAcceptance:
    def test_agent_inspects_modifies_and_produces_a_diff(self, repo: Path) -> None:
        """M1 ACCEPTANCE CRITERION."""
        gateway = build_gateway(repo, WorkerAgent.identity.permissions, agent="worker")
        agent = WorkerAgent(provider=FakeProvider(_params()), tools=gateway)

        response = agent.execute(
            AgentRequest(
                payload={
                    "path": "src/app.py",
                    "old_text": "return 'hello'",
                    "new_text": "return 'hello, world'",
                }
            )
        )

        assert response.ok, f"{response.failure_category}: {response.error}"
        assert response.output["changed_file"] == "src/app.py"
        assert response.output["diff_lines"] > 0
        assert response.output["committed_sha"]

        # Verified against the real repository, not the agent's own claim.
        content = (repo / "src" / "app.py").read_text(encoding="utf-8")
        assert "hello, world" in content
        log = subprocess.run(
            ["git", "log", "-1", "--format=%B"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
        assert "apply requested edit" in log
        assert "Edith-Agent: worker" in log

    def test_agent_cannot_escape_its_write_scope(self, repo: Path) -> None:
        """The same agent, asked to touch a file outside src/, is refused."""
        gateway = build_gateway(repo, WorkerAgent.identity.permissions, agent="worker")
        agent = WorkerAgent(provider=FakeProvider(_params()), tools=gateway)

        response = agent.execute(
            AgentRequest(
                payload={
                    "path": "docs/guide.md",
                    "old_text": "# Guide",
                    "new_text": "# Compromised",
                }
            )
        )
        assert not response.ok
        assert "# Guide" in (repo / "docs" / "guide.md").read_text(encoding="utf-8")

    def test_agent_cannot_read_secrets(self, repo: Path) -> None:
        gateway = build_gateway(repo, WorkerAgent.identity.permissions, agent="worker")
        result = gateway.execute(ToolCall(tool="filesystem.read", arguments={"path": ".env"}))
        assert not result.ok and result.denied


class TestRegistryWiring:
    def test_registry_injects_a_scoped_gateway(self, repo: Path) -> None:
        """The gateway an agent receives is bound to its own declared permissions."""
        config = build_config(repo)
        registry = AgentRegistry(
            config,
            provider_factory=lambda cfg, profile: FakeProvider(_params()),
            gateway_factory=lambda cfg, cls: build_gateway(
                repo, cls.identity.permissions, agent=cls.identity.name
            ),
        )
        registry.register(WorkerAgent)

        tools = registry.get("worker").require_tools()
        assert tools.can_use("filesystem.patch")
        assert not tools.can_use("shell.run")
        assert {s.name for s in tools.available_specs()} == {
            "filesystem.read", "filesystem.search", "filesystem.write",
            "filesystem.patch", "git.status", "git.diff", "git.commit",
        }

    def test_agent_without_tool_grants_gets_no_gateway(self, repo: Path) -> None:
        """The echo canary declares no tools, so it holds no gateway at all."""
        from edith.agents.echo import EchoAgent

        config = build_config(repo)
        registry = AgentRegistry(
            config, provider_factory=lambda cfg, profile: FakeProvider(_params())
        )
        registry.register(EchoAgent)
        assert registry.get("echo").tools is None

    def test_require_tools_fails_loudly_when_absent(self) -> None:
        from edith.agents.echo import EchoAgent
        from edith.errors import AgentExecutionError

        agent = EchoAgent(provider=FakeProvider(_params()))
        with pytest.raises(AgentExecutionError, match="requires the tool gateway"):
            agent.require_tools()


def _params():
    from edith.config.schema import ModelParams

    return ModelParams(model_name="test-model:q4")
