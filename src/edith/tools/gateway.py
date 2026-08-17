"""The Tool Gateway -- the single controlled entry point to every tool.

::

    Agent -> ToolGateway -> PermissionEngine -> Tool -> Workspace -> filesystem/process

Agents never hold a :class:`~edith.tools.base.Tool` instance. They hold a gateway bound to
their own permissions, so there is no path from agent code to an unauthorized operation.

The gateway owns what must never be skipped: permission checks, argument validation, audit
logging, and turning every outcome -- success, denial, timeout, crash -- into a structured
:class:`~edith.tools.schemas.ToolResult`. :meth:`ToolGateway.execute` never raises.
"""

from __future__ import annotations

import time

from edith.config.schema import EdithConfig
from edith.errors import (
    EdithError,
    FailureCategory,
    PermissionDeniedError,
    ToolError,
)
from edith.observability.logging import bind_context, clear_context, get_logger
from edith.schemas.agent import AgentPermissions

from .base import ToolContext
from .paths import PathPolicy
from .permissions import PermissionEngine
from .registry import ToolRegistry, build_default_registry
from .schemas import ToolCall, ToolResult, ToolSpec
from .workspace import Workspace

logger = get_logger(__name__)


class ToolGateway:
    """Executes tool calls on behalf of an agent, under that agent's permissions."""

    def __init__(
        self,
        config: EdithConfig,
        permissions: AgentPermissions,
        *,
        registry: ToolRegistry | None = None,
        agent: str | None = None,
        policy: PathPolicy | None = None,
    ) -> None:
        """
        Args:
            config: Resolved Edith configuration.
            permissions: The calling agent's granted scope. This is the *only* place a
                permission set enters the tool layer.
            registry: Tool registry; defaults to the tools shipped in this milestone.
            agent: Agent name, recorded in the audit log and commit attribution.
            policy: Pre-built path policy, mainly for tests that point at a temp workspace.
        """
        self.config = config
        self.permissions = permissions
        self.agent = agent
        self.registry = registry or build_default_registry()
        self.policy = policy or PathPolicy.create(
            config.tools.workspace_root, config.tools.paths
        )
        self.engine = PermissionEngine(permissions)
        self.workspace = Workspace(policy=self.policy, engine=self.engine)

    # -- Introspection -------------------------------------------------------------

    def available_specs(self) -> tuple[ToolSpec, ...]:
        """Return the specs of tools this agent is actually permitted to use."""
        return tuple(
            spec for spec in self.registry.specs() if self.engine.may_use_tool(spec.name)
        )

    def can_use(self, name: str) -> bool:
        """Whether the agent may use the named tool, without invoking it."""
        return name in self.registry and self.engine.may_use_tool(name)

    # -- Execution -----------------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool call and return a structured result.

        Never raises. Every failure mode -- unknown tool, denied permission, invalid
        arguments, timeout, unexpected exception -- becomes a ``ToolResult`` with
        ``ok=False`` and a :class:`~edith.errors.FailureCategory`.
        """
        started = time.monotonic()
        agent = call.agent or self.agent
        bind_context(tool=call.tool, call_id=call.call_id, agent=agent, task_id=call.task_id)
        try:
            tool = self.registry.get(call.tool)
            self.engine.authorize_tool(tool.spec, agent)

            logger.info(
                "tool.start",
                tool=call.tool,
                writes=tool.spec.writes,
                spawns_process=tool.spec.spawns_process,
            )

            context = ToolContext(
                workspace=self.workspace,
                config=self.config.tools,
                call_id=call.call_id,
                agent=agent,
                timeout_seconds=call.timeout_seconds,
            )
            output = tool.run(call.arguments, context)
            duration = time.monotonic() - started

            logger.info(
                "tool.success", tool=call.tool, duration_seconds=round(duration, 3)
            )
            return ToolResult(
                call_id=call.call_id,
                tool=call.tool,
                ok=True,
                output=output.model_dump(mode="json"),
                duration_seconds=duration,
                evidence={"agent": agent} if agent else {},
            )

        except PermissionDeniedError as exc:
            # A denial is a policy event. It is logged at WARNING with the reason so that a
            # misconfigured agent is visible, and it is never retried.
            return self._failure(call, exc, started, agent, level="denied")
        except (ToolError, EdithError) as exc:
            return self._failure(call, exc, started, agent)
        except Exception as exc:  # noqa: BLE001 - a tool must not crash its caller
            wrapped = ToolError(
                f"tool {call.tool!r} raised {type(exc).__name__}: {exc}",
                category=FailureCategory.TOOL_ERROR,
            )
            return self._failure(call, wrapped, started, agent)
        finally:
            clear_context()

    def _failure(
        self,
        call: ToolCall,
        error: EdithError,
        started: float,
        agent: str | None,
        *,
        level: str = "error",
    ) -> ToolResult:
        """Build and log a structured failure. Failures are never hidden."""
        duration = time.monotonic() - started
        event = "tool.denied" if level == "denied" else "tool.failure"
        logger.warning(
            event,
            tool=call.tool,
            agent=agent,
            category=str(error.category),
            error=error.message,
            duration_seconds=round(duration, 3),
        )
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            ok=False,
            error=error.message,
            failure_category=error.category,
            duration_seconds=duration,
            evidence={"details": error.details} if error.details else {},
        )

    def for_agent(self, permissions: AgentPermissions, agent: str) -> ToolGateway:
        """Return a gateway bound to a different agent, reusing this one's registry/policy.

        Cheap enough to build per agent, and it keeps one registry and one resolved
        workspace root shared across the process.
        """
        return ToolGateway(
            self.config,
            permissions,
            registry=self.registry,
            agent=agent,
            policy=self.policy,
        )
