"""Structured types for the tool layer.

Every tool invocation is a :class:`ToolCall` and every outcome is a :class:`ToolResult` --
success and failure alike. A tool never returns bare text and never signals failure by
raising into the agent, so the orchestrator (M2) can reason about tool outcomes uniformly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from edith.errors import FailureCategory
from edith.schemas.common import EdithModel, new_id


class AccessMode(StrEnum):
    """The kind of filesystem access an operation needs."""

    READ = "READ"
    WRITE = "WRITE"


class ToolCall(EdithModel):
    """A request to execute one tool."""

    call_id: str = Field(default_factory=lambda: new_id("call"))
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Trace linkage, populated by the orchestrator in M2.
    agent: str | None = None
    task_id: str | None = None
    #: Per-call override of the tool's timeout; ``None`` uses the configured default.
    timeout_seconds: float | None = Field(default=None, gt=0.0)


class ToolResult(EdithModel):
    """The structured outcome of a tool call.

    ``ok`` distinguishes success from failure. On failure, ``error`` and ``failure_category``
    are always populated -- there is no silent or ambiguous outcome.
    """

    call_id: str
    tool: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_category: FailureCategory | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    #: Non-sensitive execution evidence (paths touched, exit code, bytes read).
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def denied(self) -> bool:
        """True when the call was refused by policy rather than failing during execution."""
        return self.failure_category is FailureCategory.SECURITY_FAILURE


class ToolSpec(EdithModel):
    """Static declaration of what a tool is and what it may do.

    ``access`` is advisory metadata for operators and the permission report; the binding
    enforcement happens when the tool calls ``Workspace.resolve_read``/``resolve_write``,
    which cannot be bypassed.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    description: str = Field(min_length=1)
    access: frozenset[AccessMode] = frozenset({AccessMode.READ})
    #: Whether this tool starts a child process. Used by the audit log and by operators
    #: reviewing which tools carry execution risk.
    spawns_process: bool = False
    #: Whether this tool can reach the network. Gated by ``AgentPermissions.network_access``.
    uses_network: bool = False

    @property
    def writes(self) -> bool:
        """True when the tool can modify the workspace."""
        return AccessMode.WRITE in self.access


class TruncatedText(EdithModel):
    """Captured output that may have been cut to respect a size limit.

    Truncation is reported explicitly rather than silently, so an agent reasoning over the
    text knows it is incomplete.
    """

    text: str = ""
    truncated: bool = False
    original_bytes: int = Field(default=0, ge=0)


def truncate(raw: str, limit_bytes: int) -> TruncatedText:
    """Cut ``raw`` to ``limit_bytes`` UTF-8 bytes without splitting a character."""
    encoded = raw.encode("utf-8", errors="replace")
    if len(encoded) <= limit_bytes:
        return TruncatedText(text=raw, truncated=False, original_bytes=len(encoded))
    clipped = encoded[:limit_bytes].decode("utf-8", errors="ignore")
    return TruncatedText(text=clipped, truncated=True, original_bytes=len(encoded))
