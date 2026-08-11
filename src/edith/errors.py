"""Exception hierarchy and failure classification for Edith.

Every failure that crosses a subsystem boundary is an :class:`EdithError` carrying a
:class:`FailureCategory`. The categories are the vocabulary the orchestrator (M2) will use
to decide whether to retry, escalate, or abort, so they are defined here in M0 rather than
being invented ad hoc later.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class FailureCategory(StrEnum):
    """Classification of a failure. Mirrors the taxonomy in CLAUDE.md."""

    MODEL_ERROR = "MODEL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    BUILD_ERROR = "BUILD_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    REQUIREMENT_FAILURE = "REQUIREMENT_FAILURE"
    ARCHITECTURE_FAILURE = "ARCHITECTURE_FAILURE"
    SECURITY_FAILURE = "SECURITY_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class EdithError(Exception):
    """Base class for all Edith errors.

    Args:
        message: Human-readable description. Must not contain secrets.
        category: Failure classification used by retry/escalation policy.
        retryable: Whether retrying the identical operation could plausibly succeed.
        details: Structured, non-sensitive context for logs and failure reports.
    """

    category: FailureCategory = FailureCategory.UNKNOWN
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category or self.category
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a structured, log-safe representation."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "category": str(self.category),
            "retryable": self.retryable,
            "details": self.details,
        }

    def __str__(self) -> str:
        return self.message


class ConfigurationError(EdithError):
    """Configuration is missing, malformed, or fails validation."""

    category = FailureCategory.CONFIGURATION_ERROR


class ProviderError(EdithError):
    """Base class for model-provider failures."""

    category = FailureCategory.MODEL_ERROR


class ProviderUnavailableError(ProviderError):
    """The model runtime is unreachable (not installed, not running, network refused)."""

    category = FailureCategory.ENVIRONMENT_FAILURE
    default_retryable = True


class ModelNotFoundError(ProviderError):
    """The runtime is reachable but the configured model is not present."""

    category = FailureCategory.ENVIRONMENT_FAILURE
    default_retryable = False


class ProviderTimeoutError(ProviderError):
    """The model runtime did not respond within the configured timeout."""

    category = FailureCategory.TIMEOUT
    default_retryable = True


class StructuredOutputError(ProviderError):
    """The model produced output that could not be validated against the target schema."""

    category = FailureCategory.VALIDATION_FAILURE
    default_retryable = True


class AgentError(EdithError):
    """Base class for agent-level failures."""

    category = FailureCategory.UNKNOWN


class AgentNotFoundError(AgentError):
    """No agent is registered under the requested name."""

    category = FailureCategory.CONFIGURATION_ERROR


class AgentRegistrationError(AgentError):
    """An agent could not be registered (duplicate name, invalid definition)."""

    category = FailureCategory.CONFIGURATION_ERROR


class AgentValidationError(AgentError):
    """Agent input or output failed schema validation."""

    category = FailureCategory.VALIDATION_FAILURE


class AgentExecutionError(AgentError):
    """The agent's execute() raised or returned an unusable result."""

    category = FailureCategory.UNKNOWN
