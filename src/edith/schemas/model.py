"""Provider-neutral schemas for the model layer.

Nothing here mentions Ollama. Agents depend on these types only, which is what makes the
provider replaceable (CLAUDE.md invariant 15).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from .common import EdithModel


class Role(StrEnum):
    """Conversation role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(EdithModel):
    """A single conversation turn."""

    role: Role
    content: str


class GenerationOptions(EdithModel):
    """Per-call overrides on top of the configured :class:`ModelParams` profile.

    Every field is optional; ``None`` means "use the profile value". This keeps inference
    parameters configurable without scattering defaults through the codebase.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    context_length: int | None = Field(default=None, ge=512)


class TokenUsage(EdithModel):
    """Token accounting for one generation, when the runtime reports it."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class GenerationResult(EdithModel):
    """The result of a text generation call."""

    text: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = None
    #: Non-sensitive runtime metadata (e.g. eval rates) for observability.
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthState(StrEnum):
    """Health of a provider or dependency."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class ProviderHealth(EdithModel):
    """Structured provider health, with actionable remediation text.

    ``remediation`` exists so ``edith doctor`` can tell the user what to *do*, rather than
    only that something is broken.
    """

    provider: str
    state: HealthState
    detail: str = ""
    remediation: str | None = None
    endpoint: str | None = None
    available_models: tuple[str, ...] = ()
    configured_model: str | None = None
    configured_model_present: bool | None = None
    latency_ms: float | None = None

    @property
    def ok(self) -> bool:
        """True when the provider is fully usable."""
        return self.state is HealthState.HEALTHY
