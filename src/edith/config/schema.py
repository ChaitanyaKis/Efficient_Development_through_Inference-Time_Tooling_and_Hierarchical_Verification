"""Typed configuration schema.

Every tunable value in Edith is declared here and loaded from ``config/*.yaml`` with
environment-variable overrides. Source code must never hard-code a model name, timeout,
context length, host, or path -- it reads them from :class:`EdithConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json"]


class StrictModel(BaseModel):
    """Base for config models: unknown keys are an error, not a silent typo."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(StrictModel):
    """Structured logging configuration."""

    level: LogLevel = "INFO"
    format: LogFormat = "console"
    file_enabled: bool = True
    file_path: Path = Path(".edith/logs/edith.jsonl")
    redact_keys: tuple[str, ...] = (
        "api_key",
        "apikey",
        "authorization",
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "private_key",
        "access_key",
        "session",
    )


class ResourceConfig(StrictModel):
    """Resource-awareness thresholds used by doctor and (later) the scheduler."""

    max_concurrent_inferences: int = Field(default=1, ge=1, le=8)
    min_free_vram_mb: int = Field(default=2048, ge=0)
    min_free_ram_mb: int = Field(default=2048, ge=0)
    min_free_disk_mb: int = Field(default=10_240, ge=0)

    @model_validator(mode="after")
    def _warn_on_parallelism(self) -> ResourceConfig:
        # Not an error -- overridable -- but the default must stay sequential on 6 GB VRAM.
        return self


class RetryConfig(StrictModel):
    """Bounded retry policy. Autonomous loops must never retry indefinitely."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    max_backoff_seconds: float = Field(default=8.0, ge=0.0, le=300.0)


class ModelParams(StrictModel):
    """Inference parameters for one named model profile."""

    model_name: str = Field(min_length=1)
    context_length: int = Field(default=8192, ge=512, le=131_072)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=2048, ge=1, le=32_768)
    seed: int | None = None
    stop: tuple[str, ...] = ()
    supports_tools: bool = False
    keep_alive: str = "5m"
    estimated_vram_mb: int = Field(default=0, ge=0)


class OllamaProviderConfig(StrictModel):
    """Connection settings for a local Ollama runtime.

    ``host`` must resolve to a loopback address unless ``allow_remote`` is explicitly set;
    this enforces the local-first invariant and prevents a config typo from shipping
    prompts to a third party.
    """

    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)
    allow_remote: bool = False

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("host must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _enforce_local(self) -> OllamaProviderConfig:
        if self.allow_remote:
            return self
        local_prefixes = (
            "http://127.0.0.1",
            "http://localhost",
            "http://[::1]",
            "https://127.0.0.1",
            "https://localhost",
        )
        if not self.host.startswith(local_prefixes):
            raise ValueError(
                f"host {self.host!r} is not loopback; set allow_remote: true to permit it. "
                "Edith is local-first and must not send prompts off-machine by accident."
            )
        return self


class ModelsConfig(StrictModel):
    """Model provider selection and the profiles agents may reference by role."""

    provider: str = "ollama"
    ollama: OllamaProviderConfig = OllamaProviderConfig()
    retry: RetryConfig = RetryConfig()
    default_profile: str = "default"
    profiles: dict[str, ModelParams]

    @model_validator(mode="after")
    def _default_profile_exists(self) -> ModelsConfig:
        if self.default_profile not in self.profiles:
            raise ValueError(
                f"default_profile {self.default_profile!r} is not defined in profiles "
                f"({sorted(self.profiles)})"
            )
        return self

    def profile(self, name: str | None = None) -> ModelParams:
        """Return a profile by name, falling back to the configured default."""
        key = name or self.default_profile
        try:
            return self.profiles[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown model profile {key!r}; available: {sorted(self.profiles)}"
            ) from exc


class AgentDefaults(StrictModel):
    """Defaults applied to every agent unless the agent overrides them."""

    model_profile: str = "default"
    max_attempts: int = Field(default=2, ge=1, le=10)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class AgentsConfig(StrictModel):
    """Agent-layer configuration.

    ``overrides`` maps agent name -> partial settings. Agent *implementations* are
    registered in code; this file only tunes them.
    """

    defaults: AgentDefaults = AgentDefaults()
    overrides: dict[str, AgentDefaults] = Field(default_factory=dict)

    def for_agent(self, name: str) -> AgentDefaults:
        """Return effective settings for ``name``."""
        return self.overrides.get(name, self.defaults)


class SystemConfig(StrictModel):
    """Top-level system settings."""

    project_name: str = "edith"
    state_dir: Path = Path(".edith")
    logging: LoggingConfig = LoggingConfig()
    resources: ResourceConfig = ResourceConfig()


class EdithConfig(StrictModel):
    """The fully-resolved configuration object passed through dependency injection."""

    system: SystemConfig = SystemConfig()
    models: ModelsConfig
    agents: AgentsConfig = AgentsConfig()

    #: Absolute directory the config was loaded from; ``None`` when built in-memory.
    config_dir: Path | None = None
