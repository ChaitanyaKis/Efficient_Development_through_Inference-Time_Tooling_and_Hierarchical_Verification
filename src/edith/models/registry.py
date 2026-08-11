"""Provider registry and factory.

Adding a future runtime (llama.cpp server, vLLM, LM Studio) means registering a builder
here. No agent, CLI command, or test changes.
"""

from __future__ import annotations

from collections.abc import Callable

from edith.config.schema import EdithConfig, ModelParams
from edith.errors import ConfigurationError

from .base import ModelProvider
from .ollama import PROVIDER_NAME as OLLAMA
from .ollama import OllamaProvider

#: provider key -> builder taking the full config and a resolved model profile.
ProviderBuilder = Callable[[EdithConfig, ModelParams], ModelProvider]

_BUILDERS: dict[str, ProviderBuilder] = {}


def register_provider(key: str, builder: ProviderBuilder, *, replace: bool = False) -> None:
    """Register a provider builder under ``key``."""
    normalized = key.strip().lower()
    if not normalized:
        raise ValueError("provider key must not be empty")
    if normalized in _BUILDERS and not replace:
        raise ConfigurationError(f"provider {normalized!r} is already registered")
    _BUILDERS[normalized] = builder


def available_providers() -> tuple[str, ...]:
    """Return the registered provider keys."""
    return tuple(sorted(_BUILDERS))


def build_provider(config: EdithConfig, profile: str | None = None) -> ModelProvider:
    """Construct the configured provider for a model profile.

    Args:
        config: Fully-resolved configuration.
        profile: Profile key from ``models.yaml``; ``None`` uses ``default_profile``.

    Raises:
        ConfigurationError: The provider key or profile name is unknown.
    """
    key = config.models.provider.strip().lower()
    builder = _BUILDERS.get(key)
    if builder is None:
        raise ConfigurationError(
            f"unknown model provider {key!r}; available: {list(available_providers())}",
            details={"provider": key},
        )
    try:
        params = config.models.profile(profile)
    except KeyError as exc:
        raise ConfigurationError(str(exc), details={"profile": profile}) from exc
    return builder(config, params)


def _build_ollama(config: EdithConfig, params: ModelParams) -> ModelProvider:
    return OllamaProvider(config.models.ollama, params)


register_provider(OLLAMA, _build_ollama)
