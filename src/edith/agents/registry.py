"""Agent registry: name -> agent class, with lazy, config-driven instantiation.

The registry owns provider wiring so that agents never construct their own model client.
That keeps the provider replaceable and makes every agent trivially testable with a fake.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from edith.config.schema import EdithConfig
from edith.errors import AgentNotFoundError, AgentRegistrationError
from edith.models.base import ModelProvider
from edith.models.registry import build_provider
from edith.observability.logging import get_logger
from edith.schemas.agent import AgentHealth, AgentIdentity

from .base import Agent

logger = get_logger(__name__)

ProviderFactory = Callable[[EdithConfig, str | None], ModelProvider]


class AgentRegistry:
    """A collection of agent classes that can be instantiated on demand.

    Instances are cached per registry so a provider (and its HTTP connection pool) is not
    rebuilt on every invocation.
    """

    def __init__(
        self,
        config: EdithConfig,
        *,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        """
        Args:
            config: Resolved configuration used to wire agents.
            provider_factory: Builds a provider from (config, profile). Injected in tests to
                avoid touching a real runtime.
        """
        self._config = config
        self._provider_factory: ProviderFactory = provider_factory or (
            lambda cfg, profile: build_provider(cfg, profile)
        )
        self._classes: dict[str, type[Agent]] = {}
        self._instances: dict[str, Agent] = {}
        self._providers: dict[str, ModelProvider] = {}

    def register(self, agent_cls: type[Agent], *, replace: bool = False) -> None:
        """Register an agent class under its declared identity name."""
        # Abstractness is checked first: "X is abstract" is a more useful diagnostic than
        # the missing-identity error an abstract base would otherwise trigger.
        if getattr(agent_cls, "__abstractmethods__", None):
            raise AgentRegistrationError(
                f"{agent_cls.__name__} is abstract and cannot be registered",
                details={"class": agent_cls.__name__},
            )
        identity = getattr(agent_cls, "identity", None)
        if not isinstance(identity, AgentIdentity):
            raise AgentRegistrationError(
                f"{agent_cls.__name__} has no valid class-level `identity`",
                details={"class": agent_cls.__name__},
            )
        name = identity.name
        if name in self._classes and not replace:
            raise AgentRegistrationError(
                f"agent {name!r} is already registered by "
                f"{self._classes[name].__name__}; pass replace=True to override",
                details={"agent": name},
            )
        self._classes[name] = agent_cls
        # Drop any cached instance so a replacement takes effect immediately.
        self._instances.pop(name, None)
        logger.debug("agent.registered", agent=name, cls=agent_cls.__name__)

    def unregister(self, name: str) -> None:
        """Remove an agent and any cached instance."""
        if name not in self._classes:
            raise AgentNotFoundError(f"agent {name!r} is not registered", details={"agent": name})
        del self._classes[name]
        self._instances.pop(name, None)

    def __contains__(self, name: object) -> bool:
        return name in self._classes

    def __len__(self) -> int:
        return len(self._classes)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._classes))

    def names(self) -> tuple[str, ...]:
        """Return registered agent names, sorted."""
        return tuple(sorted(self._classes))

    def identities(self) -> tuple[AgentIdentity, ...]:
        """Return the identity of every registered agent, sorted by name."""
        return tuple(self._classes[name].identity for name in self.names())

    def get_class(self, name: str) -> type[Agent]:
        """Return the registered class for ``name``."""
        try:
            return self._classes[name]
        except KeyError as exc:
            raise AgentNotFoundError(
                f"agent {name!r} is not registered; available: {list(self.names())}",
                details={"agent": name},
            ) from exc

    def _provider_for(self, profile: str) -> ModelProvider:
        """Return a provider for ``profile``, reusing one per profile."""
        if profile not in self._providers:
            self._providers[profile] = self._provider_factory(self._config, profile)
        return self._providers[profile]

    def get(self, name: str) -> Agent:
        """Return a ready-to-use agent instance, constructing and caching it if needed."""
        cached = self._instances.get(name)
        if cached is not None:
            return cached

        agent_cls = self.get_class(name)
        settings = self._config.agents.for_agent(name)
        # Identity-level profile wins over the agents.yaml default: an agent that declares
        # it needs a specific model must not be silently downgraded by config.
        profile = agent_cls.identity.model_profile or settings.model_profile
        instance = agent_cls(provider=self._provider_for(profile), settings=settings)
        self._instances[name] = instance
        return instance

    def health_check(self) -> tuple[AgentHealth, ...]:
        """Health-check every registered agent."""
        results: list[AgentHealth] = []
        for name in self.names():
            try:
                results.append(self.get(name).health_check())
            except Exception as exc:  # noqa: BLE001 - one bad agent must not hide the rest
                results.append(
                    AgentHealth(agent=name, healthy=False, detail=f"{type(exc).__name__}: {exc}")
                )
        return tuple(results)

    def close(self) -> None:
        """Release every provider this registry created."""
        for provider in self._providers.values():
            try:
                provider.close()
            except Exception:  # noqa: BLE001 - cleanup must be best-effort
                logger.warning("provider.close_failed", provider=provider.name)
        self._providers.clear()
        self._instances.clear()


def build_default_registry(
    config: EdithConfig, *, provider_factory: ProviderFactory | None = None
) -> AgentRegistry:
    """Return a registry populated with the agents shipped in this milestone.

    M0 ships only the ``echo`` kernel self-test agent. Later milestones append their agents
    here; nothing else needs to change.
    """
    # Local import: echo imports from .base, which this module also imports, so a
    # top-level import here would make the package import graph cyclic.
    from .echo import EchoAgent  # noqa: PLC0415

    registry = AgentRegistry(config, provider_factory=provider_factory)
    registry.register(EchoAgent)
    return registry
