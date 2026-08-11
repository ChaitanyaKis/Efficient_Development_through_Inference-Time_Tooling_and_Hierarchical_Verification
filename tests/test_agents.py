"""Agent contract, lifecycle, and registry."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import BaseModel

from edith.agents.base import Agent
from edith.agents.echo import EchoAgent, EchoInput, EchoOutput
from edith.agents.registry import AgentRegistry, build_default_registry
from edith.config.schema import AgentDefaults, EdithConfig, ModelParams
from edith.errors import (
    AgentNotFoundError,
    AgentRegistrationError,
    EdithError,
    FailureCategory,
    ProviderUnavailableError,
)
from edith.models.base import ModelProvider
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    AgentStatus,
    Capability,
)
from edith.schemas.common import EdithModel

from .fakes import FakeProvider


class SimpleInput(EdithModel):
    value: int


class SimpleOutput(EdithModel):
    doubled: int


class SimpleAgent(Agent):
    """Deterministic agent needing no model, for lifecycle testing."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="simple", description="Doubles a number", permissions=AgentPermissions()
    )
    input_schema: ClassVar[type[BaseModel]] = SimpleInput
    output_schema: ClassVar[type[BaseModel]] = SimpleOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, SimpleInput)
        return SimpleOutput(doubled=payload.value * 2)


class DictReturningAgent(SimpleAgent):
    """Returns a mapping instead of a model; it must still be validated."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="dict_agent", description="Returns a dict"
    )

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, SimpleInput)
        return {"doubled": payload.value * 2}  # type: ignore[return-value]


class BadOutputAgent(SimpleAgent):
    """Returns output that violates its own declared schema."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="bad_output", description="Returns the wrong shape"
    )

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        return {"wrong_field": "x"}  # type: ignore[return-value]


class RaisingAgent(SimpleAgent):
    """Raises a plain Python exception."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="raising", description="Explodes"
    )

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        raise RuntimeError("unexpected internal explosion")


class EdithRaisingAgent(SimpleAgent):
    """Raises a classified Edith error."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="edith_raising", description="Raises a classified error"
    )

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        raise ProviderUnavailableError("ollama is down")


class NeedsProviderAgent(SimpleAgent):
    """Requires a provider that was never injected."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="needs_provider", description="Requires a model"
    )

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        self.require_provider()
        return SimpleOutput(doubled=0)


class TestAgentDefinition:
    def test_incomplete_agent_rejected_at_class_definition(self) -> None:
        """Catch a malformed agent at import time, not at first invocation."""
        with pytest.raises(TypeError, match="must define a class-level"):

            class Incomplete(Agent):  # type: ignore[misc]
                identity = AgentIdentity(name="incomplete", description="d")

                def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
                    return SimpleOutput(doubled=0)

    def test_repr_is_informative(self) -> None:
        assert "simple" in repr(SimpleAgent())


class TestAgentLifecycle:
    def test_successful_execution(self) -> None:
        response = SimpleAgent().execute(AgentRequest(payload={"value": 21}))
        assert response.ok
        assert response.status is AgentStatus.SUCCESS
        assert response.output == {"doubled": 42}
        assert response.failure_category is None
        assert response.duration_seconds >= 0.0

    def test_response_echoes_request_id(self) -> None:
        request = AgentRequest(payload={"value": 1})
        assert SimpleAgent().execute(request).request_id == request.request_id

    def test_invalid_input_is_rejected_not_failed(self) -> None:
        """A caller error is REJECTED; an execution error is FAILURE. The distinction
        tells the orchestrator whether retrying could ever help."""
        response = SimpleAgent().execute(AgentRequest(payload={"value": "not a number"}))
        assert response.status is AgentStatus.REJECTED
        assert response.failure_category is FailureCategory.VALIDATION_FAILURE
        assert response.error is not None

    def test_missing_required_input_rejected(self) -> None:
        assert SimpleAgent().execute(AgentRequest(payload={})).status is AgentStatus.REJECTED

    def test_extra_input_field_rejected(self) -> None:
        response = SimpleAgent().execute(AgentRequest(payload={"value": 1, "extra": 2}))
        assert response.status is AgentStatus.REJECTED

    def test_dict_output_is_coerced_and_validated(self) -> None:
        response = DictReturningAgent().execute(AgentRequest(payload={"value": 3}))
        assert response.ok and response.output == {"doubled": 6}

    def test_bad_output_is_caught_by_the_lifecycle(self) -> None:
        """An agent cannot emit unvalidated output; the gate is not its own code."""
        response = BadOutputAgent().execute(AgentRequest(payload={"value": 1}))
        assert response.status is AgentStatus.REJECTED
        assert response.failure_category is FailureCategory.VALIDATION_FAILURE

    def test_unexpected_exception_does_not_escape(self) -> None:
        response = RaisingAgent().execute(AgentRequest(payload={"value": 1}))
        assert response.status is AgentStatus.FAILURE
        assert response.failure_category is FailureCategory.UNKNOWN
        assert "unexpected internal explosion" in (response.error or "")

    def test_classified_error_preserves_its_category(self) -> None:
        response = EdithRaisingAgent().execute(AgentRequest(payload={"value": 1}))
        assert response.status is AgentStatus.FAILURE
        assert response.failure_category is FailureCategory.ENVIRONMENT_FAILURE

    def test_missing_provider_is_a_configuration_error(self) -> None:
        response = NeedsProviderAgent().execute(AgentRequest(payload={"value": 1}))
        assert response.failure_category is FailureCategory.CONFIGURATION_ERROR

    def test_execute_never_raises(self) -> None:
        """Whatever happens, the caller receives a structured response."""
        for agent_cls in (RaisingAgent, BadOutputAgent, EdithRaisingAgent, NeedsProviderAgent):
            response = agent_cls().execute(AgentRequest(payload={"value": 1}))
            assert response.error is not None and not response.ok

    def test_model_name_recorded_when_provider_present(
        self, model_params: ModelParams
    ) -> None:
        agent = SimpleAgent(provider=FakeProvider(model_params))
        assert agent.execute(AgentRequest(payload={"value": 1})).model == "test-model:q4"


class FlakyAgent(SimpleAgent):
    """Fails with a retryable error a configurable number of times, then succeeds."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="flaky", description="Fails then recovers"
    )
    failures_remaining: ClassVar[int] = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.remaining = type(self).failures_remaining
        self.runs = 0

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        self.runs += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ProviderUnavailableError("transient outage")
        assert isinstance(payload, SimpleInput)
        return SimpleOutput(doubled=payload.value * 2)


class TestBoundedRetry:
    """CLAUDE.md invariant 11: autonomous loops must be bounded."""

    def _agent(self, failures: int, max_attempts: int) -> FlakyAgent:
        agent = FlakyAgent(settings=AgentDefaults(max_attempts=max_attempts))
        agent.remaining = failures
        return agent

    def test_transient_failure_is_retried(self) -> None:
        agent = self._agent(failures=1, max_attempts=3)
        response = agent.execute(AgentRequest(payload={"value": 5}))
        assert response.ok and response.output == {"doubled": 10}
        assert response.attempts == 2 and agent.runs == 2

    def test_attempts_are_capped_by_config(self) -> None:
        agent = self._agent(failures=99, max_attempts=2)
        response = agent.execute(AgentRequest(payload={"value": 5}))
        assert not response.ok
        assert response.attempts == 2 and agent.runs == 2

    def test_single_attempt_configuration_does_not_retry(self) -> None:
        agent = self._agent(failures=1, max_attempts=1)
        response = agent.execute(AgentRequest(payload={"value": 5}))
        assert not response.ok and agent.runs == 1

    def test_successful_run_reports_one_attempt(self) -> None:
        response = SimpleAgent().execute(AgentRequest(payload={"value": 1}))
        assert response.attempts == 1

    def test_non_retryable_failure_is_not_retried(self) -> None:
        """Re-running a bad output cannot help; retrying it only burns inference budget."""
        agent = BadOutputAgent(settings=AgentDefaults(max_attempts=3))
        response = agent.execute(AgentRequest(payload={"value": 1}))
        assert response.status is AgentStatus.REJECTED and response.attempts == 1

    def test_invalid_input_is_validated_once(self) -> None:
        agent = SimpleAgent(settings=AgentDefaults(max_attempts=3))
        response = agent.execute(AgentRequest(payload={"value": "nope"}))
        assert response.status is AgentStatus.REJECTED and response.attempts == 0


class TestAgentHealth:
    def test_healthy_without_provider(self) -> None:
        health = SimpleAgent().health_check()
        assert health.healthy and health.provider_state is None

    def test_reflects_provider_health(self, model_params: ModelParams) -> None:
        from edith.schemas.model import HealthState, ProviderHealth

        unhealthy = ProviderHealth(
            provider="fake", state=HealthState.UNAVAILABLE, detail="down"
        )
        agent = SimpleAgent(provider=FakeProvider(model_params, health=unhealthy))
        health = agent.health_check()
        assert not health.healthy and health.provider_state == "UNAVAILABLE"


class TestEchoAgent:
    def test_identity_is_least_privilege(self) -> None:
        """The canary needs no privileges; it must not quietly acquire any."""
        permissions = EchoAgent.identity.permissions
        assert permissions.read_only
        assert not permissions.allowed_tools
        assert not permissions.network_access
        assert Capability.SELF_TEST in EchoAgent.identity.capabilities

    def test_end_to_end_with_fake_provider(self, fake_provider: FakeProvider) -> None:
        agent = EchoAgent(provider=fake_provider)
        response = agent.execute(AgentRequest(payload={"statement": "Edith runs locally."}))
        assert response.ok
        assert response.output["summary"] == "a summary"
        assert response.output["confidence"] == 0.9

    def test_statement_reaches_the_prompt(self, fake_provider: FakeProvider) -> None:
        EchoAgent(provider=fake_provider).execute(
            AgentRequest(payload={"statement": "UNIQUE_MARKER_TEXT"})
        )
        prompt = fake_provider.calls[0]["messages"][-1][1]
        assert "UNIQUE_MARKER_TEXT" in prompt

    def test_empty_statement_rejected(self, fake_provider: FakeProvider) -> None:
        response = EchoAgent(provider=fake_provider).execute(
            AgentRequest(payload={"statement": ""})
        )
        assert response.status is AgentStatus.REJECTED

    def test_model_returning_junk_is_a_failure(self, model_params: ModelParams) -> None:
        provider = FakeProvider(model_params, ["I cannot help with that."])
        response = EchoAgent(provider=provider).execute(
            AgentRequest(payload={"statement": "x"})
        )
        assert response.status is AgentStatus.FAILURE
        assert response.failure_category is FailureCategory.VALIDATION_FAILURE

    def test_out_of_range_confidence_is_repaired_then_accepted(
        self, model_params: ModelParams
    ) -> None:
        bad = json.dumps({"summary": "s", "keywords": [], "confidence": 5.0})
        good = json.dumps({"summary": "s", "keywords": [], "confidence": 0.5})
        provider = FakeProvider(model_params, [bad, good])
        response = EchoAgent(provider=provider).execute(
            AgentRequest(payload={"statement": "x"})
        )
        assert response.ok and len(provider.calls) == 2

    def test_output_schema_bounds(self) -> None:
        with pytest.raises(ValueError):
            EchoOutput(summary="s", confidence=1.5)
        with pytest.raises(ValueError):
            EchoInput(statement="x", max_keywords=99)


class TestRegistry:
    @pytest.fixture
    def registry(self, config: EdithConfig, model_params: ModelParams) -> AgentRegistry:
        return AgentRegistry(
            config, provider_factory=lambda cfg, profile: FakeProvider(model_params)
        )

    def test_register_and_retrieve(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        assert "simple" in registry
        assert len(registry) == 1
        assert isinstance(registry.get("simple"), SimpleAgent)

    def test_instances_are_cached(self, registry: AgentRegistry) -> None:
        """Rebuilding a provider per call would thrash the HTTP pool."""
        registry.register(SimpleAgent)
        assert registry.get("simple") is registry.get("simple")

    def test_duplicate_registration_rejected(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        with pytest.raises(AgentRegistrationError, match="already registered"):
            registry.register(SimpleAgent)

    def test_replace_is_explicit(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        registry.register(SimpleAgent, replace=True)
        assert len(registry) == 1

    def test_replacement_invalidates_the_cached_instance(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        first = registry.get("simple")
        registry.register(SimpleAgent, replace=True)
        assert registry.get("simple") is not first

    def test_unknown_agent_raises(self, registry: AgentRegistry) -> None:
        with pytest.raises(AgentNotFoundError, match="not registered"):
            registry.get("nonexistent")

    def test_abstract_class_rejected(self, registry: AgentRegistry) -> None:
        with pytest.raises(AgentRegistrationError, match="abstract"):
            registry.register(Agent)  # type: ignore[arg-type]

    def test_class_without_identity_rejected(self, registry: AgentRegistry) -> None:
        class NoIdentity:
            pass

        with pytest.raises(AgentRegistrationError, match="no valid class-level"):
            registry.register(NoIdentity)  # type: ignore[arg-type]

    def test_unregister(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        registry.unregister("simple")
        assert "simple" not in registry

    def test_unregister_unknown_raises(self, registry: AgentRegistry) -> None:
        with pytest.raises(AgentNotFoundError):
            registry.unregister("ghost")

    def test_names_are_sorted(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        registry.register(RaisingAgent)
        assert registry.names() == ("raising", "simple")

    def test_identities_exposed(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        assert registry.identities()[0].name == "simple"

    def test_settings_are_injected_from_config(self, model_params: ModelParams) -> None:
        from edith.config.schema import AgentsConfig, ModelsConfig

        cfg = EdithConfig(
            models=ModelsConfig(profiles={"default": model_params}),
            agents=AgentsConfig(
                defaults=AgentDefaults(max_attempts=5),
                overrides={"simple": AgentDefaults(max_attempts=1)},
            ),
        )
        registry = AgentRegistry(cfg, provider_factory=lambda c, p: FakeProvider(model_params))
        registry.register(SimpleAgent)
        assert registry.get("simple").settings.max_attempts == 1

    def test_identity_profile_beats_config_default(self, model_params: ModelParams) -> None:
        """An agent that declares a required model must not be silently downgraded."""
        requested: list[str | None] = []

        class PinnedAgent(SimpleAgent):
            identity: ClassVar[AgentIdentity] = AgentIdentity(
                name="pinned", description="Pinned to a profile", model_profile="fast"
            )

        from edith.config.schema import ModelsConfig

        cfg = EdithConfig(
            models=ModelsConfig(
                profiles={"default": model_params, "fast": ModelParams(model_name="small")}
            )
        )

        def factory(c: EdithConfig, profile: str | None) -> ModelProvider:
            requested.append(profile)
            return FakeProvider(model_params)

        registry = AgentRegistry(cfg, provider_factory=factory)
        registry.register(PinnedAgent)
        registry.get("pinned")
        assert requested == ["fast"]

    def test_providers_are_reused_per_profile(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        registry.register(RaisingAgent)
        assert registry.get("simple").provider is registry.get("raising").provider

    def test_close_releases_providers(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        provider = registry.get("simple").provider
        registry.close()
        assert isinstance(provider, FakeProvider) and provider.closed
        assert len(registry) == 1  # classes survive; instances do not

    def test_health_check_covers_every_agent(self, registry: AgentRegistry) -> None:
        registry.register(SimpleAgent)
        registry.register(RaisingAgent)
        assert {h.agent for h in registry.health_check()} == {"simple", "raising"}

    def test_health_check_survives_a_broken_agent(self, config: EdithConfig) -> None:
        """One agent that cannot be built must not hide the health of the others."""

        def exploding_factory(cfg: EdithConfig, profile: str | None) -> ModelProvider:
            raise EdithError("provider construction failed")

        registry = AgentRegistry(config, provider_factory=exploding_factory)
        registry.register(SimpleAgent)
        results = registry.health_check()
        assert len(results) == 1 and not results[0].healthy


class TestDefaultRegistry:
    def test_ships_the_echo_agent(self, config: EdithConfig, model_params: ModelParams) -> None:
        registry = build_default_registry(
            config, provider_factory=lambda c, p: FakeProvider(model_params)
        )
        assert registry.names() == ("echo",)

    def test_scope_is_the_milestone(
        self, config: EdithConfig, model_params: ModelParams
    ) -> None:
        """M0 must not ship half-built future agents (STEP 7)."""
        registry = build_default_registry(
            config, provider_factory=lambda c, p: FakeProvider(model_params)
        )
        for future in ("planner", "coder", "critic", "memory", "context", "debugger"):
            assert future not in registry
