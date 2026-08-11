"""Live integration tests against a real Ollama runtime and a real model.

These are the tests that actually satisfy the M0 acceptance criterion: a local model called
through our abstraction returns a *validated structured result*. Everything else in the
suite is hermetic and would pass on a machine with no model at all.

Skipped automatically when the runtime or the configured model is unavailable, so a
developer without the weights still gets a green suite -- but the skip is visible, never
silently reported as a pass.

Run explicitly with::

    pytest -m integration -v
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from edith.agents.echo import EchoOutput
from edith.agents.registry import build_default_registry
from edith.config.loader import load_config
from edith.config.schema import EdithConfig
from edith.models.base import ModelProvider
from edith.models.registry import build_provider
from edith.schemas.agent import AgentRequest
from edith.schemas.model import GenerationOptions, HealthState, Message, Role

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_config() -> EdithConfig:
    """The real shipped configuration."""
    return load_config(Path(__file__).resolve().parents[1] / "config")


@pytest.fixture(scope="module")
def live_provider(live_config: EdithConfig) -> Iterator[ModelProvider]:
    """A provider connected to the real runtime, skipping if it is not usable."""
    provider = build_provider(live_config)
    health = provider.health_check()
    if health.state is HealthState.UNAVAILABLE:
        provider.close()
        pytest.skip(f"Ollama unavailable: {health.detail}")
    if health.state is HealthState.DEGRADED:
        provider.close()
        pytest.skip(f"model not pulled: {health.detail}")
    yield provider
    provider.close()


class TestLiveProvider:
    def test_health_check_reports_healthy(self, live_provider: ModelProvider) -> None:
        health = live_provider.health_check()
        assert health.ok
        assert health.configured_model_present is True
        assert health.available_models

    def test_free_text_generation(self, live_provider: ModelProvider) -> None:
        result = live_provider.generate(
            [Message(role=Role.USER, content="Reply with exactly the word: ready")],
            GenerationOptions(max_output_tokens=16, temperature=0.0),
        )
        assert result.text.strip()
        assert result.usage.completion_tokens > 0
        assert result.provider == "ollama"

    def test_streaming_yields_chunks(self, live_provider: ModelProvider) -> None:
        chunks = list(
            live_provider.stream(
                [Message(role=Role.USER, content="Count: one two three.")],
                GenerationOptions(max_output_tokens=32, temperature=0.0),
            )
        )
        assert chunks and "".join(chunks).strip()

    def test_structured_generate_returns_a_validated_model(
        self, live_provider: ModelProvider
    ) -> None:
        """M0 ACCEPTANCE: the model's output is a validated Pydantic object, not text."""
        result = live_provider.structured_generate(
            [
                Message(role=Role.SYSTEM, content="You return structured analysis."),
                Message(
                    role=Role.USER,
                    content="Analyse: 'Edith runs local models on a 6 GB GPU.' "
                    "Give a summary, keywords, and confidence between 0 and 1.",
                ),
            ],
            EchoOutput,
        )
        assert isinstance(result, EchoOutput)
        assert result.summary.strip()
        assert 0.0 <= result.confidence <= 1.0


class TestLiveAgent:
    def test_echo_agent_end_to_end(
        self, live_config: EdithConfig, live_provider: ModelProvider
    ) -> None:
        """The full kernel path against real hardware."""
        registry = build_default_registry(live_config)
        try:
            response = registry.get("echo").execute(
                AgentRequest(
                    payload={
                        "statement": "Edith is a local-first agent platform with no API costs."
                    }
                )
            )
        finally:
            registry.close()

        assert response.ok, f"{response.failure_category}: {response.error}"
        assert response.output["summary"].strip()
        assert 0.0 <= response.output["confidence"] <= 1.0
        assert response.model == live_config.models.profile().model_name
        assert response.duration_seconds > 0
