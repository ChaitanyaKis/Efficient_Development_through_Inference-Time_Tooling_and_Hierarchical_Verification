"""The Ollama provider: request shaping, error translation, and health detection.

All HTTP is mocked with respx. These tests must pass on a machine with no Ollama installed.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from edith.config.schema import ModelParams, OllamaProviderConfig
from edith.errors import (
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from edith.models.ollama import OllamaProvider, normalize_model_name
from edith.schemas.model import GenerationOptions, HealthState, Message, Role

BASE = "http://127.0.0.1:11434"
CHAT = f"{BASE}/api/chat"
TAGS = f"{BASE}/api/tags"


@pytest.fixture
def provider(
    ollama_config: OllamaProviderConfig, model_params: ModelParams
) -> OllamaProvider:
    return OllamaProvider(ollama_config, model_params)


def chat_response(content: str = "hello", **extra: object) -> dict:
    return {
        "model": "test-model:q4",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 7,
        "total_duration": 1_500_000_000,
        **extra,
    }


def prompt() -> list[Message]:
    return [Message(role=Role.USER, content="hi")]


class TestGenerate:
    @respx.mock
    def test_successful_generation(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        result = provider.generate(prompt())
        assert result.text == "hello"
        assert result.provider == "ollama"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 7
        assert result.finish_reason == "stop"

    @respx.mock
    def test_profile_parameters_are_sent(self, provider: OllamaProvider) -> None:
        """Config values must actually reach the runtime, not merely be stored."""
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        provider.generate(prompt())
        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "test-model:q4"
        assert body["options"]["num_ctx"] == 4096
        assert body["options"]["num_predict"] == 256
        assert body["stream"] is False

    @respx.mock
    def test_per_call_options_override_the_profile(self, provider: OllamaProvider) -> None:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        provider.generate(prompt(), GenerationOptions(temperature=0.9, max_output_tokens=64))
        options = json.loads(route.calls[0].request.content)["options"]
        assert options["temperature"] == 0.9
        assert options["num_predict"] == 64
        assert options["num_ctx"] == 4096  # untouched fields keep the profile value

    @respx.mock
    def test_seed_and_stop_are_forwarded(self, ollama_config: OllamaProviderConfig) -> None:
        params = ModelParams(model_name="m:q4", seed=42, stop=("END",))
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        OllamaProvider(ollama_config, params).generate(prompt())
        options = json.loads(route.calls[0].request.content)["options"]
        assert options["seed"] == 42 and options["stop"] == ["END"]

    @respx.mock
    def test_seed_omitted_when_unset(self, provider: OllamaProvider) -> None:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        provider.generate(prompt())
        assert "seed" not in json.loads(route.calls[0].request.content)["options"]

    @respx.mock
    def test_messages_are_serialized_in_order(self, provider: OllamaProvider) -> None:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        provider.generate(
            [Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="usr")]
        )
        sent = json.loads(route.calls[0].request.content)["messages"]
        assert sent == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]


class TestStructuredGeneration:
    @respx.mock
    def test_json_schema_becomes_the_format_field(self, provider: OllamaProvider) -> None:
        """Ollama's constrained decoding must be engaged, not just prompted for."""
        from pydantic import BaseModel

        class Small(BaseModel):
            value: int

        route = respx.post(CHAT).mock(
            return_value=httpx.Response(200, json=chat_response('{"value": 3}'))
        )
        result = provider.structured_generate(prompt(), Small)
        assert result.value == 3
        assert json.loads(route.calls[0].request.content)["format"]["properties"]["value"]

    @respx.mock
    def test_format_absent_for_free_text(self, provider: OllamaProvider) -> None:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=chat_response()))
        provider.generate(prompt())
        assert "format" not in json.loads(route.calls[0].request.content)


class TestErrorTranslation:
    @respx.mock
    def test_connection_refused_is_unavailable(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.generate(prompt())
        assert excinfo.value.retryable is True
        assert "winget install" in excinfo.value.message  # actionable remediation

    @respx.mock
    def test_timeout_is_classified(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderTimeoutError) as excinfo:
            provider.generate(prompt())
        assert excinfo.value.retryable is True

    @respx.mock
    def test_404_is_model_not_found_and_not_retryable(self, provider: OllamaProvider) -> None:
        """Retrying a missing model wastes the user's time; tell them to pull it."""
        respx.post(CHAT).mock(return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(ModelNotFoundError) as excinfo:
            provider.generate(prompt())
        assert excinfo.value.retryable is False
        assert "ollama pull test-model:q4" in excinfo.value.message

    @respx.mock
    def test_500_is_retryable(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(ProviderError) as excinfo:
            provider.generate(prompt())
        assert excinfo.value.retryable is True

    @respx.mock
    def test_400_is_not_retryable(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(return_value=httpx.Response(400, text="bad request"))
        with pytest.raises(ProviderError) as excinfo:
            provider.generate(prompt())
        assert excinfo.value.retryable is False

    @respx.mock
    def test_non_json_body_is_classified(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.generate(prompt())


class TestStreaming:
    @respx.mock
    def test_chunks_are_yielded(self, provider: OllamaProvider) -> None:
        frames = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]
        respx.post(CHAT).mock(
            return_value=httpx.Response(200, content="\n".join(frames).encode())
        )
        assert "".join(provider.stream(prompt())) == "Hello!"

    @respx.mock
    def test_malformed_frame_is_skipped(self, provider: OllamaProvider) -> None:
        """One bad frame must not abort a long generation."""
        frames = [
            json.dumps({"message": {"content": "a"}, "done": False}),
            "{{not json}}",
            json.dumps({"message": {"content": "b"}, "done": True}),
        ]
        respx.post(CHAT).mock(
            return_value=httpx.Response(200, content="\n".join(frames).encode())
        )
        assert "".join(provider.stream(prompt())) == "ab"

    @respx.mock
    def test_error_frame_raises(self, provider: OllamaProvider) -> None:
        respx.post(CHAT).mock(
            return_value=httpx.Response(200, content=json.dumps({"error": "oom"}).encode())
        )
        with pytest.raises(ProviderError, match="oom"):
            list(provider.stream(prompt()))

    def test_empty_messages_rejected(self, provider: OllamaProvider) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            list(provider.stream([]))

    @respx.mock
    def test_stream_flag_is_set(self, provider: OllamaProvider) -> None:
        route = respx.post(CHAT).mock(
            return_value=httpx.Response(
                200, content=json.dumps({"message": {"content": "x"}, "done": True}).encode()
            )
        )
        list(provider.stream(prompt()))
        assert json.loads(route.calls[0].request.content)["stream"] is True


class TestHealthCheck:
    @respx.mock
    def test_healthy_when_model_present(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "test-model:q4"}]})
        )
        health = provider.health_check()
        assert health.state is HealthState.HEALTHY
        assert health.ok and health.configured_model_present is True
        assert health.latency_ms is not None

    @respx.mock
    def test_degraded_when_model_missing(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "other:q4"}]})
        )
        health = provider.health_check()
        assert health.state is HealthState.DEGRADED
        assert health.configured_model_present is False
        assert health.remediation is not None
        assert "ollama pull test-model:q4" in health.remediation

    @respx.mock
    def test_unavailable_when_server_down(self, provider: OllamaProvider) -> None:
        """The commonest real failure: this must never raise, only report."""
        respx.get(TAGS).mock(side_effect=httpx.ConnectError("refused"))
        health = provider.health_check()
        assert health.state is HealthState.UNAVAILABLE
        assert not health.ok
        assert health.remediation is not None and "Install Ollama" in health.remediation

    @respx.mock
    def test_unavailable_on_timeout(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(side_effect=httpx.ConnectTimeout("slow"))
        assert provider.health_check().state is HealthState.UNAVAILABLE

    @respx.mock
    def test_unavailable_on_server_error(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(return_value=httpx.Response(500, text="boom"))
        assert provider.health_check().state is HealthState.UNAVAILABLE

    @respx.mock
    def test_latest_tag_normalization(self, ollama_config: OllamaProviderConfig) -> None:
        """`foo` configured against a runtime reporting `foo:latest` must match."""
        provider = OllamaProvider(ollama_config, ModelParams(model_name="qwen"))
        respx.get(TAGS).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen:latest"}]})
        )
        assert provider.health_check().state is HealthState.HEALTHY


class TestListModels:
    @respx.mock
    def test_returns_names(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
        )
        assert provider.list_models() == ("a", "b")

    @respx.mock
    def test_empty_runtime(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(return_value=httpx.Response(200, json={"models": []}))
        assert provider.list_models() == ()

    @respx.mock
    def test_missing_key_tolerated(self, provider: OllamaProvider) -> None:
        respx.get(TAGS).mock(return_value=httpx.Response(200, json={}))
        assert provider.list_models() == ()


class TestNormalizeModelName:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [("qwen", "qwen:latest"), ("qwen:3b", "qwen:3b"), ("a/b:c", "a/b:c")],
    )
    def test_normalization(self, given: str, expected: str) -> None:
        assert normalize_model_name(given) == expected


class TestClientOwnership:
    def test_injected_client_is_not_closed(
        self, ollama_config: OllamaProviderConfig, model_params: ModelParams
    ) -> None:
        """The caller owns an injected client; closing it would surprise them."""
        client = httpx.Client(base_url=BASE)
        OllamaProvider(ollama_config, model_params, client=client).close()
        assert not client.is_closed
        client.close()

    def test_owned_client_is_closed(
        self, ollama_config: OllamaProviderConfig, model_params: ModelParams
    ) -> None:
        provider = OllamaProvider(ollama_config, model_params)
        provider.close()
        assert provider._client.is_closed
