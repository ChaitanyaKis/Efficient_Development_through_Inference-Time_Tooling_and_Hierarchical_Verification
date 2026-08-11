"""The provider registry: the seam that makes the model runtime replaceable."""

from __future__ import annotations

import pytest

from edith.config.schema import EdithConfig, ModelParams, ModelsConfig
from edith.errors import ConfigurationError
from edith.models.ollama import OllamaProvider
from edith.models.registry import available_providers, build_provider, register_provider

from .fakes import FakeProvider


class TestRegistration:
    def test_ollama_is_registered_by_default(self) -> None:
        assert "ollama" in available_providers()

    def test_key_is_normalized(self, model_params: ModelParams) -> None:
        register_provider("MyProvider", lambda cfg, p: FakeProvider(p), replace=True)
        assert "myprovider" in available_providers()

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            register_provider("   ", lambda cfg, p: FakeProvider(p))

    def test_duplicate_registration_rejected(self, model_params: ModelParams) -> None:
        register_provider("dup_probe", lambda cfg, p: FakeProvider(p), replace=True)
        with pytest.raises(ConfigurationError, match="already registered"):
            register_provider("dup_probe", lambda cfg, p: FakeProvider(p))

    def test_replace_is_explicit(self) -> None:
        register_provider("replaceable", lambda cfg, p: FakeProvider(p), replace=True)
        register_provider("replaceable", lambda cfg, p: FakeProvider(p), replace=True)


class TestBuildProvider:
    def test_builds_the_ollama_provider(self, model_params: ModelParams) -> None:
        cfg = EdithConfig(models=ModelsConfig(profiles={"default": model_params}))
        provider = build_provider(cfg)
        try:
            assert isinstance(provider, OllamaProvider)
            assert provider.params.model_name == "test-model:q4"
        finally:
            provider.close()

    def test_named_profile_is_honoured(self, model_params: ModelParams) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(
                profiles={
                    "default": model_params,
                    "fast": ModelParams(model_name="small:q4", context_length=2048),
                }
            )
        )
        provider = build_provider(cfg, "fast")
        try:
            assert provider.params.model_name == "small:q4"
            assert provider.params.context_length == 2048
        finally:
            provider.close()

    def test_unknown_provider_raises_with_available_list(
        self, model_params: ModelParams
    ) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(provider="nonexistent", profiles={"default": model_params})
        )
        with pytest.raises(ConfigurationError, match="unknown model provider"):
            build_provider(cfg)

    def test_unknown_profile_raises(self, model_params: ModelParams) -> None:
        cfg = EdithConfig(models=ModelsConfig(profiles={"default": model_params}))
        with pytest.raises(ConfigurationError, match="unknown model profile"):
            build_provider(cfg, "ghost")

    def test_provider_key_is_case_insensitive(self, model_params: ModelParams) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(provider="OLLAMA", profiles={"default": model_params})
        )
        provider = build_provider(cfg)
        provider.close()

    def test_a_fake_substitutes_for_ollama_without_other_changes(
        self, model_params: ModelParams
    ) -> None:
        """The practical test of invariant 15: the provider is replaceable by config alone."""
        register_provider("swap_probe", lambda cfg, p: FakeProvider(p), replace=True)
        cfg = EdithConfig(
            models=ModelsConfig(provider="swap_probe", profiles={"default": model_params})
        )
        provider = build_provider(cfg)
        assert isinstance(provider, FakeProvider)
        assert provider.health_check().ok
