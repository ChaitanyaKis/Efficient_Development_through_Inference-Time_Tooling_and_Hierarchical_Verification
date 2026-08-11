"""Shared fixtures.

Unit tests must never touch a real model runtime, the real config directory, or the user's
home directory. Everything here is hermetic; live-model coverage lives in
``tests/test_integration_ollama.py`` behind the ``integration`` marker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from edith.config.schema import (
    EdithConfig,
    ModelParams,
    ModelsConfig,
    OllamaProviderConfig,
)

from .fakes import FakeProvider

SAMPLE_SYSTEM_YAML: dict[str, Any] = {
    "project_name": "edith-test",
    "state_dir": ".edith-test",
    "logging": {"level": "DEBUG", "format": "json", "file_enabled": False},
    "resources": {"max_concurrent_inferences": 1, "min_free_vram_mb": 1024},
}

SAMPLE_MODELS_YAML: dict[str, Any] = {
    "provider": "ollama",
    "ollama": {"host": "http://127.0.0.1:11434", "timeout_seconds": 30.0},
    "default_profile": "default",
    "profiles": {
        "default": {
            "model_name": "test-model:q4",
            "context_length": 4096,
            "temperature": 0.1,
            "max_output_tokens": 512,
            "estimated_vram_mb": 2048,
        },
        "fast": {
            "model_name": "test-model-small:q4",
            "context_length": 2048,
            "estimated_vram_mb": 1024,
        },
    },
}

SAMPLE_AGENTS_YAML: dict[str, Any] = {
    "defaults": {"model_profile": "default", "max_attempts": 2},
    "overrides": {"echo": {"model_profile": "default", "max_attempts": 1}},
}


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A throwaway config directory containing valid YAML."""
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "system.yaml").write_text(yaml.safe_dump(SAMPLE_SYSTEM_YAML), encoding="utf-8")
    (directory / "models.yaml").write_text(yaml.safe_dump(SAMPLE_MODELS_YAML), encoding="utf-8")
    (directory / "agents.yaml").write_text(yaml.safe_dump(SAMPLE_AGENTS_YAML), encoding="utf-8")
    return directory


@pytest.fixture
def repo_config_dir() -> Path:
    """The real ``config/`` directory shipped with the repository."""
    return Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def model_params() -> ModelParams:
    """A minimal model profile."""
    return ModelParams(model_name="test-model:q4", context_length=4096, max_output_tokens=256)


@pytest.fixture
def ollama_config() -> OllamaProviderConfig:
    """Loopback Ollama settings with short timeouts."""
    return OllamaProviderConfig(
        host="http://127.0.0.1:11434", timeout_seconds=5.0, connect_timeout_seconds=1.0
    )


@pytest.fixture
def config(model_params: ModelParams) -> EdithConfig:
    """An in-memory :class:`EdithConfig` with one profile."""
    return EdithConfig(
        models=ModelsConfig(default_profile="default", profiles={"default": model_params})
    )


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``EDITH*`` variables so a developer's shell cannot alter test outcomes."""
    for key in list(os.environ):
        if key.startswith("EDITH"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_provider(model_params: ModelParams) -> FakeProvider:
    """A :class:`FakeProvider` returning one valid echo-shaped payload."""
    payload = json.dumps({"summary": "a summary", "keywords": ["local"], "confidence": 0.9})
    return FakeProvider(model_params, [payload])
