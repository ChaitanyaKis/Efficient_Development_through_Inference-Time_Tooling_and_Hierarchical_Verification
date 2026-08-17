"""Configuration loading, merging, precedence, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from edith.config.loader import default_config_dir, env_overrides, load_config
from edith.config.schema import (
    AgentsConfig,
    EdithConfig,
    ModelParams,
    ModelsConfig,
    OllamaProviderConfig,
)
from edith.errors import ConfigurationError


class TestLoading:
    def test_loads_all_sections(self, config_dir: Path) -> None:
        config = load_config(config_dir)
        assert config.system.project_name == "edith-test"
        assert config.models.provider == "ollama"
        assert config.models.profile().model_name == "test-model:q4"
        assert config.agents.for_agent("echo").max_attempts == 1
        assert config.config_dir == config_dir.resolve()

    def test_repo_config_is_valid(self, repo_config_dir: Path) -> None:
        """The shipped config must load - a broken default config breaks every command."""
        config = load_config(repo_config_dir)
        assert config.models.provider == "ollama"
        assert config.models.default_profile in config.models.profiles
        # Hard constraint: sequential inference on 6 GB VRAM.
        assert config.system.resources.max_concurrent_inferences == 1

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="config directory not found"):
            load_config(tmp_path / "absent")

    def test_missing_models_file_raises(self, tmp_path: Path) -> None:
        directory = tmp_path / "config"
        directory.mkdir()
        (directory / "system.yaml").write_text("project_name: x", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="required config file missing"):
            load_config(directory)

    def test_optional_files_may_be_absent(self, tmp_path: Path) -> None:
        directory = tmp_path / "config"
        directory.mkdir()
        (directory / "models.yaml").write_text(
            yaml.safe_dump({"profiles": {"default": {"model_name": "m:q4"}}}), encoding="utf-8"
        )
        config = load_config(directory)
        assert config.system.project_name == "edith"

    def test_malformed_yaml_raises(self, config_dir: Path) -> None:
        (config_dir / "models.yaml").write_text("profiles: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid YAML"):
            load_config(config_dir)

    def test_non_mapping_yaml_raises(self, config_dir: Path) -> None:
        (config_dir / "system.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must contain a YAML mapping"):
            load_config(config_dir)

    def test_empty_optional_file_is_tolerated(self, config_dir: Path) -> None:
        (config_dir / "agents.yaml").write_text("", encoding="utf-8")
        assert load_config(config_dir).agents.defaults.max_attempts == 2

    def test_unknown_key_is_rejected(self, config_dir: Path) -> None:
        """extra=forbid turns a config typo into a loud error, not a silent default."""
        (config_dir / "system.yaml").write_text("projct_name: typo\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="failed validation"):
            load_config(config_dir)


class TestEnvironmentOverrides:
    def test_nested_override(self) -> None:
        result = env_overrides({"EDITH__MODELS__OLLAMA__HOST": "http://localhost:9999"})
        assert result == {"models": {"ollama": {"host": "http://localhost:9999"}}}

    def test_scalars_are_typed(self) -> None:
        result = env_overrides(
            {
                "EDITH__SYSTEM__RESOURCES__MIN_FREE_RAM_MB": "4096",
                "EDITH__MODELS__OLLAMA__ALLOW_REMOTE": "true",
            }
        )
        assert result["system"]["resources"]["min_free_ram_mb"] == 4096
        assert result["models"]["ollama"]["allow_remote"] is True

    def test_unprefixed_variables_ignored(self) -> None:
        assert env_overrides({"PATH": "/usr/bin", "HOME": "/home/x"}) == {}

    def test_malformed_variable_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="malformed"):
            env_overrides({"EDITH____HOST": "x"})

    def test_env_beats_file(self, config_dir: Path) -> None:
        config = load_config(
            config_dir, environ={"EDITH__SYSTEM__LOGGING__LEVEL": "ERROR"}
        )
        assert config.system.logging.level == "ERROR"

    def test_programmatic_overrides_beat_env(self, config_dir: Path) -> None:
        config = load_config(
            config_dir,
            environ={"EDITH__SYSTEM__LOGGING__LEVEL": "ERROR"},
            overrides={"system": {"logging": {"level": "WARNING"}}},
        )
        assert config.system.logging.level == "WARNING"

    def test_deep_merge_preserves_siblings(self, config_dir: Path) -> None:
        config = load_config(
            config_dir, environ={"EDITH__MODELS__OLLAMA__TIMEOUT_SECONDS": "99"}
        )
        assert config.models.ollama.timeout_seconds == 99.0
        # The sibling key from the file survives the merge.
        assert config.models.ollama.host == "http://127.0.0.1:11434"


class TestDefaultConfigDir:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDITH_CONFIG_DIR", str(tmp_path))
        assert default_config_dir() == tmp_path.resolve()

    def test_falls_back_to_repo_config(self) -> None:
        resolved = default_config_dir()
        assert resolved.name == "config"
        assert (resolved / "models.yaml").is_file()


class TestLocalFirstEnforcement:
    """A non-loopback endpoint must be an explicit, deliberate act."""

    def test_remote_host_rejected_by_default(self) -> None:
        with pytest.raises(ValueError, match="not loopback"):
            OllamaProviderConfig(host="http://198.51.100.7:11434")

    def test_remote_host_allowed_when_opted_in(self) -> None:
        cfg = OllamaProviderConfig(host="http://198.51.100.7:11434", allow_remote=True)
        assert cfg.host == "http://198.51.100.7:11434"

    @pytest.mark.parametrize(
        "host", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"]
    )
    def test_loopback_variants_accepted(self, host: str) -> None:
        assert OllamaProviderConfig(host=host).host == host

    def test_missing_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="must start with http"):
            OllamaProviderConfig(host="127.0.0.1:11434")

    def test_trailing_slash_stripped(self) -> None:
        assert OllamaProviderConfig(host="http://127.0.0.1:11434/").host.endswith("11434")


class TestProfiles:
    def test_unknown_default_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="not defined in profiles"):
            ModelsConfig(
                default_profile="absent", profiles={"default": ModelParams(model_name="m")}
            )

    def test_profile_lookup_falls_back_to_default(self, config: EdithConfig) -> None:
        assert config.models.profile() is config.models.profile("default")

    def test_unknown_profile_raises(self, config: EdithConfig) -> None:
        with pytest.raises(KeyError, match="unknown model profile"):
            config.models.profile("nope")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("temperature", -0.1),
            ("temperature", 2.5),
            ("context_length", 0),
            ("max_output_tokens", 0),
            ("top_p", 1.5),
        ],
    )
    def test_out_of_range_values_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            ModelParams(model_name="m", **{field: value})

    def test_empty_model_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            ModelParams(model_name="")


class TestAgentsConfig:
    def test_override_applies(self) -> None:
        cfg = AgentsConfig.model_validate(
            {"defaults": {"max_attempts": 3}, "overrides": {"echo": {"max_attempts": 1}}}
        )
        assert cfg.for_agent("echo").max_attempts == 1
        assert cfg.for_agent("other").max_attempts == 3

    def test_config_is_immutable(self, config: EdithConfig) -> None:
        """Frozen config prevents a subsystem from mutating shared settings at runtime."""
        with pytest.raises(ValueError):
            config.system.logging.level = "DEBUG"  # type: ignore[misc]
