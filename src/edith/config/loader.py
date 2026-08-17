"""Configuration loading: YAML files + environment overrides -> validated :class:`EdithConfig`.

Precedence, lowest to highest:

1. Schema defaults
2. ``config/system.yaml``, ``config/models.yaml``, ``config/agents.yaml``
3. Environment variables prefixed ``EDITH__`` (double underscore separates nesting)

Example: ``EDITH__MODELS__OLLAMA__HOST=http://127.0.0.1:11500``
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from edith.errors import ConfigurationError

from .schema import EdithConfig

ENV_PREFIX = "EDITH__"
ENV_NESTING_DELIMITER = "__"

#: Config filename -> top-level key in :class:`EdithConfig`.
CONFIG_FILES: dict[str, str] = {
    "system.yaml": "system",
    "models.yaml": "models",
    "agents.yaml": "agents",
    "tools.yaml": "tools",
    "orchestration.yaml": "orchestration",
}


def default_config_dir() -> Path:
    """Return the config directory, honouring ``EDITH_CONFIG_DIR``.

    Falls back to ``<repo root>/config`` derived from this file's location, so the CLI
    works regardless of the caller's working directory.
    """
    override = os.environ.get("EDITH_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # src/edith/config/loader.py -> repo root is three parents up from `edith`.
    return (Path(__file__).resolve().parents[3] / "config").resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping, returning ``{}`` for an empty file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"invalid YAML in {path}: {exc}", details={"path": str(path)}
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"could not read {path}: {exc}", details={"path": str(path)}
        ) from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}",
            details={"path": str(path)},
        )
    return raw


def _coerce_scalar(value: str) -> Any:
    """Interpret an environment string as YAML so ``true``/``8192`` arrive typed."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a nested override dict from ``EDITH__``-prefixed environment variables."""
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in source.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in raw_key[len(ENV_PREFIX) :].split(ENV_NESTING_DELIMITER)]
        if not path or any(not part for part in path):
            raise ConfigurationError(
                f"malformed config environment variable {raw_key!r}",
                details={"variable": raw_key},
            )
        cursor = overrides
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigurationError(
                    f"environment variable {raw_key!r} conflicts with an earlier scalar override",
                    details={"variable": raw_key},
                )
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw_value)
    return overrides


def load_config(
    config_dir: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> EdithConfig:
    """Load, merge, and validate configuration.

    Args:
        config_dir: Directory containing the YAML files. Defaults to :func:`default_config_dir`.
        environ: Environment mapping to read overrides from. Defaults to ``os.environ``.
        overrides: Highest-precedence programmatic overrides, mainly for tests.

    Raises:
        ConfigurationError: A file is missing, malformed, or the merged result is invalid.
    """
    directory = (config_dir or default_config_dir()).resolve()
    if not directory.is_dir():
        raise ConfigurationError(
            f"config directory not found: {directory}",
            details={"config_dir": str(directory)},
        )

    data: dict[str, Any] = {}
    for filename, section in CONFIG_FILES.items():
        path = directory / filename
        if not path.is_file():
            # Only models.yaml is mandatory; the rest have complete schema defaults.
            if section == "models":
                raise ConfigurationError(
                    f"required config file missing: {path}",
                    details={"path": str(path)},
                )
            continue
        data[section] = _read_yaml(path)

    data = _deep_merge(data, env_overrides(environ))
    if overrides:
        data = _deep_merge(data, overrides)
    data["config_dir"] = str(directory)

    try:
        return EdithConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(
            f"configuration failed validation: {exc}",
            details={"config_dir": str(directory)},
        ) from exc
