"""Structured logging with mandatory secret redaction.

CLAUDE.md forbids logging secrets. Redaction is therefore a processor in the structlog
pipeline rather than a caller responsibility -- there is no code path that emits a log
event with an unredacted ``api_key``/``token``/``password`` field, however the caller
constructs it.

Both structlog events and plain stdlib records (from third-party libraries) flow through
the same shared processor chain, so a dependency logging a secret is redacted too.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import structlog

from edith.config.schema import LoggingConfig

REDACTED = "***REDACTED***"

#: Guard against pathological nesting in LLM-produced payloads.
_MAX_REDACTION_DEPTH = 12

#: Third-party loggers that emit one INFO line per operation. At Edith's own INFO level
#: they drown the signal (httpx logs every request), so they are raised to WARNING unless
#: the user explicitly asked for DEBUG.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio")


class SecretRedactor:
    """structlog processor that masks values whose key looks sensitive.

    Matching is case-insensitive and substring-based, so ``OLLAMA_API_TOKEN`` is caught by
    the ``token`` pattern. Nested mappings and sequences are walked.
    """

    def __init__(self, redact_keys: Sequence[str]) -> None:
        self._patterns = tuple(key.lower() for key in redact_keys)

    def _is_sensitive(self, key: str) -> bool:
        lowered = key.lower()
        return any(pattern in lowered for pattern in self._patterns)

    def _scrub(self, value: Any, depth: int) -> Any:
        if depth >= _MAX_REDACTION_DEPTH:
            return value
        if isinstance(value, Mapping):
            return {
                k: REDACTED
                if isinstance(k, str) and self._is_sensitive(k)
                else self._scrub(v, depth + 1)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(self._scrub(item, depth + 1) for item in value)
        if isinstance(value, set):
            return {self._scrub(item, depth + 1) for item in value}
        return value

    def __call__(
        self, _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for key in list(event_dict):
            if isinstance(key, str) and self._is_sensitive(key):
                event_dict[key] = REDACTED
            else:
                event_dict[key] = self._scrub(event_dict[key], 0)
        return event_dict


def _shared_processors(config: LoggingConfig) -> list[Any]:
    """Processors applied to every event, whatever its origin.

    Redaction is last so it also covers fields added by the processors above it.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        SecretRedactor(config.redact_keys),
    ]


def _make_formatter(config: LoggingConfig, *, as_json: bool) -> structlog.stdlib.ProcessorFormatter:
    """Build a formatter that renders both structlog and foreign stdlib records."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if as_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    return structlog.stdlib.ProcessorFormatter(
        # Applied only to records that did not originate from structlog.
        foreign_pre_chain=_shared_processors(config),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def configure_logging(config: LoggingConfig, *, base_dir: Path | None = None) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: calling it again reconfigures cleanly, which matters for tests and for the
    CLI adjusting verbosity after config load.

    Args:
        config: Logging settings.
        base_dir: Root that a relative ``file_path`` is resolved against. Defaults to CWD.
    """
    level = getattr(logging, config.level)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    # Logs go to stderr so that stdout stays clean for machine-readable command output.
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(_make_formatter(config, as_json=config.format == "json"))
    root.addHandler(console)

    if config.file_enabled:
        path = config.file_path
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
        except OSError:
            # A read-only or unwritable location must degrade to console-only, never crash
            # the process before it has had a chance to report the real problem.
            structlog.get_logger(__name__).warning(
                "log_file_unavailable", path=str(path), fallback="console-only"
            )
        else:
            # The file is always JSON: it is meant for machines and for later inspection.
            file_handler.setFormatter(_make_formatter(config, as_json=True))
            root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.DEBUG if level <= logging.DEBUG else logging.WARNING
        )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_shared_processors(config),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger for ``name``."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**values: Any) -> None:
    """Bind trace context (project_id, task_id, agent, model) to all subsequent events."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Clear bound trace context."""
    structlog.contextvars.clear_contextvars()
