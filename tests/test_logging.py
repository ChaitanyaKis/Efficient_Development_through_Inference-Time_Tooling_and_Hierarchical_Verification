"""Structured logging, and the guarantee that secrets are never emitted."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from edith.config.schema import LoggingConfig
from edith.observability.logging import (
    REDACTED,
    SecretRedactor,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)


@pytest.fixture
def redactor() -> SecretRedactor:
    return SecretRedactor(LoggingConfig().redact_keys)


class TestSecretRedaction:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "authorization", "token", "password", "private_key", "secret"],
    )
    def test_sensitive_top_level_keys_masked(self, redactor: SecretRedactor, key: str) -> None:
        result = redactor(None, "info", {"event": "x", key: "hunter2"})
        assert result[key] == REDACTED

    def test_substring_match(self, redactor: SecretRedactor) -> None:
        """OLLAMA_API_TOKEN must be caught by the 'token' pattern."""
        result = redactor(None, "info", {"OLLAMA_API_TOKEN": "abc"})
        assert result["OLLAMA_API_TOKEN"] == REDACTED

    def test_nested_mapping_masked(self, redactor: SecretRedactor) -> None:
        result = redactor(None, "info", {"config": {"host": "local", "password": "p"}})
        assert result["config"] == {"host": "local", "password": REDACTED}

    def test_deeply_nested_masked(self, redactor: SecretRedactor) -> None:
        result = redactor(None, "info", {"a": {"b": {"c": {"api_key": "k"}}}})
        assert result["a"]["b"]["c"]["api_key"] == REDACTED

    def test_inside_list_masked(self, redactor: SecretRedactor) -> None:
        result = redactor(None, "info", {"items": [{"token": "t"}, {"name": "safe"}]})
        assert result["items"][0]["token"] == REDACTED
        assert result["items"][1]["name"] == "safe"

    def test_benign_values_untouched(self, redactor: SecretRedactor) -> None:
        event = {"event": "agent.start", "agent": "echo", "duration_seconds": 1.5}
        assert redactor(None, "info", dict(event)) == event

    def test_recursion_is_bounded(self, redactor: SecretRedactor) -> None:
        """A pathologically nested payload must not blow the stack."""
        payload: dict = {"api_key": "leak"}
        for _ in range(200):
            payload = {"nested": payload}
        redactor(None, "info", {"data": payload})  # must return, not recurse forever

    def test_tuple_type_preserved(self, redactor: SecretRedactor) -> None:
        result = redactor(None, "info", {"items": ("a", "b")})
        assert isinstance(result["items"], tuple)


class TestConfigureLogging:
    def test_console_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(LoggingConfig(level="INFO", format="json", file_enabled=False))
        get_logger("t").info("hello", agent="echo")
        assert "hello" in capsys.readouterr().err

    def test_writes_json_file(self, tmp_path: Path) -> None:
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=Path("logs/e.jsonl")),
            base_dir=tmp_path,
        )
        get_logger("t").info("file_event", agent="echo")
        logging.getLogger().handlers[-1].flush()

        lines = (tmp_path / "logs" / "e.jsonl").read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        assert record["event"] == "file_event"
        assert record["agent"] == "echo"
        assert record["level"] == "info"

    def test_secrets_never_reach_the_log_file(self, tmp_path: Path) -> None:
        """The end-to-end guarantee, not just the processor in isolation."""
        configure_logging(
            LoggingConfig(level="INFO", file_enabled=True, file_path=Path("e.jsonl")),
            base_dir=tmp_path,
        )
        get_logger("t").info("call", api_key="sk-SUPERSECRET", nested={"password": "p4ss"})
        logging.getLogger().handlers[-1].flush()

        content = (tmp_path / "e.jsonl").read_text(encoding="utf-8")
        assert "sk-SUPERSECRET" not in content
        assert "p4ss" not in content
        assert REDACTED in content

    def test_level_is_respected(self, tmp_path: Path) -> None:
        configure_logging(
            LoggingConfig(level="WARNING", file_enabled=True, file_path=Path("e.jsonl")),
            base_dir=tmp_path,
        )
        logger = get_logger("t")
        logger.debug("suppressed")
        logger.warning("emitted")
        logging.getLogger().handlers[-1].flush()

        content = (tmp_path / "e.jsonl").read_text(encoding="utf-8")
        assert "suppressed" not in content
        assert "emitted" in content

    def test_reconfiguration_does_not_duplicate_handlers(self, tmp_path: Path) -> None:
        for _ in range(3):
            configure_logging(
                LoggingConfig(file_enabled=True, file_path=Path("e.jsonl")), base_dir=tmp_path
            )
        assert len(logging.getLogger().handlers) == 2  # console + file

    def test_noisy_third_party_loggers_are_quietened(self, tmp_path: Path) -> None:
        """httpx logs every request at INFO; unfiltered it drowns the doctor's output."""
        configure_logging(LoggingConfig(level="INFO", file_enabled=False))
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_debug_level_keeps_third_party_logs(self, tmp_path: Path) -> None:
        """When the user explicitly asks for DEBUG, they want the HTTP traffic too."""
        configure_logging(LoggingConfig(level="DEBUG", file_enabled=False))
        assert logging.getLogger("httpx").level == logging.DEBUG

    def test_unwritable_log_path_degrades(self, tmp_path: Path) -> None:
        """A bad log path must not prevent the process from reporting the real problem."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        configure_logging(
            LoggingConfig(file_enabled=True, file_path=blocker / "sub" / "e.jsonl"),
            base_dir=tmp_path,
        )
        get_logger("t").info("still works")


class TestTraceContext:
    def test_bound_context_appears_in_events(self, tmp_path: Path) -> None:
        configure_logging(
            LoggingConfig(file_enabled=True, file_path=Path("e.jsonl")), base_dir=tmp_path
        )
        bind_context(project_id="proj_1", task_id="task_1")
        try:
            get_logger("t").info("traced")
        finally:
            clear_context()
        logging.getLogger().handlers[-1].flush()

        record = json.loads(
            (tmp_path / "e.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        assert record["project_id"] == "proj_1"
        assert record["task_id"] == "task_1"

    def test_clear_context_removes_bindings(self, tmp_path: Path) -> None:
        configure_logging(
            LoggingConfig(file_enabled=True, file_path=Path("e.jsonl")), base_dir=tmp_path
        )
        bind_context(task_id="task_1")
        clear_context()
        get_logger("t").info("untraced")
        logging.getLogger().handlers[-1].flush()

        record = json.loads(
            (tmp_path / "e.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        assert "task_id" not in record
