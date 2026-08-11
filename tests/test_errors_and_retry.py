"""Error classification and bounded retry behaviour."""

from __future__ import annotations

import pytest

from edith.config.schema import RetryConfig
from edith.errors import (
    AgentValidationError,
    ConfigurationError,
    EdithError,
    FailureCategory,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from edith.models.retry import backoff_delays, with_retry


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        ("error_cls", "category"),
        [
            (ConfigurationError, FailureCategory.CONFIGURATION_ERROR),
            (ProviderError, FailureCategory.MODEL_ERROR),
            (ProviderUnavailableError, FailureCategory.ENVIRONMENT_FAILURE),
            (ModelNotFoundError, FailureCategory.ENVIRONMENT_FAILURE),
            (ProviderTimeoutError, FailureCategory.TIMEOUT),
            (StructuredOutputError, FailureCategory.VALIDATION_FAILURE),
            (AgentValidationError, FailureCategory.VALIDATION_FAILURE),
        ],
    )
    def test_default_categories(
        self, error_cls: type[EdithError], category: FailureCategory
    ) -> None:
        assert error_cls("x").category is category

    @pytest.mark.parametrize(
        ("error_cls", "retryable"),
        [
            (ProviderUnavailableError, True),
            (ProviderTimeoutError, True),
            (StructuredOutputError, True),
            (ModelNotFoundError, False),
            (ConfigurationError, False),
        ],
    )
    def test_default_retryability(self, error_cls: type[EdithError], retryable: bool) -> None:
        """Retrying a missing model or a bad config can never succeed."""
        assert error_cls("x").retryable is retryable

    def test_explicit_overrides_win(self) -> None:
        error = ModelNotFoundError("x", retryable=True, category=FailureCategory.TIMEOUT)
        assert error.retryable is True and error.category is FailureCategory.TIMEOUT

    def test_to_dict_is_structured(self) -> None:
        error = ProviderError("boom", details={"model": "m"})
        payload = error.to_dict()
        assert payload["error"] == "ProviderError"
        assert payload["category"] == "MODEL_ERROR"
        assert payload["details"] == {"model": "m"}

    def test_details_default_to_empty(self) -> None:
        assert EdithError("x").details == {}

    def test_all_errors_subclass_edith_error(self) -> None:
        for cls in (ConfigurationError, ProviderError, AgentValidationError):
            assert issubclass(cls, EdithError)

    def test_str_is_the_message(self) -> None:
        assert str(ProviderError("readable message")) == "readable message"


class TestBackoffDelays:
    def test_exponential_growth(self) -> None:
        config = RetryConfig(
            max_attempts=4,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=100.0,
        )
        assert backoff_delays(config) == [1.0, 2.0, 4.0]

    def test_capped_at_maximum(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_backoff_seconds=1.0,
            backoff_multiplier=10.0,
            max_backoff_seconds=5.0,
        )
        assert backoff_delays(config) == [1.0, 5.0, 5.0, 5.0]

    def test_single_attempt_means_no_delays(self) -> None:
        assert backoff_delays(RetryConfig(max_attempts=1)) == []


class TestWithRetry:
    def test_returns_on_first_success(self) -> None:
        calls = []

        def operation() -> str:
            calls.append(1)
            return "ok"

        assert with_retry(operation, RetryConfig(), sleep=lambda _: None) == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def operation() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ProviderUnavailableError("down")
            return "recovered"

        result = with_retry(operation, RetryConfig(max_attempts=3), sleep=lambda _: None)
        assert result == "recovered" and attempts["n"] == 3

    def test_is_bounded(self) -> None:
        """CLAUDE.md invariant 11: autonomous loops must be bounded."""
        attempts = {"n": 0}

        def operation() -> str:
            attempts["n"] += 1
            raise ProviderUnavailableError("always down")

        with pytest.raises(ProviderUnavailableError):
            with_retry(operation, RetryConfig(max_attempts=3), sleep=lambda _: None)
        assert attempts["n"] == 3

    def test_non_retryable_error_fails_immediately(self) -> None:
        attempts = {"n": 0}

        def operation() -> str:
            attempts["n"] += 1
            raise ModelNotFoundError("pull the model")

        with pytest.raises(ModelNotFoundError):
            with_retry(operation, RetryConfig(max_attempts=5), sleep=lambda _: None)
        assert attempts["n"] == 1

    def test_non_edith_exception_propagates_immediately(self) -> None:
        attempts = {"n": 0}

        def operation() -> str:
            attempts["n"] += 1
            raise ValueError("a programming bug, not a transient failure")

        with pytest.raises(ValueError):
            with_retry(operation, RetryConfig(max_attempts=3), sleep=lambda _: None)
        assert attempts["n"] == 1

    def test_backoff_delays_are_applied(self) -> None:
        slept: list[float] = []

        def operation() -> str:
            raise ProviderUnavailableError("down")

        config = RetryConfig(
            max_attempts=3, initial_backoff_seconds=0.5, backoff_multiplier=2.0
        )
        with pytest.raises(ProviderUnavailableError):
            with_retry(operation, config, sleep=slept.append)
        assert slept == [0.5, 1.0]
