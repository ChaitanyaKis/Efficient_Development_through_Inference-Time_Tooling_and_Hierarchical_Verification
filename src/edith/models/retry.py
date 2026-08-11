"""Bounded retry with exponential backoff.

Every autonomous loop in Edith must terminate (CLAUDE.md invariant 11). This helper is the
single place that decides *how* to wait; callers decide *whether* an error is retryable by
raising an :class:`EdithError` with ``retryable`` set.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from edith.config.schema import RetryConfig
from edith.errors import EdithError
from edith.observability.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def backoff_delays(config: RetryConfig) -> list[float]:
    """Return the delay before each retry, capped at ``max_backoff_seconds``."""
    delays: list[float] = []
    delay = config.initial_backoff_seconds
    for _ in range(max(config.max_attempts - 1, 0)):
        delays.append(min(delay, config.max_backoff_seconds))
        delay *= config.backoff_multiplier
    return delays


def with_retry(
    operation: Callable[[], T],
    config: RetryConfig,
    *,
    description: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation``, retrying retryable :class:`EdithError` failures.

    Non-retryable errors and non-Edith exceptions propagate immediately -- retrying a
    ``ModelNotFoundError`` three times only wastes the user's time.

    Args:
        operation: Zero-argument callable to execute.
        config: Bounded retry policy.
        description: Label used in log events.
        sleep: Injected for tests; defaults to :func:`time.sleep`.

    Raises:
        EdithError: The final failure, after the attempt budget is exhausted.
    """
    delays = backoff_delays(config)
    last_error: EdithError | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return operation()
        except EdithError as exc:
            last_error = exc
            if not exc.retryable or attempt == config.max_attempts:
                logger.warning(
                    "retry.exhausted",
                    operation=description,
                    attempt=attempt,
                    retryable=exc.retryable,
                    category=str(exc.category),
                    error=exc.message,
                )
                raise
            delay = delays[attempt - 1]
            logger.info(
                "retry.scheduled",
                operation=description,
                attempt=attempt,
                max_attempts=config.max_attempts,
                delay_seconds=delay,
                error=exc.message,
            )
            sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error if last_error else RuntimeError("retry loop terminated unexpectedly")
