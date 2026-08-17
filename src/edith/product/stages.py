"""Inference-time decomposition: many small validated steps instead of one large one.

M4 measured a monolithic UX generation call failing six consecutive times on the configured
3B model with ``<root>: Input should be an object``, while the much smaller Product Manager
call succeeded first time. The hypothesis M4.1 tests is that the difference is *schema
size*, not task difficulty, and that decomposing a large structured generation into small
independently-validated steps extracts more reliable work from a fixed model. The
measurement is recorded in ``docs/experiments/0001-stage-decomposition.md``.

This module is the machinery for that, and it is deliberately generic: a stage is a named
unit of work with a small schema, its own validation, and its own failure classification.
Nothing here knows what a flow or a component is.

Two rules shape the design, both from the milestone:

**Validation is never weakened to make a stage pass.** A stage that cannot produce its
schema fails and says why. The response to an unreliable model is to decompose further, not
to loosen the contract — a required field that becomes optional is a guarantee that silently
stops being one.

**A failed stage never destroys a successful one.** :class:`StageLedger` records each stage
independently, so a run where four stages validated and one failed is representable as
exactly that. The assembled artifact is marked incomplete and cannot be approved, but the
four good stages are not thrown away and re-generated from scratch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from edith.errors import (
    EdithError,
    FailureCategory,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from edith.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StageStatus(StrEnum):
    """What happened to one generation stage."""

    #: Produced output that validated against its schema.
    VALID = "VALID"
    #: Ran and failed. See :class:`StageFailure` for which way.
    FAILED = "FAILED"
    #: Not attempted, because a stage it depends on failed.
    SKIPPED = "SKIPPED"
    #: Deliberately not run: the product does not need this stage.
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StageFailure(StrEnum):
    """Why a stage failed.

    Distinguished because the remedies are different and a single generic failure hides
    that. A model producing malformed JSON needs a smaller schema; an unreachable runtime
    needs an operator; a timeout might just need re-running.
    """

    #: The model produced output that could not be parsed at all.
    MODEL_FAILURE = "MODEL_FAILURE"
    #: Output parsed but did not satisfy the stage's schema.
    SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
    #: The runtime did not answer in time.
    TIMEOUT = "TIMEOUT"
    #: The attempt budget was spent without a valid response.
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    #: The input this stage needed was absent or unusable.
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    #: The model runtime is missing, unreachable, or misconfigured.
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    #: Output validated as a schema but was rejected when assembled into the artifact.
    ARTIFACT_VALIDATION_FAILURE = "ARTIFACT_VALIDATION_FAILURE"

    @property
    def category(self) -> FailureCategory:
        """The platform-wide failure category this maps to."""
        return _FAILURE_CATEGORY[self]


_FAILURE_CATEGORY: dict[StageFailure, FailureCategory] = {
    StageFailure.MODEL_FAILURE: FailureCategory.MODEL_ERROR,
    StageFailure.SCHEMA_VALIDATION_FAILURE: FailureCategory.VALIDATION_FAILURE,
    StageFailure.TIMEOUT: FailureCategory.TIMEOUT,
    StageFailure.RETRY_EXHAUSTED: FailureCategory.VALIDATION_FAILURE,
    StageFailure.CONTEXT_FAILURE: FailureCategory.REQUIREMENT_FAILURE,
    StageFailure.ENVIRONMENT_FAILURE: FailureCategory.ENVIRONMENT_FAILURE,
    StageFailure.ARTIFACT_VALIDATION_FAILURE: FailureCategory.VALIDATION_FAILURE,
}


@dataclass
class StageMeasurement:
    """What one stage cost.

    M4.1 asks for the largest schema and the largest input, not just totals: the hypothesis
    is about *size per call*, so an average across stages would hide exactly the number the
    experiment is testing.
    """

    stage: str
    #: Characters of prompt sent, across every message of every attempt.
    input_chars: int = 0
    #: Characters of raw model output received.
    output_chars: int = 0
    #: Bytes of the JSON Schema this stage asked the model to satisfy.
    schema_bytes: int = 0
    model_calls: int = 0
    attempts: int = 0
    duration_seconds: float = 0.0
    #: Elements the stage produced: flows, screens, components, tasks.
    elements: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary."""
        return {
            "stage": self.stage,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "schema_bytes": self.schema_bytes,
            "model_calls": self.model_calls,
            "attempts": self.attempts,
            "duration_seconds": round(self.duration_seconds, 2),
            "elements": self.elements,
        }


@dataclass
class StageResult:
    """The outcome of one generation stage.

    Carries the parsed output when it succeeded and the classified reason when it did not.
    Never raises past the caller: a stage that threw would force the assembler to decide
    what a partial run means, which is the one decision this module exists to own.
    """

    stage: str
    status: StageStatus
    output: BaseModel | None = None
    failure: StageFailure | None = None
    detail: str = ""
    measurement: StageMeasurement | None = None

    @property
    def ok(self) -> bool:
        """Whether this stage produced usable output."""
        return self.status is StageStatus.VALID and self.output is not None

    @property
    def attempted(self) -> bool:
        """Whether the stage ran at all."""
        return self.status in {StageStatus.VALID, StageStatus.FAILED}

    def summary(self) -> str:
        """One readable line."""
        if self.ok:
            elements = self.measurement.elements if self.measurement else 0
            return f"{self.stage}: VALID ({elements} element(s))"
        if self.status is StageStatus.SKIPPED:
            return f"{self.stage}: SKIPPED - {self.detail}"
        if self.status is StageStatus.NOT_APPLICABLE:
            return f"{self.stage}: NOT APPLICABLE - {self.detail}"
        return f"{self.stage}: FAILED [{self.failure}] - {self.detail}"


@dataclass
class StageLedger:
    """Every stage of one decomposed generation, successes and failures alike.

    The whole point of decomposition is that partial success is a real state. A ledger where
    four stages validated and one failed is not "a failed run" -- it is a run that produced
    four usable artifacts' worth of content and needs one part re-generated.
    """

    results: list[StageResult] = field(default_factory=list)

    def add(self, result: StageResult) -> StageResult:
        """Record a stage outcome."""
        self.results.append(result)
        # The measurement carries its own `stage` key; drop it rather than passing the same
        # keyword twice, which structlog rejects at runtime.
        measured = result.measurement.as_dict() if result.measurement else {}
        measured.pop("stage", None)
        logger.info(
            "stage.completed",
            stage=result.stage,
            status=str(result.status),
            failure=str(result.failure) if result.failure else "",
            **measured,
        )
        return result

    def get(self, stage: str) -> StageResult | None:
        """The recorded result for one stage, if it ran."""
        for result in self.results:
            if result.stage == stage:
                return result
        return None

    def output(self, stage: str) -> BaseModel | None:
        """The validated output of one stage, or ``None`` when it did not produce any."""
        result = self.get(stage)
        return result.output if result is not None and result.ok else None

    @property
    def valid_stages(self) -> tuple[str, ...]:
        """Stages that produced usable output."""
        return tuple(result.stage for result in self.results if result.ok)

    @property
    def failed_stages(self) -> tuple[str, ...]:
        """Stages that ran and failed."""
        return tuple(
            result.stage for result in self.results if result.status is StageStatus.FAILED
        )

    @property
    def complete(self) -> bool:
        """Whether every stage that was attempted succeeded.

        Skipped and not-applicable stages do not make a run incomplete: a product with no
        visual interface legitimately has no design tokens.
        """
        return bool(self.results) and not self.failed_stages

    @property
    def failures(self) -> tuple[StageFailure, ...]:
        """The classified failure of every stage that failed."""
        return tuple(
            result.failure
            for result in self.results
            if result.status is StageStatus.FAILED and result.failure is not None
        )

    def totals(self) -> dict[str, Any]:
        """Aggregated measurements, including the per-call maxima the experiment needs."""
        measurements = [
            result.measurement for result in self.results if result.measurement
        ]
        return {
            "stages": len(self.results),
            "stages_valid": len(self.valid_stages),
            "stages_failed": len(self.failed_stages),
            "model_calls": sum(item.model_calls for item in measurements),
            "attempts": sum(item.attempts for item in measurements),
            "input_chars": sum(item.input_chars for item in measurements),
            "output_chars": sum(item.output_chars for item in measurements),
            "duration_seconds": round(
                sum(item.duration_seconds for item in measurements), 2
            ),
            # The numbers the hypothesis is actually about: size of the largest single call.
            "largest_schema_bytes": max(
                (item.schema_bytes for item in measurements), default=0
            ),
            "largest_input_chars": max(
                (item.input_chars for item in measurements), default=0
            ),
            "per_stage": [item.as_dict() for item in measurements],
        }

    def summary(self) -> str:
        """A readable report over every stage."""
        return "\n".join(result.summary() for result in self.results)


def classify_exception(exc: Exception) -> tuple[StageFailure, str]:
    """Map an exception raised during generation onto a stage failure.

    Ordered from most specific to least. ``StructuredOutputError`` is the common case and
    means the model produced something the schema rejected after every repair attempt --
    which is precisely the signal that the schema is too large for this model.
    """
    if isinstance(exc, ProviderTimeoutError):
        return (StageFailure.TIMEOUT, exc.message)
    if isinstance(exc, ProviderUnavailableError):
        return (StageFailure.ENVIRONMENT_FAILURE, exc.message)
    if isinstance(exc, StructuredOutputError):
        # The provider exhausts its repair budget before raising, so this *is* the
        # exhausted case; the message carries which validation error persisted.
        return (StageFailure.RETRY_EXHAUSTED, exc.message)
    if isinstance(exc, ValidationError):
        return (StageFailure.SCHEMA_VALIDATION_FAILURE, str(exc))
    if isinstance(exc, EdithError):
        if exc.category is FailureCategory.ENVIRONMENT_FAILURE:
            return (StageFailure.ENVIRONMENT_FAILURE, exc.message)
        if exc.category is FailureCategory.VALIDATION_FAILURE:
            return (StageFailure.SCHEMA_VALIDATION_FAILURE, exc.message)
        return (StageFailure.MODEL_FAILURE, exc.message)
    return (StageFailure.MODEL_FAILURE, f"{type(exc).__name__}: {exc}")


def run_stage(
    stage: str,
    schema: type[T],
    generate: Callable[[], T],
    *,
    prompt_chars: int,
    elements_of: Callable[[T], int] | None = None,
) -> StageResult:
    """Run one generation stage, measuring and classifying whatever happens.

    Args:
        stage: Stage name, used in the ledger and the logs.
        schema: The model the output must satisfy. Its rendered size is recorded, because
            that is the independent variable the M4.1 experiment is testing.
        generate: Performs the call. Expected to raise on failure.
        prompt_chars: Characters of prompt this stage sends, measured by the caller which
            is the only place that knows what it assembled.
        elements_of: How many elements the output represents, for the measurement.

    Returns:
        A result carrying either the validated output or a classified failure. Never raises.
    """
    import json  # noqa: PLC0415 - only needed to size the schema

    measurement = StageMeasurement(
        stage=stage,
        input_chars=prompt_chars,
        schema_bytes=len(json.dumps(schema.model_json_schema())),
        model_calls=1,
        attempts=1,
    )
    started = time.monotonic()
    try:
        output = generate()
    except Exception as exc:  # noqa: BLE001 - every failure is classified, none escapes
        failure, detail = classify_exception(exc)
        measurement.duration_seconds = time.monotonic() - started
        return StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            failure=failure,
            detail=detail[:1000],
            measurement=measurement,
        )

    measurement.duration_seconds = time.monotonic() - started
    measurement.output_chars = len(output.model_dump_json())
    measurement.elements = elements_of(output) if elements_of else 0
    return StageResult(
        stage=stage,
        status=StageStatus.VALID,
        output=output,
        measurement=measurement,
    )


def skipped(stage: str, reason: str) -> StageResult:
    """A stage that was not attempted because something it needed failed."""
    return StageResult(stage=stage, status=StageStatus.SKIPPED, detail=reason)


def not_applicable(stage: str, reason: str) -> StageResult:
    """A stage the product genuinely does not need."""
    return StageResult(stage=stage, status=StageStatus.NOT_APPLICABLE, detail=reason)
