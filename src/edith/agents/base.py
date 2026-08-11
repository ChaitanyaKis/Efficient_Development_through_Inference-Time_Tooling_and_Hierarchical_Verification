"""The base :class:`Agent` contract.

Every specialized agent in later milestones (Planner, Coder, Critic, ...) subclasses this.
The lifecycle is fixed here so that input validation, output validation, error
classification, timing, and trace binding cannot be skipped by a subclass -- subclasses
implement :meth:`_run`, and :meth:`execute` wraps it.

This is the load-bearing decision of M0: an agent physically cannot return unvalidated
output or swallow a failure, because it does not own the code path that builds the response.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from edith.config.schema import AgentDefaults, RetryConfig
from edith.errors import (
    AgentExecutionError,
    AgentValidationError,
    EdithError,
    FailureCategory,
)
from edith.models.base import ModelProvider
from edith.models.retry import with_retry
from edith.observability.logging import bind_context, clear_context, get_logger
from edith.schemas.agent import (
    AgentHealth,
    AgentIdentity,
    AgentRequest,
    AgentResponse,
    AgentStatus,
)

logger = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Agent(ABC):
    """Abstract agent with a validated, observable lifecycle.

    Subclasses declare :attr:`identity`, :attr:`input_schema`, :attr:`output_schema`, and
    implement :meth:`_run`.
    """

    #: Static description of this agent, including its declared permissions.
    identity: ClassVar[AgentIdentity]
    #: Pydantic model that ``AgentRequest.payload`` must validate against. Prefer
    #: :class:`~edith.schemas.common.EdithModel` so an unexpected field is a loud
    #: validation failure rather than silently discarded data.
    input_schema: ClassVar[type[BaseModel]]
    #: Pydantic model that :meth:`_run` must return. Same guidance as ``input_schema``.
    output_schema: ClassVar[type[BaseModel]]

    def __init__(
        self,
        provider: ModelProvider | None = None,
        settings: AgentDefaults | None = None,
    ) -> None:
        """
        Args:
            provider: Model provider, injected. ``None`` for agents that need no inference
                (and for tests), in which case :meth:`require_provider` raises if used.
            settings: Effective per-agent settings from ``agents.yaml``.
        """
        self._provider = provider
        self.settings = settings or AgentDefaults()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject incomplete agent definitions at import time, not at first invocation."""
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        for attribute in ("identity", "input_schema", "output_schema"):
            if not hasattr(cls, attribute):
                raise TypeError(f"{cls.__name__} must define a class-level `{attribute}`")

    @property
    def name(self) -> str:
        """The agent's registered name."""
        return self.identity.name

    @property
    def provider(self) -> ModelProvider | None:
        """The injected model provider, if any."""
        return self._provider

    def require_provider(self) -> ModelProvider:
        """Return the provider, raising a classified error when none was injected."""
        if self._provider is None:
            raise AgentExecutionError(
                f"agent {self.name!r} requires a model provider but none was injected",
                category=FailureCategory.CONFIGURATION_ERROR,
            )
        return self._provider

    # -- Subclass surface ----------------------------------------------------------

    @abstractmethod
    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        """Execute the agent's work.

        Args:
            payload: ``request.payload`` already validated against :attr:`input_schema`.
            request: The full request envelope, for task/trace metadata.

        Returns:
            An instance of :attr:`output_schema`. Returning anything else is a bug and is
            reported as ``AgentValidationError``.
        """

    # -- Fixed lifecycle -----------------------------------------------------------

    def validate_input(self, request: AgentRequest) -> BaseModel:
        """Validate the request payload against :attr:`input_schema`."""
        try:
            return self.input_schema.model_validate(request.payload)
        except ValidationError as exc:
            raise AgentValidationError(
                f"input to agent {self.name!r} failed {self.input_schema.__name__} validation",
                details={"agent": self.name, "errors": exc.errors(include_url=False)},
            ) from exc

    def validate_output(self, output: Any) -> BaseModel:
        """Validate a result against :attr:`output_schema`.

        Accepts either an instance of the schema or a mapping to coerce, so an agent that
        builds a dict still gets validated rather than bypassing the gate.
        """
        if isinstance(output, self.output_schema):
            return output
        try:
            return self.output_schema.model_validate(output)
        except ValidationError as exc:
            raise AgentValidationError(
                f"output of agent {self.name!r} failed "
                f"{self.output_schema.__name__} validation",
                details={"agent": self.name, "errors": exc.errors(include_url=False)},
            ) from exc

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Run the full lifecycle: validate input -> run -> validate output -> respond.

        Never raises for an agent-level failure. Every outcome is an :class:`AgentResponse`
        with an explicit status and, on failure, a :class:`FailureCategory`.
        """
        started = time.monotonic()
        bind_context(
            agent=self.name,
            request_id=request.request_id,
            task_id=request.task.task_id,
            project_id=request.task.project_id,
        )
        model_name = self._provider.params.model_name if self._provider else None
        attempts = 0
        try:
            logger.info("agent.start", agent=self.name, model=model_name)
            # Input is validated once: a malformed payload will not become valid on a
            # retry, so re-running it would only waste inference budget.
            payload = self.validate_input(request)

            def attempt() -> BaseModel:
                nonlocal attempts
                attempts += 1
                return self.validate_output(self._run(payload, request))

            validated = with_retry(
                attempt,
                self._retry_config(),
                description=f"agent.{self.name}",
            )
            duration = time.monotonic() - started
            logger.info(
                "agent.success",
                agent=self.name,
                attempts=attempts,
                duration_seconds=round(duration, 3),
            )
            return AgentResponse(
                request_id=request.request_id,
                agent=self.name,
                status=AgentStatus.SUCCESS,
                output=validated.model_dump(mode="json"),
                attempts=attempts,
                duration_seconds=duration,
                model=model_name,
            )
        except AgentValidationError as exc:
            return self._failure(
                request, exc, started, model_name, AgentStatus.REJECTED, attempts
            )
        except EdithError as exc:
            return self._failure(
                request, exc, started, model_name, AgentStatus.FAILURE, attempts
            )
        except Exception as exc:  # noqa: BLE001 - an agent must not crash its caller
            wrapped = AgentExecutionError(
                f"agent {self.name!r} raised {type(exc).__name__}: {exc}",
                category=FailureCategory.UNKNOWN,
            )
            return self._failure(
                request, wrapped, started, model_name, AgentStatus.FAILURE, attempts
            )
        finally:
            clear_context()

    def _retry_config(self) -> RetryConfig:
        """Bounded retry policy for this agent's work.

        Derived from ``agents.yaml`` so the attempt budget is configuration, not a constant
        buried in code. Backoff is shared with the model layer's policy defaults.
        """
        return RetryConfig(max_attempts=self.settings.max_attempts)

    def _failure(
        self,
        request: AgentRequest,
        error: EdithError,
        started: float,
        model_name: str | None,
        status: AgentStatus,
        attempts: int = 1,
    ) -> AgentResponse:
        """Build a structured failure response and log it. Failures are never hidden."""
        duration = time.monotonic() - started
        logger.warning(
            "agent.failure",
            agent=self.name,
            status=str(status),
            category=str(error.category),
            error=error.message,
            attempts=attempts,
            duration_seconds=round(duration, 3),
        )
        return AgentResponse(
            request_id=request.request_id,
            agent=self.name,
            status=status,
            error=error.message,
            failure_category=error.category,
            attempts=attempts,
            duration_seconds=duration,
            model=model_name,
            evidence={"details": error.details} if error.details else {},
        )

    def health_check(self) -> AgentHealth:
        """Report whether this agent and its model provider are usable."""
        if self._provider is None:
            return AgentHealth(
                agent=self.name,
                healthy=True,
                detail="no model provider required",
            )
        health = self._provider.health_check()
        return AgentHealth(
            agent=self.name,
            healthy=health.ok,
            detail=health.detail,
            provider_state=str(health.state),
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} v{self.identity.version}>"
