"""The :class:`Tool` contract.

Mirrors the M0 agent contract deliberately: the base class owns validation, and subclasses
implement only ``_run``. A tool cannot return unvalidated output because it does not own the
code that builds the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from edith.config.schema import ToolsConfig
from edith.errors import ToolValidationError

from .schemas import ToolSpec
from .workspace import Workspace


class ToolContext:
    """Everything a tool is allowed to know about the invocation.

    Notably absent: the agent's raw permissions and the workspace root. A tool cannot widen
    its own scope because it is never handed the objects that define it.
    """

    def __init__(
        self,
        workspace: Workspace,
        config: ToolsConfig,
        *,
        call_id: str,
        agent: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.call_id = call_id
        self.agent = agent
        self._timeout_seconds = timeout_seconds

    def timeout(self, default: float) -> float:
        """Return the effective timeout: the per-call override, or ``default``."""
        return self._timeout_seconds if self._timeout_seconds is not None else default


class Tool(ABC):
    """A single capability exposed to agents through the gateway."""

    #: Static declaration of the tool's name, description, and risk surface.
    spec: ClassVar[ToolSpec]
    #: Pydantic model the call arguments must validate against.
    input_schema: ClassVar[type[BaseModel]]
    #: Pydantic model :meth:`_run` must return.
    output_schema: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject an incomplete tool definition at import time, not at first call."""
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        for attribute in ("spec", "input_schema", "output_schema"):
            if not hasattr(cls, attribute):
                raise TypeError(f"{cls.__name__} must define a class-level `{attribute}`")

    @property
    def name(self) -> str:
        """The tool's registered name."""
        return self.spec.name

    @abstractmethod
    def _run(self, args: BaseModel, ctx: ToolContext) -> BaseModel:
        """Perform the tool's work.

        Args:
            args: Call arguments already validated against :attr:`input_schema`.
            ctx: Invocation context carrying the permission-bound workspace.

        Returns:
            An instance of :attr:`output_schema`.
        """

    def validate_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        """Validate raw call arguments against :attr:`input_schema`."""
        try:
            return self.input_schema.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(
                f"arguments for tool {self.name!r} failed "
                f"{self.input_schema.__name__} validation",
                details={"tool": self.name, "errors": exc.errors(include_url=False)},
            ) from exc

    def validate_output(self, output: Any) -> BaseModel:
        """Validate a result against :attr:`output_schema`."""
        if isinstance(output, self.output_schema):
            return output
        try:
            return self.output_schema.model_validate(output)
        except ValidationError as exc:
            raise ToolValidationError(
                f"result of tool {self.name!r} failed "
                f"{self.output_schema.__name__} validation",
                details={"tool": self.name, "errors": exc.errors(include_url=False)},
            ) from exc

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> BaseModel:
        """Validate arguments, execute, and validate the result."""
        args = self.validate_arguments(arguments)
        return self.validate_output(self._run(args, ctx))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
