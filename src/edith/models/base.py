"""The :class:`ModelProvider` interface.

This is the single seam between Edith and any LLM runtime. Agents import from this module
and the provider-neutral schemas; they must never import :mod:`edith.models.ollama`.

Structured generation is implemented once here, in terms of the abstract
:meth:`ModelProvider._generate_raw`, so that every provider gets identical validation,
JSON extraction, and bounded repair behaviour rather than reimplementing it (and getting
it subtly wrong) per backend.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from edith.config.schema import ModelParams
from edith.errors import StructuredOutputError
from edith.observability.logging import get_logger
from edith.schemas.model import (
    GenerationOptions,
    GenerationResult,
    Message,
    ProviderHealth,
    Role,
    StructuredMode,
)

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

#: Matches a ```json ... ``` or ``` ... ``` fence, which small models emit habitually.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

_REPAIR_INSTRUCTION = (
    "Your previous response was not valid against the required JSON schema.\n"
    "Error: {error}\n"
    "{envelope}"
    "Respond again with ONLY the corrected JSON object. No prose, no code fences."
)


def render_envelope_hint(schema: type[BaseModel]) -> str:
    """Describe a schema's required top-level shape, for a repair prompt.

    A small model that gets the *shape* wrong -- hoisting a nested key to the top level,
    say ``{"replace_file": {...}}`` where ``{"edits": [{"mode": "replace_file", ...}]}`` was
    required -- is not helped by being told its output was invalid. It needs to be told what
    the outer object looks like.

    Derived from the schema rather than hand-written per model, so a new structured output
    gets the same help without anyone remembering to add it. Nothing here relaxes
    validation: the response is still validated against the strict schema afterwards, and a
    reply that ignores this hint fails exactly as it did before.
    """
    document = schema.model_json_schema()
    properties = document.get("properties", {})
    if not properties:
        return ""

    required = set(document.get("required", []))
    keys = sorted(properties, key=lambda name: (name not in required, name))
    described = ", ".join(
        f'"{name}"{"" if name in required else " (optional)"}' for name in keys
    )
    mandatory = sorted(required)
    lines = [
        f"The JSON object's top-level keys must be exactly: {described}.",
    ]
    if mandatory:
        lines.append(
            f"Do not put any other key at the top level. {', '.join(mandatory)} "
            f"{'is' if len(mandatory) == 1 else 'are'} required."
        )
    return "\n".join(lines) + "\n"


def extract_json_object(text: str) -> str:
    """Extract the most likely JSON object from model output.

    Small quantized models wrap JSON in prose or code fences even when told not to. This
    recovers the payload deterministically rather than spending another inference pass.

    Raises:
        ValueError: No balanced JSON object or array could be located.
    """
    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return candidate[start : index + 1]
    raise ValueError("no balanced JSON object found in model output")


class ModelProvider(ABC):
    """Abstract local model runtime.

    Concrete providers implement :meth:`_generate_raw`, :meth:`stream`, :meth:`health_check`,
    :meth:`list_models`, and :meth:`close`. Everything else is shared.
    """

    #: Stable identifier used in logs and :class:`ProviderHealth`.
    name: str = "abstract"

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        #: Updated by a provider when it learns what the runtime actually enforces.
        self._structured_mode: StructuredMode = StructuredMode.UNKNOWN

    # -- Provider-specific surface -------------------------------------------------

    @abstractmethod
    def _generate_raw(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Perform one generation call.

        Args:
            messages: Conversation, system message first by convention.
            options: Per-call overrides; ``None`` uses the profile defaults.
            json_schema: When provided, the runtime should constrain output to this JSON
                Schema. Providers without native support may ignore it -- the shared
                validation and repair loop in :meth:`structured_generate` covers that case.
        """

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], options: GenerationOptions | None = None
    ) -> Iterator[str]:
        """Yield generated text incrementally."""

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Report reachability and whether the configured model is present.

        Must never raise for an expected failure (runtime down, model missing); it returns
        a structured :class:`ProviderHealth` so ``edith doctor`` can render diagnostics.
        """

    @abstractmethod
    def list_models(self) -> tuple[str, ...]:
        """Return model names available in the runtime."""

    @abstractmethod
    def close(self) -> None:
        """Release network/runtime resources."""

    # -- Shared behaviour ----------------------------------------------------------

    def supports_tools(self) -> bool:
        """Whether the configured model supports native tool calling."""
        return self.params.supports_tools

    def structured_mode(self) -> StructuredMode:
        """The guarantee structured generation actually carries on this provider.

        Providers that discover at runtime that their schema was rejected must update this,
        so an agent is never told decoding is constrained when it is not. The base class
        reports the conservative answer until something proves otherwise.
        """
        return self._structured_mode

    def generate(
        self, messages: Sequence[Message], options: GenerationOptions | None = None
    ) -> GenerationResult:
        """Generate free-form text."""
        if not messages:
            raise ValueError("messages must not be empty")
        return self._generate_raw(messages, options)

    def structured_generate(
        self,
        messages: Sequence[Message],
        schema: type[T],
        options: GenerationOptions | None = None,
        *,
        max_repair_attempts: int = 2,
    ) -> T:
        """Generate output validated against ``schema``.

        The model is asked to conform to the schema; the response is then parsed and
        validated locally. On failure the validation error is fed back for a bounded number
        of repair attempts. A model *claiming* well-formed output is never sufficient --
        only successful Pydantic validation returns.

        Args:
            messages: Conversation to send.
            schema: Pydantic model the output must satisfy.
            options: Per-call inference overrides.
            max_repair_attempts: Additional attempts after the first (bounded, never infinite).

        Raises:
            StructuredOutputError: Output never validated within the attempt budget.
        """
        if not messages:
            raise ValueError("messages must not be empty")
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")

        json_schema = schema.model_json_schema()
        # State the schema in the prompt as well as requesting constrained decoding. A
        # runtime may decline to enforce a schema it cannot compile, and a small model
        # produces far better-shaped output when it has seen the target explicitly.
        conversation: list[Message] = [
            *messages,
            Message(role=Role.SYSTEM, content=render_schema_instruction(schema)),
        ]
        last_error = "unknown"
        started = time.monotonic()

        for attempt in range(max_repair_attempts + 1):
            result = self._generate_raw(conversation, options, json_schema=json_schema)
            try:
                payload = extract_json_object(result.text)
            except ValueError as exc:
                last_error = str(exc)
            else:
                try:
                    validated = schema.model_validate_json(payload)
                except ValidationError as exc:
                    last_error = _summarize_validation_error(exc)
                else:
                    logger.debug(
                        "structured_generate.ok",
                        provider=self.name,
                        model=result.model,
                        schema=schema.__name__,
                        attempt=attempt + 1,
                    )
                    return validated

            logger.warning(
                "structured_generate.invalid",
                provider=self.name,
                schema=schema.__name__,
                attempt=attempt + 1,
                error=last_error,
            )
            if attempt < max_repair_attempts:
                conversation = [
                    *conversation,
                    Message(role=Role.ASSISTANT, content=result.text),
                    Message(
                        role=Role.USER,
                        content=_REPAIR_INSTRUCTION.format(
                            error=last_error,
                            envelope=render_envelope_hint(schema),
                        ),
                    ),
                ]

        raise StructuredOutputError(
            f"model output failed {schema.__name__} validation after "
            f"{max_repair_attempts + 1} attempt(s): {last_error}",
            details={
                "schema": schema.__name__,
                "provider": self.name,
                "model": self.params.model_name,
                "attempts": max_repair_attempts + 1,
                "duration_seconds": round(time.monotonic() - started, 3),
            },
        )

    def __enter__(self) -> ModelProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _summarize_validation_error(exc: ValidationError, *, limit: int = 5) -> str:
    """Render a compact, model-readable summary of a validation failure."""
    parts = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    remaining = len(exc.errors()) - limit
    if remaining > 0:
        parts.append(f"(+{remaining} more)")
    return "; ".join(parts)


def render_schema_instruction(schema: type[BaseModel]) -> str:
    """Build a system-prompt fragment describing the required JSON schema.

    Used for providers or models without native constrained decoding.
    """
    return (
        "You must respond with a single JSON object that validates against this JSON Schema.\n"
        "Output ONLY the JSON object. No prose, no explanation, no code fences.\n\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
