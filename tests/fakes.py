"""Test doubles for the layers above the model seam.

That a :class:`FakeProvider` can stand in for :class:`OllamaProvider` with no other change
is the practical proof that the provider abstraction holds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from edith.config.schema import ModelParams
from edith.models.base import ModelProvider
from edith.schemas.model import (
    GenerationOptions,
    GenerationResult,
    HealthState,
    Message,
    ProviderHealth,
    TokenUsage,
)


class FakeProvider(ModelProvider):
    """In-memory provider that returns queued responses.

    Responses are returned in order; the last one repeats once the queue is exhausted, so a
    test that only cares about the first call need not enumerate every subsequent one. Every
    call is recorded for assertions about prompts, options, and schema constraints.
    """

    name = "fake"

    def __init__(
        self,
        params: ModelParams,
        responses: Sequence[str] | None = None,
        *,
        health: ProviderHealth | None = None,
        raises: Exception | None = None,
    ) -> None:
        super().__init__(params)
        self.responses = list(responses or ["{}"])
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._health = health or ProviderHealth(
            provider=self.name, state=HealthState.HEALTHY, detail="fake provider is healthy"
        )

    def _generate_raw(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.calls.append(
            {
                "messages": [(str(m.role), m.content) for m in messages],
                "options": options,
                "json_schema": json_schema,
            }
        )
        if self.raises is not None:
            raise self.raises
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return GenerationResult(
            text=text,
            model=self.params.model_name,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        )

    def stream(
        self, messages: Sequence[Message], options: GenerationOptions | None = None
    ) -> Iterator[str]:
        yield self._generate_raw(messages, options).text

    def health_check(self) -> ProviderHealth:
        return self._health

    def list_models(self) -> tuple[str, ...]:
        return (self.params.model_name,)

    def close(self) -> None:
        self.closed = True
