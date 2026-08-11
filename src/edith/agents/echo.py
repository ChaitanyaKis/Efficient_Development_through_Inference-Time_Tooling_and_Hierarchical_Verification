"""The kernel self-test agent.

``echo`` is not a product feature. It is the smallest agent that exercises the entire M0
kernel end to end -- config -> registry -> provider -> constrained decoding -> schema
validation -> structured response -- and is what ``edith selftest`` runs to prove the
milestone acceptance criterion on real hardware.

Later agents replace it as the interesting ones; it stays as a permanent canary.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

from .base import Agent

SYSTEM_PROMPT = (
    "You are a precise analysis component inside a software engineering system. "
    "You read a short statement and return a structured analysis. "
    "Be terse and factual. Never invent details that are not in the statement."
)

USER_TEMPLATE = (
    "Analyse the following statement.\n\n"
    "STATEMENT:\n{statement}\n\n"
    "Return: a one-sentence summary, up to {max_keywords} lowercase keywords drawn from "
    "the statement, and your confidence between 0.0 and 1.0."
)


class EchoInput(EdithModel):
    """Input contract for :class:`EchoAgent`."""

    statement: str = Field(min_length=1, max_length=4000)
    max_keywords: int = Field(default=5, ge=1, le=10)


class EchoOutput(EdithModel):
    """Output contract for :class:`EchoAgent`.

    Deliberately mixes a string, a constrained list, and a bounded float -- enough shape
    that a model returning sloppy JSON fails validation instead of sliding through.
    """

    summary: str = Field(min_length=1, max_length=1000)
    keywords: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class EchoAgent(Agent):
    """Round-trips a statement through the local model and validates the structured result."""

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="echo",
        version="0.1.0",
        description="Kernel self-test agent: proves the model and validation path work.",
        capabilities=frozenset({Capability.SELF_TEST}),
        # Read-only, no tools, no network. The canary needs no privileges.
        permissions=AgentPermissions(),
    )
    input_schema: ClassVar[type[BaseModel]] = EchoInput
    output_schema: ClassVar[type[BaseModel]] = EchoOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, EchoInput)  # noqa: S101 - guaranteed by validate_input
        provider = self.require_provider()
        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    statement=payload.statement, max_keywords=payload.max_keywords
                ),
            ),
        ]
        return provider.structured_generate(messages, EchoOutput)
