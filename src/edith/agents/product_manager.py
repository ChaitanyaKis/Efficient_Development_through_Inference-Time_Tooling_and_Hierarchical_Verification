"""The Product Manager Agent.

Turns a product idea into a validated PRD. It reads artifacts and research; it has no write
scope and no shell, because a PM that could edit the repository would blur the line CLAUDE.md
draws between deciding what to build and building it.

**The model does not assign identifiers.** It produces a flat list of requirements; this
module numbers them ``REQ-001``, ``REQ-002``, and pairs each with an ``AC``. That is not a
stylistic choice. Requirement ids are the most load-bearing strings in the system — every
downstream artifact, task, and test references them — and a 3B model asked to emit unique,
correctly-formatted, densely-numbered ids across a nested document will eventually emit
``REQ-1``, a duplicate, or a gap. Numbering in code makes that class of defect unreachable.

The same reasoning drives the flat model-facing schema: :class:`ProductManagerOutput` is much
smaller than :class:`~edith.product.prd.PRDDocument`, because a small model producing deeply
nested objects with enums and frozensets fails constantly. :func:`draft_to_prd` is the trust
boundary where that flat output becomes a strictly-validated document.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.product.artifacts import element_id
from edith.product.prd import (
    AcceptanceCriterion,
    OpenQuestion,
    PRDDocument,
    Priority,
    Requirement,
    RequirementKind,
    Risk,
)
from edith.product.properties import ProductProperty
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

from .base import Agent

#: Ceiling on requirements per PRD. Not a product limit -- a context limit. A 3B model asked
#: for twenty requirements produces twenty shallow ones; asked for eight it produces eight
#: usable ones, and a revision can always add more.
MAX_REQUIREMENTS = 8

SYSTEM_PROMPT = """You are the product management component of a software engineering system.

You turn a product idea into precise, checkable requirements. You do NOT write code, choose
technologies, or design interfaces -- other components do that, and doing it here would
commit the project to decisions nobody has evaluated yet.

Rules:
- Every requirement states ONE thing, in language that makes it obvious whether it was met.
  Bad: "The app should be fast and easy to use."
  Good: "A search returns results within 2 seconds for a catalogue of 10,000 items."
- Give every requirement an acceptance criterion: the concrete check that proves it works.
- Do NOT invent requirement numbers or IDs. Just list the requirements in order.
- `priority` is one of MUST, SHOULD, COULD, WONT.
- `kind` is one of FUNCTIONAL, NON_FUNCTIONAL, CONSTRAINT.
- `properties` tags structural claims from this exact list, or leave it empty:
  OFFLINE_CAPABLE, REQUIRES_NETWORK, CLOUD_DEPENDENT, LOCAL_ONLY, AUTHENTICATION_REQUIRED,
  NO_AUTHENTICATION, AUTHORIZATION_REQUIRED, MULTI_USER, SINGLE_USER, MULTI_TENANT,
  MOBILE_RESPONSIVE, DESKTOP_ONLY, ACCESSIBLE, HEADLESS, PERSISTENT_STORAGE,
  EPHEMERAL_STORAGE, SENSITIVE_DATA, DATA_RESIDENCY, REAL_TIME, ASYNCHRONOUS,
  HIGH_AVAILABILITY, RESOURCE_CONSTRAINED.
- State non-goals. What the product will NOT do is as valuable as what it will.
- If something genuinely cannot be determined from the request, put it in open_questions
  rather than inventing an answer. An invented requirement is indistinguishable from a real
  one downstream, and that is how a product gets built wrong."""

USER_TEMPLATE = """PRODUCT IDEA:
{idea}

CONSTRAINTS:
{constraints}

RESEARCH EVIDENCE (advisory only -- it informs requirements, it does not set them):
{research}

Produce the product requirements."""


class DraftRequirement(EdithModel):
    """One requirement as the model produces it: flat, unnumbered, stringly-typed."""

    title: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=1000)
    kind: str = Field(default="FUNCTIONAL", max_length=40)
    priority: str = Field(default="MUST", max_length=20)
    #: The check that proves this requirement was met.
    acceptance: str = Field(default="", max_length=600)
    rationale: str = Field(default="", max_length=500)
    properties: list[str] = Field(default_factory=list, max_length=6)


class ProductManagerInput(EdithModel):
    """Input contract for :class:`ProductManagerAgent`."""

    idea: str = Field(min_length=1, max_length=4000)
    constraints: str = Field(default="", max_length=4000)
    #: Rendered research, when any was gathered. Never authoritative.
    research: str = Field(default="", max_length=8000)
    #: Prior knowledge, when the governor granted any. Empty by default: M3.2 measured
    #: automatic injection as harmful, and nothing here overrides that.
    prior_knowledge: str = Field(default="", max_length=4000)


class ProductManagerOutput(EdithModel):
    """Output contract for :class:`ProductManagerAgent`."""

    product_name: str = Field(min_length=1, max_length=200)
    problem: str = Field(min_length=1, max_length=2000)
    target_users: str = Field(default="", max_length=1000)
    goals: list[str] = Field(default_factory=list, max_length=8)
    non_goals: list[str] = Field(default_factory=list, max_length=8)
    requirements: list[DraftRequirement] = Field(min_length=1, max_length=MAX_REQUIREMENTS)
    assumptions: list[str] = Field(default_factory=list, max_length=8)
    risks: list[str] = Field(default_factory=list, max_length=6)
    open_questions: list[str] = Field(default_factory=list, max_length=6)


class ProductManagerAgent(Agent):
    """Produces a PRD from a product idea.

    Read-only by construction. It may inspect project artifacts and research; it has no
    write paths and no shell, so it cannot touch the repository at all.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="product_manager",
        description="Turns a product idea into checkable, traceable requirements.",
        capabilities=frozenset({Capability.PLANNING, Capability.DOCUMENTATION}),
        permissions=AgentPermissions(
            # No shell, no writes. A PM that could run commands is a PM that could ship.
            allowed_tools=frozenset({"filesystem.read", "filesystem.search"}),
            allowed_read_paths=("docs/**", "architecture/**", "README.md"),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = ProductManagerInput
    output_schema: ClassVar[type[BaseModel]] = ProductManagerOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, ProductManagerInput)  # noqa: S101 - validate_input guarantees
        provider = self.require_provider()
        idea = payload.idea
        if payload.prior_knowledge:
            idea = (
                f"{idea}\n\nPRIOR KNOWLEDGE (informative, not a requirement):\n"
                f"{payload.prior_knowledge}"
            )

        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    idea=idea,
                    constraints=payload.constraints or "(none stated)",
                    research=payload.research or "(no research was gathered)",
                ),
            ),
        ]
        return provider.structured_generate(
            messages, ProductManagerOutput, max_repair_attempts=2
        )


def _coerce_priority(value: str) -> Priority:
    """Map model text to a priority, defaulting to MUST.

    Defaults *up* rather than down: treating an unrecognised priority as MUST means it gets
    attention and can be demoted, whereas defaulting to COULD would let a genuine requirement
    quietly fall off the plan.
    """
    try:
        return Priority(value.strip().upper())
    except ValueError:
        return Priority.MUST


def _coerce_kind(value: str) -> RequirementKind:
    """Map model text to a requirement kind, defaulting to functional."""
    try:
        return RequirementKind(value.strip().upper())
    except ValueError:
        return RequirementKind.FUNCTIONAL


def _coerce_properties(values: list[str]) -> frozenset[ProductProperty]:
    """Map model-supplied property names to the vocabulary, dropping anything unknown.

    Dropping rather than failing is deliberate. A model inventing ``FAST`` should not take
    down a PRD that is otherwise sound; the property simply does not participate in
    contradiction detection, which is the honest outcome for a claim the system cannot
    interpret.
    """
    resolved: set[ProductProperty] = set()
    for raw in values:
        try:
            resolved.add(ProductProperty(raw.strip().upper()))
        except ValueError:
            continue
    return frozenset(resolved)


def draft_to_prd(
    draft: ProductManagerOutput,
    *,
    source: str = "",
) -> PRDDocument:
    """Translate model output into a strictly-validated PRD.

    This is the trust boundary. Everything the model produced is re-expressed through the
    strict schema, so a malformed draft becomes a validation error rather than a document
    that looks authoritative and is not. Specifically:

    - Requirement and criterion ids are assigned here, densely and in order, so they are
      unique and correctly formatted by construction.
    - Every requirement gets an acceptance criterion. When the model supplied none, one is
      synthesised that names the requirement, which is weak but traceable -- and validation
      still flags it, because a placeholder criterion is visible where a missing one is not.
    - Unknown enum values are coerced rather than rejected, since a small model's
      ``priority: "high"`` is a formatting failure rather than a product decision.

    Raises:
        ValueError: The resulting document fails PRD validation.
    """
    requirements: list[Requirement] = []
    criteria: list[AcceptanceCriterion] = []

    for index, item in enumerate(draft.requirements, start=1):
        requirement_id = element_id("REQ", index)
        requirements.append(
            Requirement(
                requirement_id=requirement_id,
                title=item.title,
                statement=item.statement,
                kind=_coerce_kind(item.kind),
                priority=_coerce_priority(item.priority),
                rationale=item.rationale,
                properties=_coerce_properties(item.properties),
                source=source,
            )
        )
        statement = item.acceptance.strip() or (
            f"Demonstrate that {item.title!r} is satisfied as stated."
        )
        criteria.append(
            AcceptanceCriterion(
                criterion_id=element_id("AC", index),
                statement=statement,
                verifies=(requirement_id,),
            )
        )

    risks = tuple(
        Risk(risk_id=element_id("RISK", index), description=text)
        for index, text in enumerate(draft.risks, start=1)
        if text.strip()
    )
    questions = tuple(
        OpenQuestion(question_id=element_id("Q", index), question=text)
        for index, text in enumerate(draft.open_questions, start=1)
        if text.strip()
    )

    return PRDDocument(
        product_name=draft.product_name,
        problem=draft.problem,
        target_users=draft.target_users,
        goals=tuple(item for item in draft.goals if item.strip()),
        non_goals=tuple(item for item in draft.non_goals if item.strip()),
        requirements=tuple(requirements),
        acceptance_criteria=tuple(criteria),
        assumptions=tuple(item for item in draft.assumptions if item.strip()),
        risks=risks,
        open_questions=questions,
    )
