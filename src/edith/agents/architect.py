"""The Architect Agent.

Consumes a PRD and a UX specification and produces a system architecture plus an
implementation plan. It writes only into the architecture area and has no shell, so it can
record a decision but cannot act on one.

The hard part of this agent is not producing a design — it is producing a design that does
not quietly change what was asked for. Three mechanisms address that, and none of them is a
prompt instruction:

**Decisions require alternatives and consequences.** The schema refuses an ADR without them,
so "we will use PostgreSQL" cannot be recorded as a decision unless something was rejected
and something was given up.

**Requirement references are filtered against the PRD.** An architecture claiming to satisfy
``REQ-042`` when the PRD defines six requirements is naming something that does not exist;
the translation drops it, and the requirement it was meant to cover then correctly shows as
uncovered.

**Contradictions are found structurally.** The architecture declares properties from the
same closed vocabulary the PRD uses, so "must work offline" against "cloud-dependent" is a
set intersection rather than a judgement — see :mod:`edith.product.contradictions`.

The implementation plan this agent produces is a *document*, not an executable DAG. M4
produces the plan and stops; converting it into tasks the loop will run is a separate,
deliberate step, because that conversion is a trust boundary and an agent should not be on
both sides of it.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.product.architecture import (
    ApiEndpoint,
    ArchitectureComponent,
    ArchitectureDecision,
    Complexity,
    ComponentKind,
    Confidence,
    DataEntity,
    DataFlowEdge,
    ImplementationPlanDocument,
    PlannedTask,
    SystemArchitectureDocument,
    TechnologyChoice,
    Threat,
)
from edith.product.artifacts import element_id
from edith.product.prd import PRDDocument
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

MAX_COMPONENTS = 8
MAX_DECISIONS = 6
MAX_TASKS = 10

SYSTEM_PROMPT = """You are the architecture component of a software engineering system.

You turn requirements and an interface specification into a system design and an
implementation plan.

Rules:
- Choose the SIMPLEST design that satisfies the requirements. Do not add a queue, a cache, a
  microservice, or a database the requirements do not need. Unnecessary infrastructure is a
  cost the project pays forever.
- Do NOT choose a technology because it is popular. State what you rejected and why, under
  the project's actual constraints: hardware, budget, deployment, team, complexity.
- Every decision must record alternatives and consequences, including the bad ones. A
  decision with no downside was not a decision.
- `satisfies` and `implements` list requirement IDs (like REQ-002) from the requirements you
  were given. Do not invent IDs that were not given to you.
- Do not change or reinterpret a requirement. If a requirement cannot be met as written, say
  so in the risks of the relevant decision.
- `properties` tags structural claims about what you are building, from this exact list, or
  leave it empty: OFFLINE_CAPABLE, REQUIRES_NETWORK, CLOUD_DEPENDENT, LOCAL_ONLY,
  AUTHENTICATION_REQUIRED, NO_AUTHENTICATION, AUTHORIZATION_REQUIRED, MULTI_USER,
  SINGLE_USER, MULTI_TENANT, MOBILE_RESPONSIVE, DESKTOP_ONLY, ACCESSIBLE, HEADLESS,
  PERSISTENT_STORAGE, EPHEMERAL_STORAGE, SENSITIVE_DATA, DATA_RESIDENCY, REAL_TIME,
  ASYNCHRONOUS, HIGH_AVAILABILITY, RESOURCE_CONSTRAINED.
- Tasks are concrete units of implementation work. Each names the components it touches and
  the earlier tasks it depends on, by name. Do not create a circular dependency.
- Do NOT invent component, decision, or task numbers. List them in order."""

USER_TEMPLATE = """PRODUCT REQUIREMENTS:
{prd}

UX SPECIFICATION:
{ux}

PROJECT CONSTRAINTS (binding):
{constraints}

RESEARCH EVIDENCE (advisory only -- you decide, research does not):
{research}

Produce the architecture and the implementation plan."""


class DraftComponent(EdithModel):
    """One component as the model produces it."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="SERVICE", max_length=30)
    responsibility: str = Field(min_length=1, max_length=600)
    #: Names of components this one depends on.
    depends_on: list[str] = Field(default_factory=list, max_length=6)
    satisfies: list[str] = Field(default_factory=list, max_length=8)
    technology: str = Field(default="", max_length=120)
    paths: list[str] = Field(default_factory=list, max_length=6)


class DraftDecision(EdithModel):
    """One architecture decision as the model produces it."""

    title: str = Field(min_length=1, max_length=200)
    context: str = Field(min_length=1, max_length=1000)
    decision: str = Field(min_length=1, max_length=600)
    alternatives: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=1, max_length=1000)
    consequences: list[str] = Field(default_factory=list, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=4)
    affects_requirements: list[str] = Field(default_factory=list, max_length=8)
    confidence: str = Field(default="MEDIUM", max_length=20)


class DraftTechnology(EdithModel):
    """One technology choice as the model produces it."""

    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=80)
    rationale: str = Field(min_length=1, max_length=600)
    alternatives_rejected: list[str] = Field(default_factory=list, max_length=5)
    constraints_considered: list[str] = Field(default_factory=list, max_length=5)


class DraftEntity(EdithModel):
    """One data entity as the model produces it."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    fields: dict[str, str] = Field(default_factory=dict)
    sensitive: bool = False
    satisfies: list[str] = Field(default_factory=list, max_length=6)


class DraftEndpoint(EdithModel):
    """One API operation as the model produces it."""

    method: str = Field(default="GET", max_length=10)
    path: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=300)
    requires_authentication: bool = True
    satisfies: list[str] = Field(default_factory=list, max_length=6)


class DraftThreat(EdithModel):
    """One threat as the model produces it."""

    asset: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=600)
    mitigation: str = Field(default="", max_length=600)
    severity: str = Field(default="MEDIUM", max_length=20)


class DraftTask(EdithModel):
    """One implementation task as the model produces it."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    agent: str = Field(default="coder", max_length=40)
    #: Names of tasks that must finish first.
    depends_on: list[str] = Field(default_factory=list, max_length=6)
    implements: list[str] = Field(default_factory=list, max_length=8)
    #: Names of components this task touches.
    components: list[str] = Field(default_factory=list, max_length=6)
    acceptance: str = Field(default="", max_length=600)
    complexity: str = Field(default="MEDIUM", max_length=20)
    paths: list[str] = Field(default_factory=list, max_length=6)


class ArchitectInput(EdithModel):
    """Input contract for :class:`ArchitectAgent`."""

    prd: str = Field(min_length=1, max_length=20_000)
    ux_spec: str = Field(default="", max_length=20_000)
    constraints: str = Field(default="", max_length=4000)
    research: str = Field(default="", max_length=8000)
    prior_knowledge: str = Field(default="", max_length=4000)


class ArchitectOutput(EdithModel):
    """Output contract for :class:`ArchitectAgent`."""

    # No product_name, for the same reason as the UX agent's: product identity is
    # system-owned and the caller already knows it.
    overview: str = Field(min_length=1, max_length=2000)
    components: list[DraftComponent] = Field(min_length=1, max_length=MAX_COMPONENTS)
    decisions: list[DraftDecision] = Field(default_factory=list, max_length=MAX_DECISIONS)
    technologies: list[DraftTechnology] = Field(default_factory=list, max_length=8)
    entities: list[DraftEntity] = Field(default_factory=list, max_length=8)
    endpoints: list[DraftEndpoint] = Field(default_factory=list, max_length=10)
    threats: list[DraftThreat] = Field(default_factory=list, max_length=6)
    tasks: list[DraftTask] = Field(default_factory=list, max_length=MAX_TASKS)
    properties: list[str] = Field(default_factory=list, max_length=8)
    constraints_considered: list[str] = Field(default_factory=list, max_length=6)
    deliberate_omissions: list[str] = Field(default_factory=list, max_length=6)


class ArchitectAgent(Agent):
    """Produces a system architecture and an implementation plan.

    Writes only architecture documents, and has no shell. It can record that a decision was
    made; it cannot act on one, which is what keeps design and implementation separable.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="architect",
        description="Turns requirements and UX into a system design and an implementation plan.",
        capabilities=frozenset({Capability.ARCHITECTURE, Capability.PLANNING}),
        permissions=AgentPermissions(
            allowed_tools=frozenset(
                {"filesystem.read", "filesystem.search", "filesystem.write"}
            ),
            allowed_read_paths=("**",),
            allowed_write_paths=("architecture/**", "docs/adr/**"),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = ArchitectInput
    output_schema: ClassVar[type[BaseModel]] = ArchitectOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, ArchitectInput)  # noqa: S101 - validate_input guarantees
        provider = self.require_provider()
        prd = payload.prd
        if payload.prior_knowledge:
            prd = f"{prd}\n\nPRIOR KNOWLEDGE (informative):\n{payload.prior_knowledge}"

        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    prd=prd,
                    ux=payload.ux_spec or "(no UX specification was produced)",
                    constraints=payload.constraints or "(none stated)",
                    research=payload.research or "(no research was gathered)",
                ),
            ),
        ]
        return provider.structured_generate(
            messages, ArchitectOutput, max_repair_attempts=2
        )


def _slug(value: str) -> str:
    """Normalise a name for matching model-supplied cross-references."""
    return "".join(char for char in value.lower() if char.isalnum())


def _coerce(enum_type: type, value: str, fallback: object) -> object:
    try:
        return enum_type(value.strip().upper())
    except ValueError:
        return fallback


def _filter_requirements(values: list[str], known: frozenset[str]) -> tuple[str, ...]:
    """Keep only requirement ids the PRD defines. See ``ux_designer._filter_requirements``."""
    return tuple(
        dict.fromkeys(item.strip().upper() for item in values if item.strip().upper() in known)
    )


def draft_to_architecture(
    draft: ArchitectOutput,
    *,
    prd: PRDDocument | None = None,
    product_name: str = "",
) -> SystemArchitectureDocument:
    """Translate model output into a strictly-validated architecture.

    The trust boundary for architecture. Ids are assigned here; component and requirement
    references are resolved by name and dropped when they resolve to nothing, so a model
    naming a component it never defined produces a missing edge rather than an invalid
    document.

    A decision the model produced without alternatives or consequences is *completed* rather
    than discarded, with an explicit placeholder saying none were recorded. Dropping it would
    hide that a decision was made; keeping it unmarked would let an unexamined choice look
    considered.

    Raises:
        ValueError: The resulting document fails architecture schema validation.
    """
    known = prd.requirement_ids if prd is not None else frozenset()

    component_ids: dict[str, str] = {}
    for index, item in enumerate(draft.components, start=1):
        component_ids[_slug(item.name)] = element_id("ARCH", index)

    components = tuple(
        ArchitectureComponent(
            component_id=component_ids[_slug(item.name)],
            name=item.name,
            kind=_coerce(ComponentKind, item.kind, ComponentKind.SERVICE),  # type: ignore[arg-type]
            responsibility=item.responsibility,
            depends_on=tuple(
                dict.fromkeys(
                    component_ids[_slug(name)]
                    for name in item.depends_on
                    if _slug(name) in component_ids
                    and component_ids[_slug(name)] != component_ids[_slug(item.name)]
                )
            ),
            satisfies=_filter_requirements(item.satisfies, known),
            technology=item.technology,
            paths=tuple(item.paths),
        )
        for item in draft.components
    )

    decisions = tuple(
        ArchitectureDecision(
            decision_id=element_id("ADR", index),
            title=item.title,
            context=item.context,
            decision=item.decision,
            alternatives=tuple(item.alternatives)
            or ("(none recorded -- no alternative was evaluated)",),
            rationale=item.rationale,
            consequences=tuple(item.consequences)
            or ("(none recorded -- the cost of this decision was not stated)",),
            affects_requirements=_filter_requirements(item.affects_requirements, known),
            risks=tuple(item.risks),
            confidence=_coerce(Confidence, item.confidence, Confidence.MEDIUM),  # type: ignore[arg-type]
        )
        for index, item in enumerate(draft.decisions, start=1)
    )

    technologies = tuple(
        TechnologyChoice(
            name=item.name,
            role=item.role,
            rationale=item.rationale,
            alternatives_rejected=tuple(item.alternatives_rejected),
            constraints_considered=tuple(item.constraints_considered),
        )
        for item in draft.technologies
    )

    entities = tuple(
        DataEntity(
            entity_id=element_id("ENT", index),
            name=item.name,
            description=item.description,
            fields=dict(item.fields),
            sensitive=item.sensitive,
            satisfies=_filter_requirements(item.satisfies, known),
        )
        for index, item in enumerate(draft.entities, start=1)
    )

    endpoints = tuple(
        ApiEndpoint(
            endpoint_id=element_id("API", index),
            method=item.method.upper()[:10],
            path=item.path,
            purpose=item.purpose,
            requires_authentication=item.requires_authentication,
            satisfies=_filter_requirements(item.satisfies, known),
        )
        for index, item in enumerate(draft.endpoints, start=1)
    )

    threats = tuple(
        Threat(
            threat_id=element_id("THR", index),
            asset=item.asset,
            description=item.description,
            mitigation=item.mitigation,
            severity=item.severity,
        )
        for index, item in enumerate(draft.threats, start=1)
    )

    # Data flow is derived from the component graph rather than asked for separately: a model
    # asked for both produces two descriptions that disagree, and the dependency edges are
    # the ones that were actually reasoned about.
    flows = tuple(
        DataFlowEdge(
            source=component.component_id,
            target=dependency,
            payload=f"data required by {component.name}",
            transport="in-process",
        )
        for component in components
        for dependency in component.depends_on
    )

    properties: set[ProductProperty] = set()
    for raw in draft.properties:
        try:
            properties.add(ProductProperty(raw.strip().upper()))
        except ValueError:
            continue

    return SystemArchitectureDocument(
        product_name=product_name or "Untitled product",
        overview=draft.overview,
        components=components,
        decisions=decisions,
        technologies=technologies,
        entities=entities,
        endpoints=endpoints,
        data_flows=flows,
        threats=threats,
        properties=frozenset(properties),
        constraints_considered=tuple(draft.constraints_considered),
        deliberate_omissions=tuple(draft.deliberate_omissions),
    )


def draft_to_plan(
    draft: ArchitectOutput,
    architecture: SystemArchitectureDocument,
    *,
    prd: PRDDocument | None = None,
    product_name: str = "",
) -> ImplementationPlanDocument:
    """Translate model output into a validated implementation plan.

    Task dependencies are resolved by name against the tasks in this plan; an edge naming
    something that is not a task is dropped. A *cycle* is deliberately **not** repaired here
    — it is left in place so :func:`~edith.product.validation.find_cycle` reports it. Silently
    breaking a cycle would hide that the Architect proposed work that cannot be ordered,
    which is a real signal about the design.

    Raises:
        ValueError: The resulting plan fails schema validation.
    """
    known = prd.requirement_ids if prd is not None else frozenset()
    component_ids = architecture.component_ids
    component_by_slug = {
        _slug(component.name): component.component_id
        for component in architecture.components
    }

    task_ids = {
        _slug(item.title): element_id("TASK", index)
        for index, item in enumerate(draft.tasks, start=1)
    }

    tasks: list[PlannedTask] = []
    for index, item in enumerate(draft.tasks, start=1):
        task_id = element_id("TASK", index)
        dependencies = tuple(
            dict.fromkeys(
                task_ids[_slug(name)]
                for name in item.depends_on
                if _slug(name) in task_ids and task_ids[_slug(name)] != task_id
            )
        )
        components = tuple(
            dict.fromkeys(
                component_by_slug[_slug(name)]
                for name in item.components
                if _slug(name) in component_by_slug
            )
        ) or tuple(
            name.strip().upper()
            for name in item.components
            if name.strip().upper() in component_ids
        )

        tasks.append(
            PlannedTask(
                task_id=task_id,
                title=item.title,
                description=item.description,
                agent=item.agent or "coder",
                depends_on=dependencies,
                implements=_filter_requirements(item.implements, known),
                components=components,
                acceptance_criteria=(item.acceptance,) if item.acceptance.strip() else (),
                complexity=_coerce(Complexity, item.complexity, Complexity.MEDIUM),  # type: ignore[arg-type]
                paths=tuple(item.paths),
            )
        )

    return ImplementationPlanDocument(
        product_name=product_name or architecture.product_name,
        goal=draft.overview[:2000],
        tasks=tuple(tasks),
    )
