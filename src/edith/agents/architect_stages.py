"""Decomposed architecture generation: six small schemas instead of one large one.

Same hypothesis as :mod:`edith.agents.ux_stages`, applied to the larger of the two agents.
The monolithic ``ArchitectOutput`` renders to well over 6,000 bytes of JSON Schema — larger
than the UX schema that failed six consecutive times — so it was never expected to work and
was never observed to, because the M4 pipeline stopped before reaching it.

The decomposition follows the dependency order the milestone names:

===================  ===============================================  =====================
stage                produces                                         depends on
===================  ===============================================  =====================
``components``       the component graph, overview, properties        PRD, UX
``data``             entities and their fields                        components
``api``              externally visible operations                    components
``decisions``        ADRs and technology choices                      components
``threats``          assets, threats, mitigations                     components, data
``plan``             implementation tasks                             components
===================  ===============================================  =====================

Every downstream stage takes the *component names* rather than the whole architecture, which
keeps each prompt small and stops context cost scaling with the number of stages.

Ids remain system-owned throughout. The model names things; :func:`assemble_architecture`
numbers them, resolves references, and drops anything that resolves to nothing.
"""

from __future__ import annotations

from pydantic import Field

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
from edith.schemas.common import EdithModel

MAX_COMPONENTS = 6
MAX_DECISIONS = 4
MAX_TASKS = 8

# -- Stage 1: components -----------------------------------------------------------------

COMPONENTS_SYSTEM = """You are the architecture component of a software engineering system.

Decompose this product into the smallest set of parts that satisfies the requirements.

Rules:
- Choose the SIMPLEST design. Do not add a queue, a cache, a microservice, or a database
  the requirements do not need. Unnecessary infrastructure is a cost paid forever.
- `kind` is exactly one of: UI, SERVICE, LIBRARY, DATASTORE, JOB, EXTERNAL, CLI.
- Use EXTERNAL only for something the project does not own and cannot run locally.
- `depends_on` names OTHER components in this same list, by name.
- `satisfies` lists requirement IDs (like REQ-002) from the requirements you were given.
- `properties` tags structural claims about what you are building, from exactly this list,
  or leave it empty: OFFLINE_CAPABLE, REQUIRES_NETWORK, CLOUD_DEPENDENT, LOCAL_ONLY,
  AUTHENTICATION_REQUIRED, NO_AUTHENTICATION, AUTHORIZATION_REQUIRED, MULTI_USER,
  SINGLE_USER, MULTI_TENANT, MOBILE_RESPONSIVE, DESKTOP_ONLY, ACCESSIBLE, HEADLESS,
  PERSISTENT_STORAGE, EPHEMERAL_STORAGE, SENSITIVE_DATA, DATA_RESIDENCY, REAL_TIME,
  ASYNCHRONOUS, HIGH_AVAILABILITY, RESOURCE_CONSTRAINED.
- Do NOT invent component numbers or IDs."""

COMPONENTS_USER = """REQUIREMENTS:
{requirements}

USER INTERFACE:
{ux}

PROJECT CONSTRAINTS (binding):
{constraints}

Decompose the system."""


class ComponentSketch(EdithModel):
    """One component as the model produces it."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="SERVICE", max_length=30)
    responsibility: str = Field(min_length=1, max_length=500)
    depends_on: list[str] = Field(default_factory=list, max_length=5)
    satisfies: list[str] = Field(default_factory=list, max_length=6)
    technology: str = Field(default="", max_length=100)


class ComponentsOutput(EdithModel):
    """Stage 1 output: the component graph."""

    overview: str = Field(min_length=1, max_length=1000)
    components: list[ComponentSketch] = Field(min_length=1, max_length=MAX_COMPONENTS)
    properties: list[str] = Field(default_factory=list, max_length=8)
    #: Things deliberately left out, so an absent layer reads as a decision.
    deliberate_omissions: list[str] = Field(default_factory=list, max_length=4)


# -- Stage 2: data model ------------------------------------------------------------------

DATA_SYSTEM = """You are the architecture component of a software engineering system.

Describe the data this product stores.

Rules:
- One entity per thing the product must remember.
- `fields` maps a field name to a plain type name such as str, int, bool, datetime.
- Mark `sensitive` true when losing or exposing that data would genuinely harm someone.
- `satisfies` lists requirement IDs from the requirements you were given.
- If this product stores nothing at all, return an empty list.
- Do NOT invent entity numbers or IDs."""

DATA_USER = """REQUIREMENTS:
{requirements}

COMPONENTS:
{components}

Describe the data model."""


class EntitySketch(EdithModel):
    """One stored entity."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    fields: dict[str, str] = Field(default_factory=dict)
    sensitive: bool = False
    satisfies: list[str] = Field(default_factory=list, max_length=6)


class DataModelOutput(EdithModel):
    """Stage 2 output: the entities."""

    entities: list[EntitySketch] = Field(default_factory=list, max_length=8)


# -- Stage 3: API contract ----------------------------------------------------------------

API_SYSTEM = """You are the architecture component of a software engineering system.

Describe the operations this product exposes to callers outside it.

Rules:
- If the product has no external interface -- it is a local tool, a library, or a CLI --
  return an empty list. Do not invent an HTTP API for something that does not need one.
- `requires_authentication` must reflect the requirements, not a default.
- `satisfies` lists requirement IDs from the requirements you were given.
- Do NOT invent endpoint numbers or IDs."""

API_USER = """REQUIREMENTS:
{requirements}

COMPONENTS:
{components}

Describe the API contract."""


class EndpointSketch(EdithModel):
    """One externally visible operation."""

    method: str = Field(default="GET", max_length=10)
    path: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=300)
    requires_authentication: bool = True
    satisfies: list[str] = Field(default_factory=list, max_length=6)


class ApiContractOutput(EdithModel):
    """Stage 3 output: the endpoints."""

    endpoints: list[EndpointSketch] = Field(default_factory=list, max_length=8)


# -- Stage 4: decisions -------------------------------------------------------------------

DECISIONS_SYSTEM = """You are the architecture component of a software engineering system.

Record the decisions behind this design.

Rules:
- Every decision MUST name at least one alternative you rejected and at least one
  consequence you are accepting, including a bad one. A decision with no downside was not
  a decision -- it was a preference.
- Do NOT choose a technology because it is popular. Justify it under the project's actual
  constraints: hardware, budget, deployment, team, complexity.
- `confidence` is exactly one of HIGH, MEDIUM, LOW.
- `affects_requirements` lists requirement IDs from the requirements you were given.
- Do NOT invent decision numbers or IDs."""

DECISIONS_USER = """REQUIREMENTS:
{requirements}

COMPONENTS:
{components}

PROJECT CONSTRAINTS (binding):
{constraints}

Record the architecture decisions and technology choices."""


class DecisionSketch(EdithModel):
    """One architecture decision."""

    title: str = Field(min_length=1, max_length=200)
    context: str = Field(min_length=1, max_length=800)
    decision: str = Field(min_length=1, max_length=500)
    alternatives: list[str] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=800)
    consequences: list[str] = Field(min_length=1, max_length=4)
    affects_requirements: list[str] = Field(default_factory=list, max_length=6)
    confidence: str = Field(default="MEDIUM", max_length=20)


class TechnologySketch(EdithModel):
    """One technology choice."""

    name: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=60)
    rationale: str = Field(min_length=1, max_length=500)
    alternatives_rejected: list[str] = Field(default_factory=list, max_length=4)
    constraints_considered: list[str] = Field(default_factory=list, max_length=4)


class DecisionsOutput(EdithModel):
    """Stage 4 output: ADRs and technology choices."""

    decisions: list[DecisionSketch] = Field(default_factory=list, max_length=MAX_DECISIONS)
    technologies: list[TechnologySketch] = Field(default_factory=list, max_length=6)


# -- Stage 5: threat model ----------------------------------------------------------------

THREATS_SYSTEM = """You are the security component of a software engineering system.

Describe what could go wrong on purpose.

Rules:
- Name the asset being protected, what an attacker would do, and what stops them.
- If a risk is being ACCEPTED rather than mitigated, say so in the mitigation field. A
  blank mitigation is indistinguishable from nobody having thought about it.
- `severity` is LOW, MEDIUM, HIGH, or CRITICAL.
- Do NOT invent threat numbers or IDs."""

THREATS_USER = """COMPONENTS:
{components}

DATA STORED:
{entities}

Describe the threats and their mitigations."""


class ThreatSketch(EdithModel):
    """One threat."""

    asset: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    mitigation: str = Field(default="", max_length=500)
    severity: str = Field(default="MEDIUM", max_length=20)


class ThreatModelOutput(EdithModel):
    """Stage 5 output: the threat model."""

    threats: list[ThreatSketch] = Field(default_factory=list, max_length=6)


# -- Stage 6: implementation plan ---------------------------------------------------------

PLAN_SYSTEM = """You are the planning component of a software engineering system.

Break this architecture into concrete implementation tasks.

Rules:
- A task is a unit of work that changes files. Do not create tasks like "investigate",
  "review", or "run the tests" -- the system runs the real tests after every task.
- `depends_on` names OTHER tasks in this same list, by name. Do NOT create a cycle.
- `components` names components from the list you were given.
- `implements` lists requirement IDs from the requirements you were given.
- `complexity` is exactly one of TRIVIAL, SMALL, MEDIUM, LARGE, UNKNOWN.
- Do NOT invent task numbers or IDs."""

PLAN_USER = """REQUIREMENTS:
{requirements}

COMPONENTS:
{components}

Break the work into implementation tasks."""


class TaskSketch(EdithModel):
    """One implementation task."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=800)
    agent: str = Field(default="coder", max_length=40)
    depends_on: list[str] = Field(default_factory=list, max_length=5)
    implements: list[str] = Field(default_factory=list, max_length=6)
    components: list[str] = Field(default_factory=list, max_length=4)
    acceptance: str = Field(default="", max_length=400)
    complexity: str = Field(default="MEDIUM", max_length=20)


class PlanOutput(EdithModel):
    """Stage 6 output: the implementation tasks."""

    tasks: list[TaskSketch] = Field(min_length=1, max_length=MAX_TASKS)


# -- Context construction ------------------------------------------------------------------


def requirements_context(prd: PRDDocument, *, limit: int = 12) -> str:
    """Requirement lines: id, priority, statement. Not the whole PRD."""
    lines = [
        f"{item.requirement_id} [{item.priority}] {item.title}: {item.statement}"
        for item in prd.requirements[:limit]
    ]
    return "\n".join(lines) or "(no requirements)"


def components_context(components: tuple[ComponentSketch, ...]) -> str:
    """Component names and responsibilities, for every downstream stage.

    Names, not the full component records. A downstream stage needs to know what exists so
    it can reference it; the dependency edges and paths are the component stage's business.
    """
    return (
        "\n".join(f"- {item.name} ({item.kind}): {item.responsibility}" for item in components)
        or "(no components)"
    )


def entities_context(entities: tuple[EntitySketch, ...]) -> str:
    """Entity names and sensitivity, for the threat stage."""
    if not entities:
        return "(this product stores nothing)"
    return "\n".join(
        f"- {item.name}{' [SENSITIVE]' if item.sensitive else ''}: {item.description}"
        for item in entities
    )


# -- Deterministic assembly -----------------------------------------------------------------


def _slug(value: str) -> str:
    """Normalise a name for matching model-supplied cross-references."""
    return "".join(char for char in value.lower() if char.isalnum())


def _coerce(enum_type: type, value: str, fallback: object) -> object:
    try:
        return enum_type(value.strip().upper())
    except ValueError:
        return fallback


def _filter_requirements(values: list[str], known: frozenset[str]) -> tuple[str, ...]:
    """Keep only requirement ids the PRD defines."""
    return tuple(
        dict.fromkeys(
            item.strip().upper() for item in values if item.strip().upper() in known
        )
    )


def assemble_architecture(
    *,
    product_name: str,
    prd: PRDDocument | None,
    components: ComponentsOutput,
    data: DataModelOutput | None,
    api: ApiContractOutput | None,
    decisions: DecisionsOutput | None,
    threats: ThreatModelOutput | None,
) -> SystemArchitectureDocument:
    """Assemble validated stage outputs into one architecture document.

    Deterministic throughout. Ids are assigned here; component and requirement references
    are resolved by name; anything unresolvable is dropped rather than emitted as a dangling
    edge. A stage that failed simply contributes nothing, which is why the architecture can
    be assembled from a partial run and still be internally consistent.

    Raises:
        ValueError: The assembled document fails architecture schema validation.
    """
    known = prd.requirement_ids if prd is not None else frozenset()

    component_ids: dict[str, str] = {
        _slug(item.name): element_id("ARCH", index)
        for index, item in enumerate(components.components, start=1)
    }

    built_components = tuple(
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
        )
        for item in components.components
    )

    built_entities = tuple(
        DataEntity(
            entity_id=element_id("ENT", index),
            name=item.name,
            description=item.description,
            fields=dict(item.fields),
            sensitive=item.sensitive,
            satisfies=_filter_requirements(item.satisfies, known),
        )
        for index, item in enumerate(data.entities if data else (), start=1)
    )

    built_endpoints = tuple(
        ApiEndpoint(
            endpoint_id=element_id("API", index),
            method=item.method.upper()[:10],
            path=item.path,
            purpose=item.purpose,
            requires_authentication=item.requires_authentication,
            satisfies=_filter_requirements(item.satisfies, known),
        )
        for index, item in enumerate(api.endpoints if api else (), start=1)
    )

    built_decisions = tuple(
        ArchitectureDecision(
            decision_id=element_id("ADR", index),
            title=item.title,
            context=item.context,
            decision=item.decision,
            alternatives=tuple(item.alternatives),
            rationale=item.rationale,
            consequences=tuple(item.consequences),
            affects_requirements=_filter_requirements(item.affects_requirements, known),
            confidence=_coerce(Confidence, item.confidence, Confidence.MEDIUM),  # type: ignore[arg-type]
        )
        for index, item in enumerate(decisions.decisions if decisions else (), start=1)
    )

    built_technologies = tuple(
        TechnologyChoice(
            name=item.name,
            role=item.role,
            rationale=item.rationale,
            alternatives_rejected=tuple(item.alternatives_rejected),
            constraints_considered=tuple(item.constraints_considered),
        )
        for item in (decisions.technologies if decisions else ())
    )

    built_threats = tuple(
        Threat(
            threat_id=element_id("THR", index),
            asset=item.asset,
            description=item.description,
            mitigation=item.mitigation,
            severity=item.severity,
        )
        for index, item in enumerate(threats.threats if threats else (), start=1)
    )

    # Data flow is derived from the component graph rather than generated: a model asked for
    # both produces two descriptions that disagree, and the dependency edges are the ones it
    # actually reasoned about.
    flows = tuple(
        DataFlowEdge(
            source=component.component_id,
            target=dependency,
            payload=f"data required by {component.name}",
            transport="in-process",
        )
        for component in built_components
        for dependency in component.depends_on
    )

    properties: set[ProductProperty] = set()
    for raw in components.properties:
        try:
            properties.add(ProductProperty(raw.strip().upper()))
        except ValueError:
            continue

    return SystemArchitectureDocument(
        product_name=product_name or "Untitled product",
        overview=components.overview,
        components=built_components,
        decisions=built_decisions,
        technologies=built_technologies,
        entities=built_entities,
        endpoints=built_endpoints,
        data_flows=flows,
        threats=built_threats,
        properties=frozenset(properties),
        deliberate_omissions=tuple(components.deliberate_omissions),
    )


def assemble_plan(
    *,
    product_name: str,
    goal: str,
    plan: PlanOutput,
    architecture: SystemArchitectureDocument,
    prd: PRDDocument | None,
) -> ImplementationPlanDocument:
    """Assemble the implementation plan.

    Task dependencies are resolved by name; an edge naming a non-task is dropped. A *cycle*
    is deliberately preserved so validation reports it — silently breaking one would hide
    that the Architect proposed work that cannot be ordered.

    Raises:
        ValueError: The assembled plan fails schema validation.
    """
    known = prd.requirement_ids if prd is not None else frozenset()
    component_by_slug = {
        _slug(component.name): component.component_id
        for component in architecture.components
    }
    task_ids = {
        _slug(item.title): element_id("TASK", index)
        for index, item in enumerate(plan.tasks, start=1)
    }

    tasks = []
    for index, item in enumerate(plan.tasks, start=1):
        task_id = element_id("TASK", index)
        tasks.append(
            PlannedTask(
                task_id=task_id,
                title=item.title,
                description=item.description,
                agent=item.agent or "coder",
                depends_on=tuple(
                    dict.fromkeys(
                        task_ids[_slug(name)]
                        for name in item.depends_on
                        if _slug(name) in task_ids and task_ids[_slug(name)] != task_id
                    )
                ),
                implements=_filter_requirements(item.implements, known),
                components=tuple(
                    dict.fromkeys(
                        component_by_slug[_slug(name)]
                        for name in item.components
                        if _slug(name) in component_by_slug
                    )
                ),
                acceptance_criteria=(item.acceptance,) if item.acceptance.strip() else (),
                complexity=_coerce(Complexity, item.complexity, Complexity.MEDIUM),  # type: ignore[arg-type]
            )
        )

    return ImplementationPlanDocument(
        product_name=product_name or "Untitled product",
        goal=goal[:2000] or "Implement the architecture.",
        tasks=tuple(tasks),
    )
