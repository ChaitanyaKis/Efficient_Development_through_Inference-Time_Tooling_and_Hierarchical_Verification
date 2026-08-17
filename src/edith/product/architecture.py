"""Architecture artifacts: components, decisions, contracts, threats, and the plan.

The Architect's output is where product intent stops being description and starts
constraining implementation, so two rules shape these schemas.

**A decision without alternatives is not a decision.** :class:`ArchitectureDecision` requires
at least one alternative and a rationale. "We will use PostgreSQL" records a preference;
"We will use PostgreSQL over SQLite because concurrent writers are a stated requirement, at
the cost of an operational dependency" records a decision someone can later disagree with on
the merits. CLAUDE.md asks for context, alternatives, rationale, and consequences; the type
refuses to be constructed without them.

**The Architect may not quietly change what was asked for.** Every component, decision, and
task references the requirement ids it serves. An architecture that declares a structural
property contradicting the PRD is caught deterministically by
:mod:`edith.product.contradictions` -- not by asking a model whether the design looks right.

The implementation plan produced here is deliberately *not* executed by M4. It is a document
shaped so that M2's Planner can eventually consume it, which is a different and much safer
thing than an agent that plans and then runs its own plan.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator, model_validator

from edith.schemas.common import EdithModel

from .artifacts import ArtifactDocument, ArtifactKind, is_element_id, register_document
from .properties import ProductProperty


class ComponentKind(StrEnum):
    """What sort of thing a component is."""

    UI = "UI"
    SERVICE = "SERVICE"
    LIBRARY = "LIBRARY"
    DATASTORE = "DATASTORE"
    JOB = "JOB"
    #: Something the project depends on but does not own.
    EXTERNAL = "EXTERNAL"
    CLI = "CLI"


class Confidence(StrEnum):
    """How sure the Architect is.

    Recorded because a low-confidence decision is not a bad decision -- it is one that should
    be revisited when evidence arrives, and that is only possible if the uncertainty was
    written down at the time.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Complexity(StrEnum):
    """Rough effort for a task. Ordinal, not an estimate in hours."""

    TRIVIAL = "TRIVIAL"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    #: Too big to plan. Must be decomposed before it can be executed.
    UNKNOWN = "UNKNOWN"


class ArchitectureComponent(EdithModel):
    """One named part of the system."""

    component_id: str = Field(pattern=r"^ARCH-\d{3,}$")
    name: str = Field(min_length=1, max_length=200)
    kind: ComponentKind = ComponentKind.SERVICE
    responsibility: str = Field(min_length=1, max_length=1000)
    #: Other components this one calls or reads, by ARCH id.
    depends_on: tuple[str, ...] = ()
    #: Requirements this component exists to satisfy.
    satisfies: tuple[str, ...] = ()
    #: Where its code will live, as repo-relative path patterns. Becomes an agent's write
    #: scope later, which is why it is recorded now rather than invented at execution time.
    paths: tuple[str, ...] = ()
    technology: str = Field(default="", max_length=200)

    @field_validator("satisfies")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"satisfies entry {identifier!r} is not a REQ id")
        return value

    @model_validator(mode="after")
    def _no_self_dependency(self) -> ArchitectureComponent:
        if self.component_id in self.depends_on:
            raise ValueError(f"component {self.component_id} cannot depend on itself")
        return self


class ArchitectureDecision(EdithModel):
    """An ADR: what was decided, against what, and at what cost.

    ``alternatives`` and ``consequences`` are required. A record that lists neither is a
    preference wearing a decision's clothes, and it gives a future reader nothing to
    re-evaluate when circumstances change.
    """

    decision_id: str = Field(pattern=r"^ADR-\d{3,}$")
    title: str = Field(min_length=1, max_length=200)
    #: The situation that forced a choice.
    context: str = Field(min_length=1, max_length=2000)
    decision: str = Field(min_length=1, max_length=1000)
    #: What else was considered. At least one; "nothing else was possible" is rarely true and
    #: should be argued explicitly in the rationale if it is.
    alternatives: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)
    #: What this costs. Including the bad parts is the point.
    consequences: tuple[str, ...] = Field(min_length=1)
    #: Requirements this decision serves or affects.
    affects_requirements: tuple[str, ...] = ()
    #: Components this decision constrains.
    affects_components: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    confidence: Confidence = Confidence.MEDIUM
    #: Research report ids that informed this. Evidence, never authority: an Architect
    #: decides, research only supplies material for the decision.
    evidence: tuple[str, ...] = ()

    @field_validator("affects_requirements")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"affects_requirements entry {identifier!r} is not a REQ id")
        return value


class TechnologyChoice(EdithModel):
    """A selected technology and the constraints it was chosen under.

    CLAUDE.md forbids picking something because it is popular. The fields here are the ones
    that make an unjustified choice visible: a technology with no rejected alternative and no
    stated constraint is one nobody actually evaluated.
    """

    name: str = Field(min_length=1, max_length=200)
    #: What it is for: ``database``, ``web framework``, ``queue``.
    role: str = Field(min_length=1, max_length=100)
    version: str = Field(default="", max_length=50)
    rationale: str = Field(min_length=1, max_length=1000)
    alternatives_rejected: tuple[str, ...] = ()
    #: The project constraints that drove the choice: hardware, cost, team, deployment.
    constraints_considered: tuple[str, ...] = ()
    #: The ADR that records the decision in full.
    decision_id: str = Field(default="", max_length=40)
    confidence: Confidence = Confidence.MEDIUM


class DataEntity(EdithModel):
    """One entity in the data model."""

    entity_id: str = Field(pattern=r"^ENT-\d{3,}$")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    #: Field name -> type, as plain strings: a data model is a contract, not a migration.
    fields: dict[str, str] = Field(default_factory=dict)
    #: Relationships to other entities, described as "one-to-many ENT-002".
    relationships: tuple[str, ...] = ()
    #: Whether this entity holds data whose exposure would be harmful.
    sensitive: bool = False
    satisfies: tuple[str, ...] = ()


class ApiEndpoint(EdithModel):
    """One externally visible operation."""

    endpoint_id: str = Field(pattern=r"^API-\d{3,}$")
    method: str = Field(default="GET", max_length=10)
    path: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=1, max_length=500)
    request: str = Field(default="", max_length=1000)
    response: str = Field(default="", max_length=1000)
    #: Whether a caller must be authenticated. Checked against the PRD's declared properties.
    requires_authentication: bool = True
    errors: tuple[str, ...] = ()
    satisfies: tuple[str, ...] = ()
    component_id: str = Field(default="", max_length=40)


class DataFlowEdge(EdithModel):
    """Data moving from one component to another."""

    source: str = Field(min_length=1, max_length=40)
    target: str = Field(min_length=1, max_length=40)
    payload: str = Field(min_length=1, max_length=500)
    #: How it travels: ``HTTP``, ``in-process``, ``file``, ``queue``.
    transport: str = Field(default="", max_length=100)
    #: Whether what moves here is sensitive. Drives the threat model.
    sensitive: bool = False


class Threat(EdithModel):
    """One thing that could go wrong on purpose."""

    threat_id: str = Field(pattern=r"^THR-\d{3,}$")
    #: What is being protected.
    asset: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    #: STRIDE category, or any classification the project uses.
    category: str = Field(default="", max_length=100)
    #: What stops it. A threat with no mitigation is an accepted risk, and saying which it is
    #: matters more than the list being long.
    mitigation: str = Field(default="", max_length=1000)
    severity: str = Field(default="MEDIUM", max_length=20)
    affects_components: tuple[str, ...] = ()


class PlannedTask(EdithModel):
    """One task in the implementation plan.

    Deliberately shaped so M2's :class:`~edith.planning.task.Task` can be built from it. It
    is *not* that type: a plan is a proposal, and a proposal that could be executed directly
    would skip the translation step where planner output is re-validated against the strict
    schema. That step is a trust boundary and M4 does not get to cross it.
    """

    task_id: str = Field(pattern=r"^TASK-\d{3,}$")
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    #: The future agent responsible: ``backend``, ``frontend``, ``database``, ``devops``.
    agent: str = Field(default="coder", max_length=60)
    #: Tasks that must finish first.
    depends_on: tuple[str, ...] = ()
    #: Requirements this task delivers.
    implements: tuple[str, ...] = ()
    #: Architecture components it touches.
    components: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    #: Checks that must pass, by kind: ``tests``, ``lint``, ``typecheck``, ``build``.
    verification: tuple[str, ...] = ("tests",)
    complexity: Complexity = Complexity.MEDIUM
    #: Repo-relative paths this task is expected to change.
    paths: tuple[str, ...] = ()

    @field_validator("implements")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"implements entry {identifier!r} is not a REQ id")
        return value

    @field_validator("verification")
    @classmethod
    def _known_verification_kinds(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {"tests", "lint", "typecheck", "build"}
        for kind in value:
            if kind not in allowed:
                raise ValueError(
                    f"verification kind {kind!r} is not one of {sorted(allowed)}; a task "
                    "cannot invent a command, only select a configured one"
                )
        return value

    @model_validator(mode="after")
    def _no_self_dependency(self) -> PlannedTask:
        if self.task_id in self.depends_on:
            raise ValueError(f"task {self.task_id} cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"task {self.task_id} has duplicate dependencies")
        return self


@register_document
class SystemArchitectureDocument(ArtifactDocument):
    """The system architecture: components, decisions, contracts, and threats.

    One document rather than seven, because the M4 spec's separate artifacts
    (``DATA_FLOW``, ``API_CONTRACT``, ``DATA_MODEL``, ``THREAT_MODEL``,
    ``TECHNOLOGY_DECISIONS``) are all views over the same component graph. Splitting them
    into independently-versioned artifacts would let a data flow reference a component that
    a newer architecture had removed -- exactly the dangling reference M4.6 forbids. They are
    exposed as separate views via :meth:`data_flow_view` and friends.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.SYSTEM_ARCHITECTURE

    product_name: str = Field(min_length=1, max_length=200)
    overview: str = Field(min_length=1, max_length=4000)
    components: tuple[ArchitectureComponent, ...] = ()
    decisions: tuple[ArchitectureDecision, ...] = ()
    technologies: tuple[TechnologyChoice, ...] = ()
    entities: tuple[DataEntity, ...] = ()
    endpoints: tuple[ApiEndpoint, ...] = ()
    data_flows: tuple[DataFlowEdge, ...] = ()
    threats: tuple[Threat, ...] = ()
    #: Structural claims this architecture makes. Checked against the PRD's requirements.
    properties: frozenset[ProductProperty] = frozenset()
    #: Constraints the Architect worked under: hardware, budget, team, deployment target.
    constraints_considered: tuple[str, ...] = ()
    #: Deliberate simplifications, so an absent layer reads as a decision rather than an
    #: oversight. CLAUDE.md forbids infrastructure the project does not need.
    deliberate_omissions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> SystemArchitectureDocument:
        identifiers = list(self.element_ids())
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise ValueError(
                f"architecture defines duplicate element ids: {', '.join(duplicates)}"
            )
        return self

    def element_ids(self) -> tuple[str, ...]:
        """Every stable id this architecture defines."""
        return (
            tuple(item.component_id for item in self.components)
            + tuple(item.decision_id for item in self.decisions)
            + tuple(item.entity_id for item in self.entities)
            + tuple(item.endpoint_id for item in self.endpoints)
            + tuple(item.threat_id for item in self.threats)
        )

    def referenced_ids(self) -> tuple[str, ...]:
        """Every id this architecture points at."""
        references: list[str] = []
        for component in self.components:
            references.extend(component.depends_on)
            references.extend(component.satisfies)
        for decision in self.decisions:
            references.extend(decision.affects_requirements)
            references.extend(decision.affects_components)
        for technology in self.technologies:
            if technology.decision_id:
                references.append(technology.decision_id)
        for entity in self.entities:
            references.extend(entity.satisfies)
        for endpoint in self.endpoints:
            references.extend(endpoint.satisfies)
            if endpoint.component_id:
                references.append(endpoint.component_id)
        for edge in self.data_flows:
            references.extend((edge.source, edge.target))
        for threat in self.threats:
            references.extend(threat.affects_components)
        return tuple(references)

    # -- Convenience -----------------------------------------------------------------

    @property
    def component_ids(self) -> frozenset[str]:
        """Ids of every component."""
        return frozenset(item.component_id for item in self.components)

    @property
    def covered_requirements(self) -> frozenset[str]:
        """Requirement ids this architecture claims to address."""
        covered: set[str] = set()
        for component in self.components:
            covered |= set(component.satisfies)
        for entity in self.entities:
            covered |= set(entity.satisfies)
        for endpoint in self.endpoints:
            covered |= set(endpoint.satisfies)
        for decision in self.decisions:
            covered |= set(decision.affects_requirements)
        return frozenset(covered)

    def unmitigated_threats(self) -> tuple[str, ...]:
        """Threats with no stated mitigation."""
        return tuple(
            threat.threat_id for threat in self.threats if not threat.mitigation.strip()
        )

    def unjustified_technologies(self) -> tuple[str, ...]:
        """Technologies chosen without naming an alternative that was rejected.

        Not automatically wrong -- some choices genuinely have no competitor -- but it is the
        signature of a technology picked because it was familiar rather than evaluated.
        """
        return tuple(
            technology.name
            for technology in self.technologies
            if not technology.alternatives_rejected
        )

    def data_flow_view(self) -> tuple[DataFlowEdge, ...]:
        """The DATA_FLOW artifact, as a view over this document."""
        return self.data_flows

    def api_contract_view(self) -> tuple[ApiEndpoint, ...]:
        """The API_CONTRACT artifact, as a view."""
        return self.endpoints

    def data_model_view(self) -> tuple[DataEntity, ...]:
        """The DATA_MODEL artifact, as a view."""
        return self.entities

    def threat_model_view(self) -> tuple[Threat, ...]:
        """The THREAT_MODEL artifact, as a view."""
        return self.threats

    def technology_decisions_view(self) -> tuple[TechnologyChoice, ...]:
        """The TECHNOLOGY_DECISIONS artifact, as a view."""
        return self.technologies

    def render(self) -> str:
        """Render for a human reader or a downstream agent's prompt."""
        lines = [f"# Architecture: {self.product_name}", "", self.overview, ""]
        if self.components:
            lines.append("## Components")
            for component in self.components:
                depends = ", ".join(component.depends_on) or "-"
                lines.append(
                    f"- {component.component_id} [{component.kind}] {component.name}: "
                    f"{component.responsibility} (depends on: {depends})"
                )
            lines.append("")
        if self.decisions:
            lines.append("## Decisions")
            for decision in self.decisions:
                lines.append(
                    f"- {decision.decision_id} {decision.title} "
                    f"[{decision.confidence}]: {decision.decision}"
                )
                lines.append(f"    alternatives: {', '.join(decision.alternatives)}")
            lines.append("")
        if self.deliberate_omissions:
            lines.append("## Deliberately omitted")
            lines.extend(f"- {item}" for item in self.deliberate_omissions)
        return "\n".join(lines).strip()


@register_document
class ImplementationPlanDocument(ArtifactDocument):
    """Tasks the M2 Planner can eventually consume.

    The body of an :attr:`~edith.product.artifacts.ArtifactKind.IMPLEMENTATION_PLAN`
    artifact. M4 produces this and stops; execution is M2's job and M7's to connect.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.IMPLEMENTATION_PLAN

    product_name: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=2000)
    tasks: tuple[PlannedTask, ...] = ()
    #: Ordered groups of task ids that can be delivered together.
    milestones: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _unique_ids(self) -> ImplementationPlanDocument:
        identifiers = [task.task_id for task in self.tasks]
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise ValueError(f"plan defines duplicate task ids: {', '.join(duplicates)}")
        return self

    def element_ids(self) -> tuple[str, ...]:
        """Every task id this plan defines."""
        return tuple(task.task_id for task in self.tasks)

    def referenced_ids(self) -> tuple[str, ...]:
        """Every id this plan points at."""
        references: list[str] = []
        for task in self.tasks:
            references.extend(task.depends_on)
            references.extend(task.implements)
            references.extend(task.components)
        for group in self.milestones.values():
            references.extend(group)
        return tuple(references)

    @property
    def task_ids(self) -> frozenset[str]:
        """Ids of every task."""
        return frozenset(task.task_id for task in self.tasks)

    @property
    def covered_requirements(self) -> frozenset[str]:
        """Requirement ids some task claims to implement."""
        covered: set[str] = set()
        for task in self.tasks:
            covered |= set(task.implements)
        return frozenset(covered)

    def render(self) -> str:
        """Render for a human reader."""
        lines = [f"# Implementation plan: {self.product_name}", "", self.goal, ""]
        for task in self.tasks:
            depends = ", ".join(task.depends_on) or "-"
            implements = ", ".join(task.implements) or "-"
            lines.append(
                f"- {task.task_id} [{task.complexity}] ({task.agent}) {task.title} "
                f"| after: {depends} | implements: {implements}"
            )
        return "\n".join(lines).strip()
