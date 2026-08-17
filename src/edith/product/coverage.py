"""Deterministic requirement coverage: is each requirement actually addressed?

M4.1 proved the decomposed pipeline produces *valid* artifacts — 10/10 trials, every schema
satisfied, every reference resolving. It also measured UX requirement coverage at 0.67. A
specification can be perfectly well-formed and still fail to mention a third of what was
asked for, and validity checks cannot see that: a requirement nothing references is not a
broken reference, it is an absence, and absences do not fail schemas.

So coverage is computed from *evidence*, not from a model's opinion:

``COVERED``
    Some element explicitly names the requirement id. The element is the evidence.
``PARTIALLY_COVERED``
    Only a weaker element names it. A screen without a flow is a place the user can reach
    with no journey that reaches it; a decision without a component is an intention nobody
    implements.
``MISSING``
    Nothing names it at all.
``CONTRADICTED``
    Something names it *and* structurally conflicts with it. Worse than missing, because it
    looks addressed.
``NOT_APPLICABLE``
    The requirement genuinely has nothing to do with this artifact kind. A performance
    budget has no user flow; forcing a mapping would manufacture coverage.

A model is never asked "is REQ-003 covered?". The Critic may supply an advisory opinion, and
:class:`CoverageEntry` keeps it in a separate field labelled as such, because an LLM-only
coverage judgement that becomes authoritative is exactly how a gap gets marked closed
without anything being built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from edith.observability.logging import get_logger

from .architecture import ImplementationPlanDocument, SystemArchitectureDocument
from .artifacts import ArtifactKind
from .contradictions import Contradiction, check_all
from .prd import PRDDocument, Priority, Requirement, RequirementKind
from .properties import ProductProperty, find_conflicts

logger = get_logger(__name__)


class CoverageState(StrEnum):
    """How well one artifact addresses one requirement."""

    COVERED = "COVERED"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    MISSING = "MISSING"
    #: Addressed, but by something that structurally conflicts with it.
    CONTRADICTED = "CONTRADICTED"
    #: This artifact kind is not where this requirement would be satisfied.
    NOT_APPLICABLE = "NOT_APPLICABLE"

    @property
    def satisfied(self) -> bool:
        """Whether this state counts as the requirement being addressed."""
        return self in {CoverageState.COVERED, CoverageState.NOT_APPLICABLE}

    @property
    def is_gap(self) -> bool:
        """Whether this state represents work not done."""
        return self in {
            CoverageState.MISSING,
            CoverageState.PARTIALLY_COVERED,
            CoverageState.CONTRADICTED,
        }


class Criticality(StrEnum):
    """How much a coverage gap matters.

    Derived from the requirement's own priority rather than guessed at, so the threshold is
    a property of what the user asked for and not of what the system found convenient.
    """

    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    OPTIONAL = "OPTIONAL"


#: Priority -> criticality. ``WONT`` is deliberately optional: it is an explicit decision not
#: to build something, so an artifact not covering it is correct rather than deficient.
_CRITICALITY: dict[Priority, Criticality] = {
    Priority.MUST: Criticality.CRITICAL,
    Priority.SHOULD: Criticality.IMPORTANT,
    Priority.COULD: Criticality.OPTIONAL,
    Priority.WONT: Criticality.OPTIONAL,
}


def criticality_of(requirement: Requirement) -> Criticality:
    """How much a gap against this requirement matters."""
    return _CRITICALITY.get(requirement.priority, Criticality.IMPORTANT)


def applies_to(requirement: Requirement, kind: ArtifactKind) -> bool:
    """Whether this requirement is one this artifact kind should address.

    The rule that stops the matrix manufacturing coverage. A non-functional requirement
    ("results within 2 seconds") has no user flow; a constraint ("runs on one laptop") has no
    screen. Both belong to the architecture. Marking them MISSING from a UX specification
    would report a gap nobody should close.

    Functional requirements apply everywhere, because a capability the user invokes needs an
    interface, a component, and a task.
    """
    if requirement.priority is Priority.WONT:
        return False
    if kind is ArtifactKind.UX_SPEC:
        return requirement.kind is RequirementKind.FUNCTIONAL
    # Architecture and the plan carry everything else: a non-functional requirement is
    # satisfied by a design decision, and a constraint by what was built.
    return True


@dataclass(frozen=True)
class CoverageEvidence:
    """A specific element that addresses a requirement.

    Evidence is an element id, never prose. "The overview mentions stock levels" is not
    coverage; ``UX-002`` claiming to satisfy ``REQ-001`` is.
    """

    element_id: str
    #: What kind of element it is: flow, screen, component, entity, endpoint, decision, task.
    element_kind: str
    detail: str = ""

    def render(self) -> str:
        return f"{self.element_id} ({self.element_kind})"


#: Element kinds that fully cover a requirement in a UX specification. A flow is a journey
#: that delivers the capability; a screen alone is somewhere the user can be.
_UX_STRONG = frozenset({"flow"})
_UX_WEAK = frozenset({"screen"})

#: Element kinds that fully cover a requirement in an architecture. A component is a thing
#: that gets built; a decision or a threat is a statement about it.
_ARCH_STRONG = frozenset({"component", "entity", "endpoint"})
_ARCH_WEAK = frozenset({"decision"})


@dataclass
class CoverageEntry:
    """One row of the coverage matrix: a requirement, across every artifact."""

    requirement_id: str
    title: str
    criticality: Criticality
    ux: CoverageState = CoverageState.MISSING
    architecture: CoverageState = CoverageState.MISSING
    plan: CoverageState = CoverageState.MISSING
    ux_evidence: tuple[CoverageEvidence, ...] = ()
    architecture_evidence: tuple[CoverageEvidence, ...] = ()
    plan_evidence: tuple[CoverageEvidence, ...] = ()
    #: An advisory opinion, when one was requested. Never authoritative, and never allowed
    #: to change a computed state.
    critic_note: str = ""

    def state_for(self, kind: ArtifactKind) -> CoverageState:
        """The computed state for one artifact kind."""
        if kind is ArtifactKind.UX_SPEC:
            return self.ux
        if kind is ArtifactKind.IMPLEMENTATION_PLAN:
            return self.plan
        return self.architecture

    def evidence_for(self, kind: ArtifactKind) -> tuple[CoverageEvidence, ...]:
        """The evidence behind one artifact kind's state."""
        if kind is ArtifactKind.UX_SPEC:
            return self.ux_evidence
        if kind is ArtifactKind.IMPLEMENTATION_PLAN:
            return self.plan_evidence
        return self.architecture_evidence

    @property
    def fully_covered(self) -> bool:
        """Whether every applicable artifact addresses this requirement."""
        return all(
            state.satisfied for state in (self.ux, self.architecture, self.plan)
        )

    def render(self) -> str:
        """One matrix row."""
        evidence = ", ".join(
            item.element_id
            for item in (*self.ux_evidence, *self.architecture_evidence)
        )
        return (
            f"{self.requirement_id} [{self.criticality}] | {self.ux} | "
            f"{self.architecture} | {evidence or '-'}"
        )


@dataclass(frozen=True)
class CoverageGap:
    """A requirement an artifact does not address.

    The structured ``COVERAGE_GAP`` M4.2 item 3 specifies: which requirement, which artifact,
    which stage would produce it, how much it matters, and what evidence exists.
    """

    requirement_id: str
    title: str
    artifact: ArtifactKind
    state: CoverageState
    criticality: Criticality
    #: The generation stage that would close this gap.
    stage: str
    evidence: tuple[CoverageEvidence, ...] = ()
    detail: str = ""

    @property
    def blocking(self) -> bool:
        """Whether this gap must prevent approval.

        Critical requirements block. So does a contradiction at any criticality: an artifact
        that claims to address a requirement while conflicting with it is worse than one that
        omits it, because it looks finished.
        """
        return (
            self.criticality is Criticality.CRITICAL
            or self.state is CoverageState.CONTRADICTED
        )

    @property
    def code(self) -> str:
        """The stable issue code for this gap."""
        return "COVERAGE_GAP" if self.blocking else "ADVISORY_COVERAGE_GAP"

    def render(self) -> str:
        marker = "" if self.blocking else " (advisory)"
        return (
            f"[{self.code}]{marker} {self.requirement_id} is {self.state} in "
            f"{self.artifact.value} (stage {self.stage}): {self.title}"
        )


@dataclass
class CoverageThreshold:
    """The explicit, testable policy for what coverage is good enough.

    M4.2 item 6 forbids declaring 100% mandatory by default. Critical requirements must be
    covered; the rest have a fractional floor that a caller can state and a test can check.
    """

    #: Every CRITICAL requirement must be COVERED (or NOT_APPLICABLE).
    require_all_critical: bool = True
    #: Minimum fraction of IMPORTANT requirements that must be covered.
    minimum_important: float = 0.5
    #: Minimum fraction across every applicable requirement.
    minimum_overall: float = 0.0

    def describe(self) -> str:
        """A readable statement of the policy."""
        parts = []
        if self.require_all_critical:
            parts.append("every critical requirement covered")
        if self.minimum_important > 0:
            parts.append(f"at least {self.minimum_important:.0%} of important requirements")
        if self.minimum_overall > 0:
            parts.append(f"at least {self.minimum_overall:.0%} overall")
        return "; ".join(parts) or "no coverage requirement"


DEFAULT_THRESHOLD = CoverageThreshold()


@dataclass
class CoverageMatrix:
    """Requirement coverage across every artifact in a project."""

    entries: list[CoverageEntry] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)
    contradictions: tuple[Contradiction, ...] = ()

    def entry(self, requirement_id: str) -> CoverageEntry | None:
        """One requirement's row."""
        for item in self.entries:
            if item.requirement_id == requirement_id:
                return item
        return None

    def gaps_for(self, kind: ArtifactKind) -> tuple[CoverageGap, ...]:
        """Gaps against one artifact kind."""
        return tuple(gap for gap in self.gaps if gap.artifact is kind)

    @property
    def blocking_gaps(self) -> tuple[CoverageGap, ...]:
        """Gaps that must prevent approval."""
        return tuple(gap for gap in self.gaps if gap.blocking)

    def coverage(self, kind: ArtifactKind) -> float:
        """Fraction of applicable requirements this artifact kind covers.

        Not-applicable requirements are excluded from both numerator and denominator: a
        performance budget with no user flow should neither count against a UX specification
        nor inflate its score.
        """
        applicable = [
            entry
            for entry in self.entries
            if entry.state_for(kind) is not CoverageState.NOT_APPLICABLE
        ]
        if not applicable:
            return 1.0
        covered = sum(
            1 for entry in applicable if entry.state_for(kind) is CoverageState.COVERED
        )
        return covered / len(applicable)

    def coverage_at(self, kind: ArtifactKind, criticality: Criticality) -> float:
        """Coverage restricted to one criticality band."""
        applicable = [
            entry
            for entry in self.entries
            if entry.criticality is criticality
            and entry.state_for(kind) is not CoverageState.NOT_APPLICABLE
        ]
        if not applicable:
            return 1.0
        covered = sum(
            1 for entry in applicable if entry.state_for(kind) is CoverageState.COVERED
        )
        return covered / len(applicable)

    def missing(self, kind: ArtifactKind) -> tuple[str, ...]:
        """Requirement ids this artifact kind does not address at all."""
        return tuple(
            entry.requirement_id
            for entry in self.entries
            if entry.state_for(kind) is CoverageState.MISSING
        )

    def partial(self, kind: ArtifactKind) -> tuple[str, ...]:
        """Requirement ids this artifact kind only weakly addresses."""
        return tuple(
            entry.requirement_id
            for entry in self.entries
            if entry.state_for(kind) is CoverageState.PARTIALLY_COVERED
        )

    def satisfies(
        self, kind: ArtifactKind, threshold: CoverageThreshold = DEFAULT_THRESHOLD
    ) -> bool:
        """Whether coverage of one artifact kind meets the policy."""
        if threshold.require_all_critical and self.coverage_at(
            kind, Criticality.CRITICAL
        ) < 1.0:
            return False
        if self.coverage_at(kind, Criticality.IMPORTANT) < threshold.minimum_important:
            return False
        return self.coverage(kind) >= threshold.minimum_overall

    def render(self) -> str:
        """The coverage matrix as a readable table."""
        lines = ["Requirement | UX | Architecture | Evidence", "-" * 72]
        lines.extend(entry.render() for entry in self.entries)
        if self.gaps:
            lines.extend(["", "Gaps:"])
            lines.extend(f"  {gap.render()}" for gap in self.gaps)
        return "\n".join(lines)


def _collect_ux(ux: object, requirement_id: str) -> list[tuple[str, CoverageEvidence]]:
    """Every UX element that claims to satisfy a requirement, with its strength."""
    found: list[tuple[str, CoverageEvidence]] = []
    for flow in getattr(ux, "flows", ()):
        if requirement_id in flow.satisfies:
            found.append(
                ("flow", CoverageEvidence(flow.flow_id, "flow", flow.name))
            )
    for screen in getattr(ux, "screens", ()):
        if requirement_id in screen.satisfies:
            found.append(
                ("screen", CoverageEvidence(screen.screen_id, "screen", screen.name))
            )
    return found


def _collect_architecture(
    architecture: SystemArchitectureDocument, requirement_id: str
) -> list[tuple[str, CoverageEvidence]]:
    """Every architecture element that claims to satisfy a requirement."""
    found: list[tuple[str, CoverageEvidence]] = []
    for component in architecture.components:
        if requirement_id in component.satisfies:
            found.append(
                (
                    "component",
                    CoverageEvidence(component.component_id, "component", component.name),
                )
            )
    for entity in architecture.entities:
        if requirement_id in entity.satisfies:
            found.append(
                ("entity", CoverageEvidence(entity.entity_id, "entity", entity.name))
            )
    for endpoint in architecture.endpoints:
        if requirement_id in endpoint.satisfies:
            found.append(
                (
                    "endpoint",
                    CoverageEvidence(endpoint.endpoint_id, "endpoint", endpoint.path),
                )
            )
    for decision in architecture.decisions:
        if requirement_id in decision.affects_requirements:
            found.append(
                (
                    "decision",
                    CoverageEvidence(decision.decision_id, "decision", decision.title),
                )
            )
    return found


def _state_from(
    found: list[tuple[str, CoverageEvidence]],
    strong: frozenset[str],
    weak: frozenset[str],
) -> tuple[CoverageState, tuple[CoverageEvidence, ...]]:
    """Reduce collected evidence to a coverage state.

    Strong evidence covers. Weak evidence alone is partial -- it means something in the
    artifact acknowledges the requirement but nothing delivers it.
    """
    if not found:
        return (CoverageState.MISSING, ())
    evidence = tuple(item for _, item in found)
    kinds = {kind for kind, _ in found}
    if kinds & strong:
        return (CoverageState.COVERED, evidence)
    # Weak evidence, or evidence of a kind nothing classified: either way something in the
    # artifact acknowledges the requirement but nothing delivers it.
    _ = weak
    return (CoverageState.PARTIALLY_COVERED, evidence)


def _contradicted(
    requirement: Requirement, properties: frozenset[ProductProperty]
) -> bool:
    """Whether an artifact's declared properties conflict with this requirement's."""
    return bool(find_conflicts(requirement.properties, properties))


def analyse_coverage(
    prd: PRDDocument,
    *,
    ux: object | None = None,
    architecture: SystemArchitectureDocument | None = None,
    plan: ImplementationPlanDocument | None = None,
    include_contradictions: bool = True,
) -> CoverageMatrix:
    """Build the coverage matrix for a project.

    Entirely deterministic: every state comes from an element explicitly naming a
    requirement id, or from a structural property conflict. Nothing is inferred from prose
    and no model is consulted.
    """
    matrix = CoverageMatrix()
    if include_contradictions:
        matrix.contradictions = check_all(
            prd=prd,
            ux=ux,  # type: ignore[arg-type]
            architecture=architecture,
            include_hints=False,
        )

    ux_properties = frozenset(getattr(ux, "properties", frozenset()))
    architecture_properties = (
        architecture.properties if architecture is not None else frozenset()
    )

    for requirement in prd.requirements:
        criticality = criticality_of(requirement)
        entry = CoverageEntry(
            requirement_id=requirement.requirement_id,
            title=requirement.title,
            criticality=criticality,
        )

        # -- UX ---------------------------------------------------------------------
        if ux is None:
            entry.ux = CoverageState.MISSING
        elif not applies_to(requirement, ArtifactKind.UX_SPEC):
            entry.ux = CoverageState.NOT_APPLICABLE
        else:
            found = _collect_ux(ux, requirement.requirement_id)
            state, evidence = _state_from(found, _UX_STRONG, _UX_WEAK)
            if evidence and _contradicted(requirement, ux_properties):
                state = CoverageState.CONTRADICTED
            entry.ux, entry.ux_evidence = state, evidence

        # -- Architecture -------------------------------------------------------------
        if architecture is None:
            entry.architecture = CoverageState.MISSING
        elif not applies_to(requirement, ArtifactKind.SYSTEM_ARCHITECTURE):
            entry.architecture = CoverageState.NOT_APPLICABLE
        else:
            found = _collect_architecture(architecture, requirement.requirement_id)
            state, evidence = _state_from(found, _ARCH_STRONG, _ARCH_WEAK)
            if evidence and _contradicted(requirement, architecture_properties):
                state = CoverageState.CONTRADICTED
            entry.architecture, entry.architecture_evidence = state, evidence

        # -- Plan ----------------------------------------------------------------------
        if plan is None or not applies_to(requirement, ArtifactKind.IMPLEMENTATION_PLAN):
            entry.plan = CoverageState.NOT_APPLICABLE
        else:
            tasks = tuple(
                CoverageEvidence(task.task_id, "task", task.title)
                for task in plan.tasks
                if requirement.requirement_id in task.implements
            )
            entry.plan = CoverageState.COVERED if tasks else CoverageState.MISSING
            entry.plan_evidence = tasks

        matrix.entries.append(entry)

    matrix.gaps = _build_gaps(matrix, ux is not None, architecture is not None, plan is not None)
    logger.info(
        "coverage.analysed",
        requirements=len(matrix.entries),
        gaps=len(matrix.gaps),
        blocking=len(matrix.blocking_gaps),
        ux_coverage=round(matrix.coverage(ArtifactKind.UX_SPEC), 3),
        architecture_coverage=round(
            matrix.coverage(ArtifactKind.SYSTEM_ARCHITECTURE), 3
        ),
    )
    return matrix


#: Which generation stage would close a gap in each artifact kind.
GAP_STAGE: dict[ArtifactKind, str] = {
    ArtifactKind.UX_SPEC: "ux.flows",
    ArtifactKind.SYSTEM_ARCHITECTURE: "arch.components",
    ArtifactKind.IMPLEMENTATION_PLAN: "arch.plan",
}


def _build_gaps(
    matrix: CoverageMatrix, has_ux: bool, has_architecture: bool, has_plan: bool
) -> list[CoverageGap]:
    """Turn non-covering states into structured gaps."""
    gaps: list[CoverageGap] = []
    present = {
        ArtifactKind.UX_SPEC: has_ux,
        ArtifactKind.SYSTEM_ARCHITECTURE: has_architecture,
        ArtifactKind.IMPLEMENTATION_PLAN: has_plan,
    }
    for entry in matrix.entries:
        for kind, exists in present.items():
            if not exists:
                # An artifact that was never produced is a pipeline failure, reported
                # elsewhere. Emitting a gap per requirement would bury the real problem.
                continue
            state = entry.state_for(kind)
            if not state.is_gap:
                continue
            gaps.append(
                CoverageGap(
                    requirement_id=entry.requirement_id,
                    title=entry.title,
                    artifact=kind,
                    state=state,
                    criticality=entry.criticality,
                    stage=GAP_STAGE.get(kind, ""),
                    evidence=entry.evidence_for(kind),
                    detail=(
                        f"{entry.requirement_id} is {state} in {kind.value}"
                    ),
                )
            )
    return gaps
