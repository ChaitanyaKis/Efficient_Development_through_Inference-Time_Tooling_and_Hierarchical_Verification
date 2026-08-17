"""Deterministic artifact and plan validation.

M4.6 draws a hard line: an invalid artifact never enters approved project state. This module
is what makes that checkable rather than aspirational, and none of it consults a model.

Three classes of check:

**Structural.** Does the body match its kind's schema? Handled by pydantic at construction;
this module reports it when a stored body fails to parse.

**Referential.** A UX flow claims to satisfy ``REQ-999``. Does ``REQ-999`` exist in the PRD
this spec was derived from? A dangling id is the defect that survives longest, because
everything downstream keeps propagating it until something tries to build it.

**Graph.** An implementation task depends on itself through a chain of three others. Cycle
detection is Kahn's algorithm, the same one the M2 task DAG uses -- a plan that cannot be
ordered cannot be executed, and finding that out at plan time costs nothing.

Coverage checks sit alongside these as *advisory* issues: a requirement no UX flow serves is
usually a gap, but occasionally a backend requirement with no interface, so it reports
without blocking. The distinction between blocking and advisory is the whole reason
:class:`~edith.product.artifacts.ValidationIssue` carries a flag rather than a severity
string nobody agrees on.
"""

from __future__ import annotations

from edith.observability.logging import get_logger
from edith.schemas.common import utc_now

from .architecture import ImplementationPlanDocument, SystemArchitectureDocument
from .artifacts import (
    Artifact,
    ArtifactKind,
    ValidationIssue,
    ValidationOutcome,
    ValidationState,
    is_element_id,
)
from .prd import PRDDocument
from .ux import UXSpecDocument

logger = get_logger(__name__)

#: Issue codes. Stable strings so a caller can act on a specific failure without matching
#: on prose that will be reworded.
SCHEMA_INVALID = "ARTIFACT_SCHEMA_INVALID"
UNKNOWN_REFERENCE = "ARTIFACT_UNKNOWN_REFERENCE"
DUPLICATE_ID = "ARTIFACT_DUPLICATE_ID"
MISSING_ACCEPTANCE = "REQUIREMENT_WITHOUT_ACCEPTANCE_CRITERION"
UNCOVERED_REQUIREMENT = "REQUIREMENT_NOT_COVERED"
DEAD_END_STEP = "FLOW_DEAD_END"
UNREACHABLE_STEP = "FLOW_UNREACHABLE_STEP"
MISSING_SCREEN_STATE = "SCREEN_MISSING_REQUIRED_STATE"
FLOW_WITHOUT_ERROR_PATH = "FLOW_WITHOUT_ERROR_PATH"
UNMITIGATED_THREAT = "THREAT_WITHOUT_MITIGATION"
UNJUSTIFIED_TECHNOLOGY = "TECHNOLOGY_WITHOUT_ALTERNATIVES"
PLAN_CYCLE = "PLAN_CIRCULAR_DEPENDENCY"
PLAN_UNKNOWN_DEPENDENCY = "PLAN_UNKNOWN_DEPENDENCY"
PLAN_UNKNOWN_COMPONENT = "PLAN_UNKNOWN_COMPONENT"
NO_OPEN_QUESTION_OWNER = "OPEN_QUESTION_WITHOUT_OWNER"


class ReferenceIndex:
    """Every element id known to a project, and which artifact defines it.

    Built from the artifacts a document was derived from, so validation answers "does this
    reference resolve *against what this document actually read*" rather than against
    whatever happens to exist now.
    """

    def __init__(self) -> None:
        self._owner: dict[str, str] = {}

    def add_document_ids(self, artifact_id: str, identifiers: tuple[str, ...]) -> None:
        """Record that ``artifact_id`` defines these ids."""
        for identifier in identifiers:
            self._owner.setdefault(identifier, artifact_id)

    def add(self, artifact: Artifact) -> ReferenceIndex:
        """Index every id an artifact's document defines."""
        try:
            document = artifact.document()
        except ValueError:
            # A body that will not parse is reported by validate_artifact; indexing simply
            # contributes nothing rather than taking the whole index down.
            return self
        self.add_document_ids(artifact.artifact_id, document.element_ids())
        return self

    def knows(self, identifier: str) -> bool:
        """Whether this id is defined anywhere in the index."""
        return identifier in self._owner

    def owner_of(self, identifier: str) -> str | None:
        """Which artifact defines this id."""
        return self._owner.get(identifier)

    @property
    def known_ids(self) -> frozenset[str]:
        """Every id in the index."""
        return frozenset(self._owner)


def build_index(artifacts: tuple[Artifact, ...]) -> ReferenceIndex:
    """Build a reference index from a set of artifacts."""
    index = ReferenceIndex()
    for artifact in artifacts:
        index.add(artifact)
    return index


def validate_artifact(
    artifact: Artifact, index: ReferenceIndex | None = None
) -> ValidationOutcome:
    """Validate one artifact, structurally and referentially.

    Args:
        artifact: The artifact to check.
        index: Ids defined by the artifacts this one depends on. When omitted, only the
            document's own ids are considered resolvable, which is correct for a
            self-contained artifact such as a PRD.

    Returns:
        The outcome. Blocking issues make the artifact ineligible for approval; advisory
        issues are recorded and reported but do not.
    """
    issues: list[ValidationIssue] = []

    try:
        document = artifact.document()
    except ValueError as exc:
        return ValidationOutcome(
            state=ValidationState.INVALID,
            checked_at=utc_now(),
            issues=[
                ValidationIssue(
                    code=SCHEMA_INVALID,
                    message=f"the artifact body does not match its {artifact.kind} schema: {exc}",
                )
            ],
        )

    resolvable = set(document.element_ids())
    if index is not None:
        resolvable |= set(index.known_ids)

    # Referential integrity: the check M4.6 names explicitly.
    for reference in dict.fromkeys(document.referenced_ids()):
        if not is_element_id(reference):
            continue
        if reference not in resolvable:
            issues.append(
                ValidationIssue(
                    code=UNKNOWN_REFERENCE,
                    message=(
                        f"{reference} is referenced but defined nowhere in this artifact "
                        f"or the artifacts it depends on"
                    ),
                    element_id=reference,
                )
            )

    if isinstance(document, PRDDocument):
        issues.extend(_check_prd(document))
    elif isinstance(document, UXSpecDocument):
        issues.extend(_check_ux(document))
    elif isinstance(document, SystemArchitectureDocument):
        issues.extend(_check_architecture(document))
    elif isinstance(document, ImplementationPlanDocument):
        issues.extend(_check_plan(document, index))

    blocking = [issue for issue in issues if issue.blocking]
    outcome = ValidationOutcome(
        state=ValidationState.INVALID if blocking else ValidationState.VALID,
        issues=issues,
        checked_at=utc_now(),
    )
    logger.info(
        "artifact.validation",
        artifact_id=artifact.artifact_id,
        kind=str(artifact.kind),
        state=str(outcome.state),
        blocking=len(blocking),
        advisory=len(issues) - len(blocking),
    )
    return outcome


def _check_prd(document: PRDDocument) -> list[ValidationIssue]:
    """PM quality checks: is every requirement verifiable, is anything unresolved?"""
    issues: list[ValidationIssue] = []

    for requirement_id in document.unverified_requirements():
        issues.append(
            ValidationIssue(
                code=MISSING_ACCEPTANCE,
                message=(
                    f"{requirement_id} has no acceptance criterion, so nothing can "
                    f"demonstrate it was satisfied"
                ),
                element_id=requirement_id,
                # Advisory: a constraint may legitimately be verified by inspection rather
                # than by a criterion, and blocking here would make a valid PRD unapprovable.
                blocking=False,
            )
        )

    for question in document.open_questions:
        if not question.owner.strip():
            issues.append(
                ValidationIssue(
                    code=NO_OPEN_QUESTION_OWNER,
                    message=f"{question.question_id} has no owner, so nobody will answer it",
                    element_id=question.question_id,
                    blocking=False,
                )
            )
    return issues


def _check_ux(document: UXSpecDocument) -> list[ValidationIssue]:
    """UX quality checks: complete flows, specified states, error paths."""
    issues: list[ValidationIssue] = []

    for flow in document.flows:
        for step_id in flow.dead_ends():
            issues.append(
                ValidationIssue(
                    code=DEAD_END_STEP,
                    message=(
                        f"step {step_id} in flow {flow.flow_id} leads nowhere and is not "
                        f"an ending; the user would be stranded there"
                    ),
                    element_id=flow.flow_id,
                )
            )
        for step_id in flow.unreachable_steps():
            issues.append(
                ValidationIssue(
                    code=UNREACHABLE_STEP,
                    message=(
                        f"step {step_id} in flow {flow.flow_id} cannot be reached from "
                        f"its entry step"
                    ),
                    element_id=flow.flow_id,
                    blocking=False,
                )
            )

    for flow_id in document.flows_without_error_paths():
        issues.append(
            ValidationIssue(
                code=FLOW_WITHOUT_ERROR_PATH,
                message=(
                    f"flow {flow_id} specifies no failure path; every flow that can fail "
                    f"needs one, and a flow that cannot should say so"
                ),
                element_id=flow_id,
                blocking=False,
            )
        )

    for screen_id, missing in document.screens_missing_states().items():
        issues.append(
            ValidationIssue(
                code=MISSING_SCREEN_STATE,
                message=(
                    f"screen {screen_id} does not specify "
                    f"{', '.join(state.value for state in missing)}; these are the states "
                    f"users reach on their worst day"
                ),
                element_id=screen_id,
                blocking=False,
            )
        )
    return issues


def _check_architecture(document: SystemArchitectureDocument) -> list[ValidationIssue]:
    """Architect quality checks: mitigated threats, justified technology."""
    issues: list[ValidationIssue] = []

    for threat_id in document.unmitigated_threats():
        issues.append(
            ValidationIssue(
                code=UNMITIGATED_THREAT,
                message=(
                    f"{threat_id} has no mitigation; if the risk is being accepted, say so "
                    f"in the mitigation field rather than leaving it blank"
                ),
                element_id=threat_id,
                blocking=False,
            )
        )

    for name in document.unjustified_technologies():
        issues.append(
            ValidationIssue(
                code=UNJUSTIFIED_TECHNOLOGY,
                message=(
                    f"{name} was chosen without naming a rejected alternative; a choice "
                    f"nobody compared is a preference, not a decision"
                ),
                blocking=False,
            )
        )
    return issues


def _check_plan(
    document: ImplementationPlanDocument, index: ReferenceIndex | None
) -> list[ValidationIssue]:
    """Plan checks: resolvable dependencies and an orderable graph."""
    issues: list[ValidationIssue] = []
    known_tasks = document.task_ids

    for task in document.tasks:
        for dependency in task.depends_on:
            if dependency not in known_tasks:
                issues.append(
                    ValidationIssue(
                        code=PLAN_UNKNOWN_DEPENDENCY,
                        message=(
                            f"{task.task_id} depends on {dependency}, which is not a task "
                            f"in this plan"
                        ),
                        element_id=task.task_id,
                    )
                )
        if index is not None:
            for component in task.components:
                if is_element_id(component, "ARCH") and not index.knows(component):
                    issues.append(
                        ValidationIssue(
                            code=PLAN_UNKNOWN_COMPONENT,
                            message=(
                                f"{task.task_id} names component {component}, which the "
                                f"architecture does not define"
                            ),
                            element_id=task.task_id,
                        )
                    )

    cycle = find_cycle(document)
    if cycle:
        issues.append(
            ValidationIssue(
                code=PLAN_CYCLE,
                message=(
                    f"the plan contains a circular dependency and cannot be ordered: "
                    f"{' -> '.join(cycle)}"
                ),
            )
        )
    return issues


def find_cycle(document: ImplementationPlanDocument) -> tuple[str, ...]:
    """Return a task cycle, or an empty tuple when the plan is a DAG.

    Kahn's algorithm: peel off tasks with no unmet dependencies until none remain. Whatever
    is left is exactly the set involved in a cycle. The same approach the M2 task graph uses,
    so a plan that passes here will not surprise the executor later.
    """
    remaining = {task.task_id: set(task.depends_on) & document.task_ids for task in document.tasks}

    progressed = True
    while progressed:
        progressed = False
        ready = [task_id for task_id, pending in remaining.items() if not pending]
        for task_id in ready:
            del remaining[task_id]
            progressed = True
            for pending in remaining.values():
                pending.discard(task_id)

    if not remaining:
        return ()

    # Walk the residue to produce a readable cycle rather than an unordered set.
    start = sorted(remaining)[0]
    path: list[str] = [start]
    seen = {start}
    current = start
    while True:
        candidates = sorted(remaining.get(current, set()))
        if not candidates:
            break
        following = candidates[0]
        path.append(following)
        if following in seen:
            break
        seen.add(following)
        current = following
    return tuple(path)


def coverage_issues(
    prd: PRDDocument,
    *,
    ux: UXSpecDocument | None = None,
    architecture: SystemArchitectureDocument | None = None,
    plan: ImplementationPlanDocument | None = None,
) -> list[ValidationIssue]:
    """Requirements that a downstream artifact does not address.

    Advisory by design. A backend requirement legitimately has no UX flow, and a
    documentation requirement legitimately has no component. What matters is that the gap is
    visible, not that it is forbidden.
    """
    issues: list[ValidationIssue] = []
    required = {
        requirement.requirement_id
        for requirement in prd.requirements
        if requirement.priority.value in {"MUST", "SHOULD"}
    }

    for label, covered in (
        ("UX specification", ux.covered_requirements if ux else None),
        (
            "architecture",
            architecture.covered_requirements if architecture else None,
        ),
        ("implementation plan", plan.covered_requirements if plan else None),
    ):
        if covered is None:
            continue
        for requirement_id in sorted(required - covered):
            issues.append(
                ValidationIssue(
                    code=UNCOVERED_REQUIREMENT,
                    message=f"{requirement_id} is not addressed by the {label}",
                    element_id=requirement_id,
                    blocking=False,
                )
            )
    return issues


def validate_against_dependencies(
    artifact: Artifact, dependencies: tuple[Artifact, ...]
) -> ValidationOutcome:
    """Validate an artifact against the artifacts it was derived from.

    The normal path for anything downstream of a PRD: the index is built from the
    dependencies, so a reference to a requirement that PRD never defined is caught here
    rather than by whoever tries to implement it.
    """
    index = build_index((*dependencies, artifact))
    return validate_artifact(artifact, index)


def kind_supports_validation(kind: ArtifactKind) -> bool:
    """Whether a document type has checks beyond schema conformance."""
    return kind in {
        ArtifactKind.PRD,
        ArtifactKind.UX_SPEC,
        ArtifactKind.SYSTEM_ARCHITECTURE,
        ArtifactKind.IMPLEMENTATION_PLAN,
    }
