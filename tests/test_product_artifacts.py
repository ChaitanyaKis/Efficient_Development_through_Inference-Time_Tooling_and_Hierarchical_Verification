"""M4: the artifact foundation — envelope, versioning, storage, validation, contradictions.

Everything here is deterministic and offline. No model is involved, which is the point: the
checks M4.6 and M4.8 specify must not depend on a model's judgement, and a test suite that
needed inference to verify them would be proving the opposite.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from edith.authority import (
    AuthorityLevel,
    is_advisory,
    may_override,
    outranks,
    strongest,
)
from edith.product.architecture import (
    ApiEndpoint,
    ArchitectureComponent,
    ArchitectureDecision,
    ComponentKind,
    DataEntity,
    ImplementationPlanDocument,
    PlannedTask,
    SystemArchitectureDocument,
    TechnologyChoice,
)
from edith.product.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactStatus,
    ValidationOutcome,
    ValidationState,
    build_artifact,
    can_transition,
    element_id,
    is_element_id,
)
from edith.product.contradictions import (
    ContradictionSeverity,
    check_all,
    check_prd_against_architecture,
    check_ux_against_architecture,
)
from edith.product.prd import (
    AcceptanceCriterion,
    OpenQuestion,
    PRDDocument,
    Requirement,
)
from edith.product.properties import ProductProperty as P
from edith.product.properties import expand, find_conflicts
from edith.product.review import (
    FindingSeverity,
    overall_verdict,
    review_plan,
    review_prd_for_feasibility,
    review_prd_for_ux_coverage,
    review_requirement_quality,
)
from edith.product.store import (
    ArtifactConflictError,
    ProductStore,
    approve,
    open_artifacts,
)
from edith.product.ux import (
    Flow,
    FlowStep,
    Screen,
    ScreenState,
    StepKind,
    UXSpecDocument,
)
from edith.product.validation import (
    PLAN_CYCLE,
    UNKNOWN_REFERENCE,
    build_index,
    find_cycle,
    validate_against_dependencies,
    validate_artifact,
)
from edith.schemas.common import Verdict

VALID = ValidationOutcome(state=ValidationState.VALID)


def prd_document(**overrides: object) -> PRDDocument:
    """A small, valid PRD."""
    defaults: dict[str, object] = {
        "product_name": "Stockroom",
        "problem": "Shop staff cannot see which items are running low.",
        "requirements": (
            Requirement(
                requirement_id="REQ-001",
                title="Record stock levels",
                statement="The system records the quantity of every item.",
            ),
            Requirement(
                requirement_id="REQ-002",
                title="Low stock alert",
                statement="Items at or below their threshold are listed as low.",
            ),
        ),
        "acceptance_criteria": (
            AcceptanceCriterion(
                criterion_id="AC-001",
                statement="Adding an item records its quantity.",
                verifies=("REQ-001",),
            ),
            AcceptanceCriterion(
                criterion_id="AC-002",
                statement="An item at its threshold appears in the low list.",
                verifies=("REQ-002",),
            ),
        ),
        "non_goals": ("Purchasing",),
    }
    defaults.update(overrides)
    return PRDDocument.model_validate(defaults)


def prd_artifact(document: PRDDocument | None = None, *, project: str = "p1") -> Artifact:
    """A validated PRD artifact ready to store."""
    artifact = build_artifact(
        kind=ArtifactKind.PRD,
        project_id=project,
        title="Stockroom PRD",
        author="product_manager",
        document=document or prd_document(),
    )
    return artifact.model_copy(update={"validation": validate_artifact(artifact)})


@pytest.fixture
def store(tmp_path: Path) -> ProductStore:
    with open_artifacts(tmp_path / "product") as opened:
        yield opened


class TestAuthorityHierarchy:
    """M4's requirement authority order, represented structurally rather than in prose."""

    def test_the_order_is_the_one_the_milestone_specifies(self) -> None:
        order = [
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
            AuthorityLevel.PROJECT_POLICY,
            AuthorityLevel.APPROVED_ARCHITECTURE_DECISION,
            AuthorityLevel.TASK_ACCEPTANCE_CRITERIA,
            AuthorityLevel.AGENT_RECOMMENDATION,
            AuthorityLevel.REPOSITORY_CONTENT,
            AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
        ]
        for higher, lower in itertools.pairwise(order):
            assert outranks(higher, lower), f"{higher} must outrank {lower}"

    def test_a_web_page_cannot_override_a_requirement(self) -> None:
        """The defect the hierarchy exists to prevent."""
        assert not may_override(
            AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
        )

    def test_a_repository_comment_cannot_override_anything(self) -> None:
        """M2.1's lesson: a fixture's comment talked the model out of fixing a bug."""
        for level in AuthorityLevel:
            assert not may_override(AuthorityLevel.REPOSITORY_CONTENT, level)

    def test_an_agent_recommendation_is_advisory_even_against_weaker_sources(self) -> None:
        """An agent outranking a web page still does not get to decide."""
        assert outranks(
            AuthorityLevel.AGENT_RECOMMENDATION,
            AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
        )
        assert not may_override(
            AuthorityLevel.AGENT_RECOMMENDATION,
            AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
        )

    def test_equal_authority_never_overrides(self) -> None:
        """Two conflicting requirements is a contradiction, not a race."""
        assert not may_override(
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
            AuthorityLevel.USER_APPROVED_REQUIREMENT,
        )

    def test_advisory_levels_are_exactly_the_lower_three(self) -> None:
        advisory = {level for level in AuthorityLevel if is_advisory(level)}
        assert advisory == {
            AuthorityLevel.AGENT_RECOMMENDATION,
            AuthorityLevel.REPOSITORY_CONTENT,
            AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
        }

    def test_strongest_picks_the_highest_authority(self) -> None:
        assert (
            strongest(
                (
                    AuthorityLevel.REPOSITORY_CONTENT,
                    AuthorityLevel.PROJECT_POLICY,
                    AuthorityLevel.AGENT_RECOMMENDATION,
                )
            )
            is AuthorityLevel.PROJECT_POLICY
        )
        assert strongest(()) is None


class TestElementIds:
    """Stable ids are the traceability chain. Their format is load-bearing."""

    def test_ids_are_zero_padded_so_they_sort(self) -> None:
        assert element_id("REQ", 1) == "REQ-001"
        assert element_id("REQ", 42) == "REQ-042"
        assert sorted([element_id("REQ", 10), element_id("REQ", 9)]) == [
            "REQ-009",
            "REQ-010",
        ]

    def test_recognises_known_prefixes(self) -> None:
        assert is_element_id("REQ-001")
        assert is_element_id("REQ-001", "REQ")
        assert not is_element_id("REQ-001", "TASK")
        assert not is_element_id("not-an-id")
        assert not is_element_id("REQ-abc")


class TestArtifactEnvelope:
    def test_an_invalid_artifact_cannot_be_approved(self) -> None:
        """M4.6: nothing invalid enters approved project state."""
        artifact = prd_artifact()
        broken = artifact.model_copy(
            update={"validation": ValidationOutcome(state=ValidationState.INVALID)}
        )
        with pytest.raises(ValueError, match="cannot be APPROVED"):
            broken.model_copy(update={"status": ArtifactStatus.APPROVED}).model_validate(
                broken.model_dump() | {"status": "APPROVED"}
            )

    def test_an_unvalidated_artifact_cannot_be_approved(self) -> None:
        """UNVALIDATED must not be indistinguishable from passing."""
        artifact = build_artifact(
            kind=ArtifactKind.PRD,
            project_id="p1",
            title="T",
            author="product_manager",
            document=prd_document(),
        )
        assert artifact.validation.state is ValidationState.UNVALIDATED
        with pytest.raises(ValueError, match="cannot be APPROVED"):
            Artifact.model_validate(artifact.model_dump() | {"status": "APPROVED"})

    def test_a_draft_cannot_claim_approved_architecture_authority(self) -> None:
        """Otherwise an agent mints authority by writing confidently."""
        artifact = prd_artifact()
        with pytest.raises(ValueError, match="approval is what confers authority"):
            Artifact.model_validate(
                artifact.model_dump()
                | {"authority": "APPROVED_ARCHITECTURE_DECISION", "status": "DRAFT"}
            )

    def test_approved_artifacts_are_never_edited_only_superseded(self) -> None:
        assert can_transition(ArtifactStatus.APPROVED, ArtifactStatus.SUPERSEDED)
        assert not can_transition(ArtifactStatus.APPROVED, ArtifactStatus.DRAFT)
        assert not can_transition(ArtifactStatus.SUPERSEDED, ArtifactStatus.APPROVED)

    def test_an_illegal_transition_is_refused(self) -> None:
        artifact = prd_artifact()
        with pytest.raises(ValueError, match="illegal artifact transition"):
            artifact.transition_to(ArtifactStatus.SUPERSEDED)

    def test_a_revision_is_a_new_version_not_an_edit(self) -> None:
        artifact = prd_artifact()
        revision = artifact.revise(body=artifact.body, author="product_manager")
        assert revision.artifact_id == artifact.artifact_id
        assert revision.version == artifact.version + 1
        assert revision.status is ArtifactStatus.DRAFT
        assert revision.supersedes == f"{artifact.artifact_id}@1"

    def test_the_body_round_trips_through_its_typed_schema(self) -> None:
        artifact = prd_artifact()
        document = artifact.document()
        assert isinstance(document, PRDDocument)
        assert document.requirement_ids == {"REQ-001", "REQ-002"}


class TestPRDSchema:
    def test_duplicate_requirement_ids_are_rejected(self) -> None:
        """A duplicate makes every downstream reference ambiguous."""
        with pytest.raises(ValueError, match="duplicate element ids"):
            prd_document(
                requirements=(
                    Requirement(requirement_id="REQ-001", title="A", statement="a"),
                    Requirement(requirement_id="REQ-001", title="B", statement="b"),
                ),
                acceptance_criteria=(),
            )

    def test_an_acceptance_criterion_must_name_a_requirement(self) -> None:
        """A criterion verifying nothing is a test nobody asked for."""
        with pytest.raises(ValueError):
            AcceptanceCriterion(criterion_id="AC-001", statement="x", verifies=())

    def test_a_criterion_cannot_reference_a_non_requirement_id(self) -> None:
        with pytest.raises(ValueError, match="is not a REQ id"):
            AcceptanceCriterion(
                criterion_id="AC-001", statement="x", verifies=("TASK-001",)
            )

    def test_a_requirement_cannot_relate_to_itself(self) -> None:
        with pytest.raises(ValueError, match="cannot relate to itself"):
            Requirement(
                requirement_id="REQ-001",
                title="A",
                statement="a",
                related_to=("REQ-001",),
            )

    def test_unverified_requirements_are_reported(self) -> None:
        document = prd_document(
            requirements=(
                Requirement(requirement_id="REQ-001", title="A", statement="a"),
                Requirement(requirement_id="REQ-002", title="B", statement="b"),
            ),
            acceptance_criteria=(
                AcceptanceCriterion(
                    criterion_id="AC-001", statement="x", verifies=("REQ-001",)
                ),
            ),
        )
        assert document.unverified_requirements() == ("REQ-002",)


class TestUXSchema:
    def build_flow(self, **overrides: object) -> Flow:
        defaults: dict[str, object] = {
            "flow_id": "UX-001",
            "name": "Add stock",
            "entry_step": "UX-001-S1",
            "steps": (
                FlowStep(
                    step_id="UX-001-S1",
                    name="Open form",
                    next_steps=("UX-001-S2",),
                ),
                FlowStep(
                    step_id="UX-001-S2",
                    name="Submit",
                    kind=StepKind.ACTION,
                    next_steps=("UX-001-S3",),
                    error_steps=("UX-001-S4",),
                ),
                FlowStep(step_id="UX-001-S3", name="Saved", kind=StepKind.TERMINAL),
                FlowStep(step_id="UX-001-S4", name="Failed", kind=StepKind.ABORT),
            ),
        }
        defaults.update(overrides)
        return Flow.model_validate(defaults)

    def test_a_flow_step_cannot_point_at_a_step_that_does_not_exist(self) -> None:
        with pytest.raises(ValueError, match="points at unknown step"):
            self.build_flow(
                steps=(
                    FlowStep(step_id="S1", name="A", next_steps=("S99",)),
                ),
                entry_step="S1",
            )

    def test_the_entry_step_must_be_one_of_the_steps(self) -> None:
        with pytest.raises(ValueError, match="is not one of its steps"):
            self.build_flow(entry_step="nowhere")

    def test_a_dead_end_is_detected(self) -> None:
        flow = self.build_flow(
            steps=(
                FlowStep(step_id="S1", name="A", next_steps=("S2",)),
                FlowStep(step_id="S2", name="B"),
            ),
            entry_step="S1",
        )
        assert flow.dead_ends() == ("S2",)

    def test_an_unreachable_step_is_detected(self) -> None:
        flow = self.build_flow(
            steps=(
                FlowStep(step_id="S1", name="A", kind=StepKind.TERMINAL),
                FlowStep(step_id="S2", name="Orphan", kind=StepKind.TERMINAL),
            ),
            entry_step="S1",
        )
        assert flow.unreachable_steps() == ("S2",)

    def test_a_flow_with_error_steps_has_an_error_path(self) -> None:
        assert self.build_flow().has_error_path

    def test_a_happy_path_only_flow_is_flagged(self) -> None:
        flow = self.build_flow(
            steps=(
                FlowStep(step_id="S1", name="A", next_steps=("S2",)),
                FlowStep(step_id="S2", name="B", kind=StepKind.TERMINAL),
            ),
            entry_step="S1",
        )
        spec = UXSpecDocument(product_name="X", flows=(flow,))
        assert spec.flows_without_error_paths() == ("UX-001",)

    def test_a_screen_missing_required_states_is_reported(self) -> None:
        """Loading, error and default are the states an interface forgets."""
        screen = Screen(
            screen_id="SCR-001", name="List", states=frozenset({ScreenState.DEFAULT})
        )
        assert set(screen.missing_states()) == {ScreenState.LOADING, ScreenState.ERROR}


class TestArchitectureSchema:
    def test_a_decision_without_alternatives_is_rejected(self) -> None:
        """A decision with nothing rejected is a preference."""
        with pytest.raises(ValueError):
            ArchitectureDecision(
                decision_id="ADR-001",
                title="Use PostgreSQL",
                context="c",
                decision="d",
                alternatives=(),
                rationale="r",
                consequences=("x",),
            )

    def test_a_decision_without_consequences_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ArchitectureDecision(
                decision_id="ADR-001",
                title="Use PostgreSQL",
                context="c",
                decision="d",
                alternatives=("SQLite",),
                rationale="r",
                consequences=(),
            )

    def test_a_component_cannot_depend_on_itself(self) -> None:
        with pytest.raises(ValueError, match="cannot depend on itself"):
            ArchitectureComponent(
                component_id="ARCH-001",
                name="API",
                responsibility="r",
                depends_on=("ARCH-001",),
            )

    def test_a_task_cannot_invent_a_verification_command(self) -> None:
        """A planner-authored command would be a hole through the shell allowlist."""
        with pytest.raises(ValueError, match="is not one of"):
            PlannedTask(
                task_id="TASK-001",
                title="t",
                description="d",
                verification=("rm -rf /",),
            )

    def test_unjustified_technologies_are_reported(self) -> None:
        architecture = SystemArchitectureDocument(
            product_name="X",
            overview="o",
            technologies=(
                TechnologyChoice(name="Kafka", role="queue", rationale="popular"),
                TechnologyChoice(
                    name="SQLite",
                    role="database",
                    rationale="single writer, no ops",
                    alternatives_rejected=("PostgreSQL",),
                ),
            ),
        )
        assert architecture.unjustified_technologies() == ("Kafka",)


class TestArtifactStorage:
    def test_a_version_cannot_be_overwritten(self, store: ProductStore) -> None:
        """History is immutable by construction."""
        artifact = prd_artifact()
        store.save(artifact)
        with pytest.raises(ArtifactConflictError, match="already exists"):
            store.save(artifact)

    def test_approving_a_successor_supersedes_its_predecessor(
        self, store: ProductStore
    ) -> None:
        first = prd_artifact()
        store.save(first)
        approve(store, first)

        second = first.revise(body=first.body, author="product_manager")
        second = second.model_copy(update={"validation": VALID})
        store.save(second)
        approve(store, second)

        history = store.history(first.artifact_id)
        assert [(item.version, item.status) for item in history] == [
            (2, ArtifactStatus.APPROVED),
            (1, ArtifactStatus.SUPERSEDED),
        ]

    def test_a_draft_revision_does_not_retire_the_approved_version(
        self, store: ProductStore
    ) -> None:
        """A half-finished draft must not become the project's truth."""
        first = prd_artifact()
        store.save(first)
        approve(store, first)

        draft = first.revise(body=first.body, author="product_manager")
        store.save(draft)

        current = store.latest("p1", ArtifactKind.PRD, status=ArtifactStatus.APPROVED)
        assert current is not None
        assert current.version == 1

    def test_an_invalid_artifact_cannot_be_approved(self, store: ProductStore) -> None:
        artifact = build_artifact(
            kind=ArtifactKind.PRD,
            project_id="p1",
            title="T",
            author="product_manager",
            document=prd_document(),
        )
        artifact = artifact.model_copy(
            update={"validation": ValidationOutcome(state=ValidationState.INVALID)}
        )
        store.save(artifact)
        with pytest.raises(ArtifactConflictError, match="cannot be approved"):
            approve(store, artifact)

    def test_projects_are_isolated(self, store: ProductStore) -> None:
        """The M3 isolation guarantee, preserved for artifacts."""
        store.save(prd_artifact(project="p1"))
        assert store.by_kind("p2", ArtifactKind.PRD) == ()
        assert len(store.by_kind("p1", ArtifactKind.PRD)) == 1

    def test_an_approved_artifact_stays_recoverable_forever(
        self, store: ProductStore
    ) -> None:
        first = prd_artifact()
        store.save(first)
        approve(store, first)
        for _ in range(3):
            latest = store.get(first.artifact_id)
            assert latest is not None
            successor = latest.revise(body=latest.body, author="product_manager")
            store.save(successor.model_copy(update={"validation": VALID}))
            approve(store, store.get(first.artifact_id, successor.version))  # type: ignore[arg-type]

        original = store.get(first.artifact_id, 1)
        assert original is not None
        assert original.document().model_dump() == first.document().model_dump()

    def test_purging_a_project_removes_everything(self, store: ProductStore) -> None:
        store.save(prd_artifact(project="p1"))
        assert store.purge_project("p1") == 1
        assert store.count("p1") == 0


class TestValidation:
    def test_a_dangling_requirement_reference_is_blocking(self) -> None:
        """M4.6's example: a UX flow references REQ-999 which does not exist."""
        prd = prd_artifact()
        spec = UXSpecDocument(
            product_name="Stockroom",
            flows=(
                Flow(
                    flow_id="UX-001",
                    name="f",
                    entry_step="S1",
                    steps=(FlowStep(step_id="S1", name="A", kind=StepKind.TERMINAL),),
                    satisfies=("REQ-999",),
                ),
            ),
        )
        artifact = build_artifact(
            kind=ArtifactKind.UX_SPEC,
            project_id="p1",
            title="UX",
            author="ux_designer",
            document=spec,
        )
        outcome = validate_against_dependencies(artifact, (prd,))
        assert not outcome.valid
        assert any(issue.code == UNKNOWN_REFERENCE for issue in outcome.blocking_issues)

    def test_a_resolvable_reference_validates(self) -> None:
        prd = prd_artifact()
        spec = UXSpecDocument(
            product_name="Stockroom",
            flows=(
                Flow(
                    flow_id="UX-001",
                    name="f",
                    entry_step="S1",
                    steps=(FlowStep(step_id="S1", name="A", kind=StepKind.TERMINAL),),
                    satisfies=("REQ-001",),
                ),
            ),
        )
        artifact = build_artifact(
            kind=ArtifactKind.UX_SPEC,
            project_id="p1",
            title="UX",
            author="ux_designer",
            document=spec,
        )
        assert validate_against_dependencies(artifact, (prd,)).valid

    def test_a_circular_plan_dependency_is_detected(self) -> None:
        """M4.6: a circular dependency is a PLAN_VALIDATION_FAILURE."""
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(
                PlannedTask(
                    task_id="TASK-001", title="a", description="d", depends_on=("TASK-003",)
                ),
                PlannedTask(
                    task_id="TASK-002", title="b", description="d", depends_on=("TASK-001",)
                ),
                PlannedTask(
                    task_id="TASK-003", title="c", description="d", depends_on=("TASK-002",)
                ),
            ),
        )
        cycle = find_cycle(plan)
        assert cycle, "the cycle must be found"
        assert len(cycle) >= 3

        artifact = build_artifact(
            kind=ArtifactKind.IMPLEMENTATION_PLAN,
            project_id="p1",
            title="Plan",
            author="architect",
            document=plan,
        )
        outcome = validate_artifact(artifact)
        assert not outcome.valid
        assert any(issue.code == PLAN_CYCLE for issue in outcome.blocking_issues)

    def test_an_acyclic_plan_has_no_cycle(self) -> None:
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(
                PlannedTask(task_id="TASK-001", title="a", description="d"),
                PlannedTask(
                    task_id="TASK-002", title="b", description="d", depends_on=("TASK-001",)
                ),
            ),
        )
        assert find_cycle(plan) == ()

    def test_a_task_naming_an_unknown_component_is_blocking(self) -> None:
        architecture = SystemArchitectureDocument(
            product_name="X",
            overview="o",
            components=(
                ArchitectureComponent(
                    component_id="ARCH-001", name="API", responsibility="r"
                ),
            ),
        )
        architecture_artifact = build_artifact(
            kind=ArtifactKind.SYSTEM_ARCHITECTURE,
            project_id="p1",
            title="A",
            author="architect",
            document=architecture,
        )
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(
                PlannedTask(
                    task_id="TASK-001",
                    title="a",
                    description="d",
                    components=("ARCH-404",),
                ),
            ),
        )
        plan_artifact = build_artifact(
            kind=ArtifactKind.IMPLEMENTATION_PLAN,
            project_id="p1",
            title="P",
            author="architect",
            document=plan,
        )
        outcome = validate_against_dependencies(plan_artifact, (architecture_artifact,))
        assert not outcome.valid

    def test_a_missing_acceptance_criterion_is_advisory_not_blocking(self) -> None:
        """A constraint may legitimately be verified by inspection."""
        document = prd_document(
            requirements=(
                Requirement(requirement_id="REQ-001", title="A", statement="a"),
            ),
            acceptance_criteria=(),
        )
        artifact = build_artifact(
            kind=ArtifactKind.PRD,
            project_id="p1",
            title="T",
            author="product_manager",
            document=document,
        )
        outcome = validate_artifact(artifact)
        assert outcome.valid
        assert outcome.issues
        assert not outcome.blocking_issues

    def test_the_index_reports_which_artifact_owns_an_id(self) -> None:
        index = build_index((prd_artifact(),))
        assert index.knows("REQ-001")
        assert index.owner_of("REQ-001") is not None
        assert not index.knows("REQ-999")


class TestContradictionDetection:
    """M4.8's three examples, found without asking a model anything."""

    def offline_prd(self) -> PRDDocument:
        return prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Offline",
                    statement="Must work offline.",
                    properties=frozenset({P.OFFLINE_CAPABLE}),
                ),
            ),
            acceptance_criteria=(),
        )

    def test_offline_versus_cloud_only(self) -> None:
        architecture = SystemArchitectureDocument(
            product_name="X", overview="o", properties=frozenset({P.CLOUD_DEPENDENT})
        )
        findings = check_prd_against_architecture(self.offline_prd(), architecture)
        assert any(finding.blocking for finding in findings)
        assert any("OFFLINE_CAPABLE" in finding.detail for finding in findings)

    def test_authentication_required_versus_none(self) -> None:
        prd = prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Auth",
                    statement="Authentication required.",
                    properties=frozenset({P.AUTHENTICATION_REQUIRED}),
                ),
            ),
            acceptance_criteria=(),
        )
        architecture = SystemArchitectureDocument(
            product_name="X", overview="o", properties=frozenset({P.NO_AUTHENTICATION})
        )
        findings = check_prd_against_architecture(prd, architecture)
        assert any(finding.blocking for finding in findings)

    def test_mobile_responsive_versus_desktop_only(self) -> None:
        ux = UXSpecDocument(product_name="X", properties=frozenset({P.MOBILE_RESPONSIVE}))
        architecture = SystemArchitectureDocument(
            product_name="X", overview="o", properties=frozenset({P.DESKTOP_ONLY})
        )
        findings = check_ux_against_architecture(ux, architecture)
        assert any(finding.blocking for finding in findings)

    def test_implications_are_applied_before_comparison(self) -> None:
        """CLOUD_DEPENDENT implies REQUIRES_NETWORK, which offline contradicts."""
        assert P.REQUIRES_NETWORK in expand(frozenset({P.CLOUD_DEPENDENT}))
        conflicts = find_conflicts(
            frozenset({P.OFFLINE_CAPABLE}), frozenset({P.CLOUD_DEPENDENT})
        )
        assert conflicts

    def test_authentication_is_checked_against_the_endpoints_not_only_the_label(
        self,
    ) -> None:
        """An architecture can claim auth and expose everything anonymously."""
        prd = prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Auth",
                    statement="Authentication required.",
                    properties=frozenset({P.AUTHENTICATION_REQUIRED}),
                ),
            ),
            acceptance_criteria=(),
        )
        architecture = SystemArchitectureDocument(
            product_name="X",
            overview="o",
            properties=frozenset({P.AUTHENTICATION_REQUIRED}),
            endpoints=(
                ApiEndpoint(
                    endpoint_id="API-001",
                    path="/items",
                    purpose="list",
                    requires_authentication=False,
                ),
            ),
        )
        findings = check_prd_against_architecture(prd, architecture)
        assert any(finding.code == "AUTHENTICATION_MISMATCH" for finding in findings)

    def test_offline_versus_an_external_component(self) -> None:
        architecture = SystemArchitectureDocument(
            product_name="X",
            overview="o",
            components=(
                ArchitectureComponent(
                    component_id="ARCH-001",
                    name="Payments",
                    kind=ComponentKind.EXTERNAL,
                    responsibility="charge cards",
                ),
            ),
        )
        findings = check_prd_against_architecture(self.offline_prd(), architecture)
        assert any(
            finding.code == "OFFLINE_VS_EXTERNAL_DEPENDENCY" for finding in findings
        )

    def test_sensitive_data_with_no_threat_model_is_blocking(self) -> None:
        prd = prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Records",
                    statement="Stores medical records.",
                    properties=frozenset({P.SENSITIVE_DATA}),
                ),
            ),
            acceptance_criteria=(),
        )
        architecture = SystemArchitectureDocument(
            product_name="X",
            overview="o",
            entities=(DataEntity(entity_id="ENT-001", name="Record", sensitive=True),),
        )
        findings = check_prd_against_architecture(prd, architecture)
        assert any(
            finding.code == "SENSITIVE_DATA_WITHOUT_THREAT_MODEL" for finding in findings
        )

    def test_agreeing_documents_produce_no_findings(self) -> None:
        """The check must not fire on documents that agree."""
        architecture = SystemArchitectureDocument(
            product_name="X", overview="o", properties=frozenset({P.LOCAL_ONLY})
        )
        findings = check_prd_against_architecture(self.offline_prd(), architecture)
        assert [finding for finding in findings if finding.blocking] == []

    def test_a_prose_hint_is_advisory_never_blocking(self) -> None:
        """A keyword cannot tell "must work offline" from "will not work offline"."""
        prd = prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Connectivity",
                    statement="The tool will never work offline; it requires internet.",
                ),
            ),
            acceptance_criteria=(),
        )
        findings = check_all(prd=prd, include_hints=True)
        assert findings
        assert all(
            finding.severity is ContradictionSeverity.ADVISORY for finding in findings
        )

    def test_missing_documents_are_skipped_not_treated_as_empty(self) -> None:
        assert check_all(prd=self.offline_prd(), include_hints=False) == ()


class TestCrossAgentReview:
    def test_ux_review_reports_uncovered_requirements(self) -> None:
        prd = prd_document()
        spec = UXSpecDocument(
            product_name="Stockroom",
            flows=(
                Flow(
                    flow_id="UX-001",
                    name="f",
                    entry_step="S1",
                    steps=(FlowStep(step_id="S1", name="A", kind=StepKind.TERMINAL),),
                    satisfies=("REQ-001",),
                ),
            ),
        )
        review = review_prd_for_ux_coverage(prd, spec)
        assert any(
            finding.element_id == "REQ-002" for finding in review.findings
        ), "the uncovered requirement must be reported"

    def test_architecture_review_surfaces_contradictions(self) -> None:
        prd = prd_document(
            requirements=(
                Requirement(
                    requirement_id="REQ-001",
                    title="Offline",
                    statement="Must work offline.",
                    properties=frozenset({P.OFFLINE_CAPABLE}),
                ),
            ),
            acceptance_criteria=(),
        )
        architecture = SystemArchitectureDocument(
            product_name="X", overview="o", properties=frozenset({P.CLOUD_DEPENDENT})
        )
        review = review_prd_for_feasibility(prd, architecture)
        assert review.verdict is Verdict.FAIL
        assert review.blockers

    def test_the_verdict_is_computed_not_asserted(self) -> None:
        """M2.1's principle: "looks good" is not a verification result."""
        review = review_requirement_quality(prd_document())
        assert review.verdict is Verdict.PASS
        assert not review.blockers

    def test_a_requirement_without_acceptance_is_reported(self) -> None:
        document = prd_document(
            requirements=(
                Requirement(requirement_id="REQ-001", title="A", statement="a"),
            ),
            acceptance_criteria=(),
        )
        review = review_requirement_quality(document)
        assert any(
            finding.code == "REQUIREMENT_WITHOUT_ACCEPTANCE_CRITERION"
            for finding in review.findings
        )

    def test_open_questions_are_reported_as_information_not_defects(self) -> None:
        document = prd_document(
            open_questions=(
                OpenQuestion(question_id="Q-001", question="Which currency?"),
            )
        )
        review = review_requirement_quality(document)
        question_findings = [
            finding for finding in review.findings if finding.code == "OPEN_QUESTION"
        ]
        assert question_findings
        assert all(
            finding.severity is FindingSeverity.INFO for finding in question_findings
        )
        assert review.verdict is Verdict.PASS

    def test_plan_review_blocks_on_a_cycle(self) -> None:
        architecture = SystemArchitectureDocument(product_name="X", overview="o")
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(
                PlannedTask(
                    task_id="TASK-001", title="a", description="d", depends_on=("TASK-002",)
                ),
                PlannedTask(
                    task_id="TASK-002", title="b", description="d", depends_on=("TASK-001",)
                ),
            ),
        )
        review = review_plan(plan, architecture)
        assert review.verdict is Verdict.FAIL
        assert any(
            finding.code == "PLAN_CIRCULAR_DEPENDENCY" for finding in review.findings
        )

    def test_one_blocker_anywhere_fails_the_set(self) -> None:
        good = review_requirement_quality(prd_document())
        architecture = SystemArchitectureDocument(product_name="X", overview="o")
        bad = review_plan(
            ImplementationPlanDocument(
                product_name="X",
                goal="g",
                tasks=(
                    PlannedTask(
                        task_id="TASK-001",
                        title="a",
                        description="d",
                        depends_on=("TASK-002",),
                    ),
                    PlannedTask(
                        task_id="TASK-002",
                        title="b",
                        description="d",
                        depends_on=("TASK-001",),
                    ),
                ),
            ),
            architecture,
        )
        assert overall_verdict((good,)) is Verdict.PASS
        assert overall_verdict((good, bad)) is Verdict.FAIL
