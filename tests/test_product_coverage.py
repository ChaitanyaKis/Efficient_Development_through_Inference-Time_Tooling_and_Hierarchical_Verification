"""M4.2: requirement coverage, gap detection, and targeted completion.

M4.1 proved the decomposed pipeline produces *valid* artifacts and measured UX coverage at
0.67. Validity checks cannot see that gap: a requirement nothing references is an absence,
and absences do not fail schemas.

These tests pin down the coverage model — computed from evidence, never from a model's
opinion — the approval gate it feeds, and the targeted completion pass that tries to close
gaps without regenerating anything.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from edith.product.architecture import (
    ApiEndpoint,
    ArchitectureComponent,
    ArchitectureDecision,
    DataEntity,
    ImplementationPlanDocument,
    PlannedTask,
    SystemArchitectureDocument,
)
from edith.product.artifacts import ArtifactKind
from edith.product.completion import (
    complete_architecture_coverage,
    complete_ux_coverage,
    merge_component,
    merge_flow,
)
from edith.product.coverage import (
    CoverageState,
    CoverageThreshold,
    Criticality,
    analyse_coverage,
    applies_to,
    criticality_of,
)
from edith.product.prd import (
    AcceptanceCriterion,
    PRDDocument,
    Priority,
    Requirement,
    RequirementKind,
)
from edith.product.properties import ProductProperty as P
from edith.product.ux import Flow, FlowStep, Screen, StepKind, UXSpecDocument

from .test_product_stages import (
    PARAMS,
    SequenceProvider,
    presentation_json,
    screens_json,
    steps_json,
)


def requirement(
    identifier: str,
    title: str = "R",
    *,
    priority: Priority = Priority.MUST,
    kind: RequirementKind = RequirementKind.FUNCTIONAL,
    properties: frozenset[P] = frozenset(),
) -> Requirement:
    return Requirement(
        requirement_id=identifier,
        title=title,
        statement=f"The system must {title.lower()}.",
        priority=priority,
        kind=kind,
        properties=properties,
    )


def prd(*requirements: Requirement) -> PRDDocument:
    items = requirements or (
        requirement("REQ-001", "Record stock"),
        requirement("REQ-002", "List low stock"),
        requirement("REQ-003", "Adjust quantity"),
    )
    return PRDDocument(
        product_name="Stockroom",
        problem="Staff cannot see which items are running low.",
        requirements=items,
        acceptance_criteria=tuple(
            AcceptanceCriterion(
                criterion_id=f"AC-{index:03d}",
                statement="checked",
                verifies=(item.requirement_id,),
            )
            for index, item in enumerate(items, start=1)
        ),
    )


def flow(flow_id: str, *satisfies: str, name: str = "Flow") -> Flow:
    return Flow(
        flow_id=flow_id,
        name=name,
        entry_step=f"{flow_id}-S1",
        steps=(FlowStep(step_id=f"{flow_id}-S1", name="Do", kind=StepKind.TERMINAL),),
        satisfies=satisfies,
    )


def ux(*flows: Flow, screens: tuple[Screen, ...] = ()) -> UXSpecDocument:
    return UXSpecDocument(product_name="Stockroom", flows=flows, screens=screens)


def architecture(
    *components: ArchitectureComponent, **kwargs: Any
) -> SystemArchitectureDocument:
    return SystemArchitectureDocument(
        product_name="Stockroom", overview="o", components=components, **kwargs
    )


class TestCriticality:
    """Coverage policy derives from what the user asked for, not what was convenient."""

    def test_must_is_critical_and_should_is_important(self) -> None:
        assert criticality_of(requirement("REQ-001", priority=Priority.MUST)) is (
            Criticality.CRITICAL
        )
        assert criticality_of(requirement("REQ-002", priority=Priority.SHOULD)) is (
            Criticality.IMPORTANT
        )
        assert criticality_of(requirement("REQ-003", priority=Priority.COULD)) is (
            Criticality.OPTIONAL
        )

    def test_a_wont_requirement_is_not_expected_anywhere(self) -> None:
        """An explicit decision not to build something is not a coverage gap."""
        item = requirement("REQ-009", priority=Priority.WONT)
        assert not applies_to(item, ArtifactKind.UX_SPEC)
        assert not applies_to(item, ArtifactKind.SYSTEM_ARCHITECTURE)

    def test_a_non_functional_requirement_needs_no_user_flow(self) -> None:
        """Forcing a mapping would manufacture coverage nobody should close."""
        item = requirement("REQ-004", kind=RequirementKind.NON_FUNCTIONAL)
        assert not applies_to(item, ArtifactKind.UX_SPEC)
        assert applies_to(item, ArtifactKind.SYSTEM_ARCHITECTURE)


class TestCoverageStates:
    def test_a_flow_covers_the_requirement_it_names(self) -> None:
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        entry = matrix.entry("REQ-001")
        assert entry is not None
        assert entry.ux is CoverageState.COVERED
        assert entry.ux_evidence[0].element_id == "UX-001"

    def test_a_screen_alone_is_partial_coverage(self) -> None:
        """Somewhere the user can reach, with no journey that reaches it."""
        screen = Screen(screen_id="SCR-001", name="List", satisfies=("REQ-002",))
        matrix = analyse_coverage(prd(), ux=ux(screens=(screen,)))
        entry = matrix.entry("REQ-002")
        assert entry is not None
        assert entry.ux is CoverageState.PARTIALLY_COVERED

    def test_an_unreferenced_requirement_is_missing(self) -> None:
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        entry = matrix.entry("REQ-003")
        assert entry is not None
        assert entry.ux is CoverageState.MISSING
        assert entry.ux_evidence == ()

    def test_a_component_covers_but_a_decision_alone_is_partial(self) -> None:
        component = ArchitectureComponent(
            component_id="ARCH-001",
            name="Store",
            responsibility="persist",
            satisfies=("REQ-001",),
        )
        decision = ArchitectureDecision(
            decision_id="ADR-001",
            title="Use SQLite",
            context="c",
            decision="d",
            alternatives=("PostgreSQL",),
            rationale="r",
            consequences=("x",),
            affects_requirements=("REQ-002",),
        )
        matrix = analyse_coverage(
            prd(), architecture=architecture(component, decisions=(decision,))
        )
        assert matrix.entry("REQ-001").architecture is CoverageState.COVERED  # type: ignore[union-attr]
        assert (
            matrix.entry("REQ-002").architecture is CoverageState.PARTIALLY_COVERED  # type: ignore[union-attr]
        )

    def test_an_entity_or_endpoint_also_covers(self) -> None:
        entity = DataEntity(entity_id="ENT-001", name="Item", satisfies=("REQ-001",))
        endpoint = ApiEndpoint(
            endpoint_id="API-001", path="/items", purpose="list", satisfies=("REQ-002",)
        )
        matrix = analyse_coverage(
            prd(), architecture=architecture(entities=(entity,), endpoints=(endpoint,))
        )
        assert matrix.entry("REQ-001").architecture is CoverageState.COVERED  # type: ignore[union-attr]
        assert matrix.entry("REQ-002").architecture is CoverageState.COVERED  # type: ignore[union-attr]

    def test_addressing_a_requirement_while_contradicting_it_is_worse_than_missing(
        self,
    ) -> None:
        """It looks finished, which is why it is a distinct state."""
        offline = requirement("REQ-001", "Work offline", properties=frozenset({P.OFFLINE_CAPABLE}))
        component = ArchitectureComponent(
            component_id="ARCH-001",
            name="API",
            responsibility="calls a hosted service",
            satisfies=("REQ-001",),
        )
        matrix = analyse_coverage(
            prd(offline),
            architecture=architecture(component, properties=frozenset({P.CLOUD_DEPENDENT})),
        )
        entry = matrix.entry("REQ-001")
        assert entry is not None
        assert entry.architecture is CoverageState.CONTRADICTED

    def test_a_task_covers_a_requirement_in_the_plan(self) -> None:
        plan = ImplementationPlanDocument(
            product_name="X",
            goal="g",
            tasks=(
                PlannedTask(
                    task_id="TASK-001",
                    title="build",
                    description="d",
                    implements=("REQ-001",),
                ),
            ),
        )
        matrix = analyse_coverage(prd(), plan=plan)
        assert matrix.entry("REQ-001").plan is CoverageState.COVERED  # type: ignore[union-attr]
        assert matrix.entry("REQ-002").plan is CoverageState.MISSING  # type: ignore[union-attr]

    def test_every_requirement_gets_an_explicit_state(self) -> None:
        """M4.2 item 2: no requirement may be silently absent from the matrix."""
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        assert len(matrix.entries) == 3
        for entry in matrix.entries:
            assert entry.ux in set(CoverageState)


class TestCoverageMath:
    def test_not_applicable_requirements_are_excluded_from_the_fraction(self) -> None:
        """A performance budget should neither count against UX nor inflate it."""
        items = (
            requirement("REQ-001", "Record"),
            requirement("REQ-002", "Fast", kind=RequirementKind.NON_FUNCTIONAL),
        )
        matrix = analyse_coverage(prd(*items), ux=ux(flow("UX-001", "REQ-001")))
        assert matrix.coverage(ArtifactKind.UX_SPEC) == 1.0

    def test_coverage_is_the_covered_fraction(self) -> None:
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        assert matrix.coverage(ArtifactKind.UX_SPEC) == pytest.approx(1 / 3)

    def test_missing_and_partial_are_reported_separately(self) -> None:
        screen = Screen(screen_id="SCR-001", name="L", satisfies=("REQ-002",))
        matrix = analyse_coverage(
            prd(), ux=ux(flow("UX-001", "REQ-001"), screens=(screen,))
        )
        assert matrix.missing(ArtifactKind.UX_SPEC) == ("REQ-003",)
        assert matrix.partial(ArtifactKind.UX_SPEC) == ("REQ-002",)


class TestGapDetection:
    def test_a_gap_names_the_requirement_artifact_stage_and_severity(self) -> None:
        """M4.2 item 3's structured COVERAGE_GAP."""
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        gap = next(
            item for item in matrix.gaps_for(ArtifactKind.UX_SPEC)
            if item.requirement_id == "REQ-003"
        )
        assert gap.artifact is ArtifactKind.UX_SPEC
        assert gap.state is CoverageState.MISSING
        assert gap.criticality is Criticality.CRITICAL
        assert gap.stage == "ux.flows"
        assert gap.code == "COVERAGE_GAP"

    def test_a_critical_gap_blocks_and_an_optional_one_advises(self) -> None:
        items = (
            requirement("REQ-001", "Critical", priority=Priority.MUST),
            requirement("REQ-002", "Optional", priority=Priority.COULD),
        )
        matrix = analyse_coverage(prd(*items), ux=ux())
        blocking = {gap.requirement_id for gap in matrix.blocking_gaps}
        assert blocking == {"REQ-001"}
        advisory = [gap for gap in matrix.gaps if not gap.blocking]
        assert [gap.code for gap in advisory] == ["ADVISORY_COVERAGE_GAP"]

    def test_a_contradiction_blocks_at_any_criticality(self) -> None:
        """An artifact that looks finished and is wrong is worse than one that is absent."""
        offline = requirement(
            "REQ-001",
            "Work offline",
            priority=Priority.COULD,
            properties=frozenset({P.OFFLINE_CAPABLE}),
        )
        component = ArchitectureComponent(
            component_id="ARCH-001", name="API", responsibility="r", satisfies=("REQ-001",)
        )
        matrix = analyse_coverage(
            prd(offline),
            architecture=architecture(component, properties=frozenset({P.CLOUD_DEPENDENT})),
        )
        assert matrix.blocking_gaps
        assert matrix.blocking_gaps[0].state is CoverageState.CONTRADICTED

    def test_no_gaps_are_emitted_for_an_artifact_that_was_never_produced(self) -> None:
        """A missing artifact is a pipeline failure; a gap per requirement would bury it."""
        matrix = analyse_coverage(prd())
        assert matrix.gaps == []


class TestThreshold:
    """M4.2 item 6: the policy is explicit and testable, not 100% by fiat."""

    def test_a_missing_critical_requirement_fails_the_threshold(self) -> None:
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        assert not matrix.satisfies(ArtifactKind.UX_SPEC)

    def test_full_critical_coverage_passes(self) -> None:
        spec = ux(flow("UX-001", "REQ-001", "REQ-002", "REQ-003"))
        matrix = analyse_coverage(prd(), ux=spec)
        assert matrix.satisfies(ArtifactKind.UX_SPEC)

    def test_an_optional_requirement_may_be_uncovered(self) -> None:
        items = (
            requirement("REQ-001", "Critical"),
            requirement("REQ-002", "Optional", priority=Priority.COULD),
        )
        matrix = analyse_coverage(prd(*items), ux=ux(flow("UX-001", "REQ-001")))
        assert matrix.satisfies(ArtifactKind.UX_SPEC)

    def test_the_important_floor_is_enforced(self) -> None:
        items = (
            requirement("REQ-001", "A", priority=Priority.SHOULD),
            requirement("REQ-002", "B", priority=Priority.SHOULD),
        )
        matrix = analyse_coverage(prd(*items), ux=ux())
        strict = CoverageThreshold(require_all_critical=False, minimum_important=0.5)
        assert not matrix.satisfies(ArtifactKind.UX_SPEC, strict)

        covered = analyse_coverage(prd(*items), ux=ux(flow("UX-001", "REQ-001")))
        assert covered.satisfies(ArtifactKind.UX_SPEC, strict)

    def test_the_policy_describes_itself(self) -> None:
        assert "critical" in CoverageThreshold().describe()


class TestMergeIsAdditive:
    """Targeted completion must not disturb anything already in the artifact."""

    def test_merging_a_flow_keeps_existing_ids_stable(self) -> None:
        from edith.agents.ux_stages import StepsOutput

        existing = ux(flow("UX-001", "REQ-001", name="Existing"))
        merged = merge_flow(
            existing,
            StepsOutput.model_validate(json.loads(steps_json())),
            name="New",
            description="d",
            requirement_id="REQ-003",
        )
        assert merged.flows[0].flow_id == "UX-001"
        assert merged.flows[0].name == "Existing"
        assert merged.flows[1].flow_id == "UX-002"
        assert merged.flows[1].satisfies == ("REQ-003",)

    def test_the_system_attaches_the_requirement_not_the_model(self) -> None:
        """The model produces content; Edith decides what it satisfies."""
        from edith.agents.ux_stages import StepsOutput

        merged = merge_flow(
            ux(),
            StepsOutput.model_validate(json.loads(steps_json())),
            name="New",
            description="d",
            requirement_id="REQ-002",
        )
        assert merged.flows[0].satisfies == ("REQ-002",)

    def test_merging_a_component_keeps_existing_ids_stable(self) -> None:
        from edith.product.completion import TargetedComponentOutput

        existing = architecture(
            ArchitectureComponent(
                component_id="ARCH-001", name="Store", responsibility="persist"
            )
        )
        merged = merge_component(
            existing,
            TargetedComponentOutput(
                name="Reporter", responsibility="report", depends_on=["Store"]
            ),
            requirement_id="REQ-002",
        )
        assert merged.components[0].component_id == "ARCH-001"
        assert merged.components[1].component_id == "ARCH-002"
        assert merged.components[1].depends_on == ("ARCH-001",)
        assert merged.components[1].satisfies == ("REQ-002",)


class TestTargetedCompletion:
    def provider(self, count: int = 4, **kwargs: Any) -> SequenceProvider:
        return SequenceProvider([steps_json()] * count, **kwargs)

    def test_a_gap_is_closed_and_verified_by_rechecking(self) -> None:
        document = ux(flow("UX-001", "REQ-001"))
        result = complete_ux_coverage(self.provider(), prd(), document)

        assert result.closed == 2, result.summary()
        assert result.after.coverage(ArtifactKind.UX_SPEC) == 1.0
        assert result.improvement(ArtifactKind.UX_SPEC) > 0

    def test_nothing_is_regenerated(self) -> None:
        """M4.2 item 6: the existing artifact must survive untouched."""
        document = ux(flow("UX-001", "REQ-001", name="Original"))
        result = complete_ux_coverage(self.provider(), prd(), document)

        assert result.ux is not None
        assert result.ux.flows[0].flow_id == "UX-001"
        assert result.ux.flows[0].name == "Original"
        assert len(result.ux.flows) == 3

    def test_one_model_call_per_gap_not_a_full_regeneration(self) -> None:
        document = ux(flow("UX-001", "REQ-001"))
        result = complete_ux_coverage(self.provider(), prd(), document)
        assert result.completion_calls == 2, "one call per remaining gap"

    def test_a_satisfied_artifact_spends_nothing(self) -> None:
        document = ux(flow("UX-001", "REQ-001", "REQ-002", "REQ-003"))
        result = complete_ux_coverage(self.provider(), prd(), document)
        assert result.completion_calls == 0
        assert result.attempts == []

    def test_a_failed_completion_leaves_the_gap_open(self) -> None:
        """A pass that could not generate must not report the gap as closed."""
        result = complete_ux_coverage(
            self.provider(fail_at=1), prd(), ux(flow("UX-001", "REQ-001"))
        )
        assert result.attempts
        first = result.attempts[0]
        assert not first.merged
        assert not first.closed
        assert "generation failed" in first.rejected_reason

    def test_a_partial_failure_still_closes_the_other_gap(self) -> None:
        result = complete_ux_coverage(
            self.provider(fail_at=1), prd(), ux(flow("UX-001", "REQ-001"))
        )
        assert result.closed == 1
        assert result.after.coverage(ArtifactKind.UX_SPEC) > result.before.coverage(
            ArtifactKind.UX_SPEC
        )

    def test_the_attempt_budget_is_bounded(self) -> None:
        many = prd(*(requirement(f"REQ-{index:03d}") for index in range(1, 9)))
        result = complete_ux_coverage(
            SequenceProvider([steps_json()] * 10), many, ux(), limit=3
        )
        assert len(result.attempts) == 3

    def test_critical_gaps_are_attempted_first(self) -> None:
        items = (
            requirement("REQ-001", "Optional", priority=Priority.COULD),
            requirement("REQ-002", "Critical", priority=Priority.MUST),
        )
        result = complete_ux_coverage(
            SequenceProvider([steps_json()]), prd(*items), ux(), limit=1
        )
        assert result.attempts[0].requirement_id == "REQ-002"

    def test_architecture_gaps_are_closed_the_same_way(self) -> None:
        component_json = json.dumps(
            {"name": "Reporter", "kind": "SERVICE", "responsibility": "report"}
        )
        existing = architecture(
            ArchitectureComponent(
                component_id="ARCH-001",
                name="Store",
                responsibility="persist",
                satisfies=("REQ-001",),
            )
        )
        result = complete_architecture_coverage(
            SequenceProvider([component_json] * 4), prd(), existing
        )
        assert result.closed == 2
        assert result.after.coverage(ArtifactKind.SYSTEM_ARCHITECTURE) == 1.0
        assert result.architecture is not None
        assert result.architecture.components[0].component_id == "ARCH-001"


class TestNoQualityGaming:
    """M4.2 item 10: coverage must be earned, never asserted."""

    def test_coverage_requires_an_element_reference_not_prose(self) -> None:
        """An overview mentioning a requirement is not coverage."""
        spec = UXSpecDocument(
            product_name="X",
            overview="This design fully addresses REQ-001, REQ-002 and REQ-003.",
            flows=(flow("UX-001"),),
        )
        matrix = analyse_coverage(prd(), ux=spec)
        assert matrix.coverage(ArtifactKind.UX_SPEC) == 0.0

    def test_a_hallucinated_requirement_reference_grants_nothing(self) -> None:
        """A flow claiming REQ-404 covers no real requirement."""
        spec = ux(flow("UX-001", "REQ-001"))
        matrix = analyse_coverage(prd(), ux=spec)
        assert matrix.entry("REQ-002").ux is CoverageState.MISSING  # type: ignore[union-attr]

    def test_the_recheck_is_computed_from_the_merged_document(self) -> None:
        """A completion cannot mark itself successful; the matrix decides."""
        document = ux(flow("UX-001", "REQ-001"))
        result = complete_ux_coverage(
            SequenceProvider([steps_json()] * 4), prd(), document
        )
        recomputed = analyse_coverage(prd(), ux=result.ux)
        assert recomputed.coverage(ArtifactKind.UX_SPEC) == result.after.coverage(
            ArtifactKind.UX_SPEC
        )

    def test_a_critic_note_never_changes_a_computed_state(self) -> None:
        """An advisory opinion is stored beside the evidence, never instead of it."""
        matrix = analyse_coverage(prd(), ux=ux(flow("UX-001", "REQ-001")))
        entry = matrix.entry("REQ-003")
        assert entry is not None
        entry.critic_note = "the model believes this is covered by the list screen"
        assert entry.ux is CoverageState.MISSING


class TestApprovalGate:
    """M4.2 item 12: a critical gap must stop an artifact becoming project truth."""

    def build(self, tmp_path: Any, responses: list[str], **kwargs: Any) -> Any:
        from edith.config.schema import EdithConfig, ModelsConfig
        from edith.product.service import ProductService
        from edith.product.store import ProductStore

        config = EdithConfig(models=ModelsConfig(profiles={"default": PARAMS}))
        store = ProductStore(tmp_path / "artifacts.db")
        return (
            ProductService(
                config, store, provider=SequenceProvider(responses, **kwargs)
            ),
            store,
        )

    def seed(self, store: Any, document: PRDDocument) -> Any:
        from edith.product.artifacts import build_artifact
        from edith.product.validation import validate_artifact

        artifact = build_artifact(
            kind=ArtifactKind.PRD,
            project_id="p1",
            title="PRD",
            author="product_manager",
            document=document,
        )
        artifact = artifact.model_copy(update={"validation": validate_artifact(artifact)})
        store.save(artifact)
        return artifact

    def flows_covering(self, *identifiers: str) -> str:
        return json.dumps(
            {"flows": [{"name": "Main", "satisfies": list(identifiers)}]}
        )

    def test_an_uncovered_critical_requirement_blocks_approval(
        self, tmp_path: Any
    ) -> None:
        from edith.product.store import ArtifactConflictError

        document = prd(requirement("REQ-001"), requirement("REQ-002"))
        service, store = self.build(
            tmp_path,
            [
                self.flows_covering("REQ-001"),
                screens_json(),
                steps_json(),
                presentation_json(),
            ],
        )
        self.seed(store, document)

        outcome = service.create_ux_spec("p1")
        assert outcome.ok
        artifact = outcome.artifact
        assert artifact is not None
        assert not artifact.validation.valid
        assert any(
            issue.code == "COVERAGE_GAP" for issue in artifact.validation.blocking_issues
        )

        with pytest.raises(ArtifactConflictError):
            service.approve_artifact(artifact.artifact_id)
        store.close()

    def test_full_critical_coverage_is_approvable(self, tmp_path: Any) -> None:
        from edith.product.artifacts import ArtifactStatus

        document = prd(requirement("REQ-001"), requirement("REQ-002"))
        service, store = self.build(
            tmp_path,
            [
                self.flows_covering("REQ-001", "REQ-002"),
                screens_json(),
                steps_json(),
                presentation_json(),
            ],
        )
        self.seed(store, document)

        outcome = service.create_ux_spec("p1")
        artifact = outcome.artifact
        assert artifact is not None
        assert artifact.validation.valid, artifact.validation.summary()
        approved = service.approve_artifact(artifact.artifact_id)
        assert approved.status is ArtifactStatus.APPROVED
        store.close()

    def test_an_optional_gap_is_advisory_and_does_not_block(self, tmp_path: Any) -> None:
        from edith.product.artifacts import ArtifactStatus

        document = prd(
            requirement("REQ-001"),
            requirement("REQ-002", priority=Priority.COULD),
        )
        service, store = self.build(
            tmp_path,
            [
                self.flows_covering("REQ-001"),
                screens_json(),
                steps_json(),
                presentation_json(),
            ],
        )
        self.seed(store, document)

        outcome = service.create_ux_spec("p1")
        artifact = outcome.artifact
        assert artifact is not None
        assert artifact.validation.valid
        assert any(
            issue.code == "ADVISORY_COVERAGE_GAP" for issue in artifact.validation.issues
        )
        assert (
            service.approve_artifact(artifact.artifact_id).status
            is ArtifactStatus.APPROVED
        )
        store.close()

    def test_targeted_completion_can_turn_a_blocked_artifact_into_an_approvable_one(
        self, tmp_path: Any
    ) -> None:
        """The end-to-end claim M4.2 is testing, on a scripted model."""
        from edith.config.schema import EdithConfig, ModelsConfig
        from edith.product.service import ProductService
        from edith.product.store import ProductStore

        document = prd(requirement("REQ-001"), requirement("REQ-002"))
        config = EdithConfig(models=ModelsConfig(profiles={"default": PARAMS}))
        store = ProductStore(tmp_path / "artifacts.db")
        provider = SequenceProvider(
            [
                self.flows_covering("REQ-001"),
                screens_json(),
                steps_json(),
                presentation_json(),
                steps_json(),
            ]
        )
        service = ProductService(
            config, store, provider=provider, targeted_completion=True
        )
        self.seed(store, document)

        outcome = service.create_ux_spec("p1")
        artifact = outcome.artifact
        assert artifact is not None
        assert artifact.validation.valid, artifact.validation.summary()
        store.close()
