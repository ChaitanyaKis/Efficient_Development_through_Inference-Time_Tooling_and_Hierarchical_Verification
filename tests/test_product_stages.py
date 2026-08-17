"""M4.1: stage decomposition, failure isolation, and deterministic assembly.

M4 measured a monolithic UX call failing six consecutive times on a 3B model. These tests
pin down the machinery that replaced it: small stages, a ledger that keeps successes and
failures apart, and an assembler that builds a coherent artifact from a partial run.

Everything here is offline. A scripted provider stands in for the model, and a provider that
raises stands in for every way a stage can fail.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from edith.agents.architect_stages import (
    ComponentsOutput,
    DataModelOutput,
    DecisionsOutput,
    PlanOutput,
    ThreatModelOutput,
    assemble_architecture,
    assemble_plan,
    components_context,
    entities_context,
)
from edith.agents.ux_stages import (
    FlowsOutput,
    PresentationOutput,
    ScreensOutput,
    StepsOutput,
    assemble_ux_spec,
    requirements_context,
    requirements_for,
)
from edith.config.schema import ModelParams
from edith.errors import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from edith.product.architect_pipeline import (
    STAGE_COMPONENTS,
    STAGE_DATA,
    STAGE_PLAN,
    run_architect_pipeline,
)
from edith.product.prd import AcceptanceCriterion, PRDDocument, Requirement
from edith.product.stages import (
    StageFailure,
    StageLedger,
    StageStatus,
    classify_exception,
    not_applicable,
    run_stage,
    skipped,
)
from edith.product.ux import StepKind
from edith.product.ux_pipeline import (
    STAGE_FLOWS,
    STAGE_PRESENTATION,
    STAGE_SCREENS,
    run_ux_pipeline,
)

from .fakes import FakeProvider

PARAMS = ModelParams(model_name="test-model:q4")


def prd() -> PRDDocument:
    """A small PRD the stages can serve."""
    return PRDDocument(
        product_name="Stockroom",
        problem="Staff cannot see which items are running low.",
        requirements=(
            Requirement(
                requirement_id="REQ-001",
                title="Record stock",
                statement="The system records the quantity of every item.",
            ),
            Requirement(
                requirement_id="REQ-002",
                title="Low list",
                statement="Items at or below threshold are listed as low.",
            ),
        ),
        acceptance_criteria=(
            AcceptanceCriterion(
                criterion_id="AC-001", statement="Quantity recorded.", verifies=("REQ-001",)
            ),
        ),
    )


# -- Model output fixtures ---------------------------------------------------------------


def flows_json(**overrides: Any) -> str:
    payload = {"flows": [{"name": "Add stock", "satisfies": ["REQ-001"]}]}
    payload.update(overrides)
    return json.dumps(payload)


def steps_json(**overrides: Any) -> str:
    payload = {
        "steps": [
            {"name": "Open form", "kind": "VIEW"},
            {
                "name": "Submit",
                "kind": "ACTION",
                "next_steps": ["Saved"],
                "error_steps": ["Failed"],
            },
            {"name": "Saved", "kind": "TERMINAL"},
            {"name": "Failed", "kind": "ABORT"},
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


def screens_json(**overrides: Any) -> str:
    payload = {
        "screens": [
            {"name": "Stock list", "states": ["DEFAULT"], "satisfies": ["REQ-002"]},
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


def presentation_json(**overrides: Any) -> str:
    payload = {
        "components": [{"name": "Item row", "states": ["default"]}],
        "accessibility": ["Every control is keyboard reachable"],
        "design_tokens": [{"name": "danger", "category": "color", "value": "#c00"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


def components_json(**overrides: Any) -> str:
    payload = {
        "overview": "One local application with a SQLite store.",
        "components": [
            {
                "name": "Storage",
                "kind": "DATASTORE",
                "responsibility": "Persist items.",
                "satisfies": ["REQ-001"],
            },
            {
                "name": "UI",
                "kind": "UI",
                "responsibility": "Show stock.",
                "depends_on": ["Storage"],
                "satisfies": ["REQ-002"],
            },
        ],
        "properties": ["LOCAL_ONLY"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def data_json(**overrides: Any) -> str:
    payload = {
        "entities": [
            {"name": "Item", "fields": {"name": "str"}, "satisfies": ["REQ-001"]}
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


def api_json() -> str:
    return json.dumps({"endpoints": []})


def decisions_json(**overrides: Any) -> str:
    payload = {
        "decisions": [
            {
                "title": "Use SQLite",
                "context": "One machine, no operations team.",
                "decision": "Store data in a local SQLite file.",
                "alternatives": ["PostgreSQL"],
                "rationale": "No concurrent writers.",
                "consequences": ["No multi-machine access"],
                "affects_requirements": ["REQ-001"],
            }
        ],
        "technologies": [
            {
                "name": "SQLite",
                "role": "database",
                "rationale": "Zero operations.",
                "alternatives_rejected": ["PostgreSQL"],
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def threats_json() -> str:
    return json.dumps(
        {
            "threats": [
                {
                    "asset": "Stock data",
                    "description": "A stolen laptop exposes the file.",
                    "mitigation": "Full-disk encryption assumed.",
                }
            ]
        }
    )


def plan_json(**overrides: Any) -> str:
    payload = {
        "tasks": [
            {
                "title": "Create the store",
                "description": "Create the schema and repository.",
                "implements": ["REQ-001"],
                "components": ["Storage"],
            },
            {
                "title": "Build the list",
                "description": "Render the stock list.",
                "depends_on": ["Create the store"],
                "implements": ["REQ-002"],
                "components": ["UI"],
            },
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


class SequenceProvider(FakeProvider):
    """Returns a fixed sequence, raising a supplied exception at a chosen call index."""

    def __init__(
        self, responses: list[str], *, fail_at: int | None = None, error: Exception | None = None
    ) -> None:
        super().__init__(PARAMS, responses)
        self.fail_at = fail_at
        self.error = error or StructuredOutputError("the model would not comply")
        self.call_index = 0

    def _generate_raw(self, messages: Any, options: Any = None, **kwargs: Any) -> Any:
        self.call_index += 1
        if self.fail_at is not None and self.call_index == self.fail_at:
            raise self.error
        return super()._generate_raw(messages, options, **kwargs)


# -- Stage framework -----------------------------------------------------------------------


class TestStageFramework:
    def test_a_successful_stage_records_its_schema_size(self) -> None:
        """Schema size per call is the independent variable of the M4.1 experiment."""
        result = run_stage(
            "demo",
            FlowsOutput,
            lambda: FlowsOutput.model_validate(json.loads(flows_json())),
            prompt_chars=100,
            elements_of=lambda output: len(output.flows),
        )
        assert result.ok
        assert result.measurement is not None
        assert result.measurement.schema_bytes > 0
        assert result.measurement.input_chars == 100
        assert result.measurement.elements == 1

    def test_a_stage_never_raises(self) -> None:
        """A caller must be able to render a partial run without catching anything."""

        def explode() -> FlowsOutput:
            raise RuntimeError("boom")

        result = run_stage("demo", FlowsOutput, explode, prompt_chars=10)
        assert not result.ok
        assert result.status is StageStatus.FAILED
        assert result.failure is StageFailure.MODEL_FAILURE

    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (ProviderTimeoutError("slow"), StageFailure.TIMEOUT),
            (ProviderUnavailableError("down"), StageFailure.ENVIRONMENT_FAILURE),
            (StructuredOutputError("bad"), StageFailure.RETRY_EXHAUSTED),
            (RuntimeError("odd"), StageFailure.MODEL_FAILURE),
        ],
    )
    def test_failures_are_classified_not_lumped_together(
        self, exception: Exception, expected: StageFailure
    ) -> None:
        """M4.1 item 4: a generic failure hides which remedy applies."""
        failure, _ = classify_exception(exception)
        assert failure is expected

    def test_a_validation_error_is_a_schema_failure(self) -> None:
        try:
            FlowsOutput.model_validate({"flows": []})
        except ValidationError as exc:
            failure, _ = classify_exception(exc)
            assert failure is StageFailure.SCHEMA_VALIDATION_FAILURE
        else:  # pragma: no cover - the schema requires at least one flow
            pytest.fail("an empty flow list must not validate")

    def test_every_failure_maps_to_a_platform_category(self) -> None:
        for failure in StageFailure:
            assert failure.category is not None

    def test_a_ledger_keeps_successes_and_failures_apart(self) -> None:
        ledger = StageLedger()
        ledger.add(
            run_stage(
                "good",
                FlowsOutput,
                lambda: FlowsOutput.model_validate(json.loads(flows_json())),
                prompt_chars=10,
            )
        )
        ledger.add(skipped("later", "a dependency failed"))

        def explode() -> FlowsOutput:
            raise RuntimeError("boom")

        ledger.add(run_stage("bad", FlowsOutput, explode, prompt_chars=10))

        assert ledger.valid_stages == ("good",)
        assert ledger.failed_stages == ("bad",)
        assert not ledger.complete

    def test_a_skipped_stage_does_not_make_a_run_incomplete_on_its_own(self) -> None:
        """Skipped and not-applicable are different facts from failed."""
        ledger = StageLedger()
        ledger.add(
            run_stage(
                "good",
                FlowsOutput,
                lambda: FlowsOutput.model_validate(json.loads(flows_json())),
                prompt_chars=10,
            )
        )
        ledger.add(not_applicable("tokens", "this product has no interface"))
        assert ledger.complete

    def test_totals_report_the_largest_single_call(self) -> None:
        """A mean would hide the number the hypothesis is about."""
        ledger = StageLedger()
        for index, chars in enumerate((100, 900, 300), start=1):
            ledger.add(
                run_stage(
                    f"s{index}",
                    FlowsOutput,
                    lambda: FlowsOutput.model_validate(json.loads(flows_json())),
                    prompt_chars=chars,
                )
            )
        totals = ledger.totals()
        assert totals["largest_input_chars"] == 900
        assert totals["input_chars"] == 1300


# -- UX decomposition -----------------------------------------------------------------------


class TestUXDecomposition:
    def responses(self) -> list[str]:
        return [flows_json(), screens_json(), steps_json(), presentation_json()]

    def test_the_happy_path_assembles_a_complete_specification(self) -> None:
        provider = SequenceProvider(self.responses())
        result = run_ux_pipeline(provider, prd())

        assert result.ok
        assert result.complete
        document = result.document
        assert document is not None
        assert len(document.flows) == 1
        assert len(document.screens) == 1
        assert document.flows[0].flow_id == "UX-001"
        assert document.screens[0].screen_id == "SCR-001"

    def test_every_stage_schema_is_far_smaller_than_the_monolithic_one(self) -> None:
        """The hypothesis, stated as an assertion about the artifacts under test."""
        from edith.agents.ux_designer import UXDesignerOutput

        monolithic = len(json.dumps(UXDesignerOutput.model_json_schema()))
        for schema in (FlowsOutput, StepsOutput, ScreensOutput, PresentationOutput):
            size = len(json.dumps(schema.model_json_schema()))
            assert size < monolithic / 2, f"{schema.__name__} is {size} vs {monolithic}"

    def test_a_failed_flow_stage_skips_the_rest_rather_than_failing_it(self) -> None:
        provider = SequenceProvider(self.responses(), fail_at=1)
        result = run_ux_pipeline(provider, prd())

        assert not result.ok
        ledger = result.ledger
        assert ledger.get(STAGE_FLOWS).status is StageStatus.FAILED  # type: ignore[union-attr]
        assert ledger.get(STAGE_SCREENS).status is StageStatus.SKIPPED  # type: ignore[union-attr]
        assert ledger.get(STAGE_PRESENTATION).status is StageStatus.SKIPPED  # type: ignore[union-attr]

    def test_a_failed_presentation_stage_keeps_the_flows_and_screens(self) -> None:
        """M4.1 item 3: a failed stage must not destroy a validated one."""
        provider = SequenceProvider(self.responses(), fail_at=4)
        result = run_ux_pipeline(provider, prd())

        assert result.ok, "flows and screens survived"
        assert not result.complete, "the run is incomplete and cannot be approved"
        document = result.document
        assert document is not None
        assert document.flows and document.screens
        assert document.components == ()

    def test_a_failed_step_stage_omits_that_flow_rather_than_inventing_one(self) -> None:
        """A fabricated journey in a specification is worse than a missing one."""
        provider = SequenceProvider(
            [flows_json(), screens_json(), steps_json(), presentation_json()], fail_at=3
        )
        result = run_ux_pipeline(provider, prd())

        assert result.ok
        assert not result.complete
        document = result.document
        assert document is not None
        assert document.flows == (), "the flow whose steps failed must be omitted"
        assert document.screens, "screens are unaffected"

    def test_ids_are_assigned_by_the_system(self) -> None:
        """The model names things; Edith numbers them."""
        provider = SequenceProvider(self.responses())
        document = run_ux_pipeline(provider, prd()).document
        assert document is not None
        assert document.flows[0].flow_id == "UX-001"
        assert document.flows[0].steps[0].step_id == "UX-001-S1"
        assert document.components[0].component_id == "CMP-001"
        assert document.design_tokens[0].token_id == "TOK-001"  # noqa: S105 - element id

    def test_a_requirement_the_prd_never_defined_is_dropped(self) -> None:
        provider = SequenceProvider(
            [
                flows_json(flows=[{"name": "F", "satisfies": ["REQ-001", "REQ-404"]}]),
                screens_json(),
                steps_json(),
                presentation_json(),
            ]
        )
        document = run_ux_pipeline(provider, prd()).document
        assert document is not None
        assert document.flows[0].satisfies == ("REQ-001",)

    def test_required_screen_states_are_present(self) -> None:
        provider = SequenceProvider(self.responses())
        document = run_ux_pipeline(provider, prd()).document
        assert document is not None
        assert document.screens_missing_states() == {}

    def test_an_environment_failure_is_classified_as_one(self) -> None:
        provider = SequenceProvider(
            self.responses(), fail_at=1, error=ProviderUnavailableError("ollama is down")
        )
        result = run_ux_pipeline(provider, prd())
        assert result.ledger.failures == (StageFailure.ENVIRONMENT_FAILURE,)


class TestUXContextControl:
    """M4.1 item 5: each stage gets what it needs, not the whole PRD."""

    def test_requirement_context_is_lines_not_the_whole_document(self) -> None:
        document = prd()
        context = requirements_context(document)
        assert "REQ-001" in context
        assert document.problem not in context

    def test_a_flow_gets_only_the_requirements_it_serves(self) -> None:
        context = requirements_for(prd(), ("REQ-001",))
        assert "REQ-001" in context
        assert "REQ-002" not in context

    def test_a_flow_serving_nothing_gets_everything_rather_than_nothing(self) -> None:
        """A stage with no context produces worse output than one with slightly too much."""
        context = requirements_for(prd(), ())
        assert "REQ-001" in context
        assert "REQ-002" in context

    def test_stage_prompts_stay_smaller_than_the_monolithic_prompt(self) -> None:
        provider = SequenceProvider(
            [flows_json(), screens_json(), steps_json(), presentation_json()]
        )
        result = run_ux_pipeline(provider, prd())
        totals = result.ledger.totals()
        monolithic_prompt = len(prd().render())
        assert totals["largest_input_chars"] < monolithic_prompt * 4


# -- Architect decomposition -------------------------------------------------------------


class TestArchitectDecomposition:
    def responses(self) -> list[str]:
        return [
            components_json(),
            data_json(),
            api_json(),
            decisions_json(),
            threats_json(),
            plan_json(),
        ]

    def test_the_happy_path_assembles_an_architecture_and_a_plan(self) -> None:
        provider = SequenceProvider(self.responses())
        result = run_architect_pipeline(provider, prd())

        assert result.ok
        assert result.complete
        architecture = result.architecture
        plan = result.plan
        assert architecture is not None and plan is not None
        assert len(architecture.components) == 2
        assert architecture.decisions[0].decision_id == "ADR-001"
        assert plan.tasks[1].depends_on == ("TASK-001",)

    def test_component_failure_skips_every_dependent_stage(self) -> None:
        """M4.1 item 7: no stage consumes unverified upstream output."""
        provider = SequenceProvider(self.responses(), fail_at=1)
        result = run_architect_pipeline(provider, prd())

        assert not result.ok
        ledger = result.ledger
        assert ledger.get(STAGE_COMPONENTS).status is StageStatus.FAILED  # type: ignore[union-attr]
        for stage in (STAGE_DATA, STAGE_PLAN):
            assert ledger.get(stage).status is StageStatus.SKIPPED  # type: ignore[union-attr]

    def test_a_failed_threat_stage_still_yields_an_architecture(self) -> None:
        provider = SequenceProvider(self.responses(), fail_at=5)
        result = run_architect_pipeline(provider, prd())

        assert result.ok
        assert not result.complete
        architecture = result.architecture
        assert architecture is not None
        assert architecture.components
        assert architecture.threats == ()

    def test_component_references_are_resolved_by_name(self) -> None:
        provider = SequenceProvider(self.responses())
        architecture = run_architect_pipeline(provider, prd()).architecture
        assert architecture is not None
        ui = next(item for item in architecture.components if item.name == "UI")
        assert ui.depends_on == ("ARCH-001",)

    def test_data_flow_is_derived_from_the_component_graph(self) -> None:
        provider = SequenceProvider(self.responses())
        architecture = run_architect_pipeline(provider, prd()).architecture
        assert architecture is not None
        assert architecture.data_flows
        assert architecture.data_flows[0].source == "ARCH-002"

    def test_a_decision_still_requires_alternatives_and_consequences(self) -> None:
        """M4.1 item 8: decomposition must not weaken a guarantee."""
        with pytest.raises(ValidationError):
            DecisionsOutput.model_validate(
                {
                    "decisions": [
                        {
                            "title": "t",
                            "context": "c",
                            "decision": "d",
                            "alternatives": [],
                            "rationale": "r",
                            "consequences": ["x"],
                        }
                    ]
                }
            )

    def test_every_stage_schema_is_smaller_than_the_monolithic_one(self) -> None:
        from edith.agents.architect import ArchitectOutput

        monolithic = len(json.dumps(ArchitectOutput.model_json_schema()))
        for schema in (
            ComponentsOutput,
            DataModelOutput,
            DecisionsOutput,
            ThreatModelOutput,
            PlanOutput,
        ):
            size = len(json.dumps(schema.model_json_schema()))
            assert size < monolithic / 2, f"{schema.__name__} is {size} vs {monolithic}"

    def test_downstream_stages_receive_component_names_not_the_architecture(self) -> None:
        components = ComponentsOutput.model_validate(json.loads(components_json()))
        context = components_context(tuple(components.components))
        assert "Storage" in context
        assert "ARCH-001" not in context, "ids are assigned after generation, not before"

    def test_the_threat_stage_is_told_which_data_is_sensitive(self) -> None:
        data = DataModelOutput.model_validate(
            json.loads(data_json(entities=[{"name": "Record", "sensitive": True}]))
        )
        assert "[SENSITIVE]" in entities_context(tuple(data.entities))


class TestDeterministicAssembly:
    """Assembly is pure: the same stage outputs always produce the same artifact."""

    def test_ux_assembly_is_reproducible(self) -> None:
        arguments: dict[str, Any] = {
            "product_name": "Stockroom",
            "prd": prd(),
            "flows": tuple(
                FlowsOutput.model_validate(json.loads(flows_json())).flows
            ),
            "steps_by_flow": {
                "addstock": StepsOutput.model_validate(json.loads(steps_json()))
            },
            "screens": tuple(
                ScreensOutput.model_validate(json.loads(screens_json())).screens
            ),
            "presentation": PresentationOutput.model_validate(
                json.loads(presentation_json())
            ),
        }
        first = assemble_ux_spec(**arguments)
        second = assemble_ux_spec(**arguments)
        assert first.model_dump() == second.model_dump()

    def test_architecture_assembly_is_reproducible(self) -> None:
        components = ComponentsOutput.model_validate(json.loads(components_json()))
        arguments: dict[str, Any] = {
            "product_name": "Stockroom",
            "prd": prd(),
            "components": components,
            "data": DataModelOutput.model_validate(json.loads(data_json())),
            "api": None,
            "decisions": DecisionsOutput.model_validate(json.loads(decisions_json())),
            "threats": ThreatModelOutput.model_validate(json.loads(threats_json())),
        }
        first = assemble_architecture(**arguments)
        second = assemble_architecture(**arguments)
        assert first.model_dump() == second.model_dump()

    def test_a_plan_cycle_is_preserved_for_validation_to_report(self) -> None:
        components = ComponentsOutput.model_validate(json.loads(components_json()))
        architecture = assemble_architecture(
            product_name="X",
            prd=prd(),
            components=components,
            data=None,
            api=None,
            decisions=None,
            threats=None,
        )
        plan = assemble_plan(
            product_name="X",
            goal="g",
            plan=PlanOutput.model_validate(
                json.loads(
                    plan_json(
                        tasks=[
                            {"title": "A", "description": "d", "depends_on": ["B"]},
                            {"title": "B", "description": "d", "depends_on": ["A"]},
                        ]
                    )
                )
            ),
            architecture=architecture,
            prd=prd(),
        )
        from edith.product.validation import find_cycle

        assert find_cycle(plan)

    def test_assembly_never_invents_a_step_for_a_missing_flow(self) -> None:
        document = assemble_ux_spec(
            product_name="X",
            prd=prd(),
            flows=tuple(FlowsOutput.model_validate(json.loads(flows_json())).flows),
            steps_by_flow={},
            screens=(),
            presentation=None,
        )
        assert document.flows == ()

    def test_a_terminal_step_is_not_reported_as_a_dead_end(self) -> None:
        document = assemble_ux_spec(
            product_name="X",
            prd=prd(),
            flows=tuple(FlowsOutput.model_validate(json.loads(flows_json())).flows),
            steps_by_flow={
                "addstock": StepsOutput.model_validate(
                    json.loads(json.dumps({"steps": [{"name": "Only", "kind": "VIEW"}]}))
                )
            },
            screens=(),
            presentation=None,
        )
        assert document.flows[0].steps[0].kind is StepKind.TERMINAL
        assert document.flows[0].dead_ends() == ()


class TestPartialRunsAreSafe:
    """M4.1 items 3 and 9: partial success is representable and never approvable."""

    def build_service(self, tmp_path: Any, responses: list[str], **kwargs: Any) -> Any:
        from edith.config.schema import EdithConfig, ModelsConfig
        from edith.product.service import ProductService
        from edith.product.store import ProductStore

        config = EdithConfig(models=ModelsConfig(profiles={"default": PARAMS}))
        store = ProductStore(tmp_path / "artifacts.db")
        provider = SequenceProvider(responses, **kwargs)
        return (ProductService(config, store, provider=provider), store)

    def seed_prd(self, service: Any, store: Any) -> Any:
        from edith.product.artifacts import ArtifactKind, build_artifact
        from edith.product.validation import validate_artifact

        artifact = build_artifact(
            kind=ArtifactKind.PRD,
            project_id="p1",
            title="Stockroom PRD",
            author="product_manager",
            document=prd(),
        )
        artifact = artifact.model_copy(update={"validation": validate_artifact(artifact)})
        store.save(artifact)
        return artifact

    def test_a_partial_ux_run_is_stored_but_cannot_be_approved(self, tmp_path: Any) -> None:
        """The four good stages survive; the artifact still cannot become project truth."""
        from edith.product.store import ArtifactConflictError

        service, store = self.build_service(
            tmp_path,
            [flows_json(), screens_json(), steps_json(), presentation_json()],
            fail_at=4,
        )
        self.seed_prd(service, store)

        outcome = service.create_ux_spec("p1")
        assert outcome.ok, "the successful stages produced an artifact"
        artifact = outcome.artifact
        assert artifact is not None
        assert not artifact.validation.valid
        assert any(
            issue.code == "STAGE_INCOMPLETE" for issue in artifact.validation.issues
        )

        with pytest.raises(ArtifactConflictError):
            service.approve_artifact(artifact.artifact_id)
        store.close()

    def test_a_complete_ux_run_is_approvable(self, tmp_path: Any) -> None:
        """Complete *and* sufficient: every critical requirement delivered by a flow.

        M4.2 made coverage part of the approval gate, so a specification whose stages all
        succeeded is no longer automatically approvable. This one covers both requirements,
        which is what approval now requires.
        """
        from edith.product.artifacts import ArtifactStatus

        service, store = self.build_service(
            tmp_path,
            [
                flows_json(
                    flows=[{"name": "Add stock", "satisfies": ["REQ-001", "REQ-002"]}]
                ),
                screens_json(),
                steps_json(),
                presentation_json(),
            ],
        )
        self.seed_prd(service, store)

        outcome = service.create_ux_spec("p1")
        assert outcome.ok
        artifact = outcome.artifact
        assert artifact is not None
        assert artifact.validation.valid

        approved = service.approve_artifact(artifact.artifact_id)
        assert approved.status is ArtifactStatus.APPROVED
        store.close()

    def test_a_generated_artifact_is_never_authoritative(self, tmp_path: Any) -> None:
        """M4.1 item 2: authority is system-owned and starts as a recommendation."""
        service, store = self.build_service(
            tmp_path, [flows_json(), screens_json(), steps_json(), presentation_json()]
        )
        self.seed_prd(service, store)

        artifact = service.create_ux_spec("p1").artifact
        assert artifact is not None
        assert artifact.authority.value == "AGENT_RECOMMENDATION"
        assert artifact.status.value == "DRAFT"
        store.close()

    def test_a_failed_first_stage_reports_rather_than_storing_nothing_silently(
        self, tmp_path: Any
    ) -> None:
        service, store = self.build_service(
            tmp_path,
            [flows_json(), screens_json(), steps_json(), presentation_json()],
            fail_at=1,
        )
        self.seed_prd(service, store)

        outcome = service.create_ux_spec("p1")
        assert not outcome.ok
        assert outcome.error
        assert outcome.failure_category is not None
        store.close()


class TestNoValidationWeakening:
    """M4.1 item 8, asserted directly against the schemas."""

    def test_the_model_is_never_asked_for_system_owned_identity(self) -> None:
        """Product identity, ids, versions and status are Edith's, not the model's."""
        from edith.agents.architect import ArchitectOutput
        from edith.agents.ux_designer import UXDesignerOutput

        forbidden = {
            "product_name",
            "artifact_id",
            "version",
            "status",
            "authority",
            "project_id",
            "supersedes",
            "created_at",
            "token_id",
        }
        for schema in (
            UXDesignerOutput,
            ArchitectOutput,
            FlowsOutput,
            StepsOutput,
            ScreensOutput,
            PresentationOutput,
            ComponentsOutput,
            DataModelOutput,
            DecisionsOutput,
            ThreatModelOutput,
            PlanOutput,
        ):
            fields = set(schema.model_fields)
            leaked = fields & forbidden
            assert not leaked, f"{schema.__name__} asks the model for {leaked}"

    def test_stage_schemas_still_forbid_unknown_fields(self) -> None:
        """A smaller schema is not a looser one."""
        for schema in (FlowsOutput, ScreensOutput, ComponentsOutput, PlanOutput):
            assert schema.model_config.get("extra") == "forbid"

    def test_an_empty_required_list_is_still_rejected(self) -> None:
        for schema, payload in (
            (FlowsOutput, {"flows": []}),
            (StepsOutput, {"steps": []}),
            (ScreensOutput, {"screens": []}),
            (PlanOutput, {"tasks": []}),
        ):
            with pytest.raises(ValidationError):
                schema.model_validate(payload)

    def test_a_component_still_requires_a_responsibility(self) -> None:
        with pytest.raises(ValidationError):
            ComponentsOutput.model_validate(
                {"overview": "o", "components": [{"name": "X"}]}
            )
