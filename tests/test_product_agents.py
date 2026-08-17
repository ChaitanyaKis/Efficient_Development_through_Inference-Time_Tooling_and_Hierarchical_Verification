"""M4: the product agents, their trust boundaries, and the pipeline.

The agents themselves are thin -- they format a prompt and hand the result to a translation
function. The translation is where the interesting behaviour lives, because that is the trust
boundary: everything a model produced is re-expressed through a strict schema, ids are
assigned by the system rather than the model, and references that resolve to nothing are
dropped rather than propagated.

A scripted provider stands in for the model throughout, so these tests assert on what the
system does with model output rather than on what a model happens to produce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edith.agents.architect import (
    ArchitectAgent,
    ArchitectOutput,
    draft_to_architecture,
    draft_to_plan,
)
from edith.agents.product_manager import (
    ProductManagerAgent,
    ProductManagerOutput,
    draft_to_prd,
)
from edith.agents.registry import build_default_registry
from edith.agents.ux_designer import (
    UXDesignerAgent,
    UXDesignerOutput,
    draft_to_ux_spec,
)
from edith.config.schema import EdithConfig, ModelParams, ModelsConfig
from edith.product.artifacts import ArtifactKind, ArtifactStatus
from edith.product.prd import Priority, RequirementKind
from edith.product.properties import ProductProperty as P
from edith.product.service import ProductService, run_pipeline
from edith.product.store import ProductStore, open_artifacts
from edith.product.ux import ScreenState, StepKind
from edith.product.validation import find_cycle

from .fakes import FakeProvider


@pytest.fixture
def config() -> EdithConfig:
    return EdithConfig(
        models=ModelsConfig(profiles={"default": ModelParams(model_name="test-model:q4")})
    )


@pytest.fixture
def store(tmp_path: Path) -> ProductStore:
    with open_artifacts(tmp_path / "product") as opened:
        yield opened


# -- Model output fixtures ------------------------------------------------------------


def pm_output(**overrides: Any) -> dict[str, Any]:
    """What the Product Manager model returns."""
    payload: dict[str, Any] = {
        "product_name": "Stockroom",
        "problem": "Shop staff cannot see which items are running low.",
        "target_users": "Shop staff",
        "goals": ["Show low stock at a glance"],
        "non_goals": ["Purchasing"],
        "requirements": [
            {
                "title": "Record stock levels",
                "statement": "The system records the quantity of every item.",
                "kind": "FUNCTIONAL",
                "priority": "MUST",
                "acceptance": "Adding an item records its quantity.",
                "properties": ["PERSISTENT_STORAGE"],
            },
            {
                "title": "Low stock list",
                "statement": "Items at or below their threshold are listed as low.",
                "kind": "FUNCTIONAL",
                "priority": "MUST",
                "acceptance": "An item at its threshold appears in the low list.",
                "properties": [],
            },
        ],
        "assumptions": ["One shop"],
        "risks": ["Staff may not update quantities"],
        "open_questions": ["Should thresholds be per item or global?"],
    }
    payload.update(overrides)
    return payload


def ux_output(**overrides: Any) -> dict[str, Any]:
    """What the UX model returns."""
    payload: dict[str, Any] = {
        "overview": "Two screens: a stock list and an item form.",
        "flows": [{"name": "Add stock", "satisfies": ["REQ-001"]}],
        "steps": [
            {
                "flow": "Add stock",
                "name": "Open form",
                "kind": "VIEW",
                "screen": "Item form",
            },
            {
                "flow": "Add stock",
                "name": "Submit",
                "kind": "ACTION",
                "next_steps": ["Saved"],
                "error_steps": ["Failed"],
            },
            {"flow": "Add stock", "name": "Saved", "kind": "TERMINAL"},
            {"flow": "Add stock", "name": "Failed", "kind": "ABORT"},
        ],
        "screens": [
            {"name": "Stock list", "states": ["DEFAULT", "EMPTY"], "satisfies": ["REQ-002"]},
            {"name": "Item form", "states": ["DEFAULT"], "satisfies": ["REQ-001"]},
        ],
        "components": [{"name": "Item row", "states": ["default", "low"]}],
        "design_tokens": [{"name": "danger", "category": "color", "value": "#c0392b"}],
        "accessibility": ["Every control is reachable by keyboard"],
        "properties": ["MOBILE_RESPONSIVE"],
    }
    payload.update(overrides)
    return payload


def architect_output(**overrides: Any) -> dict[str, Any]:
    """What the Architect model returns."""
    payload: dict[str, Any] = {
        "overview": "A single local application with a SQLite store.",
        "components": [
            {
                "name": "Storage",
                "kind": "DATASTORE",
                "responsibility": "Persist items and quantities.",
                "satisfies": ["REQ-001"],
            },
            {
                "name": "Web UI",
                "kind": "UI",
                "responsibility": "Show stock and accept edits.",
                "depends_on": ["Storage"],
                "satisfies": ["REQ-002"],
            },
        ],
        "decisions": [
            {
                "title": "Use SQLite",
                "context": "One shop, one machine, no operations team.",
                "decision": "Store data in a local SQLite file.",
                "alternatives": ["PostgreSQL", "A hosted database"],
                "rationale": "No concurrent writers and no server to run.",
                "consequences": ["No multi-machine access without a rewrite"],
                "affects_requirements": ["REQ-001"],
                "confidence": "HIGH",
            }
        ],
        "technologies": [
            {
                "name": "SQLite",
                "role": "database",
                "rationale": "Zero operations, sufficient for one writer.",
                "alternatives_rejected": ["PostgreSQL"],
                "constraints_considered": ["No operations team"],
            }
        ],
        "entities": [
            {"name": "Item", "fields": {"name": "str", "quantity": "int"},
             "satisfies": ["REQ-001"]}
        ],
        "endpoints": [],
        "threats": [
            {
                "asset": "Stock data",
                "description": "A stolen laptop exposes the database file.",
                "mitigation": "Full-disk encryption is assumed.",
            }
        ],
        "tasks": [
            {
                "title": "Create the item store",
                "description": "Create the SQLite schema and the item repository.",
                "implements": ["REQ-001"],
                "components": ["Storage"],
                "acceptance": "Items round-trip through the store.",
                "complexity": "SMALL",
            },
            {
                "title": "Build the stock list",
                "description": "Render the stock list with low items highlighted.",
                "depends_on": ["Create the item store"],
                "implements": ["REQ-002"],
                "components": ["Web UI"],
                "complexity": "MEDIUM",
            },
        ],
        "properties": ["LOCAL_ONLY", "PERSISTENT_STORAGE"],
        "constraints_considered": ["Single machine", "No operations team"],
        "deliberate_omissions": ["No message queue: nothing is asynchronous"],
    }
    payload.update(overrides)
    return payload


def build_service(
    config: EdithConfig, store: ProductStore, responses: list[str]
) -> ProductService:
    """A service whose model returns scripted responses in order."""
    provider = FakeProvider(config.models.profile(), responses)
    return ProductService(config, store, provider=provider)


# -- Product Manager -------------------------------------------------------------------


class TestProductManagerTranslation:
    """The trust boundary: model output becomes a strictly-validated PRD."""

    def test_requirement_ids_are_assigned_by_the_system(self) -> None:
        """The model never numbers requirements. Ids are too load-bearing to delegate."""
        draft = ProductManagerOutput.model_validate(pm_output())
        prd = draft_to_prd(draft)
        assert [item.requirement_id for item in prd.requirements] == ["REQ-001", "REQ-002"]

    def test_ids_stay_dense_and_unique_regardless_of_model_output(self) -> None:
        draft = ProductManagerOutput.model_validate(
            pm_output(
                requirements=[
                    {"title": f"R{index}", "statement": f"s{index}"} for index in range(1, 7)
                ]
            )
        )
        prd = draft_to_prd(draft)
        identifiers = [item.requirement_id for item in prd.requirements]
        assert identifiers == [f"REQ-{index:03d}" for index in range(1, 7)]
        assert len(set(identifiers)) == len(identifiers)

    def test_every_requirement_gets_an_acceptance_criterion(self) -> None:
        """A requirement nobody can check is one nobody will notice missing."""
        draft = ProductManagerOutput.model_validate(
            pm_output(
                requirements=[{"title": "A", "statement": "a", "acceptance": ""}]
            )
        )
        prd = draft_to_prd(draft)
        assert prd.unverified_requirements() == ()
        assert prd.acceptance_criteria[0].verifies == ("REQ-001",)

    def test_an_unknown_priority_defaults_upward(self) -> None:
        """Defaulting down would let a real requirement fall off the plan."""
        draft = ProductManagerOutput.model_validate(
            pm_output(requirements=[{"title": "A", "statement": "a", "priority": "urgent"}])
        )
        assert draft_to_prd(draft).requirements[0].priority is Priority.MUST

    def test_a_known_priority_is_preserved(self) -> None:
        draft = ProductManagerOutput.model_validate(
            pm_output(requirements=[{"title": "A", "statement": "a", "priority": "could"}])
        )
        assert draft_to_prd(draft).requirements[0].priority is Priority.COULD

    def test_an_unknown_kind_falls_back_to_functional(self) -> None:
        draft = ProductManagerOutput.model_validate(
            pm_output(requirements=[{"title": "A", "statement": "a", "kind": "weird"}])
        )
        assert draft_to_prd(draft).requirements[0].kind is RequirementKind.FUNCTIONAL

    def test_an_invented_property_is_dropped_not_fatal(self) -> None:
        """A model inventing FAST must not take down an otherwise sound PRD."""
        draft = ProductManagerOutput.model_validate(
            pm_output(
                requirements=[
                    {
                        "title": "A",
                        "statement": "a",
                        "properties": ["FAST", "OFFLINE_CAPABLE"],
                    }
                ]
            )
        )
        prd = draft_to_prd(draft)
        assert prd.requirements[0].properties == frozenset({P.OFFLINE_CAPABLE})

    def test_risks_and_questions_are_numbered(self) -> None:
        prd = draft_to_prd(ProductManagerOutput.model_validate(pm_output()))
        assert prd.risks[0].risk_id == "RISK-001"
        assert prd.open_questions[0].question_id == "Q-001"

    def test_the_agent_is_read_only_and_has_no_shell(self) -> None:
        """M4.13: a PM that could run commands is a PM that could ship."""
        permissions = ProductManagerAgent.identity.permissions
        assert permissions.read_only
        assert "shell.run" not in permissions.allowed_tools
        assert not any(tool.startswith("git.") for tool in permissions.allowed_tools)


class TestProductManagerAgent:
    def test_it_produces_a_prd_from_scripted_output(self, config: EdithConfig) -> None:
        from edith.schemas.agent import AgentRequest

        provider = FakeProvider(config.models.profile(), [json.dumps(pm_output())])
        agent = ProductManagerAgent(provider=provider)
        response = agent.execute(AgentRequest(payload={"idea": "track shop stock"}))

        assert response.ok, response.error
        draft = ProductManagerOutput.model_validate(response.output)
        assert draft.product_name == "Stockroom"

    def test_malformed_model_output_is_a_classified_failure(
        self, config: EdithConfig
    ) -> None:
        from edith.schemas.agent import AgentRequest

        provider = FakeProvider(config.models.profile(), ['{"nonsense": true}'])
        agent = ProductManagerAgent(provider=provider)
        response = agent.execute(AgentRequest(payload={"idea": "x"}))

        assert not response.ok
        assert response.failure_category is not None


# -- UX designer -----------------------------------------------------------------------


class TestUXTranslation:
    def build(self, prd_document: Any = None, **overrides: Any) -> Any:
        draft = UXDesignerOutput.model_validate(ux_output(**overrides))
        return draft_to_ux_spec(draft, prd=prd_document)

    def test_flow_steps_are_wired_into_a_graph(self) -> None:
        spec = self.build()
        flow = spec.flows[0]
        assert flow.entry_step == flow.steps[0].step_id
        submit = flow.steps[1]
        assert submit.next_steps and submit.error_steps
        assert flow.dead_ends() == ()

    def test_a_step_naming_a_nonexistent_successor_never_becomes_an_edge(self) -> None:
        """A dangling transition would take the whole document down."""
        spec = self.build(
            flows=[{"name": "F"}],
            steps=[
                {"flow": "F", "name": "A", "next_steps": ["Ghost"]},
                {"flow": "F", "name": "B", "kind": "TERMINAL"},
            ],
        )
        flow = spec.flows[0]
        for step in flow.steps:
            for target in (*step.next_steps, *step.error_steps):
                assert target in flow.step_ids
        assert not any("ghost" in target.lower() for target in flow.steps[0].next_steps)

    def test_a_sequence_the_model_never_wired_is_read_as_a_sequence(self) -> None:
        """An ordered list of steps with no transitions is a description of a sequence."""
        spec = self.build(
            flows=[{"name": "F"}],
            steps=[
                {"flow": "F", "name": "A"},
                {"flow": "F", "name": "B"},
                {"flow": "F", "name": "C"},
            ],
        )
        flow = spec.flows[0]
        assert flow.steps[0].next_steps == (flow.steps[1].step_id,)
        assert flow.steps[1].next_steps == (flow.steps[2].step_id,)
        assert flow.dead_ends() == ()
        assert flow.unreachable_steps() == ()

    def test_a_final_step_with_no_successor_becomes_terminal(self) -> None:
        """Otherwise every flow ends in a dead end the model did not actually create."""
        spec = self.build(
            flows=[{"name": "F"}],
            steps=[{"flow": "F", "name": "A"}, {"flow": "F", "name": "B"}],
        )
        assert spec.flows[0].steps[-1].kind is StepKind.TERMINAL
        assert spec.flows[0].dead_ends() == ()

    def test_required_screen_states_are_added_when_omitted(self) -> None:
        """The states users hit on their worst day are the ones a spec forgets."""
        spec = self.build()
        for screen in spec.screens:
            assert ScreenState.DEFAULT in screen.states
            assert ScreenState.LOADING in screen.states
            assert ScreenState.ERROR in screen.states
            assert screen.missing_states() == ()

    def test_a_requirement_the_prd_never_defined_is_dropped(self) -> None:
        """A hallucinated reference must not become a dangling id downstream."""
        prd = draft_to_prd(ProductManagerOutput.model_validate(pm_output()))
        spec = self.build(
            prd,
            flows=[{"name": "F", "satisfies": ["REQ-001", "REQ-404"]}],
            steps=[{"flow": "F", "name": "A", "kind": "TERMINAL"}],
        )
        assert spec.flows[0].satisfies == ("REQ-001",)

    def test_screens_and_components_are_numbered_by_the_system(self) -> None:
        spec = self.build()
        assert [item.screen_id for item in spec.screens] == ["SCR-001", "SCR-002"]
        assert [item.component_id for item in spec.components] == ["CMP-001"]

    def test_component_references_are_resolved_by_name(self) -> None:
        spec = self.build(
            screens=[{"name": "Stock list", "components": ["Item row", "Ghost"]}],
        )
        assert spec.screens[0].components == ("CMP-001",)

    def test_the_agent_cannot_write_source_code(self) -> None:
        """M4.13: UX writes design artifacts, not the frontend it specifies."""
        permissions = UXDesignerAgent.identity.permissions
        assert "shell.run" not in permissions.allowed_tools
        assert all(
            pattern.startswith(("design/", "docs/")) for pattern in permissions.allowed_write_paths
        )


# -- Architect --------------------------------------------------------------------------


class TestArchitectTranslation:
    def prd(self) -> Any:
        return draft_to_prd(ProductManagerOutput.model_validate(pm_output()))

    def test_components_and_decisions_are_numbered_by_the_system(self) -> None:
        draft = ArchitectOutput.model_validate(architect_output())
        architecture = draft_to_architecture(draft, prd=self.prd())
        assert [item.component_id for item in architecture.components] == [
            "ARCH-001",
            "ARCH-002",
        ]
        assert architecture.decisions[0].decision_id == "ADR-001"

    def test_component_dependencies_are_resolved_by_name(self) -> None:
        architecture = draft_to_architecture(
            ArchitectOutput.model_validate(architect_output()), prd=self.prd()
        )
        web = next(item for item in architecture.components if item.name == "Web UI")
        assert web.depends_on == ("ARCH-001",)

    def test_a_dependency_on_an_undefined_component_is_dropped(self) -> None:
        draft = ArchitectOutput.model_validate(
            architect_output(
                components=[
                    {
                        "name": "Web UI",
                        "responsibility": "r",
                        "depends_on": ["Ghost service"],
                    }
                ]
            )
        )
        architecture = draft_to_architecture(draft, prd=self.prd())
        assert architecture.components[0].depends_on == ()

    def test_a_decision_missing_alternatives_is_completed_not_discarded(self) -> None:
        """Dropping it would hide that a decision was made."""
        draft = ArchitectOutput.model_validate(
            architect_output(
                decisions=[
                    {
                        "title": "Use SQLite",
                        "context": "c",
                        "decision": "d",
                        "alternatives": [],
                        "rationale": "r",
                        "consequences": [],
                    }
                ]
            )
        )
        architecture = draft_to_architecture(draft, prd=self.prd())
        decision = architecture.decisions[0]
        assert decision.alternatives
        assert "none recorded" in decision.alternatives[0]
        assert "none recorded" in decision.consequences[0]

    def test_data_flow_is_derived_from_the_component_graph(self) -> None:
        """Asking for it separately produces two descriptions that disagree."""
        architecture = draft_to_architecture(
            ArchitectOutput.model_validate(architect_output()), prd=self.prd()
        )
        assert architecture.data_flows
        assert architecture.data_flows[0].source == "ARCH-002"
        assert architecture.data_flows[0].target == "ARCH-001"

    def test_a_requirement_the_prd_never_defined_is_dropped(self) -> None:
        draft = ArchitectOutput.model_validate(
            architect_output(
                components=[
                    {"name": "Storage", "responsibility": "r", "satisfies": ["REQ-404"]}
                ]
            )
        )
        architecture = draft_to_architecture(draft, prd=self.prd())
        assert architecture.components[0].satisfies == ()

    def test_plan_task_dependencies_are_resolved_by_name(self) -> None:
        draft = ArchitectOutput.model_validate(architect_output())
        architecture = draft_to_architecture(draft, prd=self.prd())
        plan = draft_to_plan(draft, architecture, prd=self.prd())
        assert [task.task_id for task in plan.tasks] == ["TASK-001", "TASK-002"]
        assert plan.tasks[1].depends_on == ("TASK-001",)

    def test_a_cycle_is_preserved_so_validation_reports_it(self) -> None:
        """Silently repairing it would hide that the design cannot be ordered."""
        draft = ArchitectOutput.model_validate(
            architect_output(
                tasks=[
                    {"title": "A", "description": "d", "depends_on": ["B"]},
                    {"title": "B", "description": "d", "depends_on": ["A"]},
                ]
            )
        )
        architecture = draft_to_architecture(draft, prd=self.prd())
        plan = draft_to_plan(draft, architecture, prd=self.prd())
        assert find_cycle(plan)

    def test_the_architect_cannot_touch_source_or_run_commands(self) -> None:
        """M4.13: it can record a decision; it cannot act on one."""
        permissions = ArchitectAgent.identity.permissions
        assert "shell.run" not in permissions.allowed_tools
        assert set(permissions.allowed_write_paths) == {"architecture/**", "docs/adr/**"}


# -- Pipeline ---------------------------------------------------------------------------


class TestPipeline:
    """The monolithic pipeline, scripted end to end.

    Explicitly ``decomposed=False``: these tests script one response per stage, which is the
    monolithic shape. The decomposed pipeline is covered in ``test_product_stages.py``, and
    M4.1 experiment 0001 measured which of the two a 3B model can actually complete.
    """

    def responses(self) -> list[str]:
        return [
            json.dumps(pm_output()),
            json.dumps(ux_output()),
            json.dumps(architect_output()),
        ]

    def test_the_whole_pipeline_produces_four_artifacts(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, self.responses())
        result = run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )

        assert result.ok, result.summary()
        kinds = {artifact.kind for artifact in store.current("p1")}
        assert kinds == {
            ArtifactKind.PRD,
            ArtifactKind.UX_SPEC,
            ArtifactKind.SYSTEM_ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        }

    def test_downstream_artifacts_record_which_version_they_read(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        """"Derived from the PRD" is ambiguous once the PRD has been revised."""
        service = build_service(config, store, self.responses())
        run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )

        ux = store.latest("p1", ArtifactKind.UX_SPEC)
        prd = store.latest("p1", ArtifactKind.PRD)
        assert ux is not None and prd is not None
        assert ux.depends_on[0].artifact_id == prd.artifact_id
        assert ux.depends_on[0].version == prd.version

    def test_every_artifact_validates(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, self.responses())
        run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )
        for artifact in store.current("p1"):
            assert artifact.validation.valid, (
                f"{artifact.kind} did not validate: {artifact.validation.summary()}"
            )

    def test_a_stage_cannot_run_before_its_dependency_exists(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, [json.dumps(ux_output())])
        outcome = service.create_ux_spec("p1")
        assert not outcome.ok
        assert "no PRD" in outcome.error

    def test_a_failed_stage_stops_the_pipeline(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, ['{"broken": true}'])
        result = run_pipeline(service, "p1", "idea", decomposed=False)
        assert not result.ok
        assert len(result.stages) == 1

    def test_context_cost_is_instrumented(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        """M4.11: measure product context rather than guessing at a budget."""
        service = build_service(config, store, self.responses())
        result = run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )
        metrics = result.total_metrics()

        assert metrics["input_chars"] > 0
        assert metrics["artifact_chars"] > 0
        assert metrics["model_calls"] == 3
        assert len(metrics["stages"]) == 3
        for stage in metrics["stages"]:
            assert stage["elements"] > 0

    def test_memory_is_not_injected_automatically(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        """M4.10: the M3.2 governor stays the only autonomous injection path."""
        provider = FakeProvider(config.models.profile(), self.responses())
        service = ProductService(config, store, provider=provider)
        run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )

        for call in provider.calls:
            prompt = "\n".join(content for _, content in call["messages"])
            assert "PRIOR KNOWLEDGE" not in prompt

    def test_status_is_serialisable_for_a_ui(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        """M4.15: a UI drives the pipeline through plain data, never live handles."""
        service = build_service(config, store, self.responses())
        run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )

        status = service.status("p1")
        json.dumps(status)  # must not raise
        assert len(status["artifacts"]) == 4
        assert status["verdict"] in {"PASS", "FAIL"}

    def test_the_agent_roster_exposes_permissions(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, [])
        roster = service.available_agents()
        assert {item["name"] for item in roster} == {
            "product_manager",
            "ux_designer",
            "architect",
        }
        for entry in roster:
            json.dumps(entry)
            assert "shell.run" not in entry["tools"]

    def test_approval_requires_validation(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, self.responses())
        result = run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )
        prd = result.stages[0].artifact
        assert prd is not None

        approved = service.approve_artifact(prd.artifact_id)
        assert approved.status is ArtifactStatus.APPROVED

    def test_projects_are_isolated_through_the_service(
        self, config: EdithConfig, store: ProductStore
    ) -> None:
        service = build_service(config, store, self.responses())
        run_pipeline(
            service, "p1", "track shop stock", stop_on_block=False, decomposed=False
        )
        assert service.artifacts("p2") == ()
        assert service.status("p2")["artifacts"] == []


class TestRegistry:
    def test_the_product_agents_are_registered_and_inspectable(
        self, config: EdithConfig
    ) -> None:
        """An operator must be able to see what each agent may do."""
        registry = build_default_registry(config)
        names = set(registry.names())
        assert {"product_manager", "ux_designer", "architect"} <= names

    def test_no_product_agent_may_run_a_shell(self, config: EdithConfig) -> None:
        """M4.13, asserted over the registry rather than per agent."""
        registry = build_default_registry(config)
        product_agents = {"product_manager", "ux_designer", "architect"}
        for identity in registry.identities():
            if identity.name not in product_agents:
                continue
            assert "shell.run" not in identity.permissions.allowed_tools
            assert not identity.permissions.network_access
