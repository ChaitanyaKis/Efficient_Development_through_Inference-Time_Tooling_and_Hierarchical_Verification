"""Driving the decomposed architecture stages and assembling what survives.

The orchestration half of :mod:`edith.agents.architect_stages`, with the same two
properties as the UX pipeline: a failed stage never destroys a successful one, and no stage
consumes output that did not validate.

The dependency order is the one M4.1 item 7 specifies. ``components`` is the only stage
everything else needs, so it is the only stage whose failure aborts the run; the rest
contribute what they can. An architecture assembled without a threat model is incomplete and
unapprovable, but it is still an architecture someone can read and fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from pydantic import BaseModel

from edith.agents.architect_stages import (
    API_SYSTEM,
    API_USER,
    COMPONENTS_SYSTEM,
    COMPONENTS_USER,
    DATA_SYSTEM,
    DATA_USER,
    DECISIONS_SYSTEM,
    DECISIONS_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    THREATS_SYSTEM,
    THREATS_USER,
    ApiContractOutput,
    ComponentsOutput,
    DataModelOutput,
    DecisionsOutput,
    PlanOutput,
    ThreatModelOutput,
    assemble_architecture,
    assemble_plan,
    components_context,
    entities_context,
    requirements_context,
)
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.product.architecture import (
    ImplementationPlanDocument,
    SystemArchitectureDocument,
)
from edith.product.prd import PRDDocument
from edith.product.stages import StageLedger, run_stage, skipped
from edith.product.ux import UXSpecDocument
from edith.schemas.model import Message, Role

logger = get_logger(__name__)

S = TypeVar("S", bound=BaseModel)

STAGE_COMPONENTS = "arch.components"
STAGE_DATA = "arch.data"
STAGE_API = "arch.api"
STAGE_DECISIONS = "arch.decisions"
STAGE_THREATS = "arch.threats"
STAGE_PLAN = "arch.plan"

DEPENDENT_STAGES = (
    STAGE_DATA,
    STAGE_API,
    STAGE_DECISIONS,
    STAGE_THREATS,
    STAGE_PLAN,
)


@dataclass
class ArchitectPipelineResult:
    """What the decomposed architecture run produced."""

    ledger: StageLedger
    architecture: SystemArchitectureDocument | None = None
    plan: ImplementationPlanDocument | None = None
    assembly_error: str = ""

    @property
    def ok(self) -> bool:
        """Whether an architecture was assembled."""
        return self.architecture is not None

    @property
    def complete(self) -> bool:
        """Whether every attempted stage succeeded and both documents were produced."""
        return self.ok and self.plan is not None and self.ledger.complete


def _generate(provider: ModelProvider, system: str, user: str, schema: type[S]) -> S:
    """One structured call. Raises on failure; :func:`run_stage` classifies it."""
    return provider.structured_generate(
        [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=user),
        ],
        schema,
        max_repair_attempts=2,
    )


def run_architect_pipeline(
    provider: ModelProvider,
    prd: PRDDocument,
    *,
    ux: UXSpecDocument | None = None,
    constraints: str = "",
) -> ArchitectPipelineResult:
    """Generate an architecture and plan through six small stages.

    Returns the ledger plus whatever documents could be assembled. Never raises.
    """
    ledger = StageLedger()
    requirements = requirements_context(prd)
    ux_context = ux.render()[:2000] if ux is not None else "(no UX specification)"

    # -- Stage 1: components. The only stage everything else depends on. ----------------
    components_user = COMPONENTS_USER.format(
        requirements=requirements,
        ux=ux_context,
        constraints=constraints or "(none stated)",
    )
    components_result = ledger.add(
        run_stage(
            STAGE_COMPONENTS,
            ComponentsOutput,
            partial(_generate, provider, COMPONENTS_SYSTEM, components_user, ComponentsOutput),
            prompt_chars=len(COMPONENTS_SYSTEM) + len(components_user),
            elements_of=lambda output: len(output.components),
        )
    )
    if not components_result.ok or not isinstance(components_result.output, ComponentsOutput):
        for stage in DEPENDENT_STAGES:
            ledger.add(skipped(stage, f"{STAGE_COMPONENTS} did not produce components"))
        return ArchitectPipelineResult(ledger=ledger)

    components = components_result.output
    component_text = components_context(tuple(components.components))

    # -- Stage 2: data model -------------------------------------------------------------
    data_user = DATA_USER.format(requirements=requirements, components=component_text)
    data_result = ledger.add(
        run_stage(
            STAGE_DATA,
            DataModelOutput,
            partial(_generate, provider, DATA_SYSTEM, data_user, DataModelOutput),
            prompt_chars=len(DATA_SYSTEM) + len(data_user),
            elements_of=lambda output: len(output.entities),
        )
    )
    data = data_result.output if isinstance(data_result.output, DataModelOutput) else None

    # -- Stage 3: API contract ------------------------------------------------------------
    api_user = API_USER.format(requirements=requirements, components=component_text)
    api_result = ledger.add(
        run_stage(
            STAGE_API,
            ApiContractOutput,
            partial(_generate, provider, API_SYSTEM, api_user, ApiContractOutput),
            prompt_chars=len(API_SYSTEM) + len(api_user),
            elements_of=lambda output: len(output.endpoints),
        )
    )
    api = api_result.output if isinstance(api_result.output, ApiContractOutput) else None

    # -- Stage 4: decisions ---------------------------------------------------------------
    decisions_user = DECISIONS_USER.format(
        requirements=requirements,
        components=component_text,
        constraints=constraints or "(none stated)",
    )
    decisions_result = ledger.add(
        run_stage(
            STAGE_DECISIONS,
            DecisionsOutput,
            partial(_generate, provider, DECISIONS_SYSTEM, decisions_user, DecisionsOutput),
            prompt_chars=len(DECISIONS_SYSTEM) + len(decisions_user),
            elements_of=lambda output: len(output.decisions) + len(output.technologies),
        )
    )
    decisions = (
        decisions_result.output
        if isinstance(decisions_result.output, DecisionsOutput)
        else None
    )

    # -- Stage 5: threat model ------------------------------------------------------------
    threats_user = THREATS_USER.format(
        components=component_text,
        entities=entities_context(tuple(data.entities) if data else ()),
    )
    threats_result = ledger.add(
        run_stage(
            STAGE_THREATS,
            ThreatModelOutput,
            partial(_generate, provider, THREATS_SYSTEM, threats_user, ThreatModelOutput),
            prompt_chars=len(THREATS_SYSTEM) + len(threats_user),
            elements_of=lambda output: len(output.threats),
        )
    )
    threats = (
        threats_result.output
        if isinstance(threats_result.output, ThreatModelOutput)
        else None
    )

    # -- Assemble the architecture before planning against it -----------------------------
    try:
        architecture = assemble_architecture(
            product_name=prd.product_name,
            prd=prd,
            components=components,
            data=data,
            api=api,
            decisions=decisions,
            threats=threats,
        )
    except ValueError as exc:
        logger.warning("architecture.assembly_failed", error=str(exc))
        ledger.add(skipped(STAGE_PLAN, "the architecture could not be assembled"))
        return ArchitectPipelineResult(ledger=ledger, assembly_error=str(exc))

    # -- Stage 6: implementation plan ------------------------------------------------------
    plan_user = PLAN_USER.format(requirements=requirements, components=component_text)
    plan_result = ledger.add(
        run_stage(
            STAGE_PLAN,
            PlanOutput,
            partial(_generate, provider, PLAN_SYSTEM, plan_user, PlanOutput),
            prompt_chars=len(PLAN_SYSTEM) + len(plan_user),
            elements_of=lambda output: len(output.tasks),
        )
    )

    plan: ImplementationPlanDocument | None = None
    if isinstance(plan_result.output, PlanOutput):
        try:
            plan = assemble_plan(
                product_name=prd.product_name,
                goal=components.overview,
                plan=plan_result.output,
                architecture=architecture,
                prd=prd,
            )
        except ValueError as exc:
            logger.warning("plan.assembly_failed", error=str(exc))

    logger.info(
        "architecture.assembled",
        components=len(architecture.components),
        decisions=len(architecture.decisions),
        entities=len(architecture.entities),
        threats=len(architecture.threats),
        tasks=len(plan.tasks) if plan else 0,
        complete=ledger.complete,
    )
    return ArchitectPipelineResult(
        ledger=ledger, architecture=architecture, plan=plan
    )
