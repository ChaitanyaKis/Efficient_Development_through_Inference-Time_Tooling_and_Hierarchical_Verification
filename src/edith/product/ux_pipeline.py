"""Driving the decomposed UX stages and assembling what survives.

The orchestration half of :mod:`edith.agents.ux_stages`. It runs each stage, records what
happened, and hands the validated outputs to the deterministic assembler.

Two behaviours matter more than the sequencing:

**A failed stage does not destroy a successful one.** If ``screens`` validates and the steps
of one flow do not, the run keeps the screens, omits that flow, and reports the failure. The
resulting artifact is marked incomplete and cannot be approved â€” but four stages' worth of
work is not thrown away because a fifth failed.

**Downstream stages never consume unverified upstream output.** ``steps`` runs only if
``flows`` validated; ``presentation`` runs only if ``screens`` did. A stage whose dependency
failed is recorded as ``SKIPPED``, which is a different fact from having been tried and
failed, and the ledger keeps them distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from pydantic import BaseModel

from edith.agents.ux_stages import (
    FLOWS_SYSTEM,
    FLOWS_USER,
    PRESENTATION_SYSTEM,
    PRESENTATION_USER,
    SCREENS_SYSTEM,
    SCREENS_USER,
    STEPS_SYSTEM,
    STEPS_USER,
    FlowSketch,
    FlowsOutput,
    PresentationOutput,
    ScreenSketch,
    ScreensOutput,
    StepsOutput,
    assemble_ux_spec,
    flows_context,
    requirements_context,
    requirements_for,
    screens_context,
)
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.product.prd import PRDDocument
from edith.product.stages import (
    StageLedger,
    StageResult,
    run_stage,
    skipped,
)
from edith.product.ux import UXSpecDocument
from edith.schemas.model import Message, Role

logger = get_logger(__name__)

S = TypeVar("S", bound=BaseModel)

STAGE_FLOWS = "ux.flows"
STAGE_SCREENS = "ux.screens"
STAGE_PRESENTATION = "ux.presentation"


def steps_stage_name(flow_name: str) -> str:
    """The ledger name for the step stage of one flow."""
    return f"ux.steps[{flow_name[:40]}]"


@dataclass
class UXPipelineResult:
    """What the decomposed UX run produced."""

    ledger: StageLedger
    document: UXSpecDocument | None = None
    #: Set when assembly itself failed, as opposed to a stage failing.
    assembly_error: str = ""

    @property
    def ok(self) -> bool:
        """Whether a specification was assembled at all."""
        return self.document is not None

    @property
    def complete(self) -> bool:
        """Whether every attempted stage succeeded and a document was produced.

        An incomplete run may still have produced a usable document; ``complete`` is what
        gates approval, and :attr:`ok` is what says whether there is anything to look at.
        """
        return self.ok and self.ledger.complete


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


def run_ux_pipeline(
    provider: ModelProvider,
    prd: PRDDocument,
    *,
    constraints: str = "",
    overview: str = "",
) -> UXPipelineResult:
    """Generate a UX specification through four small stages instead of one large call.

    Args:
        provider: The model provider.
        prd: The requirements this specification serves. Only the parts each stage needs
            are passed on -- see the ``*_context`` helpers.
        constraints: Product constraints, for the presentation stage.
        overview: Optional human-written overview to carry into the document.

    Returns:
        The ledger of every stage plus the assembled document, when one could be built.
        Never raises: a caller must be able to render a partial run.
    """
    ledger = StageLedger()
    requirements = requirements_context(prd)

    # -- Stage 1: which flows exist ----------------------------------------------------
    flows_user = FLOWS_USER.format(requirements=requirements)
    flows_result = ledger.add(
        run_stage(
            STAGE_FLOWS,
            FlowsOutput,
            lambda: _generate(provider, FLOWS_SYSTEM, flows_user, FlowsOutput),
            prompt_chars=len(FLOWS_SYSTEM) + len(flows_user),
            elements_of=lambda output: len(output.flows),
        )
    )

    if not flows_result.ok:
        # Nothing downstream is meaningful without flows. Recorded as skipped rather than
        # failed: these stages were never given a chance.
        for stage in (STAGE_SCREENS, STAGE_PRESENTATION):
            ledger.add(skipped(stage, f"{STAGE_FLOWS} did not produce flows"))
        return UXPipelineResult(ledger=ledger)

    flows = tuple(_flows_of(flows_result))

    # -- Stage 2: screens --------------------------------------------------------------
    screens_user = SCREENS_USER.format(
        requirements=requirements, flows=flows_context(flows)
    )
    screens_result = ledger.add(
        run_stage(
            STAGE_SCREENS,
            ScreensOutput,
            lambda: _generate(provider, SCREENS_SYSTEM, screens_user, ScreensOutput),
            prompt_chars=len(SCREENS_SYSTEM) + len(screens_user),
            elements_of=lambda output: len(output.screens),
        )
    )
    screens = tuple(_screens_of(screens_result))

    # -- Stage 3: the steps of each flow, one call per flow -----------------------------
    #
    # Per flow deliberately. One call producing every step of every flow is the nested shape
    # that failed six times in M4; one call producing the steps of a single named flow is a
    # flat list, which is the shape that works.
    steps_by_flow: dict[str, StepsOutput] = {}
    for sketch in flows:
        stage_name = steps_stage_name(sketch.name)
        steps_user = STEPS_USER.format(
            flow_name=sketch.name,
            flow_description=sketch.description or "",
            requirements=requirements_for(prd, tuple(sketch.satisfies)),
            screens=screens_context(screens),
        )
        result = ledger.add(
            run_stage(
                stage_name,
                StepsOutput,
                partial(_generate, provider, STEPS_SYSTEM, steps_user, StepsOutput),
                prompt_chars=len(STEPS_SYSTEM) + len(steps_user),
                elements_of=lambda output: len(output.steps),
            )
        )
        if result.ok and isinstance(result.output, StepsOutput):
            steps_by_flow[_slug(sketch.name)] = result.output

    # -- Stage 4: presentation ----------------------------------------------------------
    if not screens_result.ok:
        ledger.add(
            skipped(STAGE_PRESENTATION, f"{STAGE_SCREENS} did not produce screens")
        )
        presentation = None
    else:
        presentation_user = PRESENTATION_USER.format(
            screens=screens_context(screens),
            constraints=constraints or "(none stated)",
        )
        presentation_result = ledger.add(
            run_stage(
                STAGE_PRESENTATION,
                PresentationOutput,
                lambda: _generate(
                    provider, PRESENTATION_SYSTEM, presentation_user, PresentationOutput
                ),
                prompt_chars=len(PRESENTATION_SYSTEM) + len(presentation_user),
                elements_of=lambda output: len(output.components)
                + len(output.design_tokens),
            )
        )
        presentation = (
            presentation_result.output
            if presentation_result.ok
            and isinstance(presentation_result.output, PresentationOutput)
            else None
        )

    # -- Deterministic assembly ---------------------------------------------------------
    if not steps_by_flow and not screens:
        return UXPipelineResult(
            ledger=ledger,
            assembly_error="no stage produced content that could be assembled",
        )

    try:
        document = assemble_ux_spec(
            product_name=prd.product_name,
            prd=prd,
            flows=flows,
            steps_by_flow=steps_by_flow,
            screens=screens,
            presentation=presentation,
            overview=overview,
        )
    except ValueError as exc:
        logger.warning("ux.assembly_failed", error=str(exc))
        return UXPipelineResult(ledger=ledger, assembly_error=str(exc))

    logger.info(
        "ux.assembled",
        flows=len(document.flows),
        screens=len(document.screens),
        components=len(document.components),
        complete=ledger.complete,
    )
    return UXPipelineResult(ledger=ledger, document=document)


def _flows_of(result: StageResult) -> list[FlowSketch]:
    """The flow sketches a stage produced."""
    return list(result.output.flows) if isinstance(result.output, FlowsOutput) else []


def _screens_of(result: StageResult) -> list[ScreenSketch]:
    """The screen sketches a stage produced."""
    return list(result.output.screens) if isinstance(result.output, ScreensOutput) else []


def _slug(value: str) -> str:
    """Match the assembler's normalisation, so flow lookups agree."""
    return "".join(char for char in value.lower() if char.isalnum())
