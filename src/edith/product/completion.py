"""Targeted completion: close a specific coverage gap without regenerating the artifact.

M4.1 produced valid, complete UX specifications that still covered only two requirements in
three. The obvious response -- re-run the whole specification and hope -- is a generic retry,
and M4.2 item 4 rules it out. It also does not work: the second attempt has no more reason
to mention the missed requirement than the first did.

Targeted completion is a different shape:

    detect gap -> generate only what is missing -> validate -> merge -> re-check coverage

The model is asked one narrow question ("produce the user flow that delivers REQ-003"),
against one requirement, with a schema smaller than any pipeline stage. What comes back is
validated, merged into the existing artifact without renumbering anything already there, and
the coverage matrix is recomputed from evidence. If the merge did not actually close the gap,
the gap stays open -- the recheck is what stops a completion pass declaring victory.

**Coverage must be earned.** A completion whose output does not reference the requirement is
discarded rather than merged: M4.2 item 10 forbids marking a gap closed without evidence, and
the merge is the only place that rule can be enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TypeVar

from pydantic import Field

from edith.agents.ux_stages import (
    StepsOutput,
    _coerce_step_kind,
    _slug,
)
from edith.models.base import ModelProvider
from edith.observability.logging import get_logger
from edith.product.architecture import (
    ArchitectureComponent,
    ComponentKind,
    SystemArchitectureDocument,
)
from edith.product.artifacts import ArtifactKind, element_id
from edith.product.coverage import (
    DEFAULT_THRESHOLD,
    CoverageGap,
    CoverageMatrix,
    CoverageState,
    CoverageThreshold,
    analyse_coverage,
)
from edith.product.prd import PRDDocument, Requirement
from edith.product.stages import StageLedger, StageResult, run_stage
from edith.product.ux import Flow, FlowStep, StepKind, UXSpecDocument
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

logger = get_logger(__name__)

S = TypeVar("S", bound=EdithModel)

#: How many gaps one completion pass will attempt. A bound, not a target: an artifact with
#: twelve gaps has something wrong with it that twelve extra model calls will not fix.
MAX_COMPLETIONS = 4


# -- UX completion -------------------------------------------------------------------------

UX_COMPLETION_SYSTEM = """You are the UX component of a software engineering system.

An existing UX specification does not yet address one specific requirement. Produce ONLY the
user flow that delivers that requirement. Do not redesign anything else.

Rules:
- Produce one flow: a name, and the steps a user takes to satisfy the requirement.
- `kind` is exactly one of: VIEW, INPUT, ACTION, DECISION, SYSTEM, TERMINAL, ABORT.
- The step where the flow succeeds is TERMINAL. A step where it gives up is ABORT.
- Include what happens when it fails: give at least one step an `error_steps` entry.
- `next_steps` and `error_steps` name OTHER steps in this same list, by name.
- Do NOT invent flow or step numbers or IDs."""

UX_COMPLETION_USER = """REQUIREMENT TO ADDRESS:
{requirement}

FLOWS THAT ALREADY EXIST (do not repeat them):
{existing}

AVAILABLE SCREENS:
{screens}

Produce the flow that delivers this requirement."""


class TargetedStepsOutput(StepsOutput):
    """The steps of one targeted flow.

    Reuses the pipeline stage schema rather than defining a new one, so a completion pass
    cannot drift into accepting a shape the normal path would reject. The flow's name and
    the requirement it satisfies are supplied by the system, not claimed by the model.
    """


# -- Architecture completion -----------------------------------------------------------------

ARCH_COMPLETION_SYSTEM = """You are the architecture component of a software engineering system.

An existing architecture does not yet address one specific requirement. Produce ONLY the
component that satisfies it. Do not redesign anything else.

Rules:
- `kind` is exactly one of: UI, SERVICE, LIBRARY, DATASTORE, JOB, EXTERNAL, CLI.
- `depends_on` may name components that already exist, by name.
- Choose the simplest thing that satisfies the requirement.
- Do NOT invent component numbers or IDs."""

ARCH_COMPLETION_USER = """REQUIREMENT TO ADDRESS:
{requirement}

COMPONENTS THAT ALREADY EXIST:
{existing}

Produce the component that satisfies this requirement."""


class TargetedComponentOutput(EdithModel):
    """One component produced to close one coverage gap."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="SERVICE", max_length=30)
    responsibility: str = Field(min_length=1, max_length=500)
    depends_on: list[str] = Field(default_factory=list, max_length=4)
    technology: str = Field(default="", max_length=100)


# -- Results ----------------------------------------------------------------------------------


@dataclass
class CompletionAttempt:
    """One attempt to close one gap."""

    requirement_id: str
    artifact: ArtifactKind
    #: Whether the generated content was merged into the artifact.
    merged: bool = False
    #: Whether the recheck confirmed the gap actually closed.
    closed: bool = False
    #: Why it was not merged, when it was not.
    rejected_reason: str = ""
    stage: StageResult | None = None

    def summary(self) -> str:
        if self.closed:
            return f"{self.requirement_id}: closed"
        if self.merged:
            return f"{self.requirement_id}: merged but still not covered"
        return f"{self.requirement_id}: not merged - {self.rejected_reason}"


@dataclass
class CompletionResult:
    """The outcome of a completion pass over one artifact."""

    before: CoverageMatrix
    after: CoverageMatrix
    attempts: list[CompletionAttempt] = field(default_factory=list)
    ledger: StageLedger = field(default_factory=StageLedger)
    ux: UXSpecDocument | None = None
    architecture: SystemArchitectureDocument | None = None

    @property
    def closed(self) -> int:
        """Gaps the pass actually closed."""
        return sum(1 for attempt in self.attempts if attempt.closed)

    @property
    def completion_calls(self) -> int:
        """Model calls this pass spent."""
        return int(self.ledger.totals()["model_calls"])

    def improvement(self, kind: ArtifactKind) -> float:
        """Coverage gained for one artifact kind."""
        return self.after.coverage(kind) - self.before.coverage(kind)

    def summary(self) -> str:
        return "\n".join(attempt.summary() for attempt in self.attempts)


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


def _requirement_text(requirement: Requirement) -> str:
    """The one requirement a completion call is about."""
    return (
        f"{requirement.requirement_id} [{requirement.priority}] {requirement.title}: "
        f"{requirement.statement}"
    )


def merge_flow(
    document: UXSpecDocument,
    steps: StepsOutput,
    *,
    name: str,
    description: str,
    requirement_id: str,
) -> UXSpecDocument:
    """Add one flow to an existing specification without disturbing anything in it.

    Ids continue from the highest existing flow, so every element already referenced by
    another artifact keeps its identity. The requirement id is attached by the system, which
    is what makes the resulting coverage evidence trustworthy: the model produced the
    content, Edith decided what it satisfies.

    Raises:
        ValueError: The merged document fails UX schema validation.
    """
    next_index = len(document.flows) + 1
    flow_id = element_id("UX", next_index)
    screen_by_slug = {_slug(item.name): item.screen_id for item in document.screens}

    step_ids = {
        _slug(step.name): f"{flow_id}-S{position}"
        for position, step in enumerate(steps.steps, start=1)
    }

    built: list[FlowStep] = []
    for position, step in enumerate(steps.steps, start=1):
        step_id = step_ids[_slug(step.name)]
        following = tuple(
            dict.fromkeys(
                step_ids[_slug(target)]
                for target in step.next_steps
                if _slug(target) in step_ids and step_ids[_slug(target)] != step_id
            )
        )
        failing = tuple(
            dict.fromkeys(
                step_ids[_slug(target)]
                for target in step.error_steps
                if _slug(target) in step_ids and step_ids[_slug(target)] != step_id
            )
        )
        kind = _coerce_step_kind(step.kind)
        if not following and not failing:
            if position == len(steps.steps):
                kind = StepKind.TERMINAL
            elif kind not in {StepKind.TERMINAL, StepKind.ABORT}:
                following = (step_ids[_slug(steps.steps[position].name)],)

        built.append(
            FlowStep(
                step_id=step_id,
                name=step.name,
                kind=kind,
                description=step.description,
                screen_id=screen_by_slug.get(_slug(step.screen), ""),
                next_steps=following,
                error_steps=failing,
            )
        )

    flow = Flow(
        flow_id=flow_id,
        name=name,
        description=description,
        entry_step=built[0].step_id,
        steps=tuple(built),
        satisfies=(requirement_id,),
    )
    return document.model_copy(update={"flows": (*document.flows, flow)})


def merge_component(
    document: SystemArchitectureDocument,
    produced: TargetedComponentOutput,
    *,
    requirement_id: str,
) -> SystemArchitectureDocument:
    """Add one component to an existing architecture without disturbing anything in it.

    Raises:
        ValueError: The merged document fails architecture schema validation.
    """
    next_index = len(document.components) + 1
    component_id = element_id("ARCH", next_index)
    by_slug = {_slug(item.name): item.component_id for item in document.components}

    component = ArchitectureComponent(
        component_id=component_id,
        name=produced.name,
        kind=_coerce_component_kind(produced.kind),
        responsibility=produced.responsibility,
        depends_on=tuple(
            dict.fromkeys(
                by_slug[_slug(name)] for name in produced.depends_on if _slug(name) in by_slug
            )
        ),
        satisfies=(requirement_id,),
        technology=produced.technology,
    )
    return document.model_copy(
        update={"components": (*document.components, component)}
    )


def _coerce_component_kind(value: str) -> ComponentKind:
    try:
        return ComponentKind(value.strip().upper())
    except ValueError:
        return ComponentKind.SERVICE


def _selected_gaps(
    matrix: CoverageMatrix, kind: ArtifactKind, limit: int
) -> tuple[CoverageGap, ...]:
    """The gaps a completion pass will attempt, most important first.

    Blocking gaps lead. MISSING is attempted before PARTIALLY_COVERED: a requirement nothing
    mentions is a bigger hole than one something half-addresses, and the budget is finite.
    """
    candidates = [
        gap
        for gap in matrix.gaps_for(kind)
        if gap.state in {CoverageState.MISSING, CoverageState.PARTIALLY_COVERED}
    ]
    candidates.sort(
        key=lambda gap: (
            not gap.blocking,
            gap.state is not CoverageState.MISSING,
            gap.requirement_id,
        )
    )
    return tuple(candidates[:limit])


def complete_ux_coverage(
    provider: ModelProvider,
    prd: PRDDocument,
    document: UXSpecDocument,
    *,
    architecture: SystemArchitectureDocument | None = None,
    threshold: CoverageThreshold = DEFAULT_THRESHOLD,
    limit: int = MAX_COMPLETIONS,
) -> CompletionResult:
    """Close UX coverage gaps one requirement at a time.

    Nothing already in the specification is regenerated. Each pass produces one flow for one
    requirement, validates it, merges it, and recomputes coverage from evidence.
    """
    before = analyse_coverage(prd, ux=document, architecture=architecture)
    result = CompletionResult(before=before, after=before, ux=document)

    if before.satisfies(ArtifactKind.UX_SPEC, threshold):
        logger.info("completion.not_needed", artifact="UX_SPEC")
        return result

    current = document
    by_id = {item.requirement_id: item for item in prd.requirements}

    for gap in _selected_gaps(before, ArtifactKind.UX_SPEC, limit):
        requirement = by_id.get(gap.requirement_id)
        if requirement is None:  # pragma: no cover - gaps come from the PRD
            continue

        attempt = CompletionAttempt(
            requirement_id=gap.requirement_id, artifact=ArtifactKind.UX_SPEC
        )
        user = UX_COMPLETION_USER.format(
            requirement=_requirement_text(requirement),
            existing="\n".join(f"- {flow.name}" for flow in current.flows) or "(none)",
            screens="\n".join(f"- {screen.name}" for screen in current.screens) or "(none)",
        )
        stage = result.ledger.add(
            run_stage(
                f"ux.complete[{gap.requirement_id}]",
                TargetedStepsOutput,
                partial(
                    _generate, provider, UX_COMPLETION_SYSTEM, user, TargetedStepsOutput
                ),
                prompt_chars=len(UX_COMPLETION_SYSTEM) + len(user),
                elements_of=lambda output: len(output.steps),
            )
        )
        attempt.stage = stage

        if not stage.ok or not isinstance(stage.output, StepsOutput):
            attempt.rejected_reason = f"generation failed: {stage.failure}"
            result.attempts.append(attempt)
            continue

        try:
            merged = merge_flow(
                current,
                stage.output,
                name=f"{requirement.title}",
                description=requirement.statement[:400],
                requirement_id=gap.requirement_id,
            )
        except ValueError as exc:
            # The merge produced an invalid document. Discard it: a half-merged artifact is
            # worse than an incomplete one.
            attempt.rejected_reason = f"merge rejected: {exc}"
            result.attempts.append(attempt)
            continue

        attempt.merged = True
        current = merged
        result.attempts.append(attempt)

    after = analyse_coverage(prd, ux=current, architecture=architecture)
    for attempt in result.attempts:
        entry = after.entry(attempt.requirement_id)
        attempt.closed = (
            attempt.merged
            and entry is not None
            and entry.ux is CoverageState.COVERED
        )

    result.after = after
    result.ux = current
    logger.info(
        "completion.finished",
        artifact="UX_SPEC",
        attempts=len(result.attempts),
        closed=result.closed,
        before=round(before.coverage(ArtifactKind.UX_SPEC), 3),
        after=round(after.coverage(ArtifactKind.UX_SPEC), 3),
        calls=result.completion_calls,
    )
    return result


def complete_architecture_coverage(
    provider: ModelProvider,
    prd: PRDDocument,
    document: SystemArchitectureDocument,
    *,
    ux: UXSpecDocument | None = None,
    threshold: CoverageThreshold = DEFAULT_THRESHOLD,
    limit: int = MAX_COMPLETIONS,
) -> CompletionResult:
    """Close architecture coverage gaps one requirement at a time."""
    before = analyse_coverage(prd, ux=ux, architecture=document)
    result = CompletionResult(before=before, after=before, architecture=document)

    if before.satisfies(ArtifactKind.SYSTEM_ARCHITECTURE, threshold):
        logger.info("completion.not_needed", artifact="SYSTEM_ARCHITECTURE")
        return result

    current = document
    by_id = {item.requirement_id: item for item in prd.requirements}

    for gap in _selected_gaps(before, ArtifactKind.SYSTEM_ARCHITECTURE, limit):
        requirement = by_id.get(gap.requirement_id)
        if requirement is None:  # pragma: no cover
            continue

        attempt = CompletionAttempt(
            requirement_id=gap.requirement_id,
            artifact=ArtifactKind.SYSTEM_ARCHITECTURE,
        )
        user = ARCH_COMPLETION_USER.format(
            requirement=_requirement_text(requirement),
            existing="\n".join(
                f"- {item.name}: {item.responsibility}" for item in current.components
            )
            or "(none)",
        )
        stage = result.ledger.add(
            run_stage(
                f"arch.complete[{gap.requirement_id}]",
                TargetedComponentOutput,
                partial(
                    _generate, provider, ARCH_COMPLETION_SYSTEM, user, TargetedComponentOutput
                ),
                prompt_chars=len(ARCH_COMPLETION_SYSTEM) + len(user),
                elements_of=lambda output: 1,
            )
        )
        attempt.stage = stage

        if not stage.ok or not isinstance(stage.output, TargetedComponentOutput):
            attempt.rejected_reason = f"generation failed: {stage.failure}"
            result.attempts.append(attempt)
            continue

        try:
            merged = merge_component(
                current, stage.output, requirement_id=gap.requirement_id
            )
        except ValueError as exc:
            attempt.rejected_reason = f"merge rejected: {exc}"
            result.attempts.append(attempt)
            continue

        attempt.merged = True
        current = merged
        result.attempts.append(attempt)

    after = analyse_coverage(prd, ux=ux, architecture=current)
    for attempt in result.attempts:
        entry = after.entry(attempt.requirement_id)
        attempt.closed = (
            attempt.merged
            and entry is not None
            and entry.architecture is CoverageState.COVERED
        )

    result.after = after
    result.architecture = current
    logger.info(
        "completion.finished",
        artifact="SYSTEM_ARCHITECTURE",
        attempts=len(result.attempts),
        closed=result.closed,
        before=round(before.coverage(ArtifactKind.SYSTEM_ARCHITECTURE), 3),
        after=round(after.coverage(ArtifactKind.SYSTEM_ARCHITECTURE), 3),
        calls=result.completion_calls,
    )
    return result
