"""Decomposed UX generation: four small schemas instead of one large one.

M4 measured the monolithic path failing six consecutive times on the configured 3B model.
Its schema rendered to ~4,900 bytes; the Product Manager's, which succeeded first time,
renders to roughly a third of that. M4.1's hypothesis is that the difference is size per
call. See ``docs/experiments/0001-stage-decomposition.md`` for the measurement.

So the work is split into stages a small model can actually complete:

===================  ==================================================  ==========
stage                produces                                            depends on
===================  ==================================================  ==========
``flows``            flow names, descriptions, requirements served       PRD
``steps``            the steps of **one** flow, run once per flow        flows
``screens``          screens and the states each can be in               flows
``presentation``     components, accessibility, design tokens            screens
===================  ==================================================  ==========

``steps`` runs per flow deliberately. A single call producing every step of every flow is
the nested shape that failed; a call producing the steps of one named flow is a flat list,
which is the shape the working PM call uses.

**Nothing here weakens a guarantee.** Every stage output is a strict schema, ids stay
system-owned, and references are still filtered against the PRD. The system prompt never
asks the model for a product name, a version, an id, or a status: M4.1 item 2 makes those
system-owned, and a field the model is asked to echo is a failure surface with no upside.
"""

from __future__ import annotations

from pydantic import Field

from edith.product.artifacts import element_id
from edith.product.prd import PRDDocument
from edith.product.properties import ProductProperty
from edith.product.ux import (
    AccessibilityRequirement,
    Component,
    DesignToken,
    Flow,
    FlowStep,
    Screen,
    ScreenState,
    StepKind,
    UXSpecDocument,
)
from edith.schemas.common import EdithModel

MAX_FLOWS = 4
MAX_STEPS = 8
MAX_SCREENS = 8

# -- Stage 1: flows --------------------------------------------------------------------

FLOWS_SYSTEM = """You are the UX component of a software engineering system.

List the user flows this product needs. A flow is one complete journey a user takes to
achieve a goal, such as "Add an item" or "Review low stock".

Rules:
- Produce between one and four flows. Prefer fewer.
- `satisfies` lists requirement IDs (like REQ-002) from the requirements you were given.
  Use only IDs that actually appear there.
- Do NOT list steps here. Only the flows themselves.
- Do NOT invent flow numbers or IDs."""

FLOWS_USER = """REQUIREMENTS:
{requirements}

List the user flows."""


class FlowSketch(EdithModel):
    """One flow, without its steps."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    satisfies: list[str] = Field(default_factory=list, max_length=8)


class FlowsOutput(EdithModel):
    """Stage 1 output: which flows exist."""

    flows: list[FlowSketch] = Field(min_length=1, max_length=MAX_FLOWS)


# -- Stage 2: steps, one flow at a time -------------------------------------------------

STEPS_SYSTEM = """You are the UX component of a software engineering system.

Describe the steps of ONE user flow, in order.

Rules:
- Each step has a short name and a kind.
- `kind` is exactly one of: VIEW, INPUT, ACTION, DECISION, SYSTEM, TERMINAL, ABORT.
- The step where the flow succeeds is TERMINAL. A step where it gives up is ABORT.
- ALWAYS include what happens when something fails. Give at least one step an
  `error_steps` entry naming the step reached on failure. A flow with only a happy path
  strands the user the first time a request errors.
- `next_steps` and `error_steps` name OTHER steps in this same list, by name.
- Do NOT invent step numbers or IDs."""

STEPS_USER = """FLOW: {flow_name}
{flow_description}

REQUIREMENTS THIS FLOW SERVES:
{requirements}

AVAILABLE SCREENS:
{screens}

List the steps of this flow."""


class StepSketch(EdithModel):
    """One step of a flow, referencing its neighbours by name."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="VIEW", max_length=20)
    description: str = Field(default="", max_length=400)
    next_steps: list[str] = Field(default_factory=list, max_length=4)
    error_steps: list[str] = Field(default_factory=list, max_length=4)
    screen: str = Field(default="", max_length=120)


class StepsOutput(EdithModel):
    """Stage 2 output: the steps of one flow."""

    steps: list[StepSketch] = Field(min_length=1, max_length=MAX_STEPS)


# -- Stage 3: screens -------------------------------------------------------------------

SCREENS_SYSTEM = """You are the UX component of a software engineering system.

List the screens this product needs, and the states each screen can be in.

Rules:
- `states` uses only these values: DEFAULT, LOADING, EMPTY, ERROR, SUCCESS, PARTIAL,
  READ_ONLY, UNAUTHORIZED, BUSY.
- Always think about LOADING, EMPTY and ERROR. Those are the states users hit on their
  worst day and the ones a specification forgets.
- `satisfies` lists requirement IDs (like REQ-002) from the requirements you were given.
- Do NOT invent screen numbers or IDs."""

SCREENS_USER = """REQUIREMENTS:
{requirements}

USER FLOWS:
{flows}

List the screens."""


class ScreenSketch(EdithModel):
    """One screen and the states it can be in."""

    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(default="", max_length=400)
    states: list[str] = Field(default_factory=list, max_length=9)
    satisfies: list[str] = Field(default_factory=list, max_length=8)


class ScreensOutput(EdithModel):
    """Stage 3 output: the screens."""

    screens: list[ScreenSketch] = Field(min_length=1, max_length=MAX_SCREENS)


# -- Stage 4: presentation --------------------------------------------------------------

PRESENTATION_SYSTEM = """You are the UX component of a software engineering system.

Describe the reusable components, the accessibility obligations, and the design tokens.

Rules:
- A component is an interface element used on more than one screen, or a complex one used
  on a single screen. Give its interactive states and what a user can do with it.
- Accessibility entries are obligations that can be checked: keyboard reachability,
  labelling, contrast, focus order.
- Only produce design tokens if this product has a visual interface. A command-line tool
  does not need a colour palette, and inventing one is noise.
- Do NOT invent component or token numbers or IDs."""

PRESENTATION_USER = """SCREENS:
{screens}

PRODUCT CONSTRAINTS:
{constraints}

Describe the components, accessibility obligations, and design tokens."""


class ComponentSketch(EdithModel):
    """One reusable interface element."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)
    states: list[str] = Field(default_factory=list, max_length=6)
    interactions: list[str] = Field(default_factory=list, max_length=6)


class TokenSketch(EdithModel):
    """One design token."""

    name: str = Field(min_length=1, max_length=80)
    category: str = Field(default="color", max_length=40)
    value: str = Field(min_length=1, max_length=120)


class PresentationOutput(EdithModel):
    """Stage 4 output: components, accessibility, and tokens."""

    components: list[ComponentSketch] = Field(default_factory=list, max_length=8)
    accessibility: list[str] = Field(default_factory=list, max_length=8)
    design_tokens: list[TokenSketch] = Field(default_factory=list, max_length=10)
    #: Why tokens were or were not produced. CLAUDE.md's rule against unnecessary
    #: infrastructure applies to design too.
    design_system_rationale: str = Field(default="", max_length=400)


# -- Context construction ----------------------------------------------------------------
#
# M4.1 item 5: each stage receives only what it needs. Passing the whole PRD to every stage
# is the habit that makes context cost scale with the number of stages, which would turn
# decomposition into a net loss.


def requirements_context(prd: PRDDocument, *, limit: int = 12) -> str:
    """The requirement lines a UX stage needs: id, priority, and statement.

    Not the whole PRD. Personas, risks, metrics, and open questions do not help decide what
    screens exist, and every character of them competes with the code the model is writing.
    """
    lines = [
        f"{item.requirement_id} [{item.priority}] {item.title}: {item.statement}"
        for item in prd.requirements[:limit]
    ]
    return "\n".join(lines) or "(no requirements)"


def requirements_for(prd: PRDDocument, identifiers: tuple[str, ...]) -> str:
    """Only the requirements a specific flow serves.

    A step-generation call needs the one or two requirements its flow delivers, not all of
    them. When the flow names none, everything is offered rather than nothing -- a stage
    with no context produces worse output than one with slightly too much.
    """
    selected = [
        item for item in prd.requirements if item.requirement_id in set(identifiers)
    ]
    if not selected:
        return requirements_context(prd)
    return "\n".join(
        f"{item.requirement_id} {item.title}: {item.statement}" for item in selected
    )


def flows_context(flows: tuple[FlowSketch, ...]) -> str:
    """Flow names and descriptions, for a stage that needs to know what journeys exist."""
    return (
        "\n".join(f"- {item.name}: {item.description}" for item in flows)
        or "(no flows)"
    )


def screens_context(screens: tuple[ScreenSketch, ...]) -> str:
    """Screen names and purposes, for the step and presentation stages."""
    return (
        "\n".join(f"- {item.name}: {item.purpose}" for item in screens)
        or "(no screens defined yet)"
    )


# -- Deterministic assembly ---------------------------------------------------------------


def _slug(value: str) -> str:
    """Normalise a name for matching model-supplied cross-references."""
    return "".join(char for char in value.lower() if char.isalnum())


def _coerce_step_kind(value: str) -> StepKind:
    try:
        return StepKind(value.strip().upper())
    except ValueError:
        return StepKind.VIEW


def _coerce_states(values: list[str]) -> frozenset[ScreenState]:
    resolved: set[ScreenState] = set()
    for raw in values:
        try:
            resolved.add(ScreenState(raw.strip().upper()))
        except ValueError:
            continue
    return frozenset(resolved)


def _filter_requirements(values: list[str], known: frozenset[str]) -> tuple[str, ...]:
    """Keep only requirement ids the PRD defines.

    A hallucinated reference must never become a dangling id downstream; dropping it makes
    the requirement correctly report as uncovered instead.
    """
    return tuple(
        dict.fromkeys(
            item.strip().upper() for item in values if item.strip().upper() in known
        )
    )


def assemble_ux_spec(
    *,
    product_name: str,
    prd: PRDDocument | None,
    flows: tuple[FlowSketch, ...],
    steps_by_flow: dict[str, StepsOutput],
    screens: tuple[ScreenSketch, ...],
    presentation: PresentationOutput | None,
    overview: str = "",
) -> UXSpecDocument:
    """Assemble validated stage outputs into one UX specification.

    Entirely deterministic. Ids are assigned here, references are resolved by name against
    what actually exists, and anything that resolves to nothing is dropped rather than
    emitted as a dangling edge.

    A flow whose step stage failed is **omitted** rather than given a fabricated step. The
    ledger records that the stage failed, the assembled artifact is incomplete, and a
    Frontend Agent reading this document is never shown an invented journey.

    Raises:
        ValueError: The assembled document fails UX schema validation.
    """
    known = prd.requirement_ids if prd is not None else frozenset()

    component_by_slug: dict[str, str] = {}
    components: list[Component] = []
    tokens: list[DesignToken] = []
    accessibility: list[AccessibilityRequirement] = []
    rationale = ""

    if presentation is not None:
        for index, component_sketch in enumerate(presentation.components, start=1):
            component_id = element_id("CMP", index)
            component_by_slug[_slug(component_sketch.name)] = component_id
            components.append(
                Component(
                    component_id=component_id,
                    name=component_sketch.name,
                    description=component_sketch.description,
                    states=tuple(component_sketch.states),
                    interactions=tuple(component_sketch.interactions),
                )
            )
        tokens = [
            DesignToken(
                token_id=element_id("TOK", index),
                name=token_sketch.name,
                category=token_sketch.category,
                value=token_sketch.value,
            )
            for index, token_sketch in enumerate(presentation.design_tokens, start=1)
        ]
        accessibility = [
            AccessibilityRequirement(requirement=text)
            for text in presentation.accessibility
            if text.strip()
        ]
        rationale = presentation.design_system_rationale

    screen_by_slug: dict[str, str] = {}
    built_screens: list[Screen] = []
    for index, screen_sketch in enumerate(screens, start=1):
        screen_id = element_id("SCR", index)
        screen_by_slug[_slug(screen_sketch.name)] = screen_id
        # DEFAULT, LOADING and ERROR are added when omitted. Not a weakened contract: the
        # states exist whether or not anyone wrote them down, and a specification that omits
        # them produces an interface that omits them.
        states = _coerce_states(screen_sketch.states) | {
            ScreenState.DEFAULT,
            ScreenState.LOADING,
            ScreenState.ERROR,
        }
        built_screens.append(
            Screen(
                screen_id=screen_id,
                name=screen_sketch.name,
                purpose=screen_sketch.purpose,
                states=frozenset(states),
                satisfies=_filter_requirements(screen_sketch.satisfies, known),
            )
        )

    built_flows: list[Flow] = []
    for flow_index, sketch in enumerate(flows, start=1):
        produced = steps_by_flow.get(_slug(sketch.name))
        if produced is None or not produced.steps:
            # The step stage failed for this flow. Omit it rather than invent a journey.
            continue

        flow_id = element_id("UX", flow_index)
        step_ids = {
            _slug(step.name): f"{flow_id}-S{position}"
            for position, step in enumerate(produced.steps, start=1)
        }

        built_steps: list[FlowStep] = []
        for position, step in enumerate(produced.steps, start=1):
            step_id = step_ids[_slug(step.name)]
            following = tuple(
                dict.fromkeys(
                    step_ids[_slug(name)]
                    for name in step.next_steps
                    if _slug(name) in step_ids and step_ids[_slug(name)] != step_id
                )
            )
            failing = tuple(
                dict.fromkeys(
                    step_ids[_slug(name)]
                    for name in step.error_steps
                    if _slug(name) in step_ids and step_ids[_slug(name)] != step_id
                )
            )
            kind = _coerce_step_kind(step.kind)
            if not following and not failing:
                if position == len(produced.steps):
                    kind = StepKind.TERMINAL
                elif kind not in {StepKind.TERMINAL, StepKind.ABORT}:
                    # An ordered list of steps with no transitions is a description of a
                    # sequence; reading it as one beats reporting a defect nobody made.
                    following = (step_ids[_slug(produced.steps[position].name)],)

            built_steps.append(
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

        built_flows.append(
            Flow(
                flow_id=flow_id,
                name=sketch.name,
                description=sketch.description,
                entry_step=built_steps[0].step_id,
                steps=tuple(built_steps),
                satisfies=_filter_requirements(sketch.satisfies, known),
            )
        )

    properties: set[ProductProperty] = set()
    if built_screens:
        # A product with screens has an interface. Declaring it lets the contradiction
        # checker compare the UX against an architecture that claims to be headless.
        properties.add(
            ProductProperty.ACCESSIBLE
            if accessibility
            else ProductProperty.MOBILE_RESPONSIVE
        )

    return UXSpecDocument(
        product_name=product_name,
        overview=overview,
        flows=tuple(built_flows),
        screens=tuple(built_screens),
        components=tuple(components),
        design_tokens=tuple(tokens),
        accessibility=tuple(accessibility),
        properties=frozenset(properties),
        design_system_rationale=rationale,
    )
