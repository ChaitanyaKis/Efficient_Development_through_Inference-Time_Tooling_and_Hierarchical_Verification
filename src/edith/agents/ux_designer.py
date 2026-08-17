"""The UX/UI Agent.

Consumes a PRD and produces a structured UX specification: flows as graphs, screens with
their states, components, and design tokens. It writes no frontend code — the output is a
specification a future Frontend Agent consumes, and an agent that produced both the spec and
the code would make the spec a formality.

Same discipline as the Product Manager: the model emits flat, unnumbered records, and
:func:`draft_to_ux_spec` assigns ``UX-001``, ``SCR-001``, ``CMP-001`` and wires the graph.
Flow steps in particular are a place a small model reliably produces broken references — it
will name a "next" step that does not exist — so the translation resolves step names to ids
and drops edges that point nowhere rather than emitting a flow that fails schema validation.

The screen-state requirement is enforced here rather than left to the model. Loading, error,
and default states are added when the model omits them, because a specification missing them
produces an interface missing them, and the states users hit on their worst day are exactly
the ones that get skipped.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

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
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

from .base import Agent

MAX_FLOWS = 4
MAX_STEPS = 8
MAX_SCREENS = 8

SYSTEM_PROMPT = """You are the UX component of a software engineering system.

You turn requirements into a structured interface specification. You do NOT write code,
choose frameworks, or design a database -- other components do that.

Rules:
- List the flows first (name, description, which requirements they satisfy).
- Then list EVERY step of EVERY flow in one flat `steps` list, in order. Each step names
  which flow it belongs to in its `flow` field, and names the steps that follow it.
- Always include what happens when something FAILS. A flow with only a happy path is an
  interface that strands users the first time a request errors.
- `kind` for a step is one of: VIEW, INPUT, ACTION, DECISION, SYSTEM, TERMINAL, ABORT.
  The last step of a successful path is TERMINAL. A failure ending is ABORT.
- A screen lists the states it can be in: DEFAULT, LOADING, EMPTY, ERROR, SUCCESS,
  PARTIAL, READ_ONLY, UNAUTHORIZED, BUSY.
- `satisfies` lists the requirement IDs (like REQ-002) a flow or screen delivers. Use only
  IDs that appear in the requirements you were given.
- Do NOT invent flow, screen, or component numbers. List them in order; they are numbered
  for you.
- Only specify design tokens if the product actually has a visual interface. A command-line
  tool does not need a colour palette, and inventing one is noise."""

USER_TEMPLATE = """PRODUCT REQUIREMENTS:
{prd}

EXISTING DESIGN SYSTEM:
{design_system}

RESEARCH EVIDENCE (advisory only):
{research}

Produce the UX specification."""


class DraftStep(EdithModel):
    """One flow step as the model produces it, referencing others by name.

    Deliberately *not* nested inside its flow. A live 3B model asked for
    ``flows -> steps -> next_steps`` -- three levels of nesting -- returned a non-object at
    the root on six consecutive attempts. The same model produces a flat list of records
    reliably, which is the identical lesson the M2 planner learned. Steps therefore live at
    the top level and name their flow, and :func:`draft_to_ux_spec` regroups them.
    """

    #: The flow this step belongs to, by name.
    flow: str = Field(default="", max_length=200)
    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="VIEW", max_length=20)
    description: str = Field(default="", max_length=500)
    #: Names of steps that follow when things go well.
    next_steps: list[str] = Field(default_factory=list, max_length=4)
    #: Names of steps reached when they do not.
    error_steps: list[str] = Field(default_factory=list, max_length=4)
    screen: str = Field(default="", max_length=120)


class DraftFlow(EdithModel):
    """One user flow as the model produces it, without its steps.

    Steps are carried separately in :attr:`UXDesignerOutput.steps`. See :class:`DraftStep`.
    """

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    satisfies: list[str] = Field(default_factory=list, max_length=8)


class DraftScreen(EdithModel):
    """One screen as the model produces it."""

    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(default="", max_length=500)
    states: list[str] = Field(default_factory=list, max_length=9)
    components: list[str] = Field(default_factory=list, max_length=8)
    satisfies: list[str] = Field(default_factory=list, max_length=8)


class DraftComponent(EdithModel):
    """One reusable component as the model produces it."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    states: list[str] = Field(default_factory=list, max_length=6)
    interactions: list[str] = Field(default_factory=list, max_length=6)
    accessibility: list[str] = Field(default_factory=list, max_length=6)


class DraftToken(EdithModel):
    """One design token as the model produces it."""

    name: str = Field(min_length=1, max_length=80)
    category: str = Field(default="color", max_length=40)
    value: str = Field(min_length=1, max_length=120)


class UXDesignerInput(EdithModel):
    """Input contract for :class:`UXDesignerAgent`."""

    prd: str = Field(min_length=1, max_length=20_000)
    design_system: str = Field(default="", max_length=8000)
    research: str = Field(default="", max_length=8000)
    prior_knowledge: str = Field(default="", max_length=4000)


class UXDesignerOutput(EdithModel):
    """Output contract for :class:`UXDesignerAgent`."""

    # No product_name. Product identity is system-owned (M4.1 item 2): the caller already
    # knows it, and asking the model to echo it back adds a required field it can omit
    # without adding any information. Removing it is not a weakened contract -- the
    # *artifact* still requires a name, supplied from the PRD by the assembler.
    overview: str = Field(default="", max_length=2000)
    flows: list[DraftFlow] = Field(default_factory=list, max_length=MAX_FLOWS)
    #: Every step of every flow, flat. Each names its flow. See :class:`DraftStep`.
    steps: list[DraftStep] = Field(
        default_factory=list, max_length=MAX_FLOWS * MAX_STEPS
    )
    screens: list[DraftScreen] = Field(default_factory=list, max_length=MAX_SCREENS)
    components: list[DraftComponent] = Field(default_factory=list, max_length=10)
    design_tokens: list[DraftToken] = Field(default_factory=list, max_length=12)
    accessibility: list[str] = Field(default_factory=list, max_length=8)
    interaction_patterns: list[str] = Field(default_factory=list, max_length=6)
    properties: list[str] = Field(default_factory=list, max_length=8)
    design_system_rationale: str = Field(default="", max_length=600)


class UXDesignerAgent(Agent):
    """Produces a UX specification from a PRD.

    Reads artifacts; writes only into the design area through the gateway. No shell, and no
    access to source, so it cannot start implementing what it specifies.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="ux_designer",
        description="Turns requirements into flows, screens, components, and design tokens.",
        capabilities=frozenset({Capability.DOCUMENTATION}),
        permissions=AgentPermissions(
            allowed_tools=frozenset(
                {"filesystem.read", "filesystem.search", "filesystem.write"}
            ),
            allowed_read_paths=("docs/**", "design/**", "architecture/**", "README.md"),
            allowed_write_paths=("design/**", "docs/ux/**"),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = UXDesignerInput
    output_schema: ClassVar[type[BaseModel]] = UXDesignerOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, UXDesignerInput)  # noqa: S101 - validate_input guarantees
        provider = self.require_provider()
        prd = payload.prd
        if payload.prior_knowledge:
            prd = f"{prd}\n\nPRIOR KNOWLEDGE (informative):\n{payload.prior_knowledge}"

        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    prd=prd,
                    design_system=payload.design_system or "(none exists yet)",
                    research=payload.research or "(no research was gathered)",
                ),
            ),
        ]
        return provider.structured_generate(
            messages, UXDesignerOutput, max_repair_attempts=2
        )


def _slug(value: str) -> str:
    """Normalise a name for matching model-supplied cross-references."""
    return "".join(char for char in value.lower() if char.isalnum())


def _coerce_step_kind(value: str) -> StepKind:
    try:
        return StepKind(value.strip().upper())
    except ValueError:
        return StepKind.VIEW


def _coerce_states(values: list[str]) -> frozenset[ScreenState]:
    """Map model state names to the enum, keeping only ones we understand."""
    resolved: set[ScreenState] = set()
    for raw in values:
        try:
            resolved.add(ScreenState(raw.strip().upper()))
        except ValueError:
            continue
    return frozenset(resolved)


def _known_requirements(prd: PRDDocument | None) -> frozenset[str]:
    return prd.requirement_ids if prd is not None else frozenset()


def _group_steps(draft: UXDesignerOutput) -> dict[str, list[DraftStep]]:
    """Regroup the flat step list by the flow each step names.

    A step naming no flow, or naming one that does not exist, is assigned to the first flow.
    That is a guess, but a bounded one: the alternative is silently discarding a step the
    model described, which produces a flow with a hole in the middle rather than a flow with
    a step in a slightly surprising place.
    """
    if not draft.flows:
        return {}

    grouped: dict[str, list[DraftStep]] = {_slug(flow.name): [] for flow in draft.flows}
    default = _slug(draft.flows[0].name)
    for step in draft.steps:
        key = _slug(step.flow)
        grouped.setdefault(key if key in grouped else default, []).append(step)
    return grouped


def _filter_requirements(values: list[str], known: frozenset[str]) -> tuple[str, ...]:
    """Keep only requirement ids the PRD actually defines.

    A model naming ``REQ-042`` for a PRD with six requirements is hallucinating a reference.
    Dropping it here means validation reports genuine coverage gaps rather than being
    drowned in dangling ids -- and the gap itself still shows up, because the requirement
    ends up covered by nothing.
    """
    return tuple(
        dict.fromkeys(item.strip().upper() for item in values if item.strip().upper() in known)
    )


def draft_to_ux_spec(
    draft: UXDesignerOutput,
    *,
    prd: PRDDocument | None = None,
    product_name: str = "",
) -> UXSpecDocument:
    """Translate model output into a strictly-validated UX specification.

    The trust boundary for UX. Beyond assigning ids, it does three repairs that a small model
    reliably needs:

    - **Step references are resolved by name.** An edge naming a step that does not exist is
      dropped rather than emitted, since :class:`~edith.product.ux.Flow` rejects a dangling
      transition and would take the whole document down with it.
    - **A last step with no successor becomes TERMINAL.** Otherwise every flow ends in a dead
      end and validation reports a problem the model did not actually make.
    - **Required screen states are added.** DEFAULT, LOADING and ERROR are the states an
      interface needs and a specification forgets.

    Raises:
        ValueError: The resulting document fails UX schema validation.
    """
    known = _known_requirements(prd)

    screens: list[Screen] = []
    screen_by_slug: dict[str, str] = {}
    component_by_slug: dict[str, str] = {}

    for index, draft_component in enumerate(draft.components, start=1):
        component_by_slug[_slug(draft_component.name)] = element_id("CMP", index)

    for index, draft_screen in enumerate(draft.screens, start=1):
        screen_id = element_id("SCR", index)
        screen_by_slug[_slug(draft_screen.name)] = screen_id
        states = _coerce_states(draft_screen.states) | {
            ScreenState.DEFAULT,
            ScreenState.LOADING,
            ScreenState.ERROR,
        }
        screens.append(
            Screen(
                screen_id=screen_id,
                name=draft_screen.name,
                purpose=draft_screen.purpose,
                states=frozenset(states),
                components=tuple(
                    component_by_slug[_slug(name)]
                    for name in draft_screen.components
                    if _slug(name) in component_by_slug
                ),
                satisfies=_filter_requirements(draft_screen.satisfies, known),
            )
        )

    components = tuple(
        Component(
            component_id=element_id("CMP", index),
            name=draft_component.name,
            description=draft_component.description,
            states=tuple(draft_component.states),
            interactions=tuple(draft_component.interactions),
            accessibility=tuple(draft_component.accessibility),
        )
        for index, draft_component in enumerate(draft.components, start=1)
    )

    steps_by_flow = _group_steps(draft)

    flows: list[Flow] = []
    for flow_index, draft_flow in enumerate(draft.flows, start=1):
        flow_id = element_id("UX", flow_index)
        draft_steps = steps_by_flow.get(_slug(draft_flow.name), [])
        if not draft_steps:
            # A flow with no steps cannot be represented -- Flow requires at least one --
            # and inventing a placeholder would put a fabricated step into a specification
            # a Frontend Agent will read as fact.
            continue

        step_ids = {
            _slug(step.name): f"{flow_id}-S{position}"
            for position, step in enumerate(draft_steps, start=1)
        }

        steps: list[FlowStep] = []
        for position, step in enumerate(draft_steps, start=1):
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
                if position == len(draft_steps):
                    # A final step nobody follows is an ending, not a dead end.
                    kind = StepKind.TERMINAL
                elif kind not in {StepKind.TERMINAL, StepKind.ABORT}:
                    # A model given an ordered list of steps routinely describes the
                    # sequence and never wires it. Reading that list as a sequence is the
                    # only sensible interpretation, and the alternative -- a flow of
                    # disconnected dead ends -- reports a defect the model did not make.
                    following = (step_ids[_slug(draft_steps[position].name)],)

            steps.append(
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

        flows.append(
            Flow(
                flow_id=flow_id,
                name=draft_flow.name,
                description=draft_flow.description,
                entry_step=steps[0].step_id,
                steps=tuple(steps),
                satisfies=_filter_requirements(draft_flow.satisfies, known),
            )
        )

    tokens = tuple(
        DesignToken(
            token_id=element_id("TOK", index),
            name=item.name,
            category=item.category,
            value=item.value,
        )
        for index, item in enumerate(draft.design_tokens, start=1)
    )

    accessibility = tuple(
        AccessibilityRequirement(requirement=text)
        for text in draft.accessibility
        if text.strip()
    )

    properties: set[ProductProperty] = set()
    for raw in draft.properties:
        try:
            properties.add(ProductProperty(raw.strip().upper()))
        except ValueError:
            continue

    return UXSpecDocument(
        product_name=product_name or "Untitled product",
        overview=draft.overview,
        flows=tuple(flows),
        screens=tuple(screens),
        components=components,
        design_tokens=tokens,
        accessibility=accessibility,
        interaction_patterns=tuple(
            item for item in draft.interaction_patterns if item.strip()
        ),
        properties=frozenset(properties),
        design_system_rationale=draft.design_system_rationale,
    )
