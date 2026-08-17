"""The UX specification.

The output of this stage has one consumer that matters: a future Frontend Agent. That agent
will need to know which screen it is building, what states that screen can be in, which
components it composes, and what happens when things go wrong. Prose describing a "clean,
modern interface" tells it nothing.

So flows are graphs, not paragraphs. A :class:`Flow` is a sequence of identified steps with
explicit alternate and error paths, which makes two things checkable that prose cannot:
whether every step leads somewhere, and whether the failure paths were thought about at all.

The other deliberate constraint is that **every screen must declare its states**. The four
that get forgotten -- loading, empty, error, success -- are exactly the ones users hit on
their worst day, and a specification that omits them produces an interface that omits them
too. Validation reports a screen missing an error state as an issue rather than trusting
anyone to remember.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator, model_validator

from edith.schemas.common import EdithModel

from .artifacts import ArtifactDocument, ArtifactKind, is_element_id, register_document
from .properties import ProductProperty


class ScreenState(StrEnum):
    """A condition a screen can be in.

    Enumerated rather than free text because the point is to check that the unglamorous ones
    were specified. A designer will always describe the happy path.
    """

    DEFAULT = "DEFAULT"
    LOADING = "LOADING"
    EMPTY = "EMPTY"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    #: Some data is present and more is arriving.
    PARTIAL = "PARTIAL"
    #: The viewer may look but not change anything.
    READ_ONLY = "READ_ONLY"
    #: The viewer is not permitted to see this at all.
    UNAUTHORIZED = "UNAUTHORIZED"
    #: A destructive or slow action is in flight.
    BUSY = "BUSY"


#: States a screen showing remote or user-entered data must specify. Omitting one of these is
#: not a style preference -- it is a state the user will reach and the interface will not
#: have been designed for.
REQUIRED_STATES: frozenset[ScreenState] = frozenset(
    {ScreenState.DEFAULT, ScreenState.LOADING, ScreenState.ERROR}
)


class StepKind(StrEnum):
    """What happens at one step of a flow."""

    #: The user is shown something and does nothing yet.
    VIEW = "VIEW"
    #: The user provides input.
    INPUT = "INPUT"
    #: The user commits: submit, confirm, purchase.
    ACTION = "ACTION"
    #: The system decides which way the flow goes.
    DECISION = "DECISION"
    #: Something happens outside the interface: an email, a webhook, a job.
    SYSTEM = "SYSTEM"
    #: The flow ends successfully.
    TERMINAL = "TERMINAL"
    #: The flow ends without achieving its goal.
    ABORT = "ABORT"


class FlowStep(EdithModel):
    """One identified step in a user flow.

    ``next_steps`` and ``error_steps`` are what make a flow a graph. A step naming neither,
    and not marked terminal, is a dead end -- and validation says so.
    """

    step_id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=200)
    kind: StepKind = StepKind.VIEW
    description: str = Field(default="", max_length=1000)
    #: The screen this step happens on, when it happens on one.
    screen_id: str = Field(default="", max_length=40)
    #: Steps reachable when things go well.
    next_steps: tuple[str, ...] = ()
    #: Steps reachable when they do not. The half of a flow that gets skipped.
    error_steps: tuple[str, ...] = ()
    #: What must be true to enter this step.
    preconditions: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        """Whether the flow ends here."""
        return self.kind in {StepKind.TERMINAL, StepKind.ABORT}

    @property
    def dead_end(self) -> bool:
        """Whether this step leads nowhere without being an ending."""
        return not self.terminal and not self.next_steps and not self.error_steps


class Flow(EdithModel):
    """A user flow as an addressable graph of steps."""

    flow_id: str = Field(pattern=r"^UX-\d{3,}$")
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    #: The persona this flow is for, when the PRD defined one.
    persona_id: str = Field(default="", max_length=40)
    #: Where the flow begins. Must be one of the step ids.
    entry_step: str = Field(min_length=1, max_length=40)
    steps: tuple[FlowStep, ...] = Field(min_length=1)
    #: Requirements this flow delivers.
    satisfies: tuple[str, ...] = ()

    @field_validator("satisfies")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"satisfies entry {identifier!r} is not a REQ id")
        return value

    @model_validator(mode="after")
    def _steps_are_coherent(self) -> Flow:
        """Step ids are unique, the entry exists, and no transition dangles.

        Enforced in the schema rather than in a validation pass because a flow whose steps
        point at nothing is not a flow with a problem -- it is not a flow.
        """
        identifiers = [step.step_id for step in self.steps]
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise ValueError(f"flow {self.flow_id} has duplicate step ids: {duplicates}")

        known = set(identifiers)
        if self.entry_step not in known:
            raise ValueError(
                f"flow {self.flow_id} entry step {self.entry_step!r} is not one of its steps"
            )
        for step in self.steps:
            for target in (*step.next_steps, *step.error_steps):
                if target not in known:
                    raise ValueError(
                        f"flow {self.flow_id} step {step.step_id} points at unknown "
                        f"step {target!r}"
                    )
        return self

    @property
    def step_ids(self) -> frozenset[str]:
        """Ids of every step in this flow."""
        return frozenset(step.step_id for step in self.steps)

    def dead_ends(self) -> tuple[str, ...]:
        """Steps that lead nowhere without ending the flow."""
        return tuple(step.step_id for step in self.steps if step.dead_end)

    def unreachable_steps(self) -> tuple[str, ...]:
        """Steps no path from the entry can reach.

        A step nobody can arrive at is either a missing transition or dead specification;
        either way it is worth knowing before someone builds it.
        """
        reached: set[str] = set()
        frontier = [self.entry_step]
        by_id = {step.step_id: step for step in self.steps}
        while frontier:
            current = frontier.pop()
            if current in reached:
                continue
            reached.add(current)
            step = by_id.get(current)
            if step is not None:
                frontier.extend((*step.next_steps, *step.error_steps))
        return tuple(sorted(self.step_ids - reached))

    @property
    def has_error_path(self) -> bool:
        """Whether any step specifies what happens when something fails."""
        return any(step.error_steps for step in self.steps) or any(
            step.kind is StepKind.ABORT for step in self.steps
        )


class Screen(EdithModel):
    """One view the user can be looking at."""

    screen_id: str = Field(pattern=r"^SCR-\d{3,}$")
    name: str = Field(min_length=1, max_length=200)
    purpose: str = Field(default="", max_length=1000)
    #: Components composed onto this screen, by CMP id.
    components: tuple[str, ...] = ()
    #: Every state this screen can be in. See :data:`REQUIRED_STATES`.
    states: frozenset[ScreenState] = frozenset({ScreenState.DEFAULT})
    #: What the user can navigate to from here, by SCR id.
    navigates_to: tuple[str, ...] = ()
    #: Requirements this screen serves.
    satisfies: tuple[str, ...] = ()
    #: Text the screen needs that someone must actually write.
    content_requirements: tuple[str, ...] = ()

    @field_validator("satisfies")
    @classmethod
    def _require_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            if not is_element_id(identifier, "REQ"):
                raise ValueError(f"satisfies entry {identifier!r} is not a REQ id")
        return value

    def missing_states(self) -> tuple[ScreenState, ...]:
        """Required states this screen does not specify."""
        return tuple(sorted(REQUIRED_STATES - self.states, key=lambda item: item.value))


class Component(EdithModel):
    """A reusable interface element."""

    component_id: str = Field(pattern=r"^CMP-\d{3,}$")
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    #: Interactive states: hover, focus, disabled, error. Free text, since component state
    #: vocabularies are genuinely product-specific.
    states: tuple[str, ...] = ()
    #: What the user can do with it.
    interactions: tuple[str, ...] = ()
    #: Accessibility obligations: roles, labels, keyboard behaviour, contrast.
    accessibility: tuple[str, ...] = ()
    #: Components this one is built from.
    composes: tuple[str, ...] = ()


class DesignToken(EdithModel):
    """One design-system value.

    Tokens rather than literals so a colour or spacing step is named once and referenced
    everywhere, which is what makes a later theme change a change to this document rather
    than a search through a codebase.
    """

    token_id: str = Field(pattern=r"^TOK-\d{3,}$")
    name: str = Field(min_length=1, max_length=100)
    #: ``color``, ``spacing``, ``typography``, ``radius``, ``shadow``, ``breakpoint``.
    category: str = Field(min_length=1, max_length=50)
    value: str = Field(min_length=1, max_length=200)
    usage: str = Field(default="", max_length=500)


class AccessibilityRequirement(EdithModel):
    """An accessibility obligation stated so it can be checked."""

    #: The standard being claimed, e.g. ``WCAG 2.2 AA``.
    standard: str = Field(default="WCAG 2.2 AA", max_length=100)
    requirement: str = Field(min_length=1, max_length=1000)
    applies_to: tuple[str, ...] = ()


@register_document
class UXSpecDocument(ArtifactDocument):
    """A UX specification.

    The body of a :attr:`~edith.product.artifacts.ArtifactKind.UX_SPEC` artifact.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.UX_SPEC

    product_name: str = Field(min_length=1, max_length=200)
    overview: str = Field(default="", max_length=4000)
    flows: tuple[Flow, ...] = ()
    screens: tuple[Screen, ...] = ()
    components: tuple[Component, ...] = ()
    design_tokens: tuple[DesignToken, ...] = ()
    accessibility: tuple[AccessibilityRequirement, ...] = ()
    #: Breakpoint names and widths, when the product has a responsive interface.
    breakpoints: dict[str, str] = Field(default_factory=dict)
    #: Interaction patterns applied across the product: undo, confirmation, autosave.
    interaction_patterns: tuple[str, ...] = ()
    #: Structural claims this specification makes, checked against the PRD and architecture.
    properties: frozenset[ProductProperty] = frozenset()
    #: Why a design system was or was not produced. CLAUDE.md's rule against unnecessary
    #: infrastructure applies to design too: a CLI does not need design tokens, and saying so
    #: explicitly is better than silently omitting them.
    design_system_rationale: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _unique_ids(self) -> UXSpecDocument:
        identifiers = list(self.element_ids())
        duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
        if duplicates:
            raise ValueError(
                f"UX spec defines duplicate element ids: {', '.join(duplicates)}"
            )
        return self

    def element_ids(self) -> tuple[str, ...]:
        """Every stable id this specification defines."""
        return (
            tuple(item.flow_id for item in self.flows)
            + tuple(item.screen_id for item in self.screens)
            + tuple(item.component_id for item in self.components)
            + tuple(item.token_id for item in self.design_tokens)
        )

    def referenced_ids(self) -> tuple[str, ...]:
        """Every id this specification points at."""
        references: list[str] = []
        for flow in self.flows:
            references.extend(flow.satisfies)
            if flow.persona_id:
                references.append(flow.persona_id)
            references.extend(
                step.screen_id for step in flow.steps if step.screen_id
            )
        for screen in self.screens:
            references.extend(screen.satisfies)
            references.extend(screen.components)
            references.extend(screen.navigates_to)
        for component in self.components:
            references.extend(component.composes)
        for requirement in self.accessibility:
            references.extend(requirement.applies_to)
        return tuple(references)

    # -- Convenience -----------------------------------------------------------------

    @property
    def screen_ids(self) -> frozenset[str]:
        """Ids of every screen."""
        return frozenset(item.screen_id for item in self.screens)

    @property
    def covered_requirements(self) -> frozenset[str]:
        """Requirement ids this specification claims to serve."""
        covered: set[str] = set()
        for flow in self.flows:
            covered |= set(flow.satisfies)
        for screen in self.screens:
            covered |= set(screen.satisfies)
        return frozenset(covered)

    def screens_missing_states(self) -> dict[str, tuple[ScreenState, ...]]:
        """Screens that do not specify one of the required states."""
        return {
            screen.screen_id: missing
            for screen in self.screens
            if (missing := screen.missing_states())
        }

    def flows_without_error_paths(self) -> tuple[str, ...]:
        """Flows that never say what happens when something goes wrong."""
        return tuple(flow.flow_id for flow in self.flows if not flow.has_error_path)

    def render(self) -> str:
        """Render for a human reader or a downstream agent's prompt."""
        lines = [f"# UX specification: {self.product_name}", ""]
        if self.overview:
            lines.extend([self.overview, ""])
        for flow in self.flows:
            lines.append(f"## Flow {flow.flow_id}: {flow.name}")
            for step in flow.steps:
                marker = " (entry)" if step.step_id == flow.entry_step else ""
                targets = ", ".join(step.next_steps) or "-"
                errors = ", ".join(step.error_steps)
                error_note = f" | on error: {errors}" if errors else ""
                lines.append(
                    f"  {step.step_id}{marker} [{step.kind}] {step.name} "
                    f"-> {targets}{error_note}"
                )
            lines.append("")
        if self.screens:
            lines.append("## Screens")
            for screen in self.screens:
                states = ", ".join(sorted(state.value for state in screen.states))
                lines.append(f"- {screen.screen_id} {screen.name} (states: {states})")
            lines.append("")
        if self.design_tokens:
            lines.append("## Design tokens")
            for token in self.design_tokens:
                lines.append(f"- {token.name} ({token.category}): {token.value}")
        return "\n".join(lines).strip()
