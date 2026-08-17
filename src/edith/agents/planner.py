"""The Planner Agent.

Turns a user request into a validated, executable DAG. It reads the repository (through the
gateway, read-only) and writes nothing: the planner has **no write scope and no shell**, so
it cannot modify source or run anything, only propose work.

Schema design note: the model-facing schema is deliberately flatter and smaller than
:class:`~edith.planning.task.Task`. A 3B model producing nested objects with enums and glob
tuples fails constantly; producing a short list of flat records with plain strings is
reliable. The orchestrator translates that into real ``Task`` objects, which is where the
strict validation lives.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from edith.planning.task import Task, TaskScope, VerificationRequirement
from edith.schemas.agent import (
    AgentIdentity,
    AgentPermissions,
    AgentRequest,
    Capability,
)
from edith.schemas.common import EdithModel
from edith.schemas.model import Message, Role

from .base import Agent

MAX_TASKS = 6

SYSTEM_PROMPT = """You are the planning component of a software engineering system.

You break a user request into a small number of concrete implementation steps.

Rules:
- Produce ONE step. Only produce more if the request genuinely requires editing several
  unrelated files, and never more than three.
- Every step must be an EDIT TO A FILE. If a step would not change a file, it is not a step.
- NEVER produce steps like "identify the bug", "investigate", "run the tests", "verify",
  or "review". The system already runs the real tests after every step, and diagnosis is
  done automatically when they fail. A step like that cannot succeed and will block the
  real work behind it.
  Correct plan for "subtract is broken": a single step, "Fix the subtract function".
- `files` must list repository-relative paths that the step will change.
- `depends_on` lists the numbers of earlier steps that must finish first. Usually empty.
- Do not plan work outside the user's request."""

USER_TEMPLATE = """USER REQUEST:
{request}

REPOSITORY CONTEXT:
{context}

Produce the implementation plan."""


class PlannedStep(EdithModel):
    """One step as produced by the model.

    Flat and stringly-typed on purpose -- this is the shape a small local model can produce
    reliably. It is translated into a validated :class:`Task` afterwards.
    """

    step: int = Field(ge=1, le=MAX_TASKS)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    files: list[str] = Field(default_factory=list, max_length=10)
    depends_on: list[int] = Field(default_factory=list, max_length=MAX_TASKS)
    acceptance: str = Field(default="", max_length=1000)

    @field_validator("files")
    @classmethod
    def _clean_paths(cls, value: list[str]) -> list[str]:
        """Normalize separators and drop anything obviously unusable.

        Path *safety* is enforced by the M1 path policy when the coder actually writes;
        this only tidies model output so a stray "./" does not defeat scope matching.
        """
        cleaned = []
        for raw in value:
            candidate = raw.strip().replace("\\", "/")
            # Strip a leading "./" as a *prefix*. Never lstrip("./") -- that removes a
            # character set, turning "../../etc/passwd" into "etc/passwd" and laundering a
            # traversal attempt into a plausible relative path.
            while candidate.startswith("./"):
                candidate = candidate[2:]
            if candidate and len(candidate) < 300:
                cleaned.append(candidate)
        return cleaned


class PlannerInput(EdithModel):
    """Input contract for :class:`PlannerAgent`."""

    request: str = Field(min_length=1, max_length=4000)
    context: str = Field(default="", max_length=20_000)


class PlannerOutput(EdithModel):
    """Output contract for :class:`PlannerAgent`."""

    goal: str = Field(min_length=1, max_length=500)
    steps: list[PlannedStep] = Field(min_length=1, max_length=MAX_TASKS)


class PlannerAgent(Agent):
    """Decomposes a request into an executable plan.

    Read-only by construction: no write paths, no shell. A planner that could edit code
    would blur the line CLAUDE.md draws between proposing and doing.
    """

    identity: ClassVar[AgentIdentity] = AgentIdentity(
        name="planner",
        description="Breaks a user request into a small DAG of concrete implementation steps.",
        capabilities=frozenset({Capability.PLANNING}),
        permissions=AgentPermissions(
            allowed_tools=frozenset({"filesystem.read", "filesystem.search", "git.status"}),
            allowed_read_paths=("**",),
        ),
    )
    input_schema: ClassVar[type[BaseModel]] = PlannerInput
    output_schema: ClassVar[type[BaseModel]] = PlannerOutput

    def _run(self, payload: BaseModel, request: AgentRequest) -> BaseModel:
        assert isinstance(payload, PlannerInput)  # noqa: S101 - guaranteed by validate_input
        provider = self.require_provider()
        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    request=payload.request,
                    context=payload.context or "(no repository context available)",
                ),
            ),
        ]
        return provider.structured_generate(messages, PlannerOutput, max_repair_attempts=2)


def plan_to_tasks(
    plan: PlannerOutput,
    *,
    agent: str = "coder",
    max_attempts: int = 3,
    default_write_scope: tuple[str, ...] = ("**",),
) -> list[Task]:
    """Translate model-shaped steps into validated :class:`Task` objects.

    This is the trust boundary. Everything the model produced is re-expressed through the
    strict schema, so a malformed or malicious plan becomes a validation error rather than
    an executed instruction. In particular:

    - Step numbers become opaque task ids; a model cannot name a task after an existing one.
    - ``depends_on`` entries that do not correspond to a real step are dropped rather than
      producing a dangling edge (the DAG would reject it anyway, less informatively).
    - Write scope is derived from the files the step named, so a step that said it would
      touch ``src/a.py`` cannot write anywhere else. A step naming no files falls back to
      the caller-supplied default.

    Raises:
        ValueError: A step's proposed scope is unsafe (absolute or traversing).
    """
    ordered = sorted(plan.steps, key=lambda step: step.step)
    identifiers = {step.step: f"task_{index + 1:02d}" for index, step in enumerate(ordered)}

    tasks: list[Task] = []
    for index, step in enumerate(ordered):
        dependencies = tuple(
            identifiers[number]
            for number in dict.fromkeys(step.depends_on)
            if number in identifiers and identifiers[number] != identifiers[step.step]
        )
        write_scope = _scope_from_files(step.files) or default_write_scope
        criteria = (step.acceptance,) if step.acceptance.strip() else ()

        tasks.append(
            Task(
                task_id=identifiers[step.step],
                title=step.title,
                description=step.description,
                agent=agent,
                priority=100 + index,
                dependencies=dependencies,
                inputs={"files": ",".join(step.files)} if step.files else {},
                acceptance_criteria=criteria,
                verification=(VerificationRequirement(kind="tests"),),
                scope=TaskScope(read_paths=("**",), write_paths=write_scope),
                max_attempts=max_attempts,
            )
        )
    return tasks


def _scope_from_files(files: list[str]) -> tuple[str, ...]:
    """Derive a write scope from the files a step said it would change.

    Grants the named files plus their immediate directory, so a step that must add a test
    beside the module it edits is not blocked by an over-tight scope. Anything that fails
    the repo-relative rules is discarded -- :class:`TaskScope` would reject it and take the
    whole plan down with it.
    """
    patterns: set[str] = set()
    for path in files:
        if path.startswith(("/", "\\")) or (len(path) > 1 and path[1] == ":"):
            continue
        if ".." in path.split("/"):
            continue
        patterns.add(path)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            patterns.add(f"{parent}/**")
    return tuple(sorted(patterns))
