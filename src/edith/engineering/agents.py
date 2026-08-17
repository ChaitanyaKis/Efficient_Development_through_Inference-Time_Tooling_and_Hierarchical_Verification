"""The five M5 engineering agents.

Each is thin on purpose. The decisions that matter -- who may write where, what context a
task gets, whether the result is acceptable -- live in :mod:`edith.engineering.ownership`,
the M2 context engine, and the M2 verification runner respectively. An agent's job is to turn
a narrow, well-scoped task into file edits.

Three properties are shared and deliberate:

**They subclass the M2 coder.** Only the prompt differs; the sanitiser, syntax gate,
symbol-preservation check, gateway write and diff are all inherited. A specialised agent is a
different prompt and a different scope, not a different route to disk. Reimplementing the
apply pipeline per role would give five chances to weaken a guarantee M2.1 established once.

**No shell, no git.** An engineering agent proposes file changes. Running commands is the
verifier's job; committing is the orchestrator's. M2.1 established that separation for the
coder and the critic, and specialisation is not a reason to collapse it.

**The role is in the prompt and in the permissions.** The prompt tells the model what it is
for; the gateway decides what it can reach. When those disagree, the gateway wins, which is
what makes the boundary real rather than advisory.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from edith.agents.coder import CoderInput, CoderOutput, CodingAgent
from edith.schemas.agent import AgentIdentity, AgentPermissions, Capability
from edith.schemas.model import Message, Role

from .ownership import EngineeringRole, scope_for

#: Rules every engineering agent obeys. Stated once so the five prompts differ only in what
#: they are actually for, which is the point of specialising them at all.
COMMON_RULES = """
Rules that apply to every change you make:
- Make the SMALLEST change that satisfies the task. Do not refactor, reformat, or rename
  anything the task did not ask about. A large speculative rewrite is harder to verify than
  the defect it was meant to fix.
- Only write files inside the paths listed in YOUR SCOPE. A write outside it is refused by
  the system and the task fails.
- Do NOT change what the product is supposed to do. If a requirement seems wrong, implement
  it as written and say so in `notes`; the requirement is not yours to change.
- `mode` is one of: replace_file, replace_function, append.
- Produce complete, syntactically valid file content. Truncated code fails immediately.
"""


class EngineeringInput(CoderInput):
    """Input contract shared by every engineering agent.

    Extends :class:`~edith.agents.coder.CoderInput` rather than replacing it. That is what
    lets these agents inherit the coder's apply pipeline untouched: its ``_run`` is typed
    against ``CoderInput``, and an engineering payload *is* one, with the extra product
    context a specialised role needs.

    The additions are the M5 item 15 context list: the requirements and architecture that
    justify the task, the interface it must respect, and the paths it may write. Nothing
    else -- an agent that receives the whole project spends its context on material it will
    not use.
    """

    #: Only the requirements this task implements.
    requirements: str = Field(default="", max_length=4000)
    #: Only the architecture components this task touches.
    architecture: str = Field(default="", max_length=4000)
    #: The UX flows and screens this task must respect. Frontend tasks only, in practice.
    ux: str = Field(default="", max_length=4000)
    #: Repo-relative paths this task may write.
    scope: str = Field(default="", max_length=1000)


USER_TEMPLATE = """TASK: {title}
{description}

ACCEPTANCE CRITERIA:
{acceptance}

REQUIREMENTS THIS TASK IMPLEMENTS:
{requirements}

ARCHITECTURE:
{architecture}
{ux_block}
YOUR SCOPE (you may write only these paths):
{scope}

EXISTING CODE:
{context}
{evidence_block}
Produce the file changes."""


class _EngineeringBehaviour(CodingAgent):
    """Shared behaviour for the five roles.

    Subclasses :class:`~edith.agents.coder.CodingAgent` deliberately, and overrides only the
    prompt. Everything downstream of generation -- the sanitiser, the syntax gate, the
    symbol-preservation check, the gateway write, the rejection accounting, the diff -- is
    inherited unchanged.

    That is the point. A specialised agent is a *different prompt and a different scope*, not
    a different route to disk. Reimplementing the apply pipeline per role would give five
    chances to weaken a guarantee M2.1 established once.
    """

    role: ClassVar[EngineeringRole]
    system_prompt: ClassVar[str]

    input_schema: ClassVar[type[BaseModel]] = EngineeringInput
    output_schema: ClassVar[type[BaseModel]] = CoderOutput

    def _build_messages(self, payload: BaseModel) -> list[Message]:
        """Assemble the role's prompt from a task-scoped context."""
        assert isinstance(payload, EngineeringInput)  # noqa: S101 - validate_input guarantees

        ux_block = f"\nUSER INTERFACE SPECIFICATION:\n{payload.ux}\n" if payload.ux else ""
        evidence_block = (
            f"\nWHAT WENT WRONG LAST TIME:\n{payload.failure_evidence}\n"
            if payload.failure_evidence
            else ""
        )
        system = self.system_prompt + COMMON_RULES
        if payload.prior_knowledge:
            system = (
                f"{system}\nPRIOR KNOWLEDGE (informative, not a requirement):\n"
                f"{payload.prior_knowledge}\n"
            )

        messages = [
            Message(role=Role.SYSTEM, content=system),
            Message(
                role=Role.USER,
                content=USER_TEMPLATE.format(
                    title=payload.title,
                    description=payload.description,
                    acceptance="; ".join(payload.acceptance_criteria) or "(none stated)",
                    requirements=payload.requirements or "(none supplied)",
                    architecture=payload.architecture or "(none supplied)",
                    ux_block=ux_block,
                    scope=payload.scope or "(the task's assigned paths)",
                    context=payload.context or "(this is a new file)",
                    evidence_block=evidence_block,
                ),
            ),
        ]
        return messages


def _identity(role: EngineeringRole, capabilities: frozenset[Capability]) -> AgentIdentity:
    """Build an identity whose permissions come from the role's scope table."""
    scope = scope_for(role)
    return AgentIdentity(
        name=f"{role.value}_engineer",
        description=scope.description,
        capabilities=capabilities,
        permissions=scope.permissions(),
    )


FRONTEND_PROMPT = """You are the frontend component of a software engineering system.

You implement user interfaces from a UX specification. The specification is authoritative:
build the screens, states and flows it describes. If it says a screen has a loading state and
an error state, build both -- those are the states users hit on their worst day.

You do NOT design the interface, choose what the product does, write server code, or touch
the database. Other components own those.
"""

BACKEND_PROMPT = """You are the backend component of a software engineering system.

You implement API endpoints, services, business logic, validation and error handling from an
architecture and an API contract.

Every endpoint must exist because a requirement or an architecture decision asked for it.
Validate input at the boundary. Return errors the caller can act on rather than letting an
exception escape.

You do NOT write migrations, design the interface, or configure infrastructure.
"""

DATABASE_PROMPT = """You are the database component of a software engineering system.

You own schema, migrations, indexes and constraints.

Every schema change is a MIGRATION -- a new file that moves the schema forward. Never edit an
applied migration and never write code that mutates data outside one. A migration is the only
record of how a database got into its current shape.

You do NOT write application business logic or API handlers.
"""

DEVOPS_PROMPT = """You are the operations component of a software engineering system.

You produce container configuration, environment templates, CI configuration and local
development scripts.

Introduce infrastructure only where the architecture actually needs it. A single-machine tool
does not need an orchestrator, a message broker, or a multi-stage deployment pipeline, and
adding them is a cost the project pays forever.

You do NOT change application code, tests, or migrations.
"""

DEPENDENCY_PROMPT = """You are the dependency component of a software engineering system.

You reconcile what the code imports with what the manifests declare.

Add a dependency only when the source actually imports it. Never add a package because it
seems useful, and never pin to a version you have no reason to believe exists. Prefer the
standard library over a new dependency.

You do NOT write application code.
"""


class FrontendAgent(_EngineeringBehaviour):
    """Implements the user interface against the UX specification."""

    role: ClassVar[EngineeringRole] = EngineeringRole.FRONTEND
    system_prompt: ClassVar[str] = FRONTEND_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.FRONTEND, frozenset({Capability.CODE_GENERATION})
    )


class BackendAgent(_EngineeringBehaviour):
    """Implements API endpoints, services, and business logic."""

    role: ClassVar[EngineeringRole] = EngineeringRole.BACKEND
    system_prompt: ClassVar[str] = BACKEND_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.BACKEND, frozenset({Capability.CODE_GENERATION})
    )


class DatabaseAgent(_EngineeringBehaviour):
    """Owns schema and migrations."""

    role: ClassVar[EngineeringRole] = EngineeringRole.DATABASE
    system_prompt: ClassVar[str] = DATABASE_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.DATABASE, frozenset({Capability.CODE_GENERATION})
    )


class DevOpsAgent(_EngineeringBehaviour):
    """Owns containers, CI, and local development scripts."""

    role: ClassVar[EngineeringRole] = EngineeringRole.DEVOPS
    system_prompt: ClassVar[str] = DEVOPS_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.DEVOPS, frozenset({Capability.CODE_GENERATION})
    )


class DependencyAgent(_EngineeringBehaviour):
    """Reconciles manifests with what the source actually imports.

    Unlike the other four, most of this role's work is *deterministic* and lives in the M3.1
    environment foundation: discovery parses imports from the AST, manifests are reconciled by
    comparison, and install artifacts are generated from a template. The model is only asked
    for judgement the parser cannot supply. See :mod:`edith.engineering.dependency`.
    """

    role: ClassVar[EngineeringRole] = EngineeringRole.DEPENDENCY
    system_prompt: ClassVar[str] = DEPENDENCY_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.DEPENDENCY, frozenset({Capability.CODE_GENERATION})
    )


GENERIC_PROMPT = """You are the implementation component of a software engineering system.

You implement whatever the task asks for: storage, services, interfaces, configuration.
"""


class GenericAgent(_EngineeringBehaviour):
    """One agent for any task. The control arm of the M5.1 specialisation experiment.

    Identical to the five specialised agents in every respect except its prompt and its
    scope: same executor, same repair loop, same import gate, same verification, same
    gateway. That is the point -- the experiment's independent variable is specialisation,
    so everything else has to be held constant, and the only way to guarantee that is for the
    control to run through the same code.
    """

    role: ClassVar[EngineeringRole] = EngineeringRole.GENERIC
    system_prompt: ClassVar[str] = GENERIC_PROMPT
    identity: ClassVar[AgentIdentity] = _identity(
        EngineeringRole.GENERIC, frozenset({Capability.CODE_GENERATION})
    )


#: Role -> agent class. The executor resolves an assignment through this, so a role without
#: an implementation is a loud failure rather than a silently skipped task.
ENGINEERING_AGENTS: dict[EngineeringRole, type[_EngineeringBehaviour]] = {
    EngineeringRole.FRONTEND: FrontendAgent,
    EngineeringRole.BACKEND: BackendAgent,
    EngineeringRole.DATABASE: DatabaseAgent,
    EngineeringRole.DEVOPS: DevOpsAgent,
    EngineeringRole.DEPENDENCY: DependencyAgent,
    EngineeringRole.GENERIC: GenericAgent,
}


def agent_for(role: EngineeringRole) -> type[_EngineeringBehaviour]:
    """The agent class that executes a role."""
    return ENGINEERING_AGENTS[role]


def permissions_for(role: EngineeringRole) -> AgentPermissions:
    """The declared permissions of a role, for inspection."""
    return scope_for(role).permissions()
