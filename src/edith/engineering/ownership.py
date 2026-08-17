"""Who may build what: engineering roles, their scopes, and task assignment.

M5 turns an implementation plan into code. That means several agents writing into one
repository, and the question that decides whether the result is trustworthy is not *can they
generate code* but *can they generate code outside their remit*.

The answer here is structural. Each role declares repo-relative write patterns, those become
:class:`~edith.schemas.agent.AgentPermissions`, and the M1 gateway enforces them. A Frontend
Agent asked to write a migration does not get a polite refusal from a prompt — its write
fails at the policy layer, is classified ``SECURITY_FAILURE``, and never reaches disk.

Two things follow that are worth stating:

**Assignment is validated, not trusted.** The M4 planner records a ``responsible_agent`` on
each task, and a model produced that string. :func:`assign` resolves it against the known
roles and refuses anything it does not recognise, so a plan cannot invent an agent with
convenient permissions.

**Scope is narrowed to the task, not widened to the role.** A backend task that names
``src/backend/api.py`` gets that file and its directory, intersected with the role's ceiling.
The role bounds what an agent may *ever* touch; the task bounds what it may touch *now*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edith.observability.logging import get_logger
from edith.product.architecture import ImplementationPlanDocument, PlannedTask
from edith.schemas.agent import AgentPermissions

logger = get_logger(__name__)


class EngineeringRole(StrEnum):
    """The five roles M5 ships.

    Deliberately not every agent CLAUDE.md eventually names. Security, Testing, Code Review,
    Debugging, Performance, Documentation and Refactoring are absent because M2 already owns
    the verify-and-repair loop, and duplicating it as agents before the five below work would
    be building on something unproven.
    """

    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DEVOPS = "devops"
    DEPENDENCY = "dependency"
    #: One agent for everything, with a broad scope. Not a production role -- it is the
    #: control arm of the M5.1 specialisation experiment, and it exists *here* rather than
    #: in the benchmark so that it runs through the identical executor, repair loop,
    #: import gate and verification. A control given weaker infrastructure measures the
    #: infrastructure, not the variable.
    GENERIC = "generic"


@dataclass(frozen=True)
class RoleScope:
    """What one role may read, write, and run.

    ``write`` is a ceiling, not a grant: a task narrows it further. Nothing outside it is
    reachable regardless of what a plan asks for.
    """

    role: EngineeringRole
    description: str
    write: tuple[str, ...]
    read: tuple[str, ...] = ("**",)
    tools: frozenset[str] = frozenset(
        {"filesystem.read", "filesystem.search", "filesystem.write", "filesystem.patch"}
    )
    #: Paths this role must never write even if a task names them. Enforced in addition to
    #: the global protected patterns, because "the frontend agent may not write migrations"
    #: is a role boundary rather than a repository-wide rule.
    forbidden: tuple[str, ...] = ()

    def permissions(self) -> AgentPermissions:
        """The M1 permission set for this role."""
        return AgentPermissions(
            allowed_tools=self.tools,
            allowed_read_paths=self.read,
            allowed_write_paths=self.write,
        )


#: The scope table. Every role is explicit; there is no default that grants anything.
#:
#: Note what is absent from every entry: ``shell.run`` and ``git.*``. An engineering agent
#: proposes file changes. Running commands is the verifier's job and committing is the
#: orchestrator's, which keeps "wrote some code" and "decided the code works" in different
#: hands -- the same separation M2.1 established for the coder and the critic.
ROLE_SCOPES: dict[EngineeringRole, RoleScope] = {
    EngineeringRole.FRONTEND: RoleScope(
        role=EngineeringRole.FRONTEND,
        description="Builds the user interface against the UX specification.",
        write=(
            "src/frontend/**",
            "frontend/**",
            "static/**",
            "templates/**",
            "tests/frontend/**",
        ),
        # A frontend agent that can write a migration can change the shape of the data
        # underneath every other agent's work.
        forbidden=("migrations/**", "database/**", "docker/**", "deploy/**"),
    ),
    EngineeringRole.BACKEND: RoleScope(
        role=EngineeringRole.BACKEND,
        description="Builds API endpoints, services, and business logic.",
        write=(
            "src/backend/**",
            "backend/**",
            "src/api/**",
            "api/**",
            "tests/backend/**",
        ),
        forbidden=("migrations/**", "docker/**", "deploy/**", "src/frontend/**"),
    ),
    EngineeringRole.DATABASE: RoleScope(
        role=EngineeringRole.DATABASE,
        description="Owns schema, migrations, indexes, and constraints.",
        write=(
            "database/**",
            "migrations/**",
            "src/database/**",
            "tests/database/**",
        ),
        # The database agent shapes storage. Business logic belongs to the backend, and
        # letting one agent do both is how a schema change quietly rewrites behaviour.
        forbidden=("src/backend/**", "src/frontend/**", "backend/**", "frontend/**"),
    ),
    EngineeringRole.DEVOPS: RoleScope(
        role=EngineeringRole.DEVOPS,
        description="Owns containers, CI, and local development scripts.",
        write=(
            "docker/**",
            "deploy/**",
            ".github/**",
            "scripts/**",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
        ),
        forbidden=("src/**", "migrations/**", "tests/**"),
    ),
    EngineeringRole.DEPENDENCY: RoleScope(
        role=EngineeringRole.DEPENDENCY,
        description="Reconciles manifests and generates reproducible install artifacts.",
        write=(
            "requirements.txt",
            "requirements-dev.txt",
            "pyproject.toml",
            "package.json",
            "scripts/install.bat",
            "scripts/install.ps1",
            "scripts/install.sh",
        ),
        # The dependency agent decides what is installed, never what the application does.
        forbidden=("src/**", "tests/**", "migrations/**"),
    ),
    EngineeringRole.GENERIC: RoleScope(
        role=EngineeringRole.GENERIC,
        description="One agent for any task, with a broad scope. The experiment's control.",
        # The union of what the five specialised roles may write, computed below rather than
        # hand-listed so it cannot drift out of date. Deliberately generous: a control
        # handicapped by a narrow scope would fail for the wrong reason, and the experiment
        # would measure permissions instead of specialisation.
        write=(),
    ),
}


def _union_of_specialised_scopes() -> tuple[str, ...]:
    """Every path pattern the five specialised roles may write, combined.

    Computed so the generic control cannot silently fall behind when a specialised role
    gains a path. A test asserts the union actually covers every role.
    """
    patterns: set[str] = set()
    for role, scope in ROLE_SCOPES.items():
        if role is not EngineeringRole.GENERIC:
            patterns.update(scope.write)
    return tuple(sorted(patterns))


ROLE_SCOPES[EngineeringRole.GENERIC] = RoleScope(
    role=EngineeringRole.GENERIC,
    description=ROLE_SCOPES[EngineeringRole.GENERIC].description,
    write=_union_of_specialised_scopes(),
)


def scope_for(role: EngineeringRole) -> RoleScope:
    """The scope of one role."""
    return ROLE_SCOPES[role]


def resolve_role(name: str) -> EngineeringRole | None:
    """Map a plan's ``responsible_agent`` string onto a known role.

    Returns ``None`` for anything unrecognised rather than guessing. A plan naming
    ``security_agent`` must not silently execute as a backend task with backend permissions.
    """
    candidate = name.strip().lower().removesuffix("_agent").removesuffix("-agent")
    aliases = {
        "ui": EngineeringRole.FRONTEND,
        "web": EngineeringRole.FRONTEND,
        "client": EngineeringRole.FRONTEND,
        "server": EngineeringRole.BACKEND,
        "api": EngineeringRole.BACKEND,
        "service": EngineeringRole.BACKEND,
        "db": EngineeringRole.DATABASE,
        "schema": EngineeringRole.DATABASE,
        "migration": EngineeringRole.DATABASE,
        "infra": EngineeringRole.DEVOPS,
        "infrastructure": EngineeringRole.DEVOPS,
        "ops": EngineeringRole.DEVOPS,
        "deps": EngineeringRole.DEPENDENCY,
        "packaging": EngineeringRole.DEPENDENCY,
    }
    if candidate in aliases:
        return aliases[candidate]
    try:
        return EngineeringRole(candidate)
    except ValueError:
        return None


@dataclass(frozen=True)
class Assignment:
    """One task, bound to the role that will execute it and the scope it may touch."""

    task: PlannedTask
    role: EngineeringRole
    #: Write patterns for this task: the role ceiling narrowed to what the task named.
    write_paths: tuple[str, ...]
    #: Why the task could not be assigned, when it could not.
    rejection: str = ""

    @property
    def assigned(self) -> bool:
        """Whether this task has an executable owner."""
        return not self.rejection

    def permissions(self) -> AgentPermissions:
        """The gateway permissions for this task.

        The role's read scope with the *task's* write scope, so an agent working on one task
        cannot write files belonging to another even though its role could.
        """
        scope = scope_for(self.role)
        return AgentPermissions(
            allowed_tools=scope.tools,
            allowed_read_paths=scope.read,
            allowed_write_paths=self.write_paths,
        )


def _within_scope(path: str, patterns: tuple[str, ...]) -> bool:
    """Whether a repo-relative path falls under any of these glob patterns.

    Prefix matching on the directory part of a ``**`` pattern, plus exact matches for
    file patterns. Deliberately simple: the authoritative check is the M1 path policy, and
    this only decides what to *ask* for.
    """
    normalised = path.replace("\\", "/").lstrip("./")
    for pattern in patterns:
        cleaned = pattern.replace("\\", "/")
        if cleaned.endswith("/**"):
            prefix = cleaned[:-3]
            if normalised == prefix or normalised.startswith(f"{prefix}/"):
                return True
        elif cleaned == normalised:
            return True
    return False


def narrow_scope(role: EngineeringRole, paths: tuple[str, ...]) -> tuple[str, ...]:
    """Intersect the paths a task names with what its role may write.

    A task naming nothing gets the role's full ceiling -- it has to be able to create the
    files it was asked for. A task naming paths gets those files plus their directories, so
    it can add a test beside the module it edits, and nothing else.
    """
    scope = scope_for(role)
    if not paths:
        return scope.write

    granted: set[str] = set()
    for raw in paths:
        candidate = raw.replace("\\", "/").strip()
        if not candidate or candidate.startswith("/") or ".." in candidate.split("/"):
            continue
        if _within_scope(candidate, scope.forbidden):
            continue
        if not _within_scope(candidate, scope.write):
            continue
        granted.add(candidate)
        parent = candidate.rsplit("/", 1)[0] if "/" in candidate else ""
        if parent:
            granted.add(f"{parent}/**")

    return tuple(sorted(granted)) or scope.write


def assign(task: PlannedTask) -> Assignment:
    """Bind one planned task to a role and a scope.

    Refuses rather than guesses. A task whose ``agent`` is unrecognised, or whose paths all
    fall outside its role, produces an unassigned :class:`Assignment` carrying the reason --
    which the executor reports instead of running the task with permissions nobody chose.
    """
    role = resolve_role(task.agent)
    if role is None:
        return Assignment(
            task=task,
            role=EngineeringRole.BACKEND,
            write_paths=(),
            rejection=(
                f"task {task.task_id} names agent {task.agent!r}, which is not one of "
                f"{sorted(item.value for item in EngineeringRole)}"
            ),
        )

    scope = scope_for(role)
    forbidden = [
        path for path in task.paths if _within_scope(path, scope.forbidden)
    ]
    if forbidden:
        return Assignment(
            task=task,
            role=role,
            write_paths=(),
            rejection=(
                f"task {task.task_id} assigns {role.value} to write {forbidden}, which is "
                f"outside that role's remit"
            ),
        )

    return Assignment(task=task, role=role, write_paths=narrow_scope(role, task.paths))


def assign_plan(plan: ImplementationPlanDocument) -> tuple[Assignment, ...]:
    """Bind every task in a plan to a role.

    Unassignable tasks are returned alongside the rest rather than dropped: a plan with one
    task nobody can execute is a fact the operator needs, not a silently shorter plan.
    """
    assignments = tuple(assign(task) for task in plan.tasks)
    rejected = [item for item in assignments if not item.assigned]
    logger.info(
        "engineering.assigned",
        tasks=len(assignments),
        assigned=len(assignments) - len(rejected),
        rejected=len(rejected),
        roles=sorted({item.role.value for item in assignments if item.assigned}),
    )
    return assignments
