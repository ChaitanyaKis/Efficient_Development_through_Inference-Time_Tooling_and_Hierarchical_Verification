"""The engineering execution layer: turning an implementation plan into software.

M4 produces a plan. M5 executes it, through five specialised agents whose boundaries are
enforced by the M1 permission engine rather than by prompt instruction:

============  ===========================================================
role          owns
============  ===========================================================
frontend      the user interface, against the UX specification
backend       API endpoints, services, business logic
database      schema and migrations
devops        containers, CI, local development scripts
dependency    manifests and reproducible install artifacts
============  ===========================================================

Nothing here loosens an M2 guarantee. The agents emit the same ``ModelEdits`` the M2 coder
does, so the syntax gate, symbol-preservation check, scope enforcement and diff all apply
unchanged; verification and repair remain M2's; and no engineering agent holds ``shell.run``
or any git tool.
"""

from .agents import (
    ENGINEERING_AGENTS,
    BackendAgent,
    DatabaseAgent,
    DependencyAgent,
    DevOpsAgent,
    EngineeringInput,
    FrontendAgent,
    agent_for,
    permissions_for,
)
from .conflicts import ConflictKind, TaskConflict, detect_conflicts, serialise
from .dependency import (
    ProvisionOutcome,
    ReconciliationResult,
    provision_environment,
    reconcile,
    verify_imports,
)
from .ownership import (
    ROLE_SCOPES,
    Assignment,
    EngineeringRole,
    RoleScope,
    assign,
    assign_plan,
    narrow_scope,
    resolve_role,
    scope_for,
)

__all__ = [
    "ENGINEERING_AGENTS",
    "ROLE_SCOPES",
    "Assignment",
    "BackendAgent",
    "ConflictKind",
    "DatabaseAgent",
    "DependencyAgent",
    "DevOpsAgent",
    "EngineeringInput",
    "EngineeringRole",
    "FrontendAgent",
    "ProvisionOutcome",
    "ReconciliationResult",
    "RoleScope",
    "TaskConflict",
    "agent_for",
    "assign",
    "assign_plan",
    "detect_conflicts",
    "narrow_scope",
    "permissions_for",
    "provision_environment",
    "reconcile",
    "resolve_role",
    "scope_for",
    "serialise",
    "verify_imports",
]
