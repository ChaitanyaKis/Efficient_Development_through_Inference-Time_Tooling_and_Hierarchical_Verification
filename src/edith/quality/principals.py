"""The quality layer's principals, and the invariant that keeps them separate.

M2.1 established that the thing which writes code must not be the thing which decides the code
works. M6 extends that to four more roles, and the extension is where it gets easy to get
wrong: each new agent needs *some* access, and the natural way to grant it is to copy an
existing permission set and add to it. Do that twice and a reviewer holds the coder's writes.

So the permission sets are declared here, once, and the relationships between them are
asserted by tests rather than trusted. Two properties matter:

**No principal is a superset of another.** If REVIEWER's tools were a superset of CODER's,
"the reviewer cannot modify what it judges" would be false while every individual grant still
looked reasonable in isolation.

**Write scope is disjoint where it must be.** TESTER writes tests, CODER writes
implementation, and neither reaches the other's files. That is what stops a testing agent from
"fixing" a failure by editing the code under test -- the failure mode M6 item 2 names.

JUDGE holds nothing at all beyond reading. It has no shell, no writes, and no git, because a
judge that can run a command can change what it is judging.
"""

from __future__ import annotations

from enum import StrEnum

from edith.schemas.agent import AgentPermissions

#: Paths a testing agent may write. Tests only -- never implementation.
TEST_SCOPE: tuple[str, ...] = ("tests/**", "test/**", "**/test_*.py", "**/*_test.py")

#: Paths an implementation agent may write. The engineering roles narrow this further per
#: task; this is the ceiling that separates code from tests.
IMPLEMENTATION_SCOPE: tuple[str, ...] = ("src/**", "lib/**", "app/**")


class Principal(StrEnum):
    """Who is acting. Recorded in the audit log and used to select permissions."""

    CODER = "coder"
    VERIFIER = "verifier"
    TESTER = "tester"
    SECURITY = "security"
    REVIEWER = "reviewer"
    JUDGE = "judge"


#: Reads code and runs the configured checks. Inherited from M5: shell to run tests, no writes.
VERIFIER = AgentPermissions(
    allowed_tools=frozenset({"shell.run", "filesystem.read"}),
    allowed_read_paths=("**",),
)

#: Writes tests, and runs them. Cannot touch implementation: a tester that could edit the code
#: under test can make any suite green.
TESTER = AgentPermissions(
    allowed_tools=frozenset(
        {"filesystem.read", "filesystem.search", "filesystem.write", "shell.run"}
    ),
    allowed_read_paths=("**",),
    allowed_write_paths=TEST_SCOPE,
)

#: Reads everything, writes nothing, runs nothing. A security agent that could execute is a
#: security agent that could be turned into a payload; deterministic scanners are invoked by
#: the pipeline on its behalf, not by it.
SECURITY = AgentPermissions(
    allowed_tools=frozenset({"filesystem.read", "filesystem.search"}),
    allowed_read_paths=("**",),
)

#: Reads code and its history to review it. No writes, no shell.
REVIEWER = AgentPermissions(
    allowed_tools=frozenset({"filesystem.read", "filesystem.search", "git.diff"}),
    allowed_read_paths=("**",),
)

#: Reads the evidence it is adjudicating. Nothing else, deliberately.
JUDGE = AgentPermissions(
    allowed_tools=frozenset({"filesystem.read"}),
    allowed_read_paths=("**",),
)

#: Every quality principal, for the isolation properties to be asserted over.
QUALITY_PERMISSIONS: dict[Principal, AgentPermissions] = {
    Principal.VERIFIER: VERIFIER,
    Principal.TESTER: TESTER,
    Principal.SECURITY: SECURITY,
    Principal.REVIEWER: REVIEWER,
    Principal.JUDGE: JUDGE,
}

#: Tools that mutate state. No quality principal may hold any of these.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {"git.commit", "git.branch", "git.worktree", "filesystem.patch"}
)


def may_write(permissions: AgentPermissions) -> bool:
    """Whether a principal can write anything at all."""
    return bool(permissions.allowed_write_paths) and (
        "filesystem.write" in permissions.allowed_tools
        or "filesystem.patch" in permissions.allowed_tools
    )


def may_execute(permissions: AgentPermissions) -> bool:
    """Whether a principal can spawn a process."""
    return "shell.run" in permissions.allowed_tools


def may_mutate_git(permissions: AgentPermissions) -> bool:
    """Whether a principal can change repository state."""
    return bool(permissions.allowed_tools & MUTATING_TOOLS)
