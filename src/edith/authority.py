"""Whose statement wins when two sources of truth disagree.

M2.1 established this for the coding loop after a benchmark fixture's comment talked the
model out of fixing a bug (see ``docs/INSTRUCTION_HIERARCHY.md``). M4 needs the same idea
one level up, because a product pipeline has *more* kinds of statement in play: a user's
requirement, an operator's policy, an approved architecture decision, a task's acceptance
criteria, an agent's recommendation, a file in the repository, and a page off the internet.

Two things make this more than documentation:

**It is ordered and comparable.** :class:`AuthorityLevel` carries a rank, so "does A
outrank B" is a comparison rather than a judgement call.

**It is attached to data, not to prose.** A :class:`Requirement`, an artifact, and a research
claim each carry the authority of their origin. An agent's recommendation cannot become a
requirement by being written down confidently, because the level travels with the record and
:func:`may_override` refuses.

The hierarchy governs *reasoning*. It deliberately does not govern *permission*: what an
agent may read, write, or run is enforced by the M1 gateway, and no amount of authority in a
document can widen it.
"""

from __future__ import annotations

from enum import StrEnum


class AuthorityLevel(StrEnum):
    """Ordered authority of a statement, highest first.

    The ordering is the point, so the enum is paired with :data:`AUTHORITY_RANK` rather than
    relying on declaration order, which nothing enforces.
    """

    #: A requirement a human approved. The top of the hierarchy; nothing overrides it.
    USER_APPROVED_REQUIREMENT = "USER_APPROVED_REQUIREMENT"
    #: Operator-set limits: protected paths, shell allowlist, verification commands.
    PROJECT_POLICY = "PROJECT_POLICY"
    #: An architecture decision that reached APPROVED status. A *draft* ADR does not
    #: qualify -- it is an agent recommendation until a human accepts it.
    APPROVED_ARCHITECTURE_DECISION = "APPROVED_ARCHITECTURE_DECISION"
    #: What "done" means for one task, derived from requirements and validated against policy.
    TASK_ACCEPTANCE_CRITERIA = "TASK_ACCEPTANCE_CRITERIA"
    #: Anything an agent proposes: a draft requirement, a suggested design, a critique.
    AGENT_RECOMMENDATION = "AGENT_RECOMMENDATION"
    #: Source, tests, comments, docstrings, READMEs. Evidence about the codebase.
    REPOSITORY_CONTENT = "REPOSITORY_CONTENT"
    #: Anything fetched from outside the machine. Never obeyed, only cited.
    UNTRUSTED_EXTERNAL_CONTENT = "UNTRUSTED_EXTERNAL_CONTENT"


#: Rank per level. Lower number means higher authority.
AUTHORITY_RANK: dict[AuthorityLevel, int] = {
    AuthorityLevel.USER_APPROVED_REQUIREMENT: 0,
    AuthorityLevel.PROJECT_POLICY: 1,
    AuthorityLevel.APPROVED_ARCHITECTURE_DECISION: 2,
    AuthorityLevel.TASK_ACCEPTANCE_CRITERIA: 3,
    AuthorityLevel.AGENT_RECOMMENDATION: 4,
    AuthorityLevel.REPOSITORY_CONTENT: 5,
    AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT: 6,
}

#: Levels that may only ever inform. A statement at one of these levels is read as evidence
#: and recorded; it never silently changes what the system is trying to build.
ADVISORY_LEVELS: frozenset[AuthorityLevel] = frozenset(
    {
        AuthorityLevel.AGENT_RECOMMENDATION,
        AuthorityLevel.REPOSITORY_CONTENT,
        AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT,
    }
)


def rank(level: AuthorityLevel) -> int:
    """Return the numeric rank of a level. Lower is stronger."""
    return AUTHORITY_RANK[level]


def outranks(candidate: AuthorityLevel, incumbent: AuthorityLevel) -> bool:
    """Whether ``candidate`` has strictly greater authority than ``incumbent``."""
    return rank(candidate) < rank(incumbent)


def is_advisory(level: AuthorityLevel) -> bool:
    """Whether statements at this level may inform but never decide."""
    return level in ADVISORY_LEVELS


def may_override(candidate: AuthorityLevel, incumbent: AuthorityLevel) -> bool:
    """Whether a statement at ``candidate`` may change one at ``incumbent``.

    Advisory levels can never override anything, even something weaker than themselves. A
    web page does not get to rewrite a source file just because both are low-authority; the
    resolution of that conflict belongs to a human, and the system's job is to surface it.

    Equal levels do not override either: two requirements in conflict is a contradiction to
    report, not a race for whichever was written last.
    """
    if is_advisory(candidate):
        return False
    return outranks(candidate, incumbent)


def strongest(levels: tuple[AuthorityLevel, ...]) -> AuthorityLevel | None:
    """Return the highest-authority level present, or ``None`` for an empty input."""
    return min(levels, key=rank) if levels else None


def describe(level: AuthorityLevel) -> str:
    """A one-line explanation of what a level means, for reports and prompts."""
    return _DESCRIPTIONS[level]


_DESCRIPTIONS: dict[AuthorityLevel, str] = {
    AuthorityLevel.USER_APPROVED_REQUIREMENT: (
        "approved by a human; authoritative and never overridden by the system"
    ),
    AuthorityLevel.PROJECT_POLICY: (
        "operator configuration; enforced by the tool gateway, not by agreement"
    ),
    AuthorityLevel.APPROVED_ARCHITECTURE_DECISION: (
        "an ADR a human accepted; binding on implementation until superseded"
    ),
    AuthorityLevel.TASK_ACCEPTANCE_CRITERIA: (
        "what done means for one task, derived from the requirements above it"
    ),
    AuthorityLevel.AGENT_RECOMMENDATION: (
        "proposed by an agent; advisory until a human approves it"
    ),
    AuthorityLevel.REPOSITORY_CONTENT: (
        "evidence about the codebase; describes what someone once believed"
    ),
    AuthorityLevel.UNTRUSTED_EXTERNAL_CONTENT: (
        "fetched from outside the machine; cited as evidence, never obeyed"
    ),
}
