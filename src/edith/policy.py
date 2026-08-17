"""Failure classification policy: what to do when something goes wrong.

Reuses M0's :class:`~edith.errors.FailureCategory` rather than introducing a second
taxonomy. This module answers the three questions the orchestrator asks about every failure:

1. Can retrying the identical operation help?
2. Should a repair attempt (the Debugger) be made?
3. Must a human be told immediately?

The single most important rule here: **a security failure is never retried.** A denied path
or a blocked command is a policy decision, not a transient fault, and retrying it is
indistinguishable from an agent probing the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edith.errors import FailureCategory


class FailureAction(StrEnum):
    """What the orchestrator should do next after a failure."""

    #: Re-run the identical operation; the fault looked transient.
    RETRY = "RETRY"
    #: Hand the failure to the Debugging Agent for a minimal fix.
    REPAIR = "REPAIR"
    #: Stop this task and report. No further automated attempt.
    ESCALATE = "ESCALATE"
    #: Stop the entire execution immediately.
    ABORT = "ABORT"


@dataclass(frozen=True)
class FailureRule:
    """Policy for one failure category."""

    action: FailureAction
    #: Whether a human should be told even if the loop recovers.
    escalate_to_human: bool = False
    reason: str = ""

    @property
    def retryable(self) -> bool:
        """Whether the identical operation may be attempted again."""
        return self.action is FailureAction.RETRY

    @property
    def repairable(self) -> bool:
        """Whether the Debugging Agent should attempt a fix."""
        return self.action is FailureAction.REPAIR


#: The policy table. Every category in the taxonomy has an explicit entry -- an unmapped
#: category would silently inherit some default, which is exactly how a security failure
#: ends up being retried.
FAILURE_POLICY: dict[FailureCategory, FailureRule] = {
    FailureCategory.MODEL_ERROR: FailureRule(
        FailureAction.RETRY,
        reason="Local inference can fail transiently; a second attempt is cheap.",
    ),
    FailureCategory.VALIDATION_FAILURE: FailureRule(
        FailureAction.RETRY,
        reason="A small model often produces malformed output once and valid output next.",
    ),
    FailureCategory.TIMEOUT: FailureRule(
        FailureAction.RETRY,
        reason="A timeout may reflect momentary resource pressure on a constrained machine.",
    ),
    FailureCategory.TEST_FAILURE: FailureRule(
        FailureAction.REPAIR,
        reason="Failing tests are the Debugger's actual job, not a reason to give up.",
    ),
    FailureCategory.BUILD_ERROR: FailureRule(
        FailureAction.REPAIR,
        reason="A broken build is a concrete, localizable defect.",
    ),
    FailureCategory.TOOL_ERROR: FailureRule(
        FailureAction.REPAIR,
        reason="A tool refusing a bad argument usually means the agent's plan needs fixing.",
    ),
    FailureCategory.REQUIREMENT_FAILURE: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason="The request or plan is wrong; more attempts cannot discover the intent.",
    ),
    FailureCategory.ARCHITECTURE_FAILURE: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason="An architectural conflict needs a human decision, not another attempt.",
    ),
    FailureCategory.SECURITY_FAILURE: FailureRule(
        FailureAction.ABORT,
        escalate_to_human=True,
        reason=(
            "A denied operation is a policy decision. Retrying is indistinguishable from "
            "an agent probing the sandbox, so the execution stops and a human is told."
        ),
    ),
    FailureCategory.DEPENDENCY_FAILURE: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason=(
            "A missing package is not a coding defect. Sending the Debugger after code "
            "that never imported wastes the whole repair budget on a guess."
        ),
    ),
    FailureCategory.CODE_FAILURE: FailureRule(
        FailureAction.REPAIR,
        reason="A syntax or import error in the project's own code is precisely repairable.",
    ),
    FailureCategory.ENVIRONMENT_FAILURE: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason="A missing model or unreachable runtime needs an operator, not a retry.",
    ),
    FailureCategory.CONFIGURATION_ERROR: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason="Configuration cannot fix itself.",
    ),
    FailureCategory.UNKNOWN: FailureRule(
        FailureAction.ESCALATE,
        escalate_to_human=True,
        reason="An unclassified failure is reported rather than guessed at.",
    ),
}


def rule_for(category: FailureCategory | None) -> FailureRule:
    """Return the policy for a failure category.

    An absent or unrecognised category is treated as ``UNKNOWN`` and escalated, never
    silently retried.
    """
    if category is None:
        return FAILURE_POLICY[FailureCategory.UNKNOWN]
    return FAILURE_POLICY.get(category, FAILURE_POLICY[FailureCategory.UNKNOWN])


def decide(
    category: FailureCategory | None, *, attempts: int, max_attempts: int
) -> FailureAction:
    """Choose the next action, honouring the attempt budget.

    Args:
        category: Classification of the failure that just occurred.
        attempts: Attempts already spent on this task.
        max_attempts: Configured ceiling.

    Returns:
        The action to take. RETRY and REPAIR both degrade to ESCALATE once the budget is
        exhausted, so no loop can spin forever. ABORT is never downgraded.
    """
    rule = rule_for(category)
    if rule.action is FailureAction.ABORT:
        return FailureAction.ABORT
    if attempts >= max_attempts:
        return FailureAction.ESCALATE
    return rule.action


def is_security_failure(category: FailureCategory | None) -> bool:
    """Whether this failure represents a violated security boundary."""
    return category is FailureCategory.SECURITY_FAILURE
