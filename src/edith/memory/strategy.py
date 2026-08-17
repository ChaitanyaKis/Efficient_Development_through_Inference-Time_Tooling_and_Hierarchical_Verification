"""When memory should be retrieved, and how relevant it must be.

M3 measured naive always-inject memory against a no-memory baseline on the multi-defect
repair benchmark and found it made a 3B model *worse*: 5/6 down to 0/6. The mechanism is
context pressure -- roughly 1800 characters of lessons competing with the code itself inside
an 8192-token window.

The conclusion was not that memory is useless but that **injecting it everywhere is the
wrong integration**. This module makes the integration a policy decision with named,
measurable alternatives rather than an implicit always-on behaviour.

A strategy answers two questions at each point where memory *could* be retrieved:

1. Does memory apply here at all?
2. How relevant must it be to earn its place in the prompt?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RetrievalPoint(StrEnum):
    """Where in the loop memory retrieval is being considered.

    The distinction matters because the cost is not uniform. On a first attempt the model
    has the code and the task and needs nothing else; after a failure it has a concrete
    error, which is both a better retrieval key and a moment where prior experience is
    plausibly worth its context cost.
    """

    #: First implementation attempt; no failure has been observed yet.
    CODER_INITIAL = "coder_initial"
    #: A repair attempt, following at least one failed verification.
    CODER_REPAIR = "coder_repair"
    #: The Debugging Agent, diagnosing a concrete failure.
    DEBUGGER = "debugger"


class MemoryStrategy(StrEnum):
    """How memory is integrated into the loop."""

    #: No retrieval anywhere. The control arm.
    NONE = "none"
    #: Retrieve at every point. The M3 behaviour that measured worse than the control.
    ALWAYS = "always"
    #: Retrieve only after a failure, for both the repair attempt and the debugger.
    FAILURE_TRIGGERED = "failure_triggered"
    #: Retrieve only for the Debugger, leaving every coder prompt untouched.
    DEBUGGER_ONLY = "debugger_only"
    #: Retrieve anywhere, but only admit memories that clear a high relevance bar.
    HIGH_RELEVANCE = "high_relevance"


#: Score a memory must reach under :attr:`MemoryStrategy.HIGH_RELEVANCE`.
#:
#: Calibrated against :class:`~edith.memory.retrieval.LexicalRanker`: a filename or title
#: term match contributes 10-12 points, so this admits memories that share substantive
#: vocabulary with the task or error and rejects those scoring only on metadata.
HIGH_RELEVANCE_THRESHOLD = 14.0

#: Default floor for the other strategies. Relevance is always a gate -- the ranker returns
#: zero when nothing ties a memory to the query -- so this is a second, softer filter.
DEFAULT_SCORE_FLOOR = 0.0


@dataclass(frozen=True)
class StrategyPolicy:
    """The concrete behaviour of one strategy."""

    strategy: MemoryStrategy
    points: frozenset[RetrievalPoint]
    min_score: float = DEFAULT_SCORE_FLOOR
    #: Maximum memories admitted at one point, overriding the configured budget downward
    #: when a strategy wants to be stingier than the global setting.
    max_memories: int | None = None

    def applies_at(self, point: RetrievalPoint) -> bool:
        """Whether this strategy retrieves memory at ``point``."""
        return point in self.points

    @property
    def retrieves_anything(self) -> bool:
        """Whether the strategy ever retrieves."""
        return bool(self.points)


_ALL_POINTS = frozenset(
    {RetrievalPoint.CODER_INITIAL, RetrievalPoint.CODER_REPAIR, RetrievalPoint.DEBUGGER}
)
_AFTER_FAILURE = frozenset({RetrievalPoint.CODER_REPAIR, RetrievalPoint.DEBUGGER})

#: The policy table. Every strategy is explicit; there is no implicit default behaviour.
POLICIES: dict[MemoryStrategy, StrategyPolicy] = {
    MemoryStrategy.NONE: StrategyPolicy(MemoryStrategy.NONE, frozenset()),
    MemoryStrategy.ALWAYS: StrategyPolicy(MemoryStrategy.ALWAYS, _ALL_POINTS),
    MemoryStrategy.FAILURE_TRIGGERED: StrategyPolicy(
        MemoryStrategy.FAILURE_TRIGGERED, _AFTER_FAILURE
    ),
    MemoryStrategy.DEBUGGER_ONLY: StrategyPolicy(
        MemoryStrategy.DEBUGGER_ONLY, frozenset({RetrievalPoint.DEBUGGER})
    ),
    MemoryStrategy.HIGH_RELEVANCE: StrategyPolicy(
        MemoryStrategy.HIGH_RELEVANCE,
        _ALL_POINTS,
        min_score=HIGH_RELEVANCE_THRESHOLD,
        # A memory good enough to clear the bar is worth showing; several are not, because
        # the point of the strategy is to spend as little context as possible.
        max_memories=2,
    ),
}


def policy_for(strategy: MemoryStrategy) -> StrategyPolicy:
    """Return the policy for a strategy, defaulting to no retrieval.

    An unrecognised strategy retrieves nothing rather than everything: the failure mode of
    injecting unexpectedly is measurably worse than the failure mode of injecting nothing.
    """
    return POLICIES.get(strategy, POLICIES[MemoryStrategy.NONE])
