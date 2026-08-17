"""The Memory Governor: one place that decides what memory an execution may spend.

M3.1 measured the problem this module exists to fix. Per-prompt limits were respected
exactly as configured, and total injected memory still reached ~14,000 characters in a
single execution, because a repair loop retrieves again on every failure:

    retrieve (2k) -> repair -> fail -> retrieve (2k) -> repair -> fail -> retrieve (2k) ...

A budget that resets at every prompt is not a budget. So the accounting unit here is the
**execution**, not the prompt and not the agent.

Three properties follow from that, and each is a deliberate structural choice:

**Agents cannot reach the budget.** An agent receives rendered text and nothing else. The
budget lives on the governor, the governor lives on the orchestrator, and no agent input or
output schema has a field that names either. This is the same defence the Research Agent
uses against prompt injection: the capability is structurally absent, not merely
discouraged.

**It fails closed.** An exhausted budget injects nothing and says so. It never borrows
against a later prompt, never grows itself, and never retries. Memory is an optimisation;
the loop must remain able to finish without it, and M3.1 measured that it does — better,
in fact, than with it.

**Nothing is sent twice.** A memory already injected in this execution is referenced by id
on later prompts rather than re-sent in full. Re-spending 400 characters to repeat a lesson
the model has already been shown is the purest form of the waste M3.1 found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import Field

from edith.observability.logging import get_logger
from edith.schemas.common import EdithModel

from .retrieval import MemoryRetriever, RetrievalRequest
from .schema import MemoryType, ScoredMemory
from .strategy import MemoryStrategy, RetrievalPoint, policy_for

logger = get_logger(__name__)

#: Types offered to the autonomous loop. DECISION and TASK records are not excluded because
#: they are worthless, but because they are narrative rather than actionable, and this
#: budget is small enough that every admitted character has to earn its place.
LOOP_MEMORY_TYPES: tuple[MemoryType, ...] = (
    MemoryType.ENGINEERING,
    MemoryType.FAILURE,
    MemoryType.PROJECT,
)

#: Fixed characters in the memory template in :func:`_render`, including the newline that
#: joins it to the next entry. Counted, not guessed: an under-estimate would let an
#: injection overshoot the ceiling it was checked against.
_MEMORY_TEMPLATE_CHARS = len("- [] \n  \n  (source: )") + 1

#: Fixed characters in the reference line used for an already-injected memory.
_REFERENCE_TEMPLATE_CHARS = len("- (already provided earlier in this run at : [])") + 1


class GrantOutcome(StrEnum):
    """Why a memory request produced what it produced.

    Every path is named. "No memory was injected" has several very different causes, and a
    caller that cannot tell them apart cannot tell a working budget from a broken retriever.
    """

    #: Memory was retrieved and admitted.
    GRANTED = "GRANTED"
    #: The active strategy does not retrieve at this point in the loop.
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: Memory is switched off, or no store is attached.
    DISABLED = "DISABLED"
    #: Retrieval ran and nothing cleared the relevance gate.
    NOTHING_RELEVANT = "NOTHING_RELEVANT"
    #: Everything relevant had already been injected in this execution.
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"
    #: The execution's budget is spent. The defining fail-closed state.
    BUDGET_EXHAUSTED = "MEMORY_BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class MemoryBudgetLimits:
    """The ceilings one execution may not exceed.

    Frozen on purpose: limits are configuration, decided before the run starts. A mutable
    limit is a limit that something can raise mid-execution, which is the failure mode this
    whole module exists to prevent.
    """

    #: Total characters of memory this execution may ever inject.
    max_total_chars: int = 2400
    #: How many times memory may be retrieved at all.
    max_retrievals: int = 3
    #: Distinct memories that may be injected across the execution.
    max_total_memories: int = 4
    #: Ceiling for any single injection, so one prompt cannot swallow the execution.
    max_chars_per_retrieval: int = 1200
    max_memories_per_retrieval: int = 2

    def scaled(self, factor: float) -> MemoryBudgetLimits:
        """Return these limits scaled, for the budget-size ablation.

        Counts floor at 1 rather than 0: a "small" budget must still be a budget that can
        inject something, otherwise the arm is measuring the no-memory control twice.
        """
        return MemoryBudgetLimits(
            max_total_chars=max(int(self.max_total_chars * factor), 1),
            max_retrievals=max(int(self.max_retrievals * factor), 1),
            max_total_memories=max(int(self.max_total_memories * factor), 1),
            max_chars_per_retrieval=max(int(self.max_chars_per_retrieval * factor), 1),
            max_memories_per_retrieval=max(int(self.max_memories_per_retrieval * factor), 1),
        )


#: Named budgets for the ablation. Anchored to the measured reality rather than to round
#: numbers: M3.1 executions injected 6,000-14,000 characters, and the model's whole window
#: is 8,192 tokens, so even LARGE is far below what the unbudgeted arms actually spent.
BUDGET_PRESETS: dict[str, MemoryBudgetLimits] = {
    "small": MemoryBudgetLimits(
        max_total_chars=800,
        max_retrievals=1,
        max_total_memories=1,
        max_chars_per_retrieval=800,
        max_memories_per_retrieval=1,
    ),
    "medium": MemoryBudgetLimits(),
    "large": MemoryBudgetLimits(
        max_total_chars=4800,
        max_retrievals=6,
        max_total_memories=8,
        max_chars_per_retrieval=1600,
        max_memories_per_retrieval=3,
    ),
}

#: What an execution gets when nothing says otherwise.
DEFAULT_LIMITS = BUDGET_PRESETS["medium"]


@dataclass(frozen=True)
class InjectionRecord:
    """One memory, the first time it entered this execution's prompts."""

    memory_id: str
    title: str
    point: RetrievalPoint
    agent: str
    score: float
    reason: str
    chars: int


class ExecutionMemoryBudget:
    """The memory allowance for one execution.

    Counters are read-only properties over private state, and the only way to move them is
    :meth:`record_injection`, which the governor calls. Assigning to ``consumed_chars``
    raises ``AttributeError`` — the guarantee that "an agent cannot modify its own budget"
    is enforced by the type, not by a convention someone has to remember.
    """

    def __init__(
        self,
        execution_id: str,
        limits: MemoryBudgetLimits | None = None,
        *,
        consumed_chars: int = 0,
        retrieval_count: int = 0,
        injected: tuple[InjectionRecord, ...] = (),
    ) -> None:
        """
        Args:
            execution_id: The execution this budget belongs to.
            limits: Ceilings; the medium preset when omitted.
            consumed_chars: Characters already spent, when resuming an interrupted run.
            retrieval_count: Retrievals already made, when resuming.
            injected: Memories already injected, when resuming, so a restart does not
                re-send what the model was shown before the interruption.
        """
        self.execution_id = execution_id
        self.limits = limits or DEFAULT_LIMITS
        self._consumed_chars = consumed_chars
        self._retrieval_count = retrieval_count
        self._injected: dict[str, InjectionRecord] = {
            record.memory_id: record for record in injected
        }
        self._exhaustions = 0

    # -- Read-only accounting -------------------------------------------------------

    @property
    def consumed_chars(self) -> int:
        """Characters of memory injected so far in this execution."""
        return self._consumed_chars

    @property
    def retrieval_count(self) -> int:
        """Retrievals that produced an injection."""
        return self._retrieval_count

    @property
    def injected_memory_ids(self) -> frozenset[str]:
        """Every memory already shown to a model in this execution."""
        return frozenset(self._injected)

    @property
    def injections(self) -> tuple[InjectionRecord, ...]:
        """The injection ledger, in insertion order."""
        return tuple(self._injected.values())

    @property
    def exhaustions(self) -> int:
        """How many requests were refused because the budget was spent."""
        return self._exhaustions

    @property
    def remaining_chars(self) -> int:
        """Characters still available. Never negative."""
        return max(self.limits.max_total_chars - self._consumed_chars, 0)

    @property
    def remaining_retrievals(self) -> int:
        """Retrievals still available."""
        return max(self.limits.max_retrievals - self._retrieval_count, 0)

    @property
    def remaining_memories(self) -> int:
        """Distinct memories that may still be injected."""
        return max(self.limits.max_total_memories - len(self._injected), 0)

    @property
    def exhausted(self) -> bool:
        """Whether any dimension of the budget is spent.

        Any dimension, not all: a budget with characters left but no retrievals left is
        spent, and pretending otherwise is how a limit becomes advisory.
        """
        return (
            self.remaining_chars <= 0
            or self.remaining_retrievals <= 0
            or self.remaining_memories <= 0
        )

    def already_injected(self, memory_id: str) -> InjectionRecord | None:
        """Return the first injection of ``memory_id``, if it has been shown."""
        return self._injected.get(memory_id)

    def chars_allowed_now(self) -> int:
        """The largest injection permitted right now."""
        return min(self.limits.max_chars_per_retrieval, self.remaining_chars)

    def memories_allowed_now(self) -> int:
        """The most memories admissible in one injection right now."""
        return min(self.limits.max_memories_per_retrieval, self.remaining_memories)

    # -- Mutation, governor-only ----------------------------------------------------

    def record_injection(self, chars: int, records: tuple[InjectionRecord, ...]) -> None:
        """Charge an injection against the budget.

        Called by :class:`MemoryGovernor` and nothing else. Characters are charged even when
        ``records`` is empty, because a reference-only injection still occupies context.

        A negative charge is floored at zero rather than refunded: consumption only moves
        one way within an execution, so "charge minus a thousand characters" cannot become
        a budget reset with extra steps.
        """
        self._consumed_chars += max(chars, 0)
        self._retrieval_count += 1
        for record in records:
            self._injected.setdefault(record.memory_id, record)

    def record_exhaustion(self) -> None:
        """Note that a request was refused for lack of budget."""
        self._exhaustions += 1

    def snapshot(self) -> dict[str, int]:
        """Structured accounting, for logs and the experiment."""
        return {
            "consumed_chars": self._consumed_chars,
            "remaining_chars": self.remaining_chars,
            "retrievals": self._retrieval_count,
            "remaining_retrievals": self.remaining_retrievals,
            "memories": len(self._injected),
            "remaining_memories": self.remaining_memories,
            "exhaustions": self._exhaustions,
        }


class MemoryGrant(EdithModel):
    """The governor's answer to one request.

    Always returned, never raised. A caller that has to catch an exception to find out it
    got no memory will eventually forget to, and the loop must carry on regardless.
    """

    outcome: GrantOutcome
    #: Prompt text. Empty for every outcome except ``GRANTED``.
    text: str = ""
    memory_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    chars: int = 0
    #: Memories withheld because this execution had already been shown them.
    suppressed_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    remaining_chars: int = 0
    remaining_retrievals: int = 0

    @property
    def granted(self) -> bool:
        """Whether any text was produced."""
        return self.outcome is GrantOutcome.GRANTED and bool(self.text)

    @property
    def exhausted(self) -> bool:
        """Whether the request was refused for lack of budget."""
        return self.outcome is GrantOutcome.BUDGET_EXHAUSTED


@dataclass
class GovernorSettings:
    """What the governor needs to know that is not the budget itself."""

    strategy: MemoryStrategy = MemoryStrategy.NONE
    project_id: str | None = None
    min_confidence: float = 0.35
    include_global: bool = True
    #: Extra floor applied on top of the strategy's own threshold.
    min_score: float = 0.0
    types: tuple[MemoryType, ...] = field(default=LOOP_MEMORY_TYPES)


class MemoryGovernor:
    """The single gate every autonomous memory injection passes through.

    Enforces, in this order: the strategy's retrieval points, the execution budget, the
    relevance gate, the per-retrieval ceilings, duplicate suppression, and the character
    budget.

    Project isolation is enforced twice, on purpose. The governor supplies the execution's
    project scope itself — :meth:`request` takes no project id, so a caller cannot ask for
    another project's memory — and never widens the result. The store then enforces the same
    boundary in SQL. Either layer alone would hold; neither is trusted to.
    """

    def __init__(
        self,
        retriever: MemoryRetriever | None,
        budget: ExecutionMemoryBudget,
        settings: GovernorSettings,
    ) -> None:
        """
        Args:
            retriever: Source of candidates. ``None`` means memory is unavailable, which is
                a valid configuration and not an error.
            budget: This execution's allowance.
            settings: Strategy, project scope, and relevance floors.
        """
        self._retriever = retriever
        self.budget = budget
        self.settings = settings

    def request(
        self,
        *,
        execution_id: str,
        query: str,
        purpose: RetrievalPoint,
        error_text: str = "",
        paths: tuple[str, ...] = (),
        agent: str = "",
    ) -> MemoryGrant:
        """Ask for memory for one prompt.

        This is the only entry point the autonomous loop may use. It takes a *purpose*
        rather than a set of retrieval parameters, so a caller cannot quietly request more
        than its position in the loop entitles it to.

        Args:
            execution_id: Must match the budget's execution. A mismatch is refused rather
                than served, since it would spend one execution's allowance on another's.
            query: What the prompt is about.
            purpose: Where in the loop this request comes from.
            error_text: Real verification output, when the request follows a failure.
            paths: Files the task touches.
            agent: Requesting agent, recorded for accounting.
        """
        if execution_id != self.budget.execution_id:
            # Not a defensive nicety: budgets are per-execution, so serving this would let
            # one run consume another's allowance.
            return self._refuse(
                GrantOutcome.DISABLED,
                f"budget belongs to {self.budget.execution_id}, not {execution_id}",
            )

        policy = policy_for(self.settings.strategy)
        if not policy.applies_at(purpose):
            return self._refuse(
                GrantOutcome.NOT_APPLICABLE,
                f"strategy {self.settings.strategy} does not retrieve at {purpose}",
            )

        if self._retriever is None:
            return self._refuse(GrantOutcome.DISABLED, "no memory store is attached")

        if self.budget.exhausted:
            self.budget.record_exhaustion()
            logger.info(
                "memory.budget_exhausted",
                execution_id=execution_id,
                purpose=str(purpose),
                agent=agent,
                **self.budget.snapshot(),
            )
            return self._refuse(
                GrantOutcome.BUDGET_EXHAUSTED,
                "the execution's memory budget is spent; continuing without memory",
            )

        bundle = self._retriever.retrieve(
            RetrievalRequest(
                query=query,
                project_id=self.settings.project_id,
                types=self.settings.types,
                agent=agent or None,
                # Over-fetch by one so a request whose top hit is a duplicate can still be
                # served something new, without raising any ceiling.
                max_memories=self.budget.memories_allowed_now() + 1,
                max_chars=self.budget.chars_allowed_now(),
                min_confidence=self.settings.min_confidence,
                include_global=self.settings.include_global,
                error_text=error_text,
                paths=paths,
                min_score=max(policy.min_score, self.settings.min_score),
            )
        )
        if bundle.is_empty:
            return self._refuse(
                GrantOutcome.NOTHING_RELEVANT, "nothing cleared the relevance gate"
            )

        return self._admit(bundle.memories, purpose=purpose, agent=agent)

    # -- Internals -------------------------------------------------------------------

    def _admit(
        self,
        candidates: list[ScoredMemory],
        *,
        purpose: RetrievalPoint,
        agent: str,
    ) -> MemoryGrant:
        """Select, charge, and render what fits."""
        ordered = sorted(candidates, key=_priority_key)

        fresh: list[ScoredMemory] = []
        suppressed: list[str] = []
        for entry in ordered:
            if self.budget.already_injected(entry.memory.memory_id):
                suppressed.append(entry.memory.memory_id)
            else:
                fresh.append(entry)

        char_ceiling = self.budget.chars_allowed_now()
        memory_ceiling = self.budget.memories_allowed_now()

        selected: list[ScoredMemory] = []
        spent = 0
        for entry in fresh:
            if len(selected) >= memory_ceiling:
                break
            cost = _render_cost(entry)
            if spent + cost > char_ceiling:
                continue
            selected.append(entry)
            spent += cost

        # Referencing what was already shown costs a line each; charge for it honestly and
        # drop the references rather than the new content when the budget is tight.
        reference_ids: list[str] = []
        for memory_id in suppressed:
            previous = self.budget.already_injected(memory_id)
            if previous is None:  # pragma: no cover - suppressed implies a ledger entry
                continue
            cost = _reference_cost(previous)
            if spent + cost > char_ceiling:
                break
            reference_ids.append(memory_id)
            spent += cost

        if not selected and not reference_ids:
            return self._refuse(
                GrantOutcome.DUPLICATE_SUPPRESSED
                if suppressed
                else GrantOutcome.NOTHING_RELEVANT,
                "everything relevant has already been injected in this execution"
                if suppressed
                else "nothing fitted the remaining budget",
                suppressed_ids=suppressed,
            )

        records = tuple(
            InjectionRecord(
                memory_id=entry.memory.memory_id,
                title=entry.memory.title,
                point=purpose,
                agent=agent,
                score=entry.score,
                reason="; ".join(entry.reasons[:2]) or "general relevance",
                chars=_render_cost(entry),
            )
            for entry in selected
        )
        # Render before charging, and charge what was actually produced. The estimate above
        # decides what *fits*; the budget is spent on real characters, so ``memory_chars``
        # in a report means literal text that reached a prompt rather than an approximation
        # of it. The ledger must record what happened, not what was projected.
        text = _render(selected, reference_ids, self.budget)
        spent = len(text)
        self.budget.record_injection(spent, records)
        logger.info(
            "memory.granted",
            execution_id=self.budget.execution_id,
            purpose=str(purpose),
            agent=agent,
            memory_ids=[entry.memory.memory_id for entry in selected],
            scores=[entry.score for entry in selected],
            referenced=reference_ids,
            chars=spent,
            **self.budget.snapshot(),
        )
        return MemoryGrant(
            outcome=GrantOutcome.GRANTED,
            text=text,
            memory_ids=[entry.memory.memory_id for entry in selected],
            scores=[entry.score for entry in selected],
            chars=spent,
            suppressed_ids=suppressed,
            reason="; ".join(record.reason for record in records[:2]),
            remaining_chars=self.budget.remaining_chars,
            remaining_retrievals=self.budget.remaining_retrievals,
        )

    def _refuse(
        self,
        outcome: GrantOutcome,
        reason: str,
        *,
        suppressed_ids: list[str] | None = None,
    ) -> MemoryGrant:
        """Return an empty grant carrying why it is empty."""
        return MemoryGrant(
            outcome=outcome,
            reason=reason,
            suppressed_ids=suppressed_ids or [],
            remaining_chars=self.budget.remaining_chars,
            remaining_retrievals=self.budget.remaining_retrievals,
        )


def _priority_key(entry: ScoredMemory) -> tuple[float, float, int, str]:
    """Ordering for a limited budget.

    The ranker has already weighted failure and component relevance above task-text overlap,
    so score leads. The remaining terms break ties in the order the milestone specifies:
    better-evidenced memories first, then observed failures over general lessons, then the
    id so the ordering is stable and the experiment is reproducible.
    """
    type_rank = {
        MemoryType.FAILURE: 0,
        MemoryType.ENGINEERING: 1,
        MemoryType.PROJECT: 2,
    }.get(entry.memory.type, 3)
    return (-entry.score, -entry.memory.confidence, type_rank, entry.memory.memory_id)


def _render_cost(entry: ScoredMemory) -> int:
    """Exact characters one memory will occupy once rendered.

    Counted from the template in :func:`_render` rather than approximated. An estimate that
    runs under the truth lets an injection overshoot the ceiling it was checked against,
    which would make the budget advisory.
    """
    record = entry.memory
    return (
        len(str(record.type))
        + len(record.title)
        + len(record.content)
        + len(record.provenance)
        + _MEMORY_TEMPLATE_CHARS
    )


def _reference_cost(previous: InjectionRecord) -> int:
    """Exact characters a repeat costs when referenced instead of re-sent."""
    return (
        len(previous.memory_id)
        + len(previous.title)
        + len(str(previous.point))
        + _REFERENCE_TEMPLATE_CHARS
    )


def _render(
    selected: list[ScoredMemory],
    reference_ids: list[str],
    budget: ExecutionMemoryBudget,
) -> str:
    """Render admitted memories, plus id-only references to repeats."""
    lines: list[str] = []
    for entry in selected:
        record = entry.memory
        lines.append(
            f"- [{record.type}] {record.title}\n"
            f"  {record.content}\n"
            f"  (source: {record.provenance})"
        )
    for memory_id in reference_ids:
        previous = budget.already_injected(memory_id)
        if previous is None:  # pragma: no cover - only reachable if the ledger is edited
            continue
        lines.append(
            f"- (already provided earlier in this run at {previous.point}: "
            f"{previous.title} [{memory_id}])"
        )
    return "\n".join(lines)
