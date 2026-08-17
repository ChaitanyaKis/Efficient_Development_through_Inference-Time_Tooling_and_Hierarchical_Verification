"""M3.2: the execution memory budget and the governor that enforces it.

M3.1 measured the defect these tests pin down: per-prompt limits were honoured exactly and
total injected memory still reached ~14,000 characters, because a repair loop retrieves
again after every failure. The unit of accounting here is the execution.

Everything below is deterministic and offline. No model is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.memory.governor import (
    BUDGET_PRESETS,
    DEFAULT_LIMITS,
    ExecutionMemoryBudget,
    GovernorSettings,
    GrantOutcome,
    InjectionRecord,
    MemoryBudgetLimits,
    MemoryGovernor,
)
from edith.memory.retrieval import MemoryRetriever
from edith.memory.schema import (
    MemoryProposal,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from edith.memory.store import MemoryStore, open_memory
from edith.memory.strategy import MemoryStrategy, RetrievalPoint

EXECUTION = "exec_test"

#: Worded to overlap a concrete failure, which is how failure-triggered retrieval finds it.
SORTING_LESSON = (
    "When an assertion compares a result against an ordered list, the returned value must "
    "also be sorted, or the assertion fails on ordering alone."
)
THRESHOLD_LESSON = (
    "An inclusive threshold comparison must use a boundary operator that admits the "
    "boundary value itself, or items exactly at the limit are dropped."
)


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    with open_memory(tmp_path / "memory") as opened:
        yield opened


def seed(
    store: MemoryStore,
    title: str,
    content: str,
    *,
    project_id: str | None = "proj_a",
    memory_type: MemoryType = MemoryType.ENGINEERING,
    source: MemorySource = MemorySource.TEST_RESULT,
) -> str:
    """Store a lesson through the same validation gate as any other memory."""
    record, outcome = store.propose(
        MemoryProposal(
            type=memory_type,
            scope=MemoryScope.GLOBAL if project_id is None else MemoryScope.PROJECT,
            project_id=project_id,
            title=title,
            content=content,
            source=source,
            source_reference="tests/test_memory_governor.py",
        )
    )
    assert record is not None, outcome.reason
    return record.memory_id


def build_governor(
    store: MemoryStore,
    *,
    limits: MemoryBudgetLimits | None = None,
    strategy: MemoryStrategy = MemoryStrategy.ALWAYS,
    project_id: str | None = "proj_a",
    budget: ExecutionMemoryBudget | None = None,
) -> MemoryGovernor:
    """A governor over a real store, with an explicit allowance."""
    return MemoryGovernor(
        MemoryRetriever(store),
        budget or ExecutionMemoryBudget(EXECUTION, limits or DEFAULT_LIMITS),
        GovernorSettings(strategy=strategy, project_id=project_id),
    )


def ask(
    governor: MemoryGovernor,
    query: str = "the returned value was not sorted",
    *,
    purpose: RetrievalPoint = RetrievalPoint.DEBUGGER,
    error_text: str = "AssertionError: assert ['b', 'a'] == ['a', 'b'] not sorted",
    agent: str = "debugger",
):
    """Make one governed request."""
    return governor.request(
        execution_id=EXECUTION,
        query=query,
        purpose=purpose,
        error_text=error_text,
        agent=agent,
    )


class TestBudgetAccounting:
    def test_a_fresh_budget_has_its_full_allowance(self) -> None:
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        assert budget.remaining_chars == DEFAULT_LIMITS.max_total_chars
        assert budget.remaining_retrievals == DEFAULT_LIMITS.max_retrievals
        assert not budget.exhausted
        assert budget.consumed_chars == 0

    def test_consumption_accumulates_across_retrievals(self) -> None:
        """The property per-prompt limits could not provide."""
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        for _ in range(3):
            budget.record_injection(500, ())
        assert budget.consumed_chars == 1500
        assert budget.retrieval_count == 3

    def test_any_spent_dimension_exhausts_the_budget(self) -> None:
        """Characters left but no retrievals left is still spent."""
        limits = MemoryBudgetLimits(max_total_chars=10_000, max_retrievals=1)
        budget = ExecutionMemoryBudget(EXECUTION, limits)
        budget.record_injection(10, ())
        assert budget.remaining_chars > 0
        assert budget.exhausted

    def test_remaining_never_goes_negative(self) -> None:
        budget = ExecutionMemoryBudget(EXECUTION, MemoryBudgetLimits(max_total_chars=100))
        budget.record_injection(500, ())
        assert budget.remaining_chars == 0

    def test_the_per_retrieval_ceiling_shrinks_with_the_remaining_budget(self) -> None:
        limits = MemoryBudgetLimits(max_total_chars=1000, max_chars_per_retrieval=800)
        budget = ExecutionMemoryBudget(EXECUTION, limits)
        assert budget.chars_allowed_now() == 800
        budget.record_injection(600, ())
        assert budget.chars_allowed_now() == 400


class TestTheBudgetCannotBeModified:
    """Acceptance criterion 4: memory cannot modify its own budget."""

    @pytest.mark.parametrize(
        "attribute", ["consumed_chars", "retrieval_count", "remaining_chars", "exhausted"]
    )
    def test_counters_are_read_only(self, attribute: str) -> None:
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        with pytest.raises(AttributeError):
            setattr(budget, attribute, 0)

    def test_limits_are_frozen(self) -> None:
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
            budget.limits.max_total_chars = 10**9  # type: ignore[misc]

    def test_the_budget_has_no_reset_path(self) -> None:
        """Consumption only ever moves one way within an execution."""
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        budget.record_injection(500, ())
        assert not hasattr(budget, "reset")
        assert not hasattr(budget, "refund")
        budget.record_injection(0, ())
        assert budget.consumed_chars == 500

    def test_a_negative_injection_cannot_refund_the_budget(self) -> None:
        """Charging -1000 characters would be a reset with extra steps."""
        budget = ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS)
        budget.record_injection(1000, ())
        budget.record_injection(-900, ())
        assert budget.consumed_chars >= 1000

    def test_no_agent_module_can_reach_the_memory_subsystem(self) -> None:
        """The structural guarantee: agents have no import path to memory at all.

        The same defence the Research Agent uses against prompt injection -- the capability
        is absent, not merely discouraged.
        """
        agents_dir = Path(__file__).resolve().parents[1] / "src" / "edith" / "agents"
        offenders = [
            path.name
            for path in agents_dir.rglob("*.py")
            if "edith.memory" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_the_orchestrator_never_retrieves_outside_the_governor(self) -> None:
        """Acceptance criterion 2: every autonomous injection passes through the governor."""
        source = (
            Path(__file__).resolve().parents[1] / "src" / "edith" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        assert "_retriever.retrieve" not in source
        assert "_governor.request" in source


class TestGovernorEnforcement:
    def test_a_relevant_memory_is_granted(self, memory: MemoryStore) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        grant = ask(build_governor(memory))
        assert grant.outcome is GrantOutcome.GRANTED
        assert grant.granted
        assert "Ordering in assertions" in grant.text
        assert grant.chars > 0

    def test_the_strategy_still_decides_where_memory_applies(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(memory, strategy=MemoryStrategy.DEBUGGER_ONLY)
        grant = ask(governor, purpose=RetrievalPoint.CODER_INITIAL)
        assert grant.outcome is GrantOutcome.NOT_APPLICABLE
        assert grant.text == ""
        assert governor.budget.consumed_chars == 0, "a refusal must cost nothing"

    def test_an_irrelevant_memory_is_never_granted(self, memory: MemoryStore) -> None:
        """The relevance gate stays mandatory regardless of available budget."""
        seed(
            memory,
            "Kubernetes ingress annotations",
            "Ingress annotations must be namespaced under the controller prefix.",
        )
        grant = ask(build_governor(memory))
        assert grant.outcome is GrantOutcome.NOTHING_RELEVANT
        assert grant.text == ""

    def test_another_projects_memory_is_invisible(self, memory: MemoryStore) -> None:
        """M3 isolation is preserved: the governor never widens what the store returns."""
        seed(memory, "Ordering in assertions", SORTING_LESSON, project_id="proj_other")
        grant = ask(build_governor(memory, project_id="proj_a"))
        assert not grant.granted
        assert "Ordering" not in grant.text

    def test_a_grant_for_another_execution_is_refused(self, memory: MemoryStore) -> None:
        """A budget is one execution's allowance and cannot be spent by another."""
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(memory)
        grant = governor.request(
            execution_id="exec_someone_else",
            query="sorted",
            purpose=RetrievalPoint.DEBUGGER,
        )
        assert grant.outcome is GrantOutcome.DISABLED
        assert governor.budget.consumed_chars == 0

    def test_no_store_means_a_clean_refusal(self) -> None:
        governor = MemoryGovernor(
            None,
            ExecutionMemoryBudget(EXECUTION, DEFAULT_LIMITS),
            GovernorSettings(strategy=MemoryStrategy.ALWAYS),
        )
        grant = ask(governor)
        assert grant.outcome is GrantOutcome.DISABLED
        assert grant.text == ""

    def test_the_per_retrieval_memory_ceiling_is_enforced(
        self, memory: MemoryStore
    ) -> None:
        for index in range(6):
            seed(memory, f"Ordering in assertions {index}", SORTING_LESSON)
        limits = MemoryBudgetLimits(
            max_total_chars=100_000, max_memories_per_retrieval=2, max_total_memories=10
        )
        grant = ask(build_governor(memory, limits=limits))
        assert len(grant.memory_ids) <= 2


class TestFailClosed:
    """Acceptance criterion 3, and the behaviour the whole milestone turns on."""

    def test_an_exhausted_budget_injects_nothing(self, memory: MemoryStore) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        limits = MemoryBudgetLimits(max_total_chars=1, max_retrievals=1)
        governor = build_governor(memory, limits=limits)
        governor.budget.record_injection(1, ())

        grant = ask(governor)
        assert grant.outcome is GrantOutcome.BUDGET_EXHAUSTED
        assert grant.text == ""
        assert grant.exhausted

    def test_exhaustion_is_reported_as_a_named_state(self, memory: MemoryStore) -> None:
        """The structured MEMORY_BUDGET_EXHAUSTED result the milestone specifies."""
        assert GrantOutcome.BUDGET_EXHAUSTED.value == "MEMORY_BUDGET_EXHAUSTED"

    def test_exhaustion_does_not_silently_grow_the_budget(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        limits = MemoryBudgetLimits(max_total_chars=50, max_retrievals=1)
        governor = build_governor(memory, limits=limits)
        governor.budget.record_injection(50, ())

        for _ in range(5):
            assert ask(governor).outcome is GrantOutcome.BUDGET_EXHAUSTED

        assert governor.budget.limits.max_total_chars == 50
        assert governor.budget.consumed_chars == 50, "refusals must not consume budget"
        assert governor.budget.exhaustions == 5

    def test_exhaustion_never_raises(self, memory: MemoryStore) -> None:
        """A caller that must catch an exception to continue will eventually forget to."""
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory, limits=MemoryBudgetLimits(max_total_chars=0, max_retrievals=0)
        )
        grant = ask(governor)
        assert grant.outcome is GrantOutcome.BUDGET_EXHAUSTED


class TestDuplicateSuppression:
    """Acceptance criterion 5. Re-sending a lesson the model has seen is pure waste."""

    def test_the_same_memory_is_not_injected_twice_in_full(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory, limits=MemoryBudgetLimits(max_total_chars=100_000, max_retrievals=5)
        )

        first = ask(governor)
        second = ask(governor)

        assert first.granted
        assert SORTING_LESSON in first.text
        assert SORTING_LESSON not in second.text
        assert second.outcome in {
            GrantOutcome.DUPLICATE_SUPPRESSED,
            GrantOutcome.GRANTED,
        }

    def test_a_repeat_is_referenced_by_id_not_re_sent(self, memory: MemoryStore) -> None:
        memory_id = seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory, limits=MemoryBudgetLimits(max_total_chars=100_000, max_retrievals=5)
        )
        first = ask(governor)
        second = ask(governor)

        assert memory_id in first.memory_ids
        assert memory_id in second.suppressed_ids
        assert second.chars < first.chars, "a reference must be cheaper than the content"

    def test_a_new_memory_still_reaches_a_later_prompt(self, memory: MemoryStore) -> None:
        """Suppression must not become "one retrieval and then nothing ever again"."""
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory,
            limits=MemoryBudgetLimits(
                max_total_chars=100_000,
                max_retrievals=5,
                max_total_memories=5,
                max_memories_per_retrieval=1,
            ),
        )
        ask(governor)
        seed(memory, "Inclusive thresholds", THRESHOLD_LESSON)

        later = governor.request(
            execution_id=EXECUTION,
            query="items exactly at the threshold boundary were dropped",
            purpose=RetrievalPoint.DEBUGGER,
            error_text="AssertionError: an item at the inclusive boundary was missing",
            agent="debugger",
        )
        assert later.granted
        assert "Inclusive thresholds" in later.text

    def test_the_ledger_records_where_a_memory_was_first_shown(
        self, memory: MemoryStore
    ) -> None:
        memory_id = seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(memory)
        ask(governor, purpose=RetrievalPoint.DEBUGGER)

        record = governor.budget.already_injected(memory_id)
        assert record is not None
        assert record.point is RetrievalPoint.DEBUGGER
        assert record.title == "Ordering in assertions"


class TestPrioritisation:
    """Acceptance criterion 6: a limited budget buys the most relevant memory first."""

    def test_the_failure_relevant_memory_wins_a_single_slot(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "Inclusive thresholds", THRESHOLD_LESSON)
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory,
            limits=MemoryBudgetLimits(
                max_total_chars=100_000,
                max_memories_per_retrieval=1,
                max_total_memories=1,
            ),
        )
        grant = ask(governor)
        assert "Ordering in assertions" in grant.text
        assert "Inclusive thresholds" not in grant.text

    def test_a_better_evidenced_memory_outranks_an_equally_relevant_one(
        self, memory: MemoryStore
    ) -> None:
        """Provenance breaks the tie, in the order the milestone specifies."""
        weak, _ = memory.propose(
            MemoryProposal(
                type=MemoryType.ENGINEERING,
                project_id="proj_a",
                title="Ordering in assertions",
                content=SORTING_LESSON,
                source=MemorySource.USER,
                source_reference="a person said so",
                confidence=0.40,
            )
        )
        assert weak is not None
        strong, _ = memory.propose(
            MemoryProposal(
                type=MemoryType.FAILURE,
                project_id="proj_a",
                title="Ordering in assertions",
                content=SORTING_LESSON,
                source=MemorySource.TEST_RESULT,
                source_reference="a test that ran",
            )
        )
        assert strong is not None

        governor = build_governor(
            memory,
            limits=MemoryBudgetLimits(
                max_total_chars=100_000,
                max_memories_per_retrieval=1,
                max_total_memories=1,
            ),
        )
        grant = ask(governor)
        assert grant.memory_ids == [strong.memory_id]


class TestRepairLoopAccumulation:
    """Acceptance criterion 7, measured directly: the M3.1 defect must not recur."""

    def seed_many(self, store: MemoryStore, count: int = 8) -> None:
        for index in range(count):
            seed(store, f"Ordering in assertions {index}", SORTING_LESSON)

    def test_an_unbudgeted_repair_loop_grows_without_bound(
        self, memory: MemoryStore
    ) -> None:
        """The M3.1 behaviour, reproduced so the fix has something to be measured against."""
        self.seed_many(memory)
        unbounded = MemoryBudgetLimits(
            max_total_chars=10**9,
            max_retrievals=10**6,
            max_total_memories=10**6,
            max_chars_per_retrieval=100_000,
            max_memories_per_retrieval=3,
        )
        governor = build_governor(memory, limits=unbounded)
        for _ in range(6):
            ask(governor)

        assert governor.budget.consumed_chars > 2000
        assert governor.budget.retrieval_count == 6

    def test_a_budgeted_repair_loop_stops_at_its_ceiling(
        self, memory: MemoryStore
    ) -> None:
        self.seed_many(memory)
        governor = build_governor(memory, limits=BUDGET_PRESETS["medium"])
        outcomes = [ask(governor).outcome for _ in range(6)]

        assert governor.budget.consumed_chars <= BUDGET_PRESETS["medium"].max_total_chars
        assert GrantOutcome.BUDGET_EXHAUSTED in outcomes
        assert governor.budget.retrieval_count <= BUDGET_PRESETS["medium"].max_retrievals

    def test_the_budget_bounds_cost_regardless_of_how_many_repairs_happen(
        self, memory: MemoryStore
    ) -> None:
        """Twenty failures must cost no more than three."""
        self.seed_many(memory)
        short = build_governor(memory, limits=BUDGET_PRESETS["medium"])
        for _ in range(3):
            ask(short)
        long_run = build_governor(memory, limits=BUDGET_PRESETS["medium"])
        for _ in range(20):
            ask(long_run)

        assert long_run.budget.consumed_chars == short.budget.consumed_chars

    @pytest.mark.parametrize("preset", ["small", "medium", "large"])
    def test_every_preset_bounds_an_unbounded_loop(
        self, memory: MemoryStore, preset: str
    ) -> None:
        self.seed_many(memory, count=20)
        limits = BUDGET_PRESETS[preset]
        governor = build_governor(memory, limits=limits)
        for _ in range(30):
            ask(governor)
        assert governor.budget.consumed_chars <= limits.max_total_chars


class TestExperimentPlumbing:
    """The arms must differ by the budget alone, or the comparison measures nothing."""

    def test_the_three_arms_isolate_the_budget(self) -> None:
        from edith.experiments import BUDGET_ARMS

        names = [name for name, _, _, _ in BUDGET_ARMS]
        assert names == ["A_no_memory", "B_debugger_unbudgeted", "C_debugger_budgeted"]

        _, strategy_b, _, enabled_b = BUDGET_ARMS[1]
        _, strategy_c, _, enabled_c = BUDGET_ARMS[2]
        assert strategy_b is strategy_c, "B and C must share a strategy"
        assert enabled_b is False
        assert enabled_c is True

    def test_the_ablation_holds_everything_but_the_allowance_constant(self) -> None:
        from edith.experiments import ABLATION_ARMS

        strategies = {strategy for _, strategy, _, _ in ABLATION_ARMS}
        assert len(strategies) == 1
        allowances = [limits.max_total_chars for _, _, limits, _ in ABLATION_ARMS if limits]
        assert allowances == sorted(allowances)
        assert len(set(allowances)) == 3

    def test_configure_budget_sets_the_limits_it_is_given(self) -> None:
        from edith.config.schema import EdithConfig, ModelParams, ModelsConfig
        from edith.experiments import configure_budget

        base = EdithConfig(
            models=ModelsConfig(profiles={"default": ModelParams(model_name="m:q4")})
        )
        configured = configure_budget(base, BUDGET_PRESETS["small"])
        budget = configured.orchestration.memory.budget
        assert budget.enabled
        assert budget.max_total_chars == BUDGET_PRESETS["small"].max_total_chars
        assert budget.max_retrievals == BUDGET_PRESETS["small"].max_retrievals

    def test_a_smaller_preset_is_smaller_in_every_dimension(self) -> None:
        small, medium, large = (
            BUDGET_PRESETS["small"],
            BUDGET_PRESETS["medium"],
            BUDGET_PRESETS["large"],
        )
        assert small.max_total_chars < medium.max_total_chars < large.max_total_chars
        assert small.max_retrievals <= medium.max_retrievals <= large.max_retrievals

    def test_scaling_never_produces_a_zero_ceiling(self) -> None:
        """A "small" budget must still be a budget, not a second control arm."""
        scaled = BUDGET_PRESETS["medium"].scaled(0.001)
        assert scaled.max_total_chars >= 1
        assert scaled.max_retrievals >= 1


class TestAccountingIsExactNotApproximate:
    """A ceiling checked against an estimate is a ceiling that can be overshot."""

    def test_the_charge_equals_the_text_actually_produced(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(memory)
        grant = ask(governor)
        assert grant.chars == len(grant.text)
        assert governor.budget.consumed_chars == len(grant.text)

    def test_an_injection_never_exceeds_the_per_retrieval_ceiling(
        self, memory: MemoryStore
    ) -> None:
        for index in range(6):
            seed(memory, f"Ordering in assertions {index}", SORTING_LESSON)
        limits = MemoryBudgetLimits(
            max_total_chars=100_000,
            max_chars_per_retrieval=600,
            max_memories_per_retrieval=5,
            max_total_memories=10,
        )
        grant = ask(build_governor(memory, limits=limits))
        assert len(grant.text) <= 600

    def test_reference_lines_are_charged_too(self, memory: MemoryStore) -> None:
        """A repeat is cheap, but it is not free, and free would be a leak."""
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory, limits=MemoryBudgetLimits(max_total_chars=100_000, max_retrievals=5)
        )
        ask(governor)
        before = governor.budget.consumed_chars
        second = ask(governor)
        assert second.chars > 0
        assert governor.budget.consumed_chars == before + second.chars


class TestProtectedContentNeverReachesAPrompt:
    """The M3 secret gate, closed through the governor rather than assumed."""

    def test_a_credential_bearing_memory_is_never_stored_so_never_granted(
        self, memory: MemoryStore
    ) -> None:
        record, outcome = memory.propose(
            MemoryProposal(
                type=MemoryType.PROJECT,
                project_id="proj_a",
                title="Deployment credentials",
                content=(
                    "The sorted results service authenticates with "
                    "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"
                ),
                source=MemorySource.USER,
                source_reference="tests/test_memory_governor.py",
            )
        )
        assert record is None, "a credential must never become a memory"
        assert "credential" in outcome.reason

        grant = ask(build_governor(memory))
        assert not grant.granted
        assert "wJalrXUtnFEMIK" not in grant.text


class TestOneBudgetPerExecution:
    """Interleaved agents share one allowance. A per-agent budget is not a budget."""

    def test_different_agents_draw_on_the_same_allowance(
        self, memory: MemoryStore
    ) -> None:
        for index in range(6):
            seed(memory, f"Ordering in assertions {index}", SORTING_LESSON)
        governor = build_governor(memory, limits=BUDGET_PRESETS["medium"])

        ask(governor, purpose=RetrievalPoint.CODER_REPAIR, agent="coder")
        after_coder = governor.budget.consumed_chars
        ask(governor, purpose=RetrievalPoint.DEBUGGER, agent="debugger")

        assert after_coder > 0
        assert governor.budget.consumed_chars > after_coder
        assert governor.budget.consumed_chars <= BUDGET_PRESETS["medium"].max_total_chars

    def test_a_second_agent_cannot_start_from_a_clean_slate(
        self, memory: MemoryStore
    ) -> None:
        """The budget is the execution's, so switching agents does not reset it."""
        seed(memory, "Ordering in assertions", SORTING_LESSON)
        governor = build_governor(
            memory,
            limits=MemoryBudgetLimits(
                max_total_chars=100_000, max_retrievals=1, max_total_memories=5
            ),
        )
        ask(governor, agent="coder", purpose=RetrievalPoint.CODER_REPAIR)
        second = ask(governor, agent="debugger", purpose=RetrievalPoint.DEBUGGER)
        assert second.outcome is GrantOutcome.BUDGET_EXHAUSTED


class TestResumedExecutions:
    """A restart must continue the budget, not be handed a fresh one."""

    def test_a_resumed_budget_starts_from_prior_consumption(self) -> None:
        budget = ExecutionMemoryBudget(
            EXECUTION,
            DEFAULT_LIMITS,
            consumed_chars=1000,
            retrieval_count=2,
            injected=(
                InjectionRecord(
                    memory_id="mem_prior",
                    title="Seen before",
                    point=RetrievalPoint.DEBUGGER,
                    agent="debugger",
                    score=20.0,
                    reason="injected before the interruption",
                    chars=500,
                ),
            ),
        )
        assert budget.consumed_chars == 1000
        assert budget.remaining_chars == DEFAULT_LIMITS.max_total_chars - 1000
        assert "mem_prior" in budget.injected_memory_ids

    def test_a_resumed_execution_does_not_re_send_what_it_already_sent(
        self, memory: MemoryStore
    ) -> None:
        """"Crash and retry" must not become an unlimited memory supply."""
        memory_id = seed(memory, "Ordering in assertions", SORTING_LESSON)
        resumed = ExecutionMemoryBudget(
            EXECUTION,
            DEFAULT_LIMITS,
            consumed_chars=200,
            retrieval_count=1,
            injected=(
                InjectionRecord(
                    memory_id=memory_id,
                    title="Ordering in assertions",
                    point=RetrievalPoint.DEBUGGER,
                    agent="debugger",
                    score=20.0,
                    reason="injected before the interruption",
                    chars=200,
                ),
            ),
        )
        grant = ask(build_governor(memory, budget=resumed))
        assert SORTING_LESSON not in grant.text
        assert memory_id in grant.suppressed_ids
