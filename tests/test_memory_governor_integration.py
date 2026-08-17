"""The governor inside the real loop: bounding, accounting, persistence, and bypass.

The unit tests prove the governor enforces its rules. These prove the *loop* cannot get
memory any other way, that an execution's cost is bounded no matter how many repairs it
attempts, and that the accounting survives a restart.

A scripted model records the prompts it was handed, so what actually reached each agent is
asserted on rather than assumed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from edith.config.schema import EdithConfig
from edith.experiments import configure_budget, configure_strategy
from edith.memory.governor import BUDGET_PRESETS, MemoryBudgetLimits
from edith.memory.schema import MemoryProposal, MemoryScope, MemorySource, MemoryType
from edith.memory.store import MemoryStore, open_memory
from edith.memory.strategy import MemoryStrategy
from edith.orchestrator import Orchestrator, create_execution
from edith.workspaces import ProjectWorkspace

from .test_orchestrator import (
    BAD_CODE,
    GOOD_CODE,
    ScriptedProvider,
    diagnosis,
    edits,
    plan,
    verdict,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

# Fixture functions imported to register them in this module.
from .test_orchestrator import config, repo, store, workspace  # noqa: F401

LESSON = (
    "When an assertion compares a result against an ordered list, the returned value must "
    "also be sorted, or the assertion fails on ordering alone."
)


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    with open_memory(tmp_path / "memory") as opened:
        yield opened


def seed(store: MemoryStore, project_id: str | None, title: str, content: str) -> str:
    record, outcome = store.propose(
        MemoryProposal(
            type=MemoryType.ENGINEERING,
            scope=MemoryScope.GLOBAL if project_id is None else MemoryScope.PROJECT,
            project_id=project_id,
            title=title,
            content=content,
            source=MemorySource.TEST_RESULT,
            source_reference="tests/test_memory_governor_integration.py",
        )
    )
    assert record is not None, outcome.reason
    return record.memory_id


def never_succeeds() -> dict[str, list[str]]:
    """A run that keeps failing, so the loop repairs until its retry budget is spent.

    This is the shape that produced M3.1's ~14,000 characters: every failure triggers
    another retrieval.
    """
    return {
        "PlannerOutput": [
            plan(
                ["calc.py"],
                "Return sorted results",
                "The helper must return results matching the expected ordered list.",
            )
        ],
        "ModelEdits": [edits("calc.py", BAD_CODE.replace("a + b", "a * b"))],
        "CriticOutput": [verdict("FAIL")],
        "DebuggerOutput": [diagnosis()],
    }


def repairs_once() -> dict[str, list[str]]:
    """Fails once, consults the debugger, then succeeds."""
    script = never_succeeds()
    script["ModelEdits"] = [
        edits("calc.py", BAD_CODE.replace("a + b", "a * b")),
        edits("calc.py", GOOD_CODE),
    ]
    script["CriticOutput"] = [verdict("FAIL"), verdict("PASS")]
    return script


def build(
    config: EdithConfig,
    workspace: ProjectWorkspace,
    state: Any,
    memory: MemoryStore | None,
    script: dict[str, list[str]],
    *,
    strategy: MemoryStrategy = MemoryStrategy.DEBUGGER_ONLY,
    limits: MemoryBudgetLimits | None = None,
    budget_enabled: bool = True,
) -> tuple[Orchestrator, ScriptedProvider]:
    """An orchestrator with an explicit memory strategy and execution budget."""
    provider = ScriptedProvider(config.models.profile(), script)
    selected = configure_budget(
        configure_strategy(config, strategy), limits, enabled=budget_enabled
    )
    return (
        Orchestrator(selected, state, workspace, provider=provider, memory=memory),
        provider,
    )


class TestTheBudgetBoundsARealExecution:
    def test_a_repair_heavy_run_stays_within_its_allowance(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """The M3.2 objective, measured end to end."""
        for index in range(8):
            seed(memory, workspace.project_id, f"Ordering in assertions {index}", LESSON)
        limits = BUDGET_PRESETS["medium"]
        orchestrator, _ = build(
            config, workspace, store, memory, never_succeeds(), limits=limits
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        assert result.repairs_attempted >= 1, "the loop must have repaired repeatedly"
        assert result.memory_chars <= limits.max_total_chars
        assert result.memory_retrievals <= limits.max_retrievals

    def test_the_same_run_without_a_budget_costs_more(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """B versus C from the experiment, as a deterministic assertion.

        Identical script, identical strategy, identical retry limits. The only difference is
        the execution ceiling, so any difference in cost is the budget.
        """
        for index in range(8):
            seed(memory, workspace.project_id, f"Ordering in assertions {index}", LESSON)

        unbudgeted, _ = build(
            config, workspace, store, memory, never_succeeds(), budget_enabled=False
        )
        _, execution_b = create_execution(store, workspace, "fix subtract so tests pass")
        result_b = unbudgeted.run(execution_b)

        budgeted, _ = build(
            config,
            workspace,
            store,
            memory,
            never_succeeds(),
            limits=BUDGET_PRESETS["medium"],
        )
        _, execution_c = create_execution(store, workspace, "fix subtract so tests pass")
        result_c = budgeted.run(execution_c)

        assert result_b.memory_chars > result_c.memory_chars
        assert result_c.memory_chars <= BUDGET_PRESETS["medium"].max_total_chars

    def test_exhaustion_is_counted_not_hidden(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        for index in range(8):
            seed(memory, workspace.project_id, f"Ordering in assertions {index}", LESSON)
        orchestrator, _ = build(
            config,
            workspace,
            store,
            memory,
            never_succeeds(),
            limits=MemoryBudgetLimits(
                max_total_chars=400,
                max_retrievals=1,
                max_total_memories=1,
                max_chars_per_retrieval=400,
                max_memories_per_retrieval=1,
            ),
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        assert result.budget_exhaustions >= 1, (
            "a repair-heavy run under a one-retrieval budget must hit the ceiling"
        )

    def test_the_loop_still_finishes_when_memory_runs_out(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """Memory is an optimisation, never a prerequisite for execution."""
        seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        orchestrator, _ = build(
            config,
            workspace,
            store,
            memory,
            repairs_once(),
            limits=MemoryBudgetLimits(
                max_total_chars=0,
                max_retrievals=0,
                max_total_memories=0,
                max_chars_per_retrieval=0,
                max_memories_per_retrieval=0,
            ),
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        from edith.schemas.common import Verdict

        assert result.verdict is Verdict.PASS
        assert result.memory_chars == 0


class TestContextAccounting:
    def test_every_injection_is_recorded_with_its_cost(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        memory_id = seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        orchestrator, _ = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        orchestrator.run(execution)

        injections = store.memory_injections(execution.execution_id)
        assert injections, "an injection must leave an auditable record"
        record = injections[0]
        assert memory_id in record.memory_ids
        assert record.chars > 0
        assert record.point == "debugger"
        assert record.agent == "debugger"
        assert record.scores and record.scores[0] > 0

    def test_the_ledger_stores_ids_not_memory_content(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """A memory the user deletes must not go on living in the state database."""
        seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        orchestrator, _ = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        orchestrator.run(execution)

        for record in store.memory_injections(execution.execution_id):
            payload = str(record.model_dump())
            assert LESSON not in payload
            # Titles are the deliberate exception: a resumed run needs them to say what it
            # already sent without re-sending it.
            assert "Ordering in assertions" in payload

    def test_consumption_is_readable_back_from_durable_state(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        orchestrator, _ = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        spent = store.memory_consumption(execution.execution_id)
        assert spent.chars == result.memory_chars
        assert spent.retrievals == result.memory_retrievals


class TestRestartContinuesTheBudget:
    def test_a_resumed_execution_does_not_get_a_fresh_allowance(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """Otherwise "crash and retry" would be an unlimited memory supply."""
        for index in range(8):
            seed(memory, workspace.project_id, f"Ordering in assertions {index}", LESSON)
        limits = BUDGET_PRESETS["medium"]

        first, _ = build(
            config, workspace, store, memory, never_succeeds(), limits=limits
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        first_result = first.run(execution)
        assert first_result.memory_chars > 0

        # A second orchestrator resuming the same execution rebuilds the budget from state.
        second, _ = build(
            config, workspace, store, memory, never_succeeds(), limits=limits
        )
        governor = second._build_governor(execution)
        assert governor is not None
        assert governor.budget.consumed_chars == first_result.memory_chars
        assert governor.budget.retrieval_count == first_result.memory_retrievals

    def test_a_resumed_execution_remembers_what_it_already_sent(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        memory_id = seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        first, _ = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        first.run(execution)

        second, _ = build(config, workspace, store, memory, repairs_once())
        governor = second._build_governor(execution)
        assert governor is not None
        assert memory_id in governor.budget.injected_memory_ids


class TestNoBypass:
    def test_memory_never_reaches_a_coder_prompt_under_debugger_only(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        orchestrator, provider = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        orchestrator.run(execution)

        for prompt in provider.prompts_for("ModelEdits"):
            assert "Ordering in assertions" not in prompt

    def test_another_projects_memory_never_reaches_any_prompt(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """M3 isolation, re-verified through the governor rather than assumed."""
        seed(memory, "proj_someone_else", "Ordering in assertions", LESSON)
        orchestrator, provider = build(config, workspace, store, memory, repairs_once())
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        prompts = provider.prompts_for("ModelEdits") + provider.prompts_for(
            "DebuggerOutput"
        )
        assert prompts
        for prompt in prompts:
            assert "Ordering in assertions" not in prompt
        assert result.memories_used == 0

    def test_disabling_memory_leaves_no_governor_to_bypass(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Ordering in assertions", LESSON)
        disabled = config.model_copy(
            update={
                "orchestration": config.orchestration.model_copy(
                    update={
                        "memory": config.orchestration.memory.model_copy(
                            update={"enabled": False}
                        )
                    }
                )
            }
        )
        provider = ScriptedProvider(config.models.profile(), repairs_once())
        orchestrator = Orchestrator(
            disabled, store, workspace, provider=provider, memory=memory
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        assert result.memory_chars == 0
        assert result.memories_used == 0
        assert not store.memory_injections(execution.execution_id)
