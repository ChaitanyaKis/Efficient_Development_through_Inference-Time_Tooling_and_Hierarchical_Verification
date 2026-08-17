"""Memory reaching the agents, and staying inside its project boundary when it does.

Deterministic: a scripted model records the prompt it was given, so what actually reached
the agent can be asserted on rather than assumed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from edith.config.schema import EdithConfig, MemoryConfig
from edith.experiments import configure_strategy
from edith.memory.schema import (
    MemoryProposal,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from edith.memory.store import MemoryStore, open_memory
from edith.memory.strategy import MemoryStrategy
from edith.orchestrator import Orchestrator, create_execution
from edith.schemas.common import Verdict
from edith.workspaces import ProjectWorkspace

from .test_orchestrator import (
    GOOD_CODE,
    ScriptedProvider,
    edits,
    plan,
    verdict,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

# Fixture functions imported to register them in this module (see test_hardening_suite).
from .test_orchestrator import config, repo, store, workspace  # noqa: F401

LESSON = "When a test compares against an ordered list, the result must also be sorted."


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    with open_memory(tmp_path / "memory") as opened:
        yield opened


def seed(store: MemoryStore, project_id: str | None, title: str, content: str) -> None:
    """Store a lesson with real provenance."""
    record, outcome = store.propose(
        MemoryProposal(
            type=MemoryType.ENGINEERING,
            scope=MemoryScope.GLOBAL if project_id is None else MemoryScope.PROJECT,
            project_id=project_id,
            title=title,
            content=content,
            source=MemorySource.TEST_RESULT,
            source_reference="tests/test_memory_integration.py",
        )
    )
    assert record is not None, outcome.reason


def build_with_memory(
    config: EdithConfig,
    workspace: ProjectWorkspace,
    state: Any,
    memory: MemoryStore | None,
    script: dict[str, list[str]],
    strategy: MemoryStrategy = MemoryStrategy.ALWAYS,
) -> tuple[Orchestrator, ScriptedProvider]:
    """Build an orchestrator whose model records the prompts it receives.

    The strategy defaults to ``ALWAYS`` because these tests assert on the *mechanism* --
    that a retrieved memory reaches the coder's prompt carrying its provenance, and that
    another project's memory never does. Under the shipped default (``failure_triggered``)
    the coder's first attempt is deliberately left untouched, so these tests would pass
    vacuously. Which strategy fires where is measured in ``test_memory_strategy.py``.
    """
    provider = ScriptedProvider(config.models.profile(), script)
    # configure_strategy also enables memory, which would defeat the disabled-config test,
    # so a config that switched memory off keeps it off.
    selected = (
        configure_strategy(config, strategy)
        if config.orchestration.memory.enabled
        else config
    )
    orchestrator = Orchestrator(
        selected, state, workspace, provider=provider, memory=memory
    )
    return (orchestrator, provider)


#: A task whose wording genuinely relates to the seeded lesson. Retrieval is driven by the
#: *task text*, so a task about subtraction would correctly retrieve nothing about ordering
#: -- the test would then be asserting that retrieval is broken.
SORTING_TASK_TITLE = "Return sorted results"
SORTING_TASK_DESCRIPTION = (
    "The helper must return results in a sorted order matching the expected list."
)


def default_script() -> dict[str, list[str]]:
    return {
        "PlannerOutput": [
            plan(["calc.py"], SORTING_TASK_TITLE, SORTING_TASK_DESCRIPTION)
        ],
        "ModelEdits": [edits("calc.py", GOOD_CODE)],
        "CriticOutput": [verdict("PASS")],
    }


class TestMemoryReachesTheCoder:
    def test_a_relevant_lesson_is_injected(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Sorted results", LESSON)
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        result = orchestrator.run(execution)

        assert result.verdict is Verdict.PASS
        coder_prompts = provider.prompts_for("ModelEdits")
        assert coder_prompts, "the coder must have been invoked"
        assert "Sorted results" in coder_prompts[0]
        assert result.memories_used >= 1

    def test_injected_memory_carries_its_provenance(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """An agent must be able to weigh a remembered claim, which needs its source."""
        seed(memory, workspace.project_id, "Sorted results", LESSON)
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        orchestrator.run(execution)
        assert "TEST_RESULT" in provider.prompts_for("ModelEdits")[0]

    def test_memory_is_presented_as_knowledge_not_instruction(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """Memory informs the work; it must not read as a replacement for the task."""
        seed(memory, workspace.project_id, "Sorted results", LESSON)
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        orchestrator.run(execution)
        prompt = provider.prompts_for("ModelEdits")[0]
        assert "PRIOR KNOWLEDGE" in prompt
        assert prompt.index("TASK:") < prompt.index("PRIOR KNOWLEDGE")

    def test_an_irrelevant_lesson_is_not_injected(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(
            memory,
            workspace.project_id,
            "Kubernetes ingress annotations",
            "Ingress annotations must be namespaced under the controller prefix.",
        )
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)
        assert "Kubernetes ingress" not in provider.prompts_for("ModelEdits")[0]
        assert result.memories_used == 0

    def test_no_memory_store_means_no_injection(
        self, config: EdithConfig, workspace: ProjectWorkspace, store: Any
    ) -> None:
        """The experiment's control arm runs the same code path with retrieval off."""
        orchestrator, provider = build_with_memory(
            config, workspace, store, None, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)
        assert result.memories_used == 0
        assert "PRIOR KNOWLEDGE" not in provider.prompts_for("ModelEdits")[0]

    def test_disabling_memory_in_config_disables_injection(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Sorted results", LESSON)
        disabled = config.model_copy(
            update={
                "orchestration": config.orchestration.model_copy(
                    update={"memory": MemoryConfig(enabled=False)}
                )
            }
        )
        orchestrator, provider = build_with_memory(
            disabled, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract")
        result = orchestrator.run(execution)
        assert result.memories_used == 0
        assert "PRIOR KNOWLEDGE" not in provider.prompts_for("ModelEdits")[0]


class TestIsolationAcrossExecutions:
    """The privacy invariant, enforced through the whole loop rather than only in SQL."""

    def test_another_projects_memory_never_reaches_this_prompt(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(
            memory,
            "proj_someone_else",
            "Competitor pricing model",
            "The pricing service applies a bespoke discount ladder for enterprise tiers.",
        )
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix the pricing discount ladder")
        orchestrator.run(execution)

        prompt = provider.prompts_for("ModelEdits")[0]
        assert "Competitor pricing" not in prompt
        assert "discount ladder" not in prompt.replace("fix the pricing discount ladder", "")

    def test_a_global_lesson_does_reach_the_prompt(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, None, "Sorted results", LESSON)
        orchestrator, provider = build_with_memory(
            config, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract so tests pass")
        orchestrator.run(execution)
        assert "Sorted results" in provider.prompts_for("ModelEdits")[0]


class TestBudget:
    def test_memory_stays_within_its_character_budget(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """On an 8k window memory competes with the code itself, so it must stay small."""
        for index in range(30):
            seed(
                memory,
                workspace.project_id,
                f"Sorted results {index}",
                LESSON + " " + ("padding text " * 8),
            )
        tight = config.model_copy(
            update={
                "orchestration": config.orchestration.model_copy(
                    update={"memory": MemoryConfig(max_memories=2, max_chars=900)}
                )
            }
        )
        orchestrator, _ = build_with_memory(
            tight, workspace, store, memory, default_script()
        )
        _, execution = create_execution(store, workspace, "fix subtract sorted results")
        result = orchestrator.run(execution)
        # Non-zero as well as bounded: a budget that silently retrieves nothing would
        # satisfy the ceiling while proving nothing about it.
        assert result.memories_used >= 1
        assert result.memories_used <= 2
        assert result.memory_chars <= 900


class TestExperimentHarness:
    def test_seeded_lessons_are_accepted_by_validation(self, tmp_path: Path) -> None:
        """The seeds must pass the same quality gate as any other memory."""
        from edith.experiments import SEEDED_LESSONS, seed_lessons

        with open_memory(tmp_path / "memory") as store:
            stored = seed_lessons(store, "experiment")
            assert stored == len(SEEDED_LESSONS)

    def test_seeded_lessons_do_not_contain_the_answer(self) -> None:
        """A memory stating the fix would measure nothing about memory."""
        from edith.experiments import SEEDED_LESSONS

        for lesson in SEEDED_LESSONS:
            body = str(lesson["content"]).lower()
            assert "sorted(" not in body
            assert "return sorted" not in body
            assert "low_stock" not in body
            assert "inventory" not in body

    def test_arm_results_report_honestly_with_no_runs(self) -> None:
        from edith.experiments import ArmResult, ExperimentResult

        experiment = ExperimentResult(
            benchmark_id="multi_repair",
            baseline=ArmResult(name="baseline"),
            memory=ArmResult(name="memory"),
        )
        assert experiment.success_rate_delta == 0.0
        assert "No measurable difference" in experiment.verdict()

    def test_a_regression_is_reported_not_hidden(self) -> None:
        from edith.benchmarks import BenchmarkResult
        from edith.experiments import ArmResult, ExperimentResult

        baseline = ArmResult(
            name="baseline",
            runs=2,
            successes=2,
            results=[BenchmarkResult("b", True, "ok"), BenchmarkResult("b", True, "ok")],
        )
        worse = ArmResult(
            name="memory",
            runs=2,
            successes=0,
            results=[BenchmarkResult("b", False, "no"), BenchmarkResult("b", False, "no")],
        )
        experiment = ExperimentResult("multi_repair", baseline, worse)
        assert not experiment.improved
        assert "reduced" in experiment.verdict()
