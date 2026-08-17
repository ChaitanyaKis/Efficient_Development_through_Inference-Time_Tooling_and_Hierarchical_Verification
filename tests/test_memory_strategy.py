"""M3.1: *where* memory is injected, measured rather than assumed.

M3 produced the finding this module exists to act on: naive always-inject memory took the
multi-defect repair benchmark from 5/6 to 0/6 on a 3B model. The conclusion was not that
memory is useless but that injecting it everywhere is the wrong integration.

These tests pin down each of the five strategies deterministically -- a scripted model
records the prompts it was handed, so what actually reached an agent is asserted on rather
than inferred. The benchmark measures which strategy is *better*; these tests only prove
each one does what it claims.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from edith.config.schema import EdithConfig, MemoryConfig
from edith.experiments import configure_strategy
from edith.memory.retrieval import LexicalRanker, MemoryRetriever, RetrievalRequest
from edith.memory.schema import MemoryProposal, MemoryScope, MemorySource, MemoryType
from edith.memory.store import MemoryStore, open_memory
from edith.memory.strategy import (
    HIGH_RELEVANCE_THRESHOLD,
    POLICIES,
    MemoryStrategy,
    RetrievalPoint,
    policy_for,
)
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

#: A lesson worded to overlap the *failure*, not the task title. That is the whole point of
#: failure-triggered retrieval: the error names the symptom, the title only names intent.
ORDERING_LESSON = (
    "When an assertion compares a result against an ordered list, the returned value must "
    "also be sorted, or the assertion fails on ordering alone."
)


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    with open_memory(tmp_path / "memory") as opened:
        yield opened


def seed(
    store: MemoryStore,
    project_id: str | None,
    title: str,
    content: str,
    *,
    tags: tuple[str, ...] = (),
) -> None:
    """Store a lesson with real provenance, through the same gate as any other memory."""
    record, outcome = store.propose(
        MemoryProposal(
            type=MemoryType.ENGINEERING,
            scope=MemoryScope.GLOBAL if project_id is None else MemoryScope.PROJECT,
            project_id=project_id,
            title=title,
            content=content,
            tags=list(tags),
            source=MemorySource.TEST_RESULT,
            source_reference="tests/test_memory_strategy.py",
        )
    )
    assert record is not None, outcome.reason


def repair_script() -> dict[str, list[str]]:
    """A run that fails once, consults the debugger, then succeeds.

    Every retrieval point is exercised: an initial coder attempt, a debugger diagnosis, and
    a repair attempt.
    """
    return {
        "PlannerOutput": [
            plan(
                ["calc.py"],
                "Return sorted results",
                "The helper must return results matching the expected ordered list.",
            )
        ],
        "ModelEdits": [
            edits("calc.py", BAD_CODE.replace("a + b", "a * b")),
            edits("calc.py", GOOD_CODE),
        ],
        "CriticOutput": [verdict("FAIL"), verdict("PASS")],
        "DebuggerOutput": [diagnosis()],
    }


def run_with(
    config: EdithConfig,
    workspace: ProjectWorkspace,
    state: Any,
    memory: MemoryStore | None,
    strategy: MemoryStrategy,
) -> ScriptedProvider:
    """Run one repair loop under a named strategy and return the recorded prompts."""
    provider = ScriptedProvider(config.models.profile(), repair_script())
    orchestrator = Orchestrator(
        configure_strategy(config, strategy),
        state,
        workspace,
        provider=provider,
        memory=memory,
    )
    _, execution = create_execution(state, workspace, "fix subtract so the tests pass")
    orchestrator.run(execution)
    return provider


class TestPolicyTable:
    """The strategies are a closed, explicit set -- no implicit default behaviour."""

    def test_every_strategy_has_an_explicit_policy(self) -> None:
        assert set(POLICIES) == set(MemoryStrategy)

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            (MemoryStrategy.NONE, set()),
            (MemoryStrategy.ALWAYS, set(RetrievalPoint)),
            (
                MemoryStrategy.FAILURE_TRIGGERED,
                {RetrievalPoint.CODER_REPAIR, RetrievalPoint.DEBUGGER},
            ),
            (MemoryStrategy.DEBUGGER_ONLY, {RetrievalPoint.DEBUGGER}),
            (MemoryStrategy.HIGH_RELEVANCE, set(RetrievalPoint)),
        ],
    )
    def test_each_strategy_retrieves_at_its_declared_points(
        self, strategy: MemoryStrategy, expected: set[RetrievalPoint]
    ) -> None:
        policy = policy_for(strategy)
        assert {point for point in RetrievalPoint if policy.applies_at(point)} == expected

    def test_failure_triggered_never_touches_the_first_attempt(self) -> None:
        """The measured regression was context pressure on the initial prompt."""
        assert not policy_for(MemoryStrategy.FAILURE_TRIGGERED).applies_at(
            RetrievalPoint.CODER_INITIAL
        )

    def test_high_relevance_sets_a_bar_and_a_tighter_cap(self) -> None:
        policy = policy_for(MemoryStrategy.HIGH_RELEVANCE)
        assert policy.min_score == HIGH_RELEVANCE_THRESHOLD
        assert policy.max_memories is not None
        assert policy.max_memories <= 2

    def test_an_unrecognised_strategy_retrieves_nothing(self) -> None:
        """Failing open would inject unexpectedly, which is the measured-worse direction."""
        policy = policy_for("not_a_strategy")  # type: ignore[arg-type]
        assert not policy.retrieves_anything


class TestTheShippedDefault:
    def test_the_default_strategy_is_the_one_that_measured_best(self) -> None:
        """Pinned so that changing it is a deliberate act with a measurement behind it.

        Over 6 runs per arm on ``multi_repair`` (ADR 0006 §4) no strategy beat the
        no-memory control, and ``always`` scored 0/6. Until a strategy earns its context
        cost, the default spends none.
        """
        assert MemoryConfig().strategy == str(MemoryStrategy.NONE)

    def test_the_default_still_leaves_the_subsystem_one_line_from_active(self) -> None:
        """A default of `none` must mean "not injected", never "not built"."""
        assert MemoryConfig().enabled is True
        assert policy_for(MemoryStrategy.DEBUGGER_ONLY).retrieves_anything


class TestFailureRelevance:
    """Retrieval keyed on the observed error, not only the task title."""

    def test_error_text_makes_a_matching_lesson_rank_higher(
        self, memory: MemoryStore
    ) -> None:
        seed(memory, "proj_a", "Ordering in assertions", ORDERING_LESSON)
        retriever = MemoryRetriever(memory, LexicalRanker())

        without = retriever.retrieve(
            RetrievalRequest(query="update the helper", project_id="proj_a")
        )
        with_error = retriever.retrieve(
            RetrievalRequest(
                query="update the helper",
                project_id="proj_a",
                error_text=(
                    "AssertionError: assert ['b', 'a'] == ['a', 'b']\n"
                    "the returned value was not sorted"
                ),
            )
        )

        assert not without.memories, "a vague title alone must not surface a lesson"
        assert with_error.memories, "the observed error must be usable as a retrieval key"
        assert with_error.memories[0].score > 0

    def test_a_lesson_about_the_file_being_changed_is_favoured(
        self, memory: MemoryStore
    ) -> None:
        seed(
            memory,
            "proj_a",
            "Inventory thresholds",
            "The inventory helper treats the threshold as inclusive at the boundary.",
        )
        retriever = MemoryRetriever(memory, LexicalRanker())
        bundle = retriever.retrieve(
            RetrievalRequest(
                query="adjust the boundary",
                project_id="proj_a",
                paths=("src/inventory.py",),
            )
        )
        assert bundle.memories
        assert any("component" in reason for reason in bundle.memories[0].reasons)

    def test_min_score_excludes_a_weak_match(self, memory: MemoryStore) -> None:
        """The HIGH_RELEVANCE gate, exercised directly on the ranker."""
        seed(
            memory,
            "proj_a",
            "Deployment rollback",
            "A rollback must restore the previous manifest revision before scaling up.",
        )
        retriever = MemoryRetriever(memory, LexicalRanker())

        admitted = retriever.retrieve(
            RetrievalRequest(query="rollback the manifest revision", project_id="proj_a")
        )
        gated = retriever.retrieve(
            RetrievalRequest(
                query="rollback the manifest revision",
                project_id="proj_a",
                min_score=1_000.0,
            )
        )
        assert admitted.memories
        assert not gated.memories

    def test_an_unrelated_memory_is_never_admitted_on_metadata_alone(
        self, memory: MemoryStore
    ) -> None:
        seed(
            memory,
            "proj_a",
            "Kubernetes ingress annotations",
            "Ingress annotations must be namespaced under the controller prefix.",
        )
        retriever = MemoryRetriever(memory, LexicalRanker())
        bundle = retriever.retrieve(
            RetrievalRequest(
                query="make subtract return the difference",
                project_id="proj_a",
                error_text="AssertionError: assert 8 == 2",
            )
        )
        assert not bundle.memories


class TestStrategiesInTheLoop:
    """Each arm, end to end, asserting on the prompts the agents actually received."""

    def test_strategy_none_injects_nowhere(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Ordering in assertions", ORDERING_LESSON)
        provider = run_with(config, workspace, store, memory, MemoryStrategy.NONE)
        for prompt in provider.prompts_for("ModelEdits") + provider.prompts_for(
            "DebuggerOutput"
        ):
            assert "Ordering in assertions" not in prompt

    def test_strategy_always_injects_into_the_first_attempt(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """The M3 behaviour, kept available so the regression stays reproducible."""
        seed(memory, workspace.project_id, "Ordering in assertions", ORDERING_LESSON)
        provider = run_with(config, workspace, store, memory, MemoryStrategy.ALWAYS)
        assert "Ordering in assertions" in provider.prompts_for("ModelEdits")[0]

    def test_failure_triggered_spares_the_first_attempt_but_arms_the_repair(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """The shipped default: the initial prompt stays lean, the repair gets help."""
        seed(memory, workspace.project_id, "Ordering in assertions", ORDERING_LESSON)
        provider = run_with(
            config, workspace, store, memory, MemoryStrategy.FAILURE_TRIGGERED
        )
        coder_prompts = provider.prompts_for("ModelEdits")
        assert len(coder_prompts) >= 2, "the loop must have attempted a repair"
        assert "Ordering in assertions" not in coder_prompts[0]
        later = coder_prompts[1:] + provider.prompts_for("DebuggerOutput")
        assert any("Ordering in assertions" in prompt for prompt in later)

    def test_debugger_only_leaves_every_coder_prompt_untouched(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        seed(memory, workspace.project_id, "Ordering in assertions", ORDERING_LESSON)
        provider = run_with(
            config, workspace, store, memory, MemoryStrategy.DEBUGGER_ONLY
        )
        for prompt in provider.prompts_for("ModelEdits"):
            assert "Ordering in assertions" not in prompt
        debugger_prompts = provider.prompts_for("DebuggerOutput")
        assert debugger_prompts, "the debugger must have been consulted"
        assert "Ordering in assertions" in debugger_prompts[0]

    def test_high_relevance_rejects_a_memory_that_only_brushes_the_task(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """A bar high enough to keep a merely-plausible lesson out of an 8k window."""
        seed(
            memory,
            workspace.project_id,
            "Release checklist",
            "A release requires a tagged commit and a published changelog entry.",
        )
        provider = run_with(
            config, workspace, store, memory, MemoryStrategy.HIGH_RELEVANCE
        )
        for prompt in provider.prompts_for("ModelEdits") + provider.prompts_for(
            "DebuggerOutput"
        ):
            assert "Release checklist" not in prompt

    def test_another_projects_memory_is_invisible_under_every_strategy(
        self,
        config: EdithConfig,
        workspace: ProjectWorkspace,
        store: Any,
        memory: MemoryStore,
    ) -> None:
        """Isolation is not a property of the strategy; no strategy may widen the set."""
        seed(
            memory,
            "proj_someone_else",
            "Ordering in assertions",
            ORDERING_LESSON,
        )
        for strategy in MemoryStrategy:
            provider = run_with(config, workspace, store, memory, strategy)
            prompts = provider.prompts_for("ModelEdits") + provider.prompts_for(
                "DebuggerOutput"
            )
            assert prompts
            for prompt in prompts:
                assert "Ordering in assertions" not in prompt, strategy
