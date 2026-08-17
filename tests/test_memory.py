"""Memory: schema, provenance, isolation, lifecycle, retrieval, and quality control."""

from __future__ import annotations

from pathlib import Path

import pytest

from edith.errors import ConfigurationError
from edith.memory.consolidation import (
    consolidate_project,
    find_duplicate_groups,
    find_existing_match,
    similarity,
)
from edith.memory.retrieval import MemoryRetriever, RetrievalRequest
from edith.memory.schema import (
    MemoryProposal,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from edith.memory.store import MemoryStore, open_memory
from edith.memory.validation import contains_secret, redact, to_record

PROJECT_A = "proj_alpha"
PROJECT_B = "proj_beta"


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with open_memory(tmp_path / "state") as opened:
        yield opened


def record(
    title: str,
    content: str,
    *,
    project_id: str | None = PROJECT_A,
    memory_type: MemoryType = MemoryType.PROJECT,
    scope: MemoryScope = MemoryScope.PROJECT,
    source: MemorySource = MemorySource.USER,
    tags: tuple[str, ...] = (),
    importance: int = 50,
    confidence: float = 0.8,
) -> MemoryRecord:
    """Build a valid record directly, bypassing proposal validation."""
    return MemoryRecord(
        type=memory_type,
        scope=scope,
        project_id=project_id,
        title=title,
        content=content,
        source=source,
        source_reference="tests/test_memory.py",
        tags=tags,
        importance=importance,
        confidence=confidence,
    )


class TestSchemaAndProvenance:
    def test_a_valid_record_round_trips(self, store: MemoryStore) -> None:
        saved = store.save(record("Uses SQLite", "The project stores state in SQLite."))
        loaded = store.get(saved.memory_id)
        assert loaded is not None
        assert loaded.title == "Uses SQLite"
        assert loaded.source is MemorySource.USER

    def test_provenance_is_mandatory(self) -> None:
        """A memory that cannot be traced is not knowledge."""
        with pytest.raises(ValueError):
            MemoryRecord(
                type=MemoryType.PROJECT,
                project_id=PROJECT_A,
                title="t",
                content="c",
                source=MemorySource.USER,
                source_reference="",
            )

    def test_provenance_renders_for_a_human(self) -> None:
        entry = record("t", "c", source=MemorySource.TEST_RESULT, confidence=0.95)
        assert "TEST_RESULT" in entry.provenance
        assert "0.95" in entry.provenance

    def test_project_scope_requires_a_project(self) -> None:
        with pytest.raises(ValueError, match="must have a project_id"):
            MemoryRecord(
                type=MemoryType.PROJECT,
                scope=MemoryScope.PROJECT,
                project_id=None,
                title="t",
                content="c",
                source=MemorySource.USER,
                source_reference="ref",
            )

    def test_global_scope_forbids_a_project(self) -> None:
        with pytest.raises(ValueError, match="must not be bound"):
            MemoryRecord(
                type=MemoryType.ENGINEERING,
                scope=MemoryScope.GLOBAL,
                project_id=PROJECT_A,
                title="t",
                content="c",
                source=MemorySource.USER,
                source_reference="ref",
            )

    def test_project_facts_cannot_be_global(self) -> None:
        """Only genuinely reusable lessons may cross a project boundary."""
        with pytest.raises(ValueError, match="cannot be GLOBAL"):
            MemoryRecord(
                type=MemoryType.PROJECT,
                scope=MemoryScope.GLOBAL,
                project_id=None,
                title="t",
                content="c",
                source=MemorySource.USER,
                source_reference="ref",
            )

    def test_engineering_lessons_may_be_global(self) -> None:
        entry = MemoryRecord(
            type=MemoryType.ENGINEERING,
            scope=MemoryScope.GLOBAL,
            project_id=None,
            title="Small models drop functions",
            content="A 3B model rewriting a file often omits an unrelated function.",
            source=MemorySource.TEST_RESULT,
            source_reference="benchmarks/multi_repair",
        )
        assert entry.scope is MemoryScope.GLOBAL


class TestProjectIsolation:
    """The privacy invariant: one project's knowledge must never reach another."""

    def test_project_a_memory_is_invisible_to_project_b(self, store: MemoryStore) -> None:
        store.save(record("Alpha secret design", "Alpha uses a bespoke scheduler."))
        visible = store.visible_to(PROJECT_B)
        assert visible == []

    def test_retrieval_cannot_cross_projects(self, store: MemoryStore) -> None:
        store.save(record("Alpha uses PostgreSQL", "Alpha stores orders in PostgreSQL."))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="which database does this project use", project_id=PROJECT_B)
        )
        assert bundle.is_empty
        assert "PostgreSQL" not in bundle.render()

    def test_each_project_sees_only_its_own(self, store: MemoryStore) -> None:
        store.save(record("Alpha fact", "Alpha detail.", project_id=PROJECT_A))
        store.save(record("Beta fact", "Beta detail.", project_id=PROJECT_B))
        assert [entry.title for entry in store.visible_to(PROJECT_A)] == ["Alpha fact"]
        assert [entry.title for entry in store.visible_to(PROJECT_B)] == ["Beta fact"]

    def test_global_lessons_reach_every_project(self, store: MemoryStore) -> None:
        store.save(
            record(
                "Prefer explicit imports",
                "Wildcard imports break static analysis.",
                project_id=None,
                scope=MemoryScope.GLOBAL,
                memory_type=MemoryType.ENGINEERING,
            )
        )
        for project in (PROJECT_A, PROJECT_B):
            titles = [entry.title for entry in store.visible_to(project)]
            assert "Prefer explicit imports" in titles

    def test_global_can_be_excluded(self, store: MemoryStore) -> None:
        store.save(
            record(
                "A global lesson",
                "Applies everywhere.",
                project_id=None,
                scope=MemoryScope.GLOBAL,
                memory_type=MemoryType.ENGINEERING,
            )
        )
        assert store.visible_to(PROJECT_A, include_global=False) == []

    def test_no_project_context_sees_only_global(self, store: MemoryStore) -> None:
        store.save(record("Alpha fact", "Alpha detail."))
        store.save(
            record(
                "Global lesson",
                "Reusable knowledge.",
                project_id=None,
                scope=MemoryScope.GLOBAL,
                memory_type=MemoryType.ENGINEERING,
            )
        )
        assert [entry.title for entry in store.visible_to(None)] == ["Global lesson"]

    def test_purging_one_project_leaves_the_other(self, store: MemoryStore) -> None:
        store.save(record("Alpha fact", "Alpha detail.", project_id=PROJECT_A))
        store.save(record("Beta fact", "Beta detail.", project_id=PROJECT_B))
        assert store.purge_project(PROJECT_A) == 1
        assert len(store.visible_to(PROJECT_B)) == 1


class TestQualityControl:
    """Not everything a model says becomes knowledge."""

    def test_a_verified_test_result_is_stored(self, store: MemoryStore) -> None:
        stored, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.FAILURE,
                project_id=PROJECT_A,
                title="low_stock needs sorting",
                content="test_low_stock requires the result to be sorted, not just filtered.",
                source=MemorySource.TEST_RESULT,
                source_reference="pytest test_inventory.py::test_low_stock",
            )
        )
        assert outcome.accepted
        assert stored is not None and stored.status is MemoryStatus.ACTIVE

    def test_model_speculation_is_not_stored(self, store: MemoryStore) -> None:
        stored, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.ENGINEERING,
                project_id=PROJECT_A,
                title="Maybe use Redis",
                content="I think Redis is probably faster here, not sure though.",
                source=MemorySource.MODEL_SUGGESTION,
                source_reference="planner run",
            )
        )
        assert stored is None
        assert not outcome.accepted

    def test_untrusted_sources_need_approval(self, store: MemoryStore) -> None:
        proposal = MemoryProposal(
            type=MemoryType.ENGINEERING,
            project_id=PROJECT_A,
            title="Batch rounding uses ceiling division",
            content="Restock quantities round up to the next whole batch.",
            source=MemorySource.AGENT_INFERENCE,
            source_reference="coder run exec_1",
        )
        stored, outcome = store.propose(proposal)
        assert stored is None and outcome.requires_approval

        approved, outcome = store.propose(proposal, approved=True)
        assert approved is not None and approved.status is MemoryStatus.ACTIVE

    def test_missing_provenance_is_refused(self, store: MemoryStore) -> None:
        stored, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.PROJECT,
                project_id=PROJECT_A,
                title="Something",
                content="A claim with no traceable origin at all.",
                source=MemorySource.USER,
                source_reference="",
            )
        )
        assert stored is None and outcome.rejected
        assert "source reference" in outcome.reason

    def test_secrets_are_never_stored(self, store: MemoryStore) -> None:
        stored, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.PROJECT,
                project_id=PROJECT_A,
                title="Deployment credentials",
                content="The API_KEY = sk-abcdefghijklmnopqrstuvwxyz for production.",
                source=MemorySource.USER,
                source_reference="user message",
            )
        )
        assert stored is None and outcome.rejected
        assert "credential" in outcome.reason

    def test_protected_paths_cannot_become_memory(self, store: MemoryStore) -> None:
        """A file agents may not read must not become a memory they can."""
        stored, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.PROJECT,
                project_id=PROJECT_A,
                title="Environment configuration",
                content="Values were read from the environment file during setup.",
                source=MemorySource.PROJECT_ARTIFACT,
                source_reference=".env",
            )
        )
        assert stored is None and outcome.rejected
        assert "protected" in outcome.reason

    def test_confidence_is_capped_by_source(self) -> None:
        """An agent cannot promote its own guess by claiming high confidence."""
        proposal = MemoryProposal(
            type=MemoryType.ENGINEERING,
            project_id=PROJECT_A,
            title="An inference",
            content="Derived from reasoning over the diff.",
            source=MemorySource.AGENT_INFERENCE,
            source_reference="exec_1",
            confidence=0.99,
        )
        assert to_record(proposal, approved=True).confidence <= 0.4

    @pytest.mark.parametrize(
        "text",
        [
            "API_KEY = sk-abcdefghijklmnopqrst",
            "password: hunter2000",
            "Bearer abcdefghijklmnopqrstuvwx",
            "-----BEGIN RSA PRIVATE KEY-----",
        ],
    )
    def test_secret_detection(self, text: str) -> None:
        assert contains_secret(text)

    def test_ordinary_text_is_not_flagged(self) -> None:
        assert not contains_secret("The scheduler retries three times before escalating.")

    def test_redaction_masks_credentials(self) -> None:
        assert "sk-" not in redact("api_key = sk-abcdefghijklmnopqrst")


class TestLifecycleAndSupersession:
    def test_a_decision_can_be_superseded_without_losing_history(
        self, store: MemoryStore
    ) -> None:
        original = store.save(
            record(
                "Database choice",
                "The project will use PostgreSQL for persistence.",
                memory_type=MemoryType.DECISION,
            )
        )
        replacement = store.save(
            MemoryRecord(
                type=MemoryType.DECISION,
                project_id=PROJECT_A,
                title="Database choice",
                content="The project will use SQLite; PostgreSQL was too heavy for local use.",
                source=MemorySource.USER,
                source_reference="architecture review",
                supersedes=original.memory_id,
            )
        )

        old = store.get(original.memory_id)
        assert old is not None
        assert old.status is MemoryStatus.SUPERSEDED
        assert old.superseded_by == replacement.memory_id
        assert "PostgreSQL" in old.content, "history must not be destroyed"

    def test_current_belief_follows_the_chain(self, store: MemoryStore) -> None:
        first = store.save(record("Choice", "Use option A.", memory_type=MemoryType.DECISION))
        second = store.save(
            MemoryRecord(
                type=MemoryType.DECISION,
                project_id=PROJECT_A,
                title="Choice",
                content="Use option B instead.",
                source=MemorySource.USER,
                source_reference="review",
                supersedes=first.memory_id,
            )
        )
        current = store.current_belief(first.memory_id)
        assert current is not None and current.memory_id == second.memory_id

    def test_history_explains_why(self, store: MemoryStore) -> None:
        """'Why does Edith believe this' must be answerable."""
        first = store.save(record("Choice", "Use PostgreSQL.", memory_type=MemoryType.DECISION))
        second = store.save(
            MemoryRecord(
                type=MemoryType.DECISION,
                project_id=PROJECT_A,
                title="Choice",
                content="Use SQLite.",
                source=MemorySource.USER,
                source_reference="review",
                supersedes=first.memory_id,
            )
        )
        chain = store.history(second.memory_id)
        assert [entry.memory_id for entry in chain] == [second.memory_id, first.memory_id]

    def test_superseded_memories_are_not_retrieved(self, store: MemoryStore) -> None:
        first = store.save(record("Database", "Use PostgreSQL.", memory_type=MemoryType.DECISION))
        store.save(
            MemoryRecord(
                type=MemoryType.DECISION,
                project_id=PROJECT_A,
                title="Database",
                content="Use SQLite.",
                source=MemorySource.USER,
                source_reference="review",
                supersedes=first.memory_id,
            )
        )
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="database", project_id=PROJECT_A)
        )
        rendered = bundle.render()
        assert "SQLite" in rendered
        assert "PostgreSQL" not in rendered

    def test_archiving_hides_without_deleting(self, store: MemoryStore) -> None:
        saved = store.save(record("Old convention", "We used tabs."))
        assert store.archive(saved.memory_id)
        assert store.visible_to(PROJECT_A) == []
        assert store.get(saved.memory_id) is not None

    def test_deletion_is_possible(self, store: MemoryStore) -> None:
        """Memory must be deletable (CLAUDE.md)."""
        saved = store.save(record("Mistake", "Stored in error."))
        assert store.delete(saved.memory_id)
        assert store.get(saved.memory_id) is None

    def test_recurrence_can_be_incremented(self, store: MemoryStore) -> None:
        saved = store.save(
            record(
                "Recurring failure",
                "The model drops functions.",
                memory_type=MemoryType.FAILURE,
            )
        )
        updated = store.bump_recurrence(saved.memory_id)
        assert updated is not None and updated.recurrence_count == 2


class TestRetrieval:
    def test_relevance_beats_recency(self, store: MemoryStore) -> None:
        store.save(record("Unrelated note", "Something about deployment pipelines."))
        store.save(
            record("Sorting requirement", "low_stock results must be sorted alphabetically.")
        )
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="low_stock sorted", project_id=PROJECT_A)
        )
        assert bundle.memories[0].memory.title == "Sorting requirement"

    def test_rationale_explains_the_selection(self, store: MemoryStore) -> None:
        store.save(record("Sorting requirement", "low_stock must be sorted."))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="low_stock sorted", project_id=PROJECT_A)
        )
        assert bundle.rationale and "score" in bundle.rationale[0]

    def test_budget_is_respected(self, store: MemoryStore) -> None:
        for index in range(20):
            store.save(record(f"Sorting note {index}", "low_stock must be sorted. " * 10))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(
                query="low_stock sorted", project_id=PROJECT_A, max_memories=3, max_chars=600
            )
        )
        assert len(bundle.memories) <= 3
        assert bundle.estimated_context_chars <= 600
        assert bundle.truncated

    def test_irrelevant_memories_are_excluded(self, store: MemoryStore) -> None:
        store.save(record("Deployment pipeline", "Deploys run on Tuesdays."))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="zzzqqq unrelated topic", project_id=PROJECT_A)
        )
        assert bundle.is_empty

    def test_tags_boost_relevance(self, store: MemoryStore) -> None:
        store.save(record("Generic note", "Some content about the system."))
        store.save(record("Tagged note", "Some content about the system.", tags=("sorting",)))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="content system", project_id=PROJECT_A, tags=("sorting",))
        )
        assert bundle.memories[0].memory.title == "Tagged note"

    def test_type_filtering(self, store: MemoryStore) -> None:
        store.save(record("A decision", "Chose SQLite.", memory_type=MemoryType.DECISION))
        store.save(record("A failure", "Chose SQLite.", memory_type=MemoryType.FAILURE))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(
                query="SQLite", project_id=PROJECT_A, types=(MemoryType.FAILURE,)
            )
        )
        assert all(entry.memory.type is MemoryType.FAILURE for entry in bundle.memories)

    def test_low_confidence_is_filtered(self, store: MemoryStore) -> None:
        store.save(record("Weak claim", "Sorting might matter.", confidence=0.1))
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="sorting", project_id=PROJECT_A, min_confidence=0.5)
        )
        assert bundle.is_empty

    def test_rendered_output_carries_provenance(self, store: MemoryStore) -> None:
        """An agent must be able to weigh a claim, which requires knowing its source."""
        store.save(
            record(
                "Sorting requirement",
                "low_stock must be sorted.",
                source=MemorySource.TEST_RESULT,
            )
        )
        rendered = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="sorting low_stock", project_id=PROJECT_A)
        ).render()
        assert "TEST_RESULT" in rendered

    def test_access_is_recorded(self, store: MemoryStore) -> None:
        saved = store.save(record("Sorting requirement", "low_stock must be sorted."))
        MemoryRetriever(store).retrieve(
            RetrievalRequest(query="sorting low_stock", project_id=PROJECT_A)
        )
        reloaded = store.get(saved.memory_id)
        assert reloaded is not None and reloaded.access_count == 1

    def test_empty_store_returns_an_empty_bundle(self, store: MemoryStore) -> None:
        bundle = MemoryRetriever(store).retrieve(
            RetrievalRequest(query="anything", project_id=PROJECT_A)
        )
        assert bundle.is_empty
        assert bundle.render() == "(no relevant prior knowledge)"


class TestConsolidation:
    def test_similar_memories_are_grouped(self, store: MemoryStore) -> None:
        first = record(
            "Model omits optional fields",
            "The 3B model frequently leaves optional structured fields empty.",
            memory_type=MemoryType.ENGINEERING,
        )
        second = record(
            "Optional fields left empty",
            "The 3B model frequently leaves optional structured fields empty entirely.",
            memory_type=MemoryType.ENGINEERING,
        )
        groups = find_duplicate_groups([first, second])
        assert len(groups) == 1
        assert len(groups[0].records) == 2

    def test_distinct_memories_are_not_grouped(self) -> None:
        first = record("Sorting", "Results must be sorted alphabetically before returning.")
        second = record("Deployment", "Deploys happen on Tuesday afternoons via CI.")
        assert find_duplicate_groups([first, second]) == []

    def test_different_projects_never_merge(self) -> None:
        first = record("Same lesson", "Identical wording here.", project_id=PROJECT_A)
        second = record("Same lesson", "Identical wording here.", project_id=PROJECT_B)
        assert find_duplicate_groups([first, second]) == []

    def test_consolidation_preserves_originals(self, store: MemoryStore) -> None:
        """Merging must never destroy the evidence behind a claim."""
        first = store.save(
            record(
                "Model omits fields",
                "The 3B model leaves optional structured fields empty.",
                memory_type=MemoryType.ENGINEERING,
            )
        )
        second = store.save(
            record(
                "Model omits fields often",
                "The 3B model leaves optional structured fields empty often.",
                memory_type=MemoryType.ENGINEERING,
            )
        )
        merged = consolidate_project(store, PROJECT_A)
        assert merged

        survivors = [store.get(first.memory_id), store.get(second.memory_id)]
        assert all(entry is not None for entry in survivors)
        assert any(entry.status is MemoryStatus.SUPERSEDED for entry in survivors)

    def test_deterministic_source_wins_as_primary(self) -> None:
        inferred = record(
            "Sorting matters",
            "Results probably need sorting for the test to pass.",
            source=MemorySource.AGENT_INFERENCE,
            confidence=0.4,
        )
        observed = record(
            "Sorting matters",
            "Results need sorting for the test to pass.",
            source=MemorySource.TEST_RESULT,
            confidence=0.95,
        )
        groups = find_duplicate_groups([inferred, observed])
        assert groups[0].primary.source is MemorySource.TEST_RESULT

    def test_existing_match_is_found_before_duplicating(self, store: MemoryStore) -> None:
        store.save(record("Sorting matters", "Results need sorting for the test to pass."))
        candidate = record("Sorting matters", "Results need sorting for the test to pass.")
        assert find_existing_match(store, candidate) is not None

    def test_similarity_is_symmetric(self) -> None:
        first = record("A", "shared vocabulary between these two records here")
        second = record("B", "shared vocabulary between these two records here")
        assert similarity(first, second) == similarity(second, first)


class TestPersistenceAndRecovery:
    def test_memory_survives_a_restart(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        with open_memory(state_dir) as first:
            first.save(record("Persistent fact", "This must outlive the process."))

        with open_memory(state_dir) as second:
            titles = [entry.title for entry in second.visible_to(PROJECT_A)]
            assert "Persistent fact" in titles

    def test_supersession_survives_a_restart(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        with open_memory(state_dir) as first:
            original = first.save(
                record("Choice", "Use PostgreSQL.", memory_type=MemoryType.DECISION)
            )
            first.save(
                MemoryRecord(
                    type=MemoryType.DECISION,
                    project_id=PROJECT_A,
                    title="Choice",
                    content="Use SQLite.",
                    source=MemorySource.USER,
                    source_reference="review",
                    supersedes=original.memory_id,
                )
            )
            original_id = original.memory_id

        with open_memory(state_dir) as second:
            reloaded = second.get(original_id)
            assert reloaded is not None
            assert reloaded.status is MemoryStatus.SUPERSEDED

    def test_schema_version_mismatch_is_refused(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        with open_memory(state_dir) as first:
            first._connection.execute(
                "UPDATE memory_meta SET value = '99' WHERE key = 'schema_version'"
            )
        with pytest.raises(ConfigurationError, match="schema v99"):
            open_memory(state_dir)

    def test_a_corrupt_row_is_classified_not_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        from edith.memory.store import MemoryCorruptionError

        state_dir = tmp_path / "state"
        with open_memory(state_dir) as store:
            saved = store.save(record("Fact", "Some content here."))
            store._connection.execute(
                "UPDATE memories SET payload = ? WHERE memory_id = ?",
                ("{not valid json", saved.memory_id),
            )
            with pytest.raises(MemoryCorruptionError):
                store.get(saved.memory_id)
