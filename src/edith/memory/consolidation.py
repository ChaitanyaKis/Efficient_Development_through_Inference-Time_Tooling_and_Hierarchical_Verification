"""Deterministic grouping and deduplication.

Memory accumulates near-duplicates: the same failure observed in three runs, the same lesson
phrased three ways. Left alone, retrieval spends its budget saying one thing repeatedly.

Everything here is deterministic. An LLM summariser that rewrites memory in place would
destroy the provenance that makes memory trustworthy, so grouping only *identifies*
candidates and merging only ever supersedes -- the originals stay recoverable through the
supersession chain (CLAUDE.md: originals must remain recoverable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edith.observability.logging import get_logger

from .retrieval import tokenize
from .schema import MemoryRecord, MemorySource, MemoryStatus, MemoryType
from .store import MemoryStore

logger = get_logger(__name__)

#: Jaccard similarity above which two memories are considered near-duplicates. Tuned to
#: catch rephrasing without merging genuinely distinct lessons that share vocabulary.
DUPLICATE_THRESHOLD = 0.6

#: Above this, the two are treated as the same claim outright.
IDENTICAL_THRESHOLD = 0.9


def similarity(left: MemoryRecord, right: MemoryRecord) -> float:
    """Jaccard similarity over the combined title and content vocabulary."""
    left_terms = tokenize(f"{left.title} {left.content}")
    right_terms = tokenize(f"{right.title} {right.content}")
    if not left_terms or not right_terms:
        return 0.0
    intersection = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return intersection / union if union else 0.0


@dataclass
class DuplicateGroup:
    """A set of memories making substantially the same claim."""

    records: list[MemoryRecord] = field(default_factory=list)

    @property
    def primary(self) -> MemoryRecord:
        """The record best suited to represent the group.

        Chosen by evidence quality first, then by how often it has been observed: a lesson
        derived from a test that ran outranks the same lesson inferred by an agent.
        """
        return max(
            self.records,
            key=lambda record: (
                record.source in _DETERMINISTIC_SOURCES,
                record.confidence,
                record.recurrence_count,
                record.importance,
            ),
        )

    @property
    def duplicates(self) -> list[MemoryRecord]:
        """Every record in the group except the primary."""
        primary_id = self.primary.memory_id
        return [record for record in self.records if record.memory_id != primary_id]

    @property
    def total_recurrence(self) -> int:
        """Combined observation count across the group."""
        return sum(record.recurrence_count for record in self.records)


_DETERMINISTIC_SOURCES = frozenset(
    {
        MemorySource.TEST_RESULT,
        MemorySource.TOOL_OBSERVATION,
        MemorySource.USER,
        MemorySource.PROJECT_ARTIFACT,
    }
)


def find_duplicate_groups(
    records: list[MemoryRecord], *, threshold: float = DUPLICATE_THRESHOLD
) -> list[DuplicateGroup]:
    """Group near-duplicate memories.

    Only compares records of the same type and project: a FAILURE and a DECISION that share
    vocabulary are not duplicates, and two projects' facts are never merged regardless of
    how similar they read.
    """
    groups: list[DuplicateGroup] = []
    assigned: set[str] = set()

    ordered = sorted(records, key=lambda record: record.memory_id)
    for index, record in enumerate(ordered):
        if record.memory_id in assigned:
            continue
        group = DuplicateGroup(records=[record])
        assigned.add(record.memory_id)

        for other in ordered[index + 1 :]:
            if other.memory_id in assigned:
                continue
            if other.type is not record.type or other.project_id != record.project_id:
                continue
            if similarity(record, other) >= threshold:
                group.records.append(other)
                assigned.add(other.memory_id)

        if len(group.records) > 1:
            groups.append(group)
    return groups


def find_existing_match(
    store: MemoryStore,
    candidate: MemoryRecord,
    *,
    threshold: float = IDENTICAL_THRESHOLD,
) -> MemoryRecord | None:
    """Find an existing memory making the same claim as ``candidate``.

    Used before storing, so a repeated observation increments a recurrence count instead of
    creating a fourth copy of the same lesson.
    """
    for existing in store.visible_to(candidate.project_id, types=(candidate.type,)):
        if existing.memory_id == candidate.memory_id:
            continue
        if similarity(existing, candidate) >= threshold:
            return existing
    return None


def consolidate_group(
    store: MemoryStore, group: DuplicateGroup, *, merged_title: str | None = None
) -> MemoryRecord | None:
    """Fold a duplicate group into its primary record.

    The primary absorbs the group's combined recurrence count and supersedes the rest. The
    duplicates become ``SUPERSEDED`` rather than deleted, so the evidence trail behind the
    consolidated claim stays walkable.
    """
    if len(group.records) < 2:
        return None

    primary = group.primary
    primary.recurrence_count = group.total_recurrence
    if merged_title:
        primary.title = merged_title
    primary.metadata = {
        **primary.metadata,
        "consolidated_from": ",".join(
            record.memory_id for record in group.duplicates
        )[:400],
    }
    store.save(primary)

    for duplicate in group.duplicates:
        duplicate.superseded_by = primary.memory_id
        duplicate.status = MemoryStatus.SUPERSEDED
        store.save(duplicate)

    logger.info(
        "memory.consolidated",
        primary=primary.memory_id,
        absorbed=len(group.duplicates),
        recurrence=primary.recurrence_count,
    )
    return primary


def consolidate_project(
    store: MemoryStore, project_id: str | None, *, threshold: float = DUPLICATE_THRESHOLD
) -> list[MemoryRecord]:
    """Consolidate duplicates across one project's memories."""
    records = store.visible_to(project_id, include_global=False)
    consolidated: list[MemoryRecord] = []
    for group in find_duplicate_groups(records, threshold=threshold):
        merged = consolidate_group(store, group)
        if merged is not None:
            consolidated.append(merged)
    return consolidated


__all__ = [
    "DUPLICATE_THRESHOLD",
    "IDENTICAL_THRESHOLD",
    "DuplicateGroup",
    "MemoryType",
    "consolidate_group",
    "consolidate_project",
    "find_duplicate_groups",
    "find_existing_match",
    "similarity",
]
