"""Relevance-based memory retrieval.

Ranking is entirely deterministic: exact phrase, keyword overlap, tags, type, importance,
recurrence, recency, access frequency, confidence. No embeddings and no vector store --
CLAUDE.md warns against starting there, and lexical signals are enough to demonstrate
whether memory helps at all, which is the question M3 has to answer first.

:class:`MemoryRanker` is the seam. A semantic ranker implements the same protocol and slots
in without any agent changing, and only then would a vector store need justifying with
measurements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from edith.observability.logging import get_logger
from edith.schemas.common import utc_now

from .schema import (
    DEFAULT_MIN_CONFIDENCE,
    MemoryBundle,
    MemoryRecord,
    MemoryType,
    ScoredMemory,
)
from .store import MemoryStore

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

#: Words too common to signal relevance between a task and a memory.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "when", "then",
        "should", "must", "will", "would", "could", "have", "has", "had", "was", "were",
        "are", "not", "but", "all", "any", "can", "its", "use", "using", "add", "make",
        "fix", "test", "tests", "code", "file", "files", "function", "functions",
    }
)

#: Days after which recency stops contributing. Engineering lessons age slowly; a month is
#: long enough that genuinely stale project facts fade without discarding real knowledge.
RECENCY_HORIZON_DAYS = 30.0


def tokenize(text: str) -> set[str]:
    """Lower-cased content words from free text."""
    return {
        word.lower() for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS
    }


@dataclass(frozen=True)
class RetrievalRequest:
    """What retrieval is being asked for."""

    query: str
    project_id: str | None = None
    types: tuple[MemoryType, ...] = ()
    tags: tuple[str, ...] = ()
    agent: str | None = None
    max_memories: int = 6
    max_chars: int = 2000
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    include_global: bool = True
    #: Real verification output, when retrieval follows a failure. The strongest available
    #: retrieval key: an error names the actual symptom, whereas a task title only names
    #: the intent.
    error_text: str = ""
    #: Files the task touches. A lesson about a component is relevant when that component
    #: is the one being changed.
    paths: tuple[str, ...] = ()
    #: Minimum score a memory must reach to be admitted, set by the active strategy.
    min_score: float = 0.0


class MemoryRanker(Protocol):
    """Pluggable relevance strategy."""

    def score(
        self, record: MemoryRecord, request: RetrievalRequest
    ) -> tuple[float, list[str]]:
        """Return a relevance score and the reasons behind it."""


class LexicalRanker:
    """Deterministic ranking from keyword, tag, and metadata signals.

    Scores are additive and each contribution records a reason, so a retrieval that pulls
    the wrong memory can be diagnosed rather than guessed at.
    """

    def score(
        self, record: MemoryRecord, request: RetrievalRequest
    ) -> tuple[float, list[str]]:
        """Score one record against a request.

        Relevance is computed first and acts as a gate. Metadata -- importance, confidence,
        recurrence, recency -- only ever *breaks ties between relevant memories*; on its own
        it can never make an unrelated memory eligible. Without that gate every retrieval
        returns whatever happens to be most important, which is how a memory system starts
        injecting noise into every prompt.
        """
        relevance, reasons = self._relevance(record, request)
        if relevance <= 0.0:
            return (0.0, [])
        return (relevance + self._metadata_bonus(record, reasons), reasons)

    @staticmethod
    def _failure_relevance(
        record: MemoryRecord, request: RetrievalRequest
    ) -> tuple[float, list[str]]:
        """Score a memory against the observed failure and the files being changed.

        Weighted above task-text overlap on purpose. "test_low_stock failed with
        AssertionError" identifies the problem far more precisely than "fix the inventory
        module", so a lesson matching the error deserves to outrank one matching the title.
        """
        reasons: list[str] = []
        score = 0.0

        if request.error_text:
            error_terms = tokenize(request.error_text)
            record_terms = tokenize(f"{record.title} {record.content}")
            overlap = error_terms & record_terms
            if overlap:
                score += min(len(overlap) * 2.5, 15.0)
                reasons.append(
                    f"matches {len(overlap)} term(s) in the observed failure"
                )

        for path in request.paths:
            stem = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if not stem:
                continue
            haystack = f"{record.title} {record.content} {' ' .join(record.tags)}".lower()
            if stem in haystack:
                score += 5.0
                reasons.append(f"mentions the component being changed ({stem})")
                break

        return (score, reasons)

    def _relevance(
        self, record: MemoryRecord, request: RetrievalRequest
    ) -> tuple[float, list[str]]:
        """Score only the signals that tie this memory to *this* query."""
        reasons: list[str] = []
        score = 0.0

        failure_score, failure_reasons = self._failure_relevance(record, request)
        score += failure_score
        reasons.extend(failure_reasons)

        query = request.query.lower()
        query_terms = tokenize(request.query)
        title_terms = tokenize(record.title)
        content_terms = tokenize(record.content)

        # An exact phrase match is the strongest lexical signal available.
        if record.title.lower() in query or query in record.title.lower():
            score += 12.0
            reasons.append("title matches the query directly")

        title_overlap = query_terms & title_terms
        if title_overlap:
            score += min(len(title_overlap) * 3.0, 12.0)
            reasons.append(f"title shares {len(title_overlap)} term(s) with the query")

        content_overlap = query_terms & content_terms
        if content_overlap:
            score += min(len(content_overlap) * 1.5, 9.0)
            reasons.append(f"content shares {len(content_overlap)} term(s)")

        if request.tags:
            tag_overlap = {tag.lower() for tag in request.tags} & {
                tag.lower() for tag in record.tags
            }
            if tag_overlap:
                score += min(len(tag_overlap) * 4.0, 12.0)
                reasons.append(f"tagged {', '.join(sorted(tag_overlap))}")

        tag_terms = {tag.lower() for tag in record.tags}
        if query_terms & tag_terms:
            score += 2.0
            reasons.append("a tag appears in the query")

        return (score, reasons)

    @staticmethod
    def _metadata_bonus(record: MemoryRecord, reasons: list[str]) -> float:
        """Tie-breakers among already-relevant memories.

        An important, repeatedly-observed, well-evidenced, recently-touched memory should
        outrank an equally relevant one lacking those properties.
        """
        bonus = record.importance / 25.0
        if record.importance >= 75:
            reasons.append("marked high importance")

        if record.recurrence_count > 1:
            bonus += min(record.recurrence_count * 0.8, 4.0)
            reasons.append(f"observed {record.recurrence_count} times")

        bonus += record.confidence * 3.0

        age_days = (utc_now() - record.updated_at).total_seconds() / 86_400.0
        if age_days < RECENCY_HORIZON_DAYS:
            bonus += (1.0 - age_days / RECENCY_HORIZON_DAYS) * 2.0

        if record.access_count:
            bonus += min(record.access_count * 0.2, 2.0)

        return bonus


class MemoryRetriever:
    """Retrieves the most relevant memories within a budget."""

    def __init__(self, store: MemoryStore, ranker: MemoryRanker | None = None) -> None:
        self.store = store
        self.ranker: MemoryRanker = ranker or LexicalRanker()

    def retrieve(self, request: RetrievalRequest) -> MemoryBundle:
        """Return the highest-scoring memories a project may see.

        Project isolation is delegated to the store, which enforces it in SQL. This method
        never widens that set -- it only ranks and truncates what it is given.

        .. warning::

           **This is the low-level, unbudgeted path**, for the CLI, for administration, and
           for inspecting what retrieval would return. It knows nothing about an execution's
           allowance, so calling it repeatedly in a repair loop reproduces exactly the
           unbounded accumulation M3.1 measured.

           Autonomous execution must go through
           :meth:`~edith.memory.governor.MemoryGovernor.request`, which takes an execution
           id and a purpose rather than free retrieval parameters, and which is the only
           caller of this method inside the loop.
        """
        candidates = self.store.visible_to(
            request.project_id,
            types=request.types,
            include_global=request.include_global,
        )

        scored: list[ScoredMemory] = []
        for record in candidates:
            if record.confidence < request.min_confidence:
                continue
            score, reasons = self.ranker.score(record, request)
            if score <= 0 or score < request.min_score:
                continue
            scored.append(ScoredMemory(memory=record, score=round(score, 2), reasons=reasons))

        scored.sort(key=lambda entry: (-entry.score, entry.memory.memory_id))

        selected: list[ScoredMemory] = []
        rationale: list[str] = []
        used = 0
        truncated = False

        for entry in scored:
            if len(selected) >= request.max_memories:
                truncated = True
                break
            cost = len(entry.memory.title) + len(entry.memory.content) + 60
            if used + cost > request.max_chars:
                truncated = True
                continue
            selected.append(entry)
            used += cost
            rationale.append(
                f"{entry.memory.title} (score {entry.score}): "
                f"{'; '.join(entry.reasons[:2]) or 'general relevance'}"
            )

        for entry in selected:
            self.store.record_access(entry.memory)

        bundle = MemoryBundle(
            query=request.query,
            memories=selected,
            rationale=rationale,
            estimated_context_chars=used,
            considered=len(candidates),
            truncated=truncated,
        )
        logger.info(
            "memory.retrieved",
            project_id=request.project_id,
            considered=len(candidates),
            returned=len(selected),
            chars=used,
            agent=request.agent,
        )
        return bundle
