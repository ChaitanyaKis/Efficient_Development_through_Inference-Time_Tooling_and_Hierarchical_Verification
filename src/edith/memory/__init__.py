"""Memory: durable, provenance-bearing, project-scoped knowledge."""

from .consolidation import (
    DuplicateGroup,
    consolidate_project,
    find_duplicate_groups,
    find_existing_match,
    similarity,
)
from .retrieval import LexicalRanker, MemoryRanker, MemoryRetriever, RetrievalRequest
from .schema import (
    AUTO_TRUSTED_SOURCES,
    SOURCE_CONFIDENCE,
    MemoryBundle,
    MemoryProposal,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    MemoryType,
    ScoredMemory,
)
from .store import (
    MemoryCorruptionError,
    MemoryStore,
    MemoryUnavailableError,
    open_memory,
)
from .validation import ValidationOutcome, contains_secret, redact, validate

__all__ = [
    "AUTO_TRUSTED_SOURCES",
    "SOURCE_CONFIDENCE",
    "DuplicateGroup",
    "LexicalRanker",
    "MemoryBundle",
    "MemoryCorruptionError",
    "MemoryProposal",
    "MemoryRanker",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "MemoryStore",
    "MemoryType",
    "MemoryUnavailableError",
    "RetrievalRequest",
    "ScoredMemory",
    "ValidationOutcome",
    "consolidate_project",
    "contains_secret",
    "find_duplicate_groups",
    "find_existing_match",
    "open_memory",
    "redact",
    "similarity",
    "validate",
]
