"""Research: evidence-backed, provenance-preserving external information.

Optional and internet-dependent. Everything else in Edith works without it, and when it is
unavailable it says so rather than inventing an answer.
"""

from .agent import (
    ResearchAgent,
    ResearchInput,
    ResearchOutput,
    build_report,
    build_source_block,
    detect_conflicts,
    ground_claims,
)
from .extract import (
    classify_source,
    extract_text,
    fence,
    find_injection_attempts,
    neutralize,
)
from .provider import (
    DuckDuckGoProvider,
    OfflineProvider,
    ResearchCache,
    ResearchProvider,
    ResearchTimeoutError,
    ResearchUnavailableError,
    SourceUnavailableError,
)
from .schema import (
    TIER_WEIGHT,
    Claim,
    Conflict,
    Evidence,
    ResearchReport,
    RetrievalStatus,
    SearchHit,
    Source,
    SourceTier,
)

__all__ = [
    "TIER_WEIGHT",
    "Claim",
    "Conflict",
    "DuckDuckGoProvider",
    "Evidence",
    "OfflineProvider",
    "ResearchAgent",
    "ResearchCache",
    "ResearchInput",
    "ResearchOutput",
    "ResearchProvider",
    "ResearchReport",
    "ResearchTimeoutError",
    "ResearchUnavailableError",
    "RetrievalStatus",
    "SearchHit",
    "Source",
    "SourceTier",
    "SourceUnavailableError",
    "build_report",
    "build_source_block",
    "classify_source",
    "detect_conflicts",
    "extract_text",
    "fence",
    "find_injection_attempts",
    "ground_claims",
    "neutralize",
]
