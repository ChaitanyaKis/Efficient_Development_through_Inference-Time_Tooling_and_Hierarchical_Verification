"""Research types: sources, claims, evidence, and reports.

The rule this schema exists to enforce is that **a claim without a source is not research**.
A model recalling something from training is not retrieval, and the type system refuses to
let the two be confused: a :class:`Claim` cannot be constructed without at least one
:class:`Evidence` pointing at a real, fetched source.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from edith.schemas.common import EdithModel, new_id, utc_now


class SourceTier(StrEnum):
    """How much weight a source's claims deserve.

    Explicitly *not* search ranking. A result being first says something about an engine's
    relevance model, not about whether the content is authoritative.
    """

    #: Official documentation from the project being asked about.
    OFFICIAL_DOCS = "OFFICIAL_DOCS"
    #: A standard or specification (RFC, W3C, ISO, PEP).
    SPECIFICATION = "SPECIFICATION"
    #: The primary artifact itself: source code, a changelog, a release note.
    PRIMARY = "PRIMARY"
    #: Peer-reviewed or preprint academic work.
    ACADEMIC = "ACADEMIC"
    #: A reputable technical publisher or well-known engineering blog.
    REPUTABLE = "REPUTABLE"
    #: Derived reporting: tutorials, summaries, aggregators.
    SECONDARY = "SECONDARY"
    #: Forums, Q&A sites, issue comments.
    COMMUNITY = "COMMUNITY"
    #: Anything unrecognised. Deliberately the default.
    UNKNOWN = "UNKNOWN"


#: Weight per tier, used when sources disagree. Never used to hide a conflict -- only to
#: say which side a recommendation leans towards while still reporting both.
TIER_WEIGHT: dict[SourceTier, float] = {
    SourceTier.OFFICIAL_DOCS: 1.0,
    SourceTier.SPECIFICATION: 1.0,
    SourceTier.PRIMARY: 0.9,
    SourceTier.ACADEMIC: 0.8,
    SourceTier.REPUTABLE: 0.6,
    SourceTier.SECONDARY: 0.4,
    SourceTier.COMMUNITY: 0.3,
    SourceTier.UNKNOWN: 0.2,
}


class RetrievalStatus(StrEnum):
    """What happened when a source was fetched."""

    OK = "OK"
    CACHED = "CACHED"
    UNAVAILABLE = "UNAVAILABLE"
    PARSE_FAILURE = "PARSE_FAILURE"
    TIMEOUT = "TIMEOUT"
    BLOCKED = "BLOCKED"


class SearchHit(EdithModel):
    """One result from a search provider, before it has been fetched."""

    url: str = Field(min_length=1, max_length=2000)
    title: str = Field(default="", max_length=500)
    snippet: str = Field(default="", max_length=2000)
    rank: int = Field(default=0, ge=0)
    provider: str = ""


class Source(EdithModel):
    """A fetched document and everything known about its origin."""

    source_id: str = Field(default_factory=lambda: new_id("src"))
    url: str = Field(min_length=1, max_length=2000)
    title: str = Field(default="", max_length=500)
    tier: SourceTier = SourceTier.UNKNOWN
    retrieved_at: datetime = Field(default_factory=utc_now)
    published_at: datetime | None = None
    status: RetrievalStatus = RetrievalStatus.OK
    #: Digest of the fetched body in the cache. The content itself is never inlined here.
    content_reference: str = ""
    content_hash: str = ""
    #: Extracted plain text, truncated for prompt use.
    excerpt: str = Field(default="", max_length=20_000)
    error: str = ""

    @property
    def usable(self) -> bool:
        """Whether this source produced content that can be reasoned over."""
        return self.status in {RetrievalStatus.OK, RetrievalStatus.CACHED} and bool(
            self.excerpt.strip()
        )

    @property
    def weight(self) -> float:
        """Authority weight from the source's tier."""
        return TIER_WEIGHT.get(self.tier, 0.2)


class Evidence(EdithModel):
    """A specific passage in a specific source that supports a claim."""

    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=2000)
    quote: str = Field(min_length=1, max_length=2000)
    tier: SourceTier = SourceTier.UNKNOWN


class Claim(EdithModel):
    """One assertion, with the sources that support it.

    ``supported_by`` is required and must be non-empty. That constraint is the whole point:
    it makes "the model asserted this with no source" unrepresentable.
    """

    statement: str = Field(min_length=1, max_length=1000)
    supported_by: list[Evidence] = Field(min_length=1)
    #: Sources that contradict this claim, when any were found.
    contradicted_by: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_support(self) -> Claim:
        if not self.supported_by:
            raise ValueError(
                "a claim must cite at least one source; unsupported model knowledge is "
                "not research"
            )
        return self

    @property
    def disputed(self) -> bool:
        """Whether any source contradicts this claim."""
        return bool(self.contradicted_by)

    @property
    def source_urls(self) -> list[str]:
        """Every URL supporting this claim."""
        return [evidence.url for evidence in self.supported_by]


class Conflict(EdithModel):
    """A disagreement between sources, surfaced rather than resolved silently."""

    topic: str = Field(min_length=1, max_length=500)
    positions: list[str] = Field(min_length=2, max_length=6)
    evidence: list[Evidence] = Field(default_factory=list)
    note: str = Field(default="", max_length=1000)


class ResearchReport(EdithModel):
    """The structured output of a research question.

    A recommendation is traceable to claims, claims are traceable to evidence, and evidence
    is traceable to a fetched source. Every step of that chain is checkable.
    """

    report_id: str = Field(default_factory=lambda: new_id("res"))
    question: str = Field(min_length=1, max_length=1000)
    queries_used: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    summary: str = Field(default="", max_length=4000)
    recommendation: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    #: Set when research could not run at all.
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        """Whether research actually ran."""
        return not self.unavailable_reason

    @property
    def has_conflicts(self) -> bool:
        """Whether sources disagreed."""
        return bool(self.conflicts)

    @property
    def usable_sources(self) -> list[Source]:
        """Sources that yielded content."""
        return [source for source in self.sources if source.usable]

    def citations(self) -> list[str]:
        """Every cited URL, in a stable order."""
        seen: dict[str, None] = {}
        for claim in self.claims:
            for url in claim.source_urls:
                seen.setdefault(url, None)
        return list(seen)

    def render(self) -> str:
        """Render the report for a human or an Architect agent."""
        if not self.available:
            return f"RESEARCH UNAVAILABLE: {self.unavailable_reason}"

        lines = [f"QUESTION: {self.question}", ""]
        if self.summary:
            lines.extend([self.summary, ""])

        if self.claims:
            lines.append("CLAIMS:")
            for claim in self.claims:
                marker = " [DISPUTED]" if claim.disputed else ""
                lines.append(f"- {claim.statement}{marker}")
                for evidence in claim.supported_by[:3]:
                    lines.append(f"    source: {evidence.url} ({evidence.tier})")
            lines.append("")

        if self.conflicts:
            lines.append("CONFLICTS (sources disagree):")
            for conflict in self.conflicts:
                lines.append(f"- {conflict.topic}")
                for position in conflict.positions:
                    lines.append(f"    * {position}")
            lines.append("")

        if self.recommendation:
            lines.extend(
                [
                    "RECOMMENDATION (advisory only - an Architect decides):",
                    self.recommendation,
                ]
            )
        return "\n".join(lines)
