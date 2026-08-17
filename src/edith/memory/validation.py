"""What is allowed to become memory.

Memory pollution is the failure mode that kills a memory system: store everything a model
says and retrieval degrades into noise, while the confident-sounding wrong entries actively
mislead. So storing is a *gated* operation, not a side effect of an agent talking.

Three gates, in order:

1. **Provenance.** No source reference, no memory. A claim that cannot be checked is not
   knowledge.
2. **Trust.** Deterministic sources store directly. Model suggestions and agent inferences
   are held as proposals requiring approval -- they may be right, but "may be right" is not
   what a knowledge base is for.
3. **Safety.** Secrets and protected paths never enter the store, whatever the source says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from edith.errors import EdithError, FailureCategory
from edith.observability.logging import get_logger
from edith.schemas.common import utc_now

from .schema import (
    AUTO_TRUSTED_SOURCES,
    SOURCE_CONFIDENCE,
    MemoryProposal,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
)

logger = get_logger(__name__)


class MemoryValidationError(EdithError):
    """A proposed memory failed validation."""

    category = FailureCategory.VALIDATION_FAILURE


#: Content that looks like a credential. Deliberately broad: a false positive costs one
#: rejected memory, a false negative writes a secret to disk permanently.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|credential)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bAWS_SECRET_ACCESS_KEY\b"),
)

#: Paths whose *contents* must never be summarised into memory. Mirrors the M1 protected
#: list: a file an agent may not read must not become a memory it can read instead.
_PROTECTED_MARKERS: tuple[str, ...] = (
    ".env",
    "secrets/",
    "credentials/",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".pfx",
    ".ssh/",
    ".aws/",
    ".npmrc",
    ".pypirc",
)

#: Phrases that mark a claim as speculation rather than observation. A memory hedged like
#: this is a hypothesis; storing it as fact is how a knowledge base starts lying.
_SPECULATION_MARKERS: tuple[str, ...] = (
    "i think",
    "probably",
    "might be",
    "maybe",
    "not sure",
    "i believe",
    "it seems",
    "possibly",
    "could be",
    "presumably",
)

MIN_CONTENT_CHARS = 10


@dataclass(frozen=True)
class ValidationOutcome:
    """The result of validating a proposal."""

    accepted: bool
    reason: str
    #: True when the content is sound but the source is not trusted enough to auto-store.
    requires_approval: bool = False

    @property
    def rejected(self) -> bool:
        """Whether the proposal must not be stored at all."""
        return not self.accepted and not self.requires_approval


def contains_secret(text: str) -> bool:
    """Whether text looks like it carries a credential."""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def references_protected_path(reference: str, content: str) -> bool:
    """Whether a memory is derived from a location agents may not read."""
    haystack = f"{reference} {content}".lower().replace("\\", "/")
    return any(marker in haystack for marker in _PROTECTED_MARKERS)


def looks_speculative(text: str) -> bool:
    """Whether the claim hedges rather than asserts."""
    lowered = text.lower()
    return any(marker in lowered for marker in _SPECULATION_MARKERS)


def validate(proposal: MemoryProposal) -> ValidationOutcome:
    """Decide whether a proposal may be stored, and with what standing.

    Returns rather than raises: a rejected proposal is a normal, loggable event, and the
    caller usually wants to record *that* rather than crash.
    """
    if not proposal.source_reference.strip():
        return ValidationOutcome(
            False,
            "no source reference; a memory that cannot be traced is not knowledge",
        )

    if len(proposal.content.strip()) < MIN_CONTENT_CHARS:
        return ValidationOutcome(
            False, f"content is shorter than {MIN_CONTENT_CHARS} characters"
        )

    if contains_secret(proposal.content) or contains_secret(proposal.title):
        return ValidationOutcome(
            False, "content looks like it contains a credential and will not be stored"
        )

    if references_protected_path(proposal.source_reference, proposal.content):
        return ValidationOutcome(
            False,
            "derived from a protected location; a file agents may not read must not "
            "become a memory they can",
        )

    if proposal.source not in AUTO_TRUSTED_SOURCES:
        if looks_speculative(proposal.content):
            return ValidationOutcome(
                False,
                "speculative wording from an untrusted source; hypotheses are not memories",
            )
        return ValidationOutcome(
            False,
            f"{proposal.source} is not an auto-trusted source and needs explicit approval",
            requires_approval=True,
        )

    return ValidationOutcome(True, "accepted")


def to_record(proposal: MemoryProposal, *, approved: bool = False) -> MemoryRecord:
    """Turn a validated proposal into a storable record.

    Confidence defaults to the source's baseline rather than to whatever the proposer
    claimed, so an agent cannot promote its own guess by asserting a high number. An
    explicit value is still honoured, but capped at the source baseline for untrusted
    sources -- the ceiling is a property of *where the claim came from*.
    """
    baseline = SOURCE_CONFIDENCE.get(proposal.source, 0.3)
    confidence = proposal.confidence if proposal.confidence is not None else baseline
    if proposal.source not in AUTO_TRUSTED_SOURCES:
        confidence = min(confidence, baseline)

    status = MemoryStatus.ACTIVE
    if proposal.source not in AUTO_TRUSTED_SOURCES and not approved:
        status = MemoryStatus.REJECTED

    return MemoryRecord(
        type=proposal.type,
        scope=proposal.scope,
        project_id=proposal.project_id,
        title=proposal.title,
        content=proposal.content,
        tags=proposal.tags,
        source=proposal.source,
        source_reference=proposal.source_reference,
        confidence=confidence,
        importance=proposal.importance,
        status=status,
        supersedes=proposal.supersedes,
        metadata=proposal.metadata,
        last_accessed_at=None,
        updated_at=utc_now(),
    )


def redact(text: str) -> str:
    """Mask anything credential-shaped in text destined for a log or a prompt."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("***REDACTED***", redacted)
    return redacted


__all__ = [
    "MemorySource",
    "MemoryValidationError",
    "ValidationOutcome",
    "contains_secret",
    "looks_speculative",
    "redact",
    "references_protected_path",
    "to_record",
    "validate",
]
