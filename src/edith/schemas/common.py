"""Primitives shared across all Edith schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    """Return a short, sortable-enough, prefixed identifier (e.g. ``task_9f2c1a4b``)."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class EdithModel(BaseModel):
    """Base for domain schemas.

    ``extra="forbid"`` is deliberate: when an LLM invents an extra field, that is a
    validation failure the Critic should see, not silent data loss.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Severity(StrEnum):
    """Severity of a finding, used by report schemas from M2 onward."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Verdict(StrEnum):
    """Outcome of a verification gate."""

    PASS = "PASS"  # noqa: S105 - a verdict, not a credential
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class Timestamped(EdithModel):
    """Mixin providing creation/update timestamps."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        """Update ``updated_at`` to now."""
        self.updated_at = utc_now()
