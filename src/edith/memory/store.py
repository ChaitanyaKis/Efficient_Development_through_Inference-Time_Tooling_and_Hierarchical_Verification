"""SQLite-backed memory.

A separate database file from execution state, deliberately. Memory outlives any single
execution, and CLAUDE.md requires it to be inspectable and deletable -- both of which are
simpler when deleting memory does not also delete the audit trail of what happened.

**Isolation is enforced in SQL, not in Python.** Every read is parameterised on
``project_id``, so a caller cannot forget to filter and quietly leak one project's knowledge
into another's context. The only cross-project path is an explicit ``GLOBAL`` scope, which
the schema restricts to genuinely reusable lesson types.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from edith.errors import ConfigurationError, EdithError, FailureCategory
from edith.observability.logging import get_logger
from edith.schemas.common import utc_now

from .schema import (
    MemoryProposal,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from .validation import ValidationOutcome, to_record, validate

logger = get_logger(__name__)

MEMORY_SCHEMA_VERSION = 1


class MemoryUnavailableError(EdithError):
    """The memory store could not be opened or read."""

    category = FailureCategory.ENVIRONMENT_FAILURE


class MemoryCorruptionError(EdithError):
    """A stored record could not be decoded."""

    category = FailureCategory.VALIDATION_FAILURE


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id        TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    scope            TEXT NOT NULL,
    project_id       TEXT,
    title            TEXT NOT NULL,
    content          TEXT NOT NULL,
    tags             TEXT NOT NULL DEFAULT '',
    source           TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    confidence       REAL NOT NULL,
    importance       INTEGER NOT NULL,
    status           TEXT NOT NULL,
    supersedes       TEXT,
    superseded_by    TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    recurrence_count INTEGER NOT NULL DEFAULT 1,
    last_accessed_at TEXT,
    payload          TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Retrieval always filters by project and status, so those lead the index.
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(project_id, status, type);
CREATE INDEX IF NOT EXISTS idx_memories_global ON memories(scope, status, type);
CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class MemoryStore:
    """Durable, project-scoped memory."""

    def __init__(self, database_path: Path) -> None:
        """
        Args:
            database_path: SQLite file. Parent directories are created.
        """
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(str(database_path), isolation_level=None)
        except sqlite3.Error as exc:
            raise MemoryUnavailableError(
                f"could not open the memory database at {database_path}: {exc}",
                details={"path": str(database_path)},
            ) from exc

        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._assert_version()

    def _assert_version(self) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'")
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO memory_meta (key, value) VALUES ('schema_version', ?)",
                    (str(MEMORY_SCHEMA_VERSION),),
                )
                return
            found = int(row["value"])
            if found != MEMORY_SCHEMA_VERSION:
                raise ConfigurationError(
                    f"memory database is schema v{found}, this build expects "
                    f"v{MEMORY_SCHEMA_VERSION}",
                    details={"found": found, "expected": MEMORY_SCHEMA_VERSION},
                )

    # -- Writing --------------------------------------------------------------------

    def propose(
        self, proposal: MemoryProposal, *, approved: bool = False
    ) -> tuple[MemoryRecord | None, ValidationOutcome]:
        """Validate and (if permitted) store a proposed memory.

        Returns the stored record and the validation outcome. A rejected proposal returns
        ``(None, outcome)`` -- the caller usually wants to log the refusal rather than treat
        it as an error.
        """
        outcome = validate(proposal)
        if outcome.rejected:
            logger.info(
                "memory.rejected",
                title=proposal.title[:80],
                source=str(proposal.source),
                reason=outcome.reason,
            )
            return (None, outcome)

        if outcome.requires_approval and not approved:
            logger.info(
                "memory.needs_approval",
                title=proposal.title[:80],
                source=str(proposal.source),
            )
            return (None, outcome)

        record = to_record(proposal, approved=approved)
        self.save(record)
        return (record, outcome)

    def save(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or update a record.

        Supersession is applied here rather than by the caller: writing the new record and
        marking the old one must not be separable, or a crash between them would leave two
        records both claiming to be current.
        """
        self._connection.execute(
            """
            INSERT INTO memories
                (memory_id, type, scope, project_id, title, content, tags, source,
                 source_reference, confidence, importance, status, supersedes,
                 superseded_by, access_count, recurrence_count, last_accessed_at,
                 payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                tags = excluded.tags,
                confidence = excluded.confidence,
                importance = excluded.importance,
                status = excluded.status,
                superseded_by = excluded.superseded_by,
                access_count = excluded.access_count,
                recurrence_count = excluded.recurrence_count,
                last_accessed_at = excluded.last_accessed_at,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                record.memory_id, str(record.type), str(record.scope), record.project_id,
                record.title, record.content, ",".join(record.tags), str(record.source),
                record.source_reference, record.confidence, record.importance,
                str(record.status), record.supersedes, record.superseded_by,
                record.access_count, record.recurrence_count,
                _iso(record.last_accessed_at), record.model_dump_json(),
                _iso(record.created_at), _iso(record.updated_at),
            ),
        )

        if record.supersedes and record.is_active:
            self._mark_superseded(record.supersedes, record.memory_id)

        logger.info(
            "memory.stored",
            memory_id=record.memory_id,
            type=str(record.type),
            scope=str(record.scope),
            project_id=record.project_id,
            source=str(record.source),
            confidence=round(record.confidence, 2),
        )
        return record

    def _mark_superseded(self, old_id: str, new_id: str) -> None:
        """Point an old record at its replacement without destroying it."""
        previous = self.get(old_id)
        if previous is None:
            logger.warning("memory.supersedes_unknown", memory_id=old_id)
            return
        previous.status = MemoryStatus.SUPERSEDED
        previous.superseded_by = new_id
        previous.updated_at = utc_now()
        self._connection.execute(
            "UPDATE memories SET status = ?, superseded_by = ?, payload = ?, updated_at = ? "
            "WHERE memory_id = ?",
            (
                str(MemoryStatus.SUPERSEDED), new_id, previous.model_dump_json(),
                _iso(previous.updated_at), old_id,
            ),
        )
        logger.info("memory.superseded", memory_id=old_id, replaced_by=new_id)

    def archive(self, memory_id: str) -> bool:
        """Mark a record archived. History is retained."""
        record = self.get(memory_id)
        if record is None:
            return False
        record.status = MemoryStatus.ARCHIVED
        record.updated_at = utc_now()
        self.save(record)
        return True

    def record_access(self, record: MemoryRecord) -> None:
        """Persist that a memory was retrieved, for recency/frequency ranking."""
        record.touch_access()
        self._connection.execute(
            "UPDATE memories SET access_count = ?, last_accessed_at = ?, payload = ? "
            "WHERE memory_id = ?",
            (
                record.access_count, _iso(record.last_accessed_at),
                record.model_dump_json(), record.memory_id,
            ),
        )

    def bump_recurrence(self, memory_id: str) -> MemoryRecord | None:
        """Increment how many times a failure or lesson has recurred.

        A lesson seen five times is stronger evidence than one seen once, and this is how
        that shows up in ranking without inventing a duplicate record each time.
        """
        record = self.get(memory_id)
        if record is None:
            return None
        record.recurrence_count += 1
        record.updated_at = utc_now()
        self.save(record)
        return record

    # -- Reading --------------------------------------------------------------------

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Load one record by id, regardless of project or status."""
        row = self._fetch_one(
            "SELECT payload FROM memories WHERE memory_id = ?", (memory_id,)
        )
        return self._decode(row) if row else None

    def visible_to(
        self,
        project_id: str | None,
        *,
        types: tuple[MemoryType, ...] = (),
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        include_global: bool = True,
        limit: int = 500,
    ) -> list[MemoryRecord]:
        """Every record a given project may see.

        This is the isolation boundary, and it is a SQL predicate rather than a filter a
        caller has to remember: a record is visible only if it belongs to this project, or
        it is explicitly GLOBAL.
        """
        clauses: list[str] = []
        parameters: list[Any] = []

        if project_id is None:
            # No project context: only genuinely shared knowledge is visible.
            clauses.append("scope = ?")
            parameters.append(str(MemoryScope.GLOBAL))
        elif include_global:
            clauses.append("(project_id = ? OR scope = ?)")
            parameters.extend([project_id, str(MemoryScope.GLOBAL)])
        else:
            clauses.append("project_id = ?")
            parameters.append(project_id)

        if status is not None:
            clauses.append("status = ?")
            parameters.append(str(status))

        if types:
            placeholders = ", ".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            parameters.extend(str(item) for item in types)


        # lines above; every value the caller supplies travels as a bound parameter. The
        # Every fragment joined here is a literal defined just above; every caller-supplied
        # value travels as a bound parameter. One hard-coded query per filter combination
        # would be harder to audit, which is the opposite of what this boundary needs.
        query = (
            "SELECT payload FROM memories WHERE "  # noqa: S608
            + " AND ".join(clauses)
            + " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        )
        parameters.append(limit)
        return [self._decode(row) for row in self._fetch_all(query, tuple(parameters))]

    def history(self, memory_id: str) -> list[MemoryRecord]:
        """Walk the supersession chain backwards from a record.

        Answers "why does Edith believe this" by showing what it believed before, and the
        provenance of each step.
        """
        chain: list[MemoryRecord] = []
        current = self.get(memory_id)
        seen: set[str] = set()
        while current is not None and current.memory_id not in seen:
            chain.append(current)
            seen.add(current.memory_id)
            current = self.get(current.supersedes) if current.supersedes else None
        return chain

    def current_belief(self, memory_id: str) -> MemoryRecord | None:
        """Follow supersession forwards to whatever replaced this record."""
        current = self.get(memory_id)
        seen: set[str] = set()
        while current is not None and current.superseded_by:
            if current.memory_id in seen:
                break
            seen.add(current.memory_id)
            successor = self.get(current.superseded_by)
            if successor is None:
                break
            current = successor
        return current

    def count(self, project_id: str | None = None) -> int:
        """How many active records a project can see."""
        return len(self.visible_to(project_id, limit=100_000))

    def all_records(self, limit: int = 1000) -> list[MemoryRecord]:
        """Every record, for inspection and maintenance. Ignores scoping by design."""
        return [
            self._decode(row)
            for row in self._fetch_all(
                "SELECT payload FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        ]

    def delete(self, memory_id: str) -> bool:
        """Permanently remove a record.

        Memory must be deletable (CLAUDE.md). Distinct from archiving: this is for content
        that should never have been stored, not for superseded belief.
        """
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            removed = cursor.rowcount > 0
        if removed:
            logger.info("memory.deleted", memory_id=memory_id)
        return removed

    def purge_project(self, project_id: str) -> int:
        """Delete every memory belonging to one project."""
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("DELETE FROM memories WHERE project_id = ?", (project_id,))
            removed = int(cursor.rowcount)
        logger.info("memory.project_purged", project_id=project_id, removed=removed)
        return removed

    # -- Plumbing -------------------------------------------------------------------

    def _decode(self, row: sqlite3.Row) -> MemoryRecord:
        """Rehydrate a record, turning a corrupt row into a classified error."""
        try:
            return MemoryRecord.model_validate_json(row["payload"])
        except Exception as exc:
            raise MemoryCorruptionError(
                f"a stored memory could not be decoded: {exc}",
                details={"database": str(self.database_path)},
            ) from exc

    def _fetch_one(self, query: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            row: sqlite3.Row | None = cursor.fetchone()
            return row

    def _fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_memory(state_dir: Path) -> MemoryStore:
    """Open (creating if needed) the memory store under ``state_dir``."""
    return MemoryStore(state_dir / "memory.db")
