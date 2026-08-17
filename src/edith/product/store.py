"""Durable, versioned artifact storage.

A separate database from execution state and from memory, for the same reason those are
separate from each other: an artifact outlives the execution that produced it, and deleting
a project's history should not require deciding what else lived in the same file.

Three guarantees this module exists to provide:

**Versions are rows, never edits.** Saving a revision inserts a new row at ``version + 1``.
Nothing overwrites an approved artifact, so "what did the PRD say when we agreed to build
this" is always answerable.

**Approval is the only thing that supersedes.** A draft revision does not retire its
predecessor -- an approved PRD stays approved until its successor is itself approved. That
ordering is what stops a half-finished draft silently becoming the project's truth.

**Isolation is a SQL predicate.** Every read is parameterised on ``project_id``, exactly as
the memory store does it, so a caller cannot forget to filter and leak one project's
requirements into another's context.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType

from edith.errors import ConfigurationError, EdithError, FailureCategory
from edith.observability.logging import get_logger
from edith.schemas.common import new_id, utc_now

from .artifacts import Artifact, ArtifactKind, ArtifactStatus, can_transition

logger = get_logger(__name__)

PRODUCT_SCHEMA_VERSION = 1


class ArtifactStoreError(EdithError):
    """The artifact store could not be opened, read, or written."""

    category = FailureCategory.ENVIRONMENT_FAILURE


class ArtifactConflictError(EdithError):
    """A write would have destroyed or duplicated an existing version."""

    category = FailureCategory.VALIDATION_FAILURE


_SCHEMA = """
CREATE TABLE IF NOT EXISTS product_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per artifact *version*. The primary key is the (id, version) pair rather than the
-- artifact id, which is what makes history immutable by construction: there is no row to
-- overwrite when a revision is saved.
CREATE TABLE IF NOT EXISTS artifacts (
    row_id        TEXT PRIMARY KEY,
    artifact_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    project_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    author        TEXT NOT NULL,
    status        TEXT NOT NULL,
    authority     TEXT NOT NULL,
    validation    TEXT NOT NULL,
    payload       TEXT NOT NULL,
    supersedes    TEXT,
    superseded_by TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE(artifact_id, version)
);

-- Reads are almost always "the current artifacts of kind K for project P".
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, kind, status);
CREATE INDEX IF NOT EXISTS idx_artifacts_lineage ON artifacts(artifact_id, version);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class ProductStore:
    """Versioned artifact storage, scoped by project."""

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
            raise ArtifactStoreError(
                f"could not open the artifact database at {database_path}: {exc}",
                details={"path": str(database_path)},
            ) from exc

        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._assert_version()

    def _assert_version(self) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT value FROM product_meta WHERE key = 'schema_version'")
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO product_meta (key, value) VALUES ('schema_version', ?)",
                    (str(PRODUCT_SCHEMA_VERSION),),
                )
                return
            found = int(row["value"])
            if found != PRODUCT_SCHEMA_VERSION:
                raise ConfigurationError(
                    f"artifact database is schema v{found}, this build expects "
                    f"v{PRODUCT_SCHEMA_VERSION}",
                    details={"found": found, "expected": PRODUCT_SCHEMA_VERSION},
                )

    # -- Writing --------------------------------------------------------------------

    def save(self, artifact: Artifact) -> Artifact:
        """Insert an artifact version.

        Raises:
            ArtifactConflictError: That ``(artifact_id, version)`` already exists. Saving
                over an existing version is never what the caller meant -- a change is a new
                version, and silently replacing one would destroy history.
        """
        existing = self.get(artifact.artifact_id, artifact.version)
        if existing is not None:
            raise ArtifactConflictError(
                f"artifact {artifact.artifact_id} version {artifact.version} already "
                f"exists; a change must be saved as a new version",
                details={
                    "artifact_id": artifact.artifact_id,
                    "version": artifact.version,
                },
            )

        self._connection.execute(
            """
            INSERT INTO artifacts
                (row_id, artifact_id, version, project_id, kind, title, author, status,
                 authority, validation, payload, supersedes, superseded_by,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("arow"), artifact.artifact_id, artifact.version, artifact.project_id,
                str(artifact.kind), artifact.title, artifact.author, str(artifact.status),
                str(artifact.authority), artifact.validation.model_dump_json(),
                artifact.model_dump_json(), artifact.supersedes, artifact.superseded_by,
                _iso(artifact.created_at), _iso(artifact.updated_at),
            ),
        )
        logger.info(
            "artifact.saved",
            artifact_id=artifact.artifact_id,
            version=artifact.version,
            kind=str(artifact.kind),
            status=str(artifact.status),
            project_id=artifact.project_id,
            author=artifact.author,
        )
        return artifact

    def set_status(self, artifact: Artifact, target: ArtifactStatus) -> Artifact:
        """Move an artifact to a new status and persist the change.

        Approval has a side effect: the previous approved version of the same artifact is
        marked ``SUPERSEDED``. That is the only path by which an approved artifact stops
        being current, which is what keeps a draft from displacing it.

        Raises:
            ValueError: The transition is illegal, or approval was requested for an
                artifact that has not validated.
        """
        updated = artifact.transition_to(target)
        self._write_status(updated)

        if target is ArtifactStatus.APPROVED:
            for previous in self.versions(artifact.artifact_id):
                if (
                    previous.version != updated.version
                    and previous.status is ArtifactStatus.APPROVED
                ):
                    retired = previous.model_copy(
                        update={
                            "status": ArtifactStatus.SUPERSEDED,
                            "superseded_by": f"{updated.artifact_id}@{updated.version}",
                            "updated_at": utc_now(),
                        }
                    )
                    self._write_status(retired)
                    logger.info(
                        "artifact.superseded",
                        artifact_id=retired.artifact_id,
                        version=retired.version,
                        by_version=updated.version,
                    )

        logger.info(
            "artifact.status_changed",
            artifact_id=updated.artifact_id,
            version=updated.version,
            status=str(target),
        )
        return updated

    def _write_status(self, artifact: Artifact) -> None:
        """Persist an artifact's mutable fields for one version row."""
        self._connection.execute(
            """
            UPDATE artifacts
               SET status = ?, authority = ?, validation = ?, payload = ?,
                   superseded_by = ?, updated_at = ?
             WHERE artifact_id = ? AND version = ?
            """,
            (
                str(artifact.status), str(artifact.authority),
                artifact.validation.model_dump_json(), artifact.model_dump_json(),
                artifact.superseded_by, _iso(artifact.updated_at),
                artifact.artifact_id, artifact.version,
            ),
        )

    def record_validation(self, artifact: Artifact) -> Artifact:
        """Persist a re-validated artifact's outcome without changing its status."""
        self._write_status(artifact)
        logger.info(
            "artifact.validated",
            artifact_id=artifact.artifact_id,
            version=artifact.version,
            state=str(artifact.validation.state),
            issues=len(artifact.validation.issues),
        )
        return artifact

    # -- Reading --------------------------------------------------------------------

    def get(self, artifact_id: str, version: int | None = None) -> Artifact | None:
        """Return one artifact version, or the latest when ``version`` is omitted."""
        if version is None:
            row = self._fetch_one(
                "SELECT payload FROM artifacts WHERE artifact_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (artifact_id,),
            )
        else:
            row = self._fetch_one(
                "SELECT payload FROM artifacts WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            )
        return self._decode(row) if row is not None else None

    def versions(self, artifact_id: str) -> tuple[Artifact, ...]:
        """Every version of one artifact, oldest first."""
        rows = self._fetch_all(
            "SELECT payload FROM artifacts WHERE artifact_id = ? ORDER BY version",
            (artifact_id,),
        )
        return tuple(self._decode(row) for row in rows)

    def latest(
        self,
        project_id: str,
        kind: ArtifactKind,
        *,
        status: ArtifactStatus | None = None,
    ) -> Artifact | None:
        """The newest artifact of a kind for a project, optionally filtered by status.

        Passing ``status=APPROVED`` is how a downstream agent asks for what it is allowed to
        build on, rather than for whatever happens to be newest.
        """
        candidates = self.by_kind(project_id, kind, status=status)
        return candidates[-1] if candidates else None

    def by_kind(
        self,
        project_id: str,
        kind: ArtifactKind,
        *,
        status: ArtifactStatus | None = None,
    ) -> tuple[Artifact, ...]:
        """Artifacts of one kind for one project, oldest first.

        Project isolation lives in this predicate. Nothing above this layer is trusted to
        remember to filter.
        """
        query = "SELECT payload FROM artifacts WHERE project_id = ? AND kind = ?"
        parameters: list[object] = [project_id, str(kind)]
        if status is not None:
            query += " AND status = ?"
            parameters.append(str(status))
        query += " ORDER BY created_at, version"
        return tuple(self._decode(row) for row in self._fetch_all(query, tuple(parameters)))

    def current(self, project_id: str) -> tuple[Artifact, ...]:
        """Every artifact for a project that still represents current intent."""
        rows = self._fetch_all(
            "SELECT payload FROM artifacts WHERE project_id = ? AND status IN "
            "('DRAFT', 'REVIEW', 'APPROVED') ORDER BY kind, created_at, version",
            (project_id,),
        )
        return tuple(self._decode(row) for row in rows)

    def history(self, artifact_id: str) -> tuple[Artifact, ...]:
        """Every version of an artifact, newest first. Nothing is ever omitted."""
        return tuple(reversed(self.versions(artifact_id)))

    def count(self, project_id: str | None = None) -> int:
        """How many artifact versions are stored."""
        if project_id is None:
            row = self._fetch_one("SELECT COUNT(*) AS total FROM artifacts")
        else:
            row = self._fetch_one(
                "SELECT COUNT(*) AS total FROM artifacts WHERE project_id = ?",
                (project_id,),
            )
        return int(row["total"]) if row else 0

    def projects(self) -> tuple[str, ...]:
        """Every project id with at least one artifact."""
        rows = self._fetch_all(
            "SELECT DISTINCT project_id FROM artifacts ORDER BY project_id"
        )
        return tuple(str(row["project_id"]) for row in rows)

    def next_version(self, artifact_id: str) -> int:
        """The version a revision of this artifact would take."""
        row = self._fetch_one(
            "SELECT MAX(version) AS top FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        top = row["top"] if row else None
        return int(top) + 1 if top is not None else 1

    def purge_project(self, project_id: str) -> int:
        """Delete every artifact for a project. Returns how many rows were removed.

        Deliberately explicit and total: CLAUDE.md requires stored knowledge to be
        deletable, and a partial delete that leaves an approved PRD behind would be worse
        than none.
        """
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("DELETE FROM artifacts WHERE project_id = ?", (project_id,))
            removed = cursor.rowcount
        logger.info("artifact.purged", project_id=project_id, removed=removed)
        return int(removed)

    # -- Plumbing -------------------------------------------------------------------

    @staticmethod
    def _decode(row: sqlite3.Row) -> Artifact:
        """Rebuild an artifact from its stored payload."""
        try:
            return Artifact.model_validate(json.loads(row["payload"]))
        except (ValueError, TypeError) as exc:
            raise ArtifactStoreError(
                f"a stored artifact could not be decoded: {exc}"
            ) from exc

    def _fetch_one(
        self, query: str, parameters: tuple[object, ...] = ()
    ) -> sqlite3.Row | None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            row: sqlite3.Row | None = cursor.fetchone()
            return row

    def _fetch_all(
        self, query: str, parameters: tuple[object, ...] = ()
    ) -> list[sqlite3.Row]:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> ProductStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@contextmanager
def open_artifacts(state_dir: Path) -> Iterator[ProductStore]:
    """Open the artifact store under a state directory."""
    store = ProductStore(state_dir / "artifacts.db")
    try:
        yield store
    finally:
        store.close()


def approve(store: ProductStore, artifact: Artifact) -> Artifact:
    """Approve an artifact, refusing anything that has not validated.

    A thin helper, but it is the one place approval semantics are stated: an artifact that
    is invalid or unvalidated cannot become project truth, and the refusal names which.
    """
    if not artifact.validation.valid:
        raise ArtifactConflictError(
            f"artifact {artifact.artifact_id} cannot be approved: "
            f"{artifact.validation.summary()}",
            details={
                "artifact_id": artifact.artifact_id,
                "state": str(artifact.validation.state),
                "issues": [issue.code for issue in artifact.validation.blocking_issues],
            },
        )
    if not can_transition(artifact.status, ArtifactStatus.APPROVED):
        raise ArtifactConflictError(
            f"artifact {artifact.artifact_id} cannot move from {artifact.status} "
            f"to APPROVED"
        )
    return store.set_status(artifact, ArtifactStatus.APPROVED)
