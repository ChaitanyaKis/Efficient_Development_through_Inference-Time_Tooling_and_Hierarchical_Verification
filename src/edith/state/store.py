"""SQLite-backed execution state.

Uses the stdlib ``sqlite3`` directly. An ORM would add a dependency and an abstraction
layer for perhaps a dozen tables that are written in exactly one place each -- not a trade
worth making (CLAUDE.md: no unnecessary frameworks).

**Artifacts.** Prompts, diffs, and captured test output are large and are never queried by
content. They are written to a content-addressed file store and referenced from rows by
SHA-256 digest, keeping the database small and identical payloads stored once.

Every write commits immediately. The process may be killed at any point, and whatever was
recorded up to that moment is durable -- that is what makes restart possible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from edith.errors import ConfigurationError, FailureCategory
from edith.observability.logging import get_logger
from edith.planning.task import Task, TaskStatus
from edith.schemas.common import new_id, utc_now

from .schema import (
    AgentRun,
    Execution,
    FailureRecord,
    MemoryConsumption,
    MemoryInjectionRecord,
    Project,
    ProjectState,
    StateTransition,
    ToolExecution,
    VerificationRecord,
)

logger = get_logger(__name__)

#: v2 adds ``memory_injections`` (M3.2 context accounting). Every change so far has been an
#: additive ``CREATE TABLE IF NOT EXISTS``, so an older database is migrated forward on open
#: rather than rejected — see :meth:`StateStore._assert_schema_version`.
SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id     TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    workspace_root TEXT NOT NULL,
    repository     TEXT,
    description    TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    execution_id   TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL REFERENCES projects(project_id),
    request        TEXT NOT NULL,
    state          TEXT NOT NULL,
    branch         TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    result_summary TEXT NOT NULL DEFAULT '',
    finished_at    TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executions_project ON executions(project_id);

CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT NOT NULL,
    execution_id  TEXT NOT NULL REFERENCES executions(execution_id),
    payload       TEXT NOT NULL,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    priority      INTEGER NOT NULL DEFAULT 100,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (execution_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_execution ON tasks(execution_id);

CREATE TABLE IF NOT EXISTS task_dependencies (
    execution_id  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    depends_on    TEXT NOT NULL,
    PRIMARY KEY (execution_id, task_id, depends_on)
);

CREATE TABLE IF NOT EXISTS state_transitions (
    transition_id TEXT PRIMARY KEY,
    execution_id  TEXT NOT NULL REFERENCES executions(execution_id),
    from_state    TEXT NOT NULL,
    to_state      TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_execution ON state_transitions(execution_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id            TEXT PRIMARY KEY,
    execution_id      TEXT NOT NULL REFERENCES executions(execution_id),
    task_id           TEXT,
    agent             TEXT NOT NULL,
    attempt           INTEGER NOT NULL DEFAULT 1,
    status            TEXT NOT NULL,
    model             TEXT,
    duration_seconds  REAL NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    output_ref        TEXT,
    error             TEXT,
    failure_category  TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_execution ON agent_runs(execution_id);

CREATE TABLE IF NOT EXISTS tool_executions (
    tool_execution_id TEXT PRIMARY KEY,
    execution_id      TEXT NOT NULL REFERENCES executions(execution_id),
    run_id            TEXT,
    tool              TEXT NOT NULL,
    ok                INTEGER NOT NULL,
    duration_seconds  REAL NOT NULL DEFAULT 0,
    error             TEXT,
    failure_category  TEXT,
    detail_ref        TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_exec_execution ON tool_executions(execution_id);

CREATE TABLE IF NOT EXISTS verifications (
    verification_id  TEXT PRIMARY KEY,
    execution_id     TEXT NOT NULL REFERENCES executions(execution_id),
    task_id          TEXT,
    kind             TEXT NOT NULL,
    command          TEXT NOT NULL,
    exit_code        INTEGER NOT NULL,
    passed           INTEGER NOT NULL,
    duration_seconds REAL NOT NULL DEFAULT 0,
    tests_passed     INTEGER,
    tests_failed     INTEGER,
    output_ref       TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verifications_execution ON verifications(execution_id);

CREATE TABLE IF NOT EXISTS failures (
    failure_id   TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
    task_id      TEXT,
    category     TEXT NOT NULL,
    action       TEXT NOT NULL,
    message      TEXT NOT NULL,
    attempt      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_execution ON failures(execution_id);

-- Context accounting for memory. One row per injection that actually reached a prompt,
-- which is what makes an execution's memory cost auditable after the fact and lets a
-- resumed run pick up its budget where the interrupted one left off.
--
-- Memory *content* is deliberately absent: the ids are enough to reconstruct what was sent
-- from the memory store, and duplicating claim text here would spread it across two
-- databases with two deletion paths.
CREATE TABLE IF NOT EXISTS memory_injections (
    injection_id    TEXT PRIMARY KEY,
    execution_id    TEXT NOT NULL REFERENCES executions(execution_id),
    task_id         TEXT,
    agent           TEXT NOT NULL DEFAULT '',
    point           TEXT NOT NULL DEFAULT '',
    memory_ids      TEXT NOT NULL DEFAULT '[]',
    scores          TEXT NOT NULL DEFAULT '[]',
    titles          TEXT NOT NULL DEFAULT '[]',
    chars           INTEGER NOT NULL DEFAULT 0,
    reason          TEXT NOT NULL DEFAULT '',
    remaining_chars INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_injections_execution
    ON memory_injections(execution_id);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class ArtifactStore:
    """Content-addressed storage for large text payloads.

    Identical content is stored once. The digest is what rows carry, so a row can always be
    resolved back to exactly the bytes that were produced.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: str) -> str:
        """Store ``content`` and return its digest."""
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        target = self._path_for(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return digest

    def put_json(self, payload: Any) -> str:
        """Store a JSON-serialisable payload and return its digest."""
        return self.put(json.dumps(payload, indent=2, default=str))

    def get(self, digest: str) -> str | None:
        """Return stored content, or ``None`` when the digest is unknown."""
        target = self._path_for(digest)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    def _path_for(self, digest: str) -> Path:
        # Shard by the first two characters: a flat directory with thousands of entries is
        # slow to enumerate on Windows.
        return self.root / digest[:2] / f"{digest}.txt"


class StateStore:
    """Durable execution state.

    Not thread-safe by design: M2 executes sequentially, and a connection-per-thread pool
    would be infrastructure for a concurrency model that does not exist yet.
    """

    def __init__(self, database_path: Path, artifact_root: Path | None = None) -> None:
        """
        Args:
            database_path: SQLite file. Parent directories are created.
            artifact_root: Directory for large payloads. Defaults to ``<db parent>/artifacts``.
        """
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(artifact_root or database_path.parent / "artifacts")

        self._connection = sqlite3.connect(str(database_path), isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        # WAL survives an abrupt process exit far better than the default journal, which is
        # exactly the restart case this store exists for.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)
        self._assert_schema_version()

    def _assert_schema_version(self) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT value FROM meta WHERE key = 'schema_version'")
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                return
            found = int(row["value"])
            if found == SCHEMA_VERSION:
                return
            if found < SCHEMA_VERSION:
                # Every migration so far is additive, and the schema script above has
                # already run with IF NOT EXISTS, so the tables are present by now. Refusing
                # here would strand a project's history for no reason.
                cursor.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )
                logger.info(
                    "state.schema_migrated", from_version=found, to_version=SCHEMA_VERSION
                )
                return
            raise ConfigurationError(
                f"state database is schema v{found}, newer than this build's "
                f"v{SCHEMA_VERSION}: {self.database_path}",
                details={"found": found, "expected": SCHEMA_VERSION},
            )

    # -- Projects -------------------------------------------------------------------

    def save_project(self, project: Project) -> Project:
        """Insert or update a project."""
        self._connection.execute(
            """
            INSERT INTO projects
                (project_id, name, workspace_root, repository, description,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                name = excluded.name,
                workspace_root = excluded.workspace_root,
                repository = excluded.repository,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            (
                project.project_id,
                project.name,
                project.workspace_root,
                project.repository,
                project.description,
                _iso(project.created_at),
                _iso(project.updated_at),
            ),
        )
        return project

    def get_project(self, project_id: str) -> Project | None:
        """Load a project by id."""
        row = self._fetch_one("SELECT * FROM projects WHERE project_id = ?", (project_id,))
        return Project.model_validate(dict(row)) if row else None

    def find_project_by_name(self, name: str) -> Project | None:
        """Load a project by its unique-in-practice name."""
        row = self._fetch_one("SELECT * FROM projects WHERE name = ?", (name,))
        return Project.model_validate(dict(row)) if row else None

    def list_projects(self) -> tuple[Project, ...]:
        """Every known project, newest first."""
        rows = self._fetch_all("SELECT * FROM projects ORDER BY created_at DESC")
        return tuple(Project.model_validate(dict(row)) for row in rows)

    # -- Executions -----------------------------------------------------------------

    def save_execution(self, execution: Execution) -> Execution:
        """Insert or update an execution."""
        self._connection.execute(
            """
            INSERT INTO executions
                (execution_id, project_id, request, state, branch, attempts,
                 result_summary, finished_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id) DO UPDATE SET
                state = excluded.state,
                branch = excluded.branch,
                attempts = excluded.attempts,
                result_summary = excluded.result_summary,
                finished_at = excluded.finished_at,
                updated_at = excluded.updated_at
            """,
            (
                execution.execution_id,
                execution.project_id,
                execution.request,
                str(execution.state),
                execution.branch,
                execution.attempts,
                execution.result_summary,
                _iso(execution.finished_at),
                _iso(execution.created_at),
                _iso(execution.updated_at),
            ),
        )
        return execution

    def get_execution(self, execution_id: str) -> Execution | None:
        """Load an execution by id."""
        row = self._fetch_one(
            "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
        )
        return Execution.model_validate(dict(row)) if row else None

    def list_executions(self, project_id: str | None = None) -> tuple[Execution, ...]:
        """Executions, newest first, optionally filtered by project."""
        if project_id:
            rows = self._fetch_all(
                "SELECT * FROM executions WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            )
        else:
            rows = self._fetch_all("SELECT * FROM executions ORDER BY created_at DESC")
        return tuple(Execution.model_validate(dict(row)) for row in rows)

    def record_transition(
        self, execution: Execution, target: ProjectState, reason: str = ""
    ) -> Execution:
        """Transition an execution and persist both the new state and an audit row."""
        previous = execution.state
        execution.transition_to(target)
        if target.terminal:
            execution.finished_at = utc_now()
        self.save_execution(execution)
        transition = StateTransition(
            execution_id=execution.execution_id,
            from_state=str(previous),
            to_state=str(target),
            reason=reason,
        )
        self._connection.execute(
            """
            INSERT INTO state_transitions
                (transition_id, execution_id, from_state, to_state, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transition.transition_id,
                transition.execution_id,
                transition.from_state,
                transition.to_state,
                transition.reason,
                _iso(transition.created_at),
            ),
        )
        logger.info(
            "execution.transition",
            execution_id=execution.execution_id,
            from_state=str(previous),
            to_state=str(target),
        )
        return execution

    def transitions(self, execution_id: str) -> tuple[StateTransition, ...]:
        """Full transition history for an execution."""
        rows = self._fetch_all(
            "SELECT * FROM state_transitions WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(StateTransition.model_validate(dict(row)) for row in rows)

    # -- Tasks ----------------------------------------------------------------------

    def save_task(self, execution_id: str, task: Task) -> Task:
        """Persist a task and its dependency edges.

        The task is stored as JSON plus a few promoted columns. Those columns exist so
        status can be queried without deserialising every row; the JSON remains the source
        of truth for the task itself.
        """
        self._connection.execute(
            """
            INSERT INTO tasks
                (task_id, execution_id, payload, status, attempts, priority, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id, task_id) DO UPDATE SET
                payload = excluded.payload,
                status = excluded.status,
                attempts = excluded.attempts,
                priority = excluded.priority,
                updated_at = excluded.updated_at
            """,
            (
                task.task_id,
                execution_id,
                task.model_dump_json(),
                str(task.status),
                task.attempts,
                task.priority,
                _iso(utc_now()),
            ),
        )
        self._connection.execute(
            "DELETE FROM task_dependencies WHERE execution_id = ? AND task_id = ?",
            (execution_id, task.task_id),
        )
        for dependency in task.dependencies:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO task_dependencies (execution_id, task_id, depends_on)
                VALUES (?, ?, ?)
                """,
                (execution_id, task.task_id, dependency),
            )
        return task

    def save_tasks(self, execution_id: str, tasks: list[Task]) -> None:
        """Persist many tasks in one transaction."""
        with self._transaction():
            for task in tasks:
                self.save_task(execution_id, task)

    def load_tasks(self, execution_id: str) -> list[Task]:
        """Load every task for an execution, in deterministic order.

        This is the heart of restart: the reconstructed tasks carry their statuses and
        attempt counts, so the DAG resumes exactly where it stopped.
        """
        rows = self._fetch_all(
            "SELECT payload FROM tasks WHERE execution_id = ? ORDER BY priority, task_id",
            (execution_id,),
        )
        return [Task.model_validate_json(row["payload"]) for row in rows]

    def task_dependencies(self, execution_id: str) -> dict[str, tuple[str, ...]]:
        """Dependency edges as stored, for verification against the task payloads."""
        rows = self._fetch_all(
            "SELECT task_id, depends_on FROM task_dependencies WHERE execution_id = ?",
            (execution_id,),
        )
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(row["task_id"], []).append(row["depends_on"])
        return {key: tuple(sorted(value)) for key, value in mapping.items()}

    # -- Evidence -------------------------------------------------------------------

    def save_agent_run(self, run: AgentRun) -> AgentRun:
        """Insert or update an agent run."""
        self._connection.execute(
            """
            INSERT INTO agent_runs
                (run_id, execution_id, task_id, agent, attempt, status, model,
                 duration_seconds, prompt_tokens, completion_tokens, output_ref,
                 error, failure_category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status,
                duration_seconds = excluded.duration_seconds,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                output_ref = excluded.output_ref,
                error = excluded.error,
                failure_category = excluded.failure_category,
                updated_at = excluded.updated_at
            """,
            (
                run.run_id, run.execution_id, run.task_id, run.agent, run.attempt,
                run.status, run.model, run.duration_seconds, run.prompt_tokens,
                run.completion_tokens, run.output_ref, run.error,
                str(run.failure_category) if run.failure_category else None,
                _iso(run.created_at), _iso(run.updated_at),
            ),
        )
        return run

    def agent_runs(self, execution_id: str) -> tuple[AgentRun, ...]:
        """Every agent run for an execution, oldest first."""
        rows = self._fetch_all(
            "SELECT * FROM agent_runs WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(AgentRun.model_validate(dict(row)) for row in rows)

    def save_tool_execution(self, record: ToolExecution) -> ToolExecution:
        """Record one tool call."""
        self._connection.execute(
            """
            INSERT INTO tool_executions
                (tool_execution_id, execution_id, run_id, tool, ok, duration_seconds,
                 error, failure_category, detail_ref, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.tool_execution_id, record.execution_id, record.run_id, record.tool,
                int(record.ok), record.duration_seconds, record.error,
                str(record.failure_category) if record.failure_category else None,
                record.detail_ref, _iso(record.created_at),
            ),
        )
        return record

    def tool_executions(self, execution_id: str) -> tuple[ToolExecution, ...]:
        """Every tool call for an execution, oldest first."""
        rows = self._fetch_all(
            "SELECT * FROM tool_executions WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(
            ToolExecution.model_validate({**dict(row), "ok": bool(row["ok"])}) for row in rows
        )

    def save_verification(self, record: VerificationRecord) -> VerificationRecord:
        """Record verification evidence."""
        self._connection.execute(
            """
            INSERT INTO verifications
                (verification_id, execution_id, task_id, kind, command, exit_code,
                 passed, duration_seconds, tests_passed, tests_failed, output_ref, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.verification_id, record.execution_id, record.task_id, record.kind,
                record.command, record.exit_code, int(record.passed),
                record.duration_seconds, record.tests_passed, record.tests_failed,
                record.output_ref, _iso(record.created_at),
            ),
        )
        return record

    def verifications(self, execution_id: str) -> tuple[VerificationRecord, ...]:
        """Every verification record for an execution, oldest first."""
        rows = self._fetch_all(
            "SELECT * FROM verifications WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(
            VerificationRecord.model_validate({**dict(row), "passed": bool(row["passed"])})
            for row in rows
        )

    def save_failure(self, record: FailureRecord) -> FailureRecord:
        """Record a classified failure."""
        self._connection.execute(
            """
            INSERT INTO failures
                (failure_id, execution_id, task_id, category, action, message,
                 attempt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.failure_id, record.execution_id, record.task_id,
                str(record.category), record.action, record.message, record.attempt,
                _iso(record.created_at),
            ),
        )
        return record

    def failures(self, execution_id: str) -> tuple[FailureRecord, ...]:
        """Every failure for an execution, oldest first."""
        rows = self._fetch_all(
            "SELECT * FROM failures WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(
            FailureRecord.model_validate(
                {**dict(row), "category": FailureCategory(row["category"])}
            )
            for row in rows
        )

    # -- Memory accounting ----------------------------------------------------------

    def record_memory_injection(
        self,
        *,
        execution_id: str,
        task_id: str | None,
        agent_name: str,
        point: str,
        memory_ids: list[str],
        scores: list[float],
        chars: int,
        reason: str,
        remaining_chars: int,
        titles: list[str] | None = None,
    ) -> str:
        """Record that memory reached a prompt, and what it cost.

        Ids and scores only. The claim text lives in the memory store, which is where it can
        be inspected and deleted; copying it here would create a second place a user has to
        remember to purge.
        """
        injection_id = new_id("minj")
        self._connection.execute(
            """
            INSERT INTO memory_injections
                (injection_id, execution_id, task_id, agent, point, memory_ids, scores,
                 titles, chars, reason, remaining_chars, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                injection_id, execution_id, task_id, agent_name, point,
                json.dumps(memory_ids), json.dumps(scores), json.dumps(titles or []),
                chars, reason[:500], remaining_chars, _iso(utc_now()),
            ),
        )
        return injection_id

    def memory_injections(self, execution_id: str) -> tuple[MemoryInjectionRecord, ...]:
        """Every memory injection for an execution, oldest first."""
        rows = self._fetch_all(
            "SELECT * FROM memory_injections WHERE execution_id = ? ORDER BY created_at",
            (execution_id,),
        )
        return tuple(
            MemoryInjectionRecord(
                injection_id=row["injection_id"],
                execution_id=row["execution_id"],
                task_id=row["task_id"],
                agent=row["agent"],
                point=row["point"],
                memory_ids=tuple(json.loads(row["memory_ids"])),
                scores=tuple(json.loads(row["scores"])),
                titles=tuple(json.loads(row["titles"])),
                chars=int(row["chars"]),
                reason=row["reason"],
                remaining_chars=int(row["remaining_chars"]),
            )
            for row in rows
        )

    def memory_consumption(self, execution_id: str) -> MemoryConsumption:
        """What an execution has already spent, so a resumed run continues its budget.

        Without this, an interrupted execution would restart with a full allowance, and
        "crash and retry" would become an unlimited memory supply.
        """
        injections = self.memory_injections(execution_id)
        return MemoryConsumption(
            chars=sum(injection.chars for injection in injections),
            retrievals=len(injections),
            injected=tuple(
                (
                    memory_id,
                    injection.titles[index] if index < len(injection.titles) else "",
                    injection.point,
                    injection.agent,
                    injection.scores[index] if index < len(injection.scores) else 0.0,
                )
                for injection in injections
                for index, memory_id in enumerate(injection.memory_ids)
            ),
        )

    # -- Plumbing -------------------------------------------------------------------

    def _fetch_one(self, query: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            row: sqlite3.Row | None = cursor.fetchone()
            return row

    def _fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()

    def _transaction(self) -> Any:
        """Return a context manager wrapping a single transaction."""
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_store(state_dir: Path) -> StateStore:
    """Open (creating if needed) the state store under ``state_dir``."""
    return StateStore(state_dir / "edith.db", state_dir / "artifacts")


__all__ = [
    "ArtifactStore",
    "StateStore",
    "TaskStatus",
    "open_store",
]
