"""A local control surface over the existing EDITH engine.

This is a *client*. It starts executions and reads state; it decides nothing. Every guarantee
EDITH makes -- gateway permissions, workspace isolation, verification, repair policy, merge
containment, artifact authority -- is enforced in the engine and remains enforced whether or
not this server is running. There is no code path here that writes a file, runs a command,
touches git, or approves anything.

Three consequences follow, and they are the reason this module is as small as it is:

**Execution goes through the orchestrator, unchanged.** :meth:`ExecutionManager.start` builds
the same ``Orchestrator`` the CLI builds and calls ``run()``. There is no parallel engine and
no mocked result. If EDITH refuses, the UI shows the refusal.

**Observability comes from the durable state store.** The engine already persists agent runs,
tool executions, verifications, failures and state transitions; the UI reads those rather than
being fed a private event stream. So the picture survives a restart, and it cannot drift from
what actually happened.

**Nothing is served that the engine would not tell the CLI.** The API is a projection of
:class:`StateStore` and :class:`EdithConfig`.

Written on the standard library. A local single-user control panel does not justify a web
framework, and CLAUDE.md is explicit about not adding one. Binds to loopback only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from edith.config.schema import EdithConfig
from edith.errors import ConfigurationError, EdithError
from edith.observability.logging import get_logger
from edith.state.store import StateStore, open_store
from edith.system.resources import ModelFit, classify_fit, snapshot

logger = get_logger(__name__)

STATIC_ROOT = Path(__file__).parent / "static"

#: Loopback only. Exposing an autonomous code executor to a network is not a default.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Request bodies are small JSON documents; anything larger is refused rather than buffered.
MAX_BODY_BYTES = 64 * 1024


@dataclass
class RunHandle:
    """One execution started from the UI."""

    execution_id: str
    project_id: str
    project_name: str
    request: str
    #: The model profile this run was actually started with. Recorded rather than assumed,
    #: so the panel reports the model that ran, not the one currently selected.
    profile: str = ""
    model_name: str = ""
    thread: threading.Thread | None = None
    error: str = ""
    finished: bool = False
    #: Set when the orchestrator raised rather than returning a verdict.
    failure_category: str = ""
    verdict: str = ""
    changed_files: tuple[str, ...] = ()
    model_calls: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "request": self.request,
            "profile": self.profile,
            "model_name": self.model_name,
            "running": bool(self.thread and self.thread.is_alive()),
            "finished": self.finished,
            "error": self.error,
            "failure_category": self.failure_category,
            "verdict": self.verdict,
            "changed_files": list(self.changed_files),
            "model_calls": self.model_calls,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ExecutionManager:
    """Starts executions and exposes what the engine recorded about them.

    Holds no authority of its own. The orchestrator it constructs is the one the CLI
    constructs, with the same config, the same store and the same workspace rules.
    """

    config: EdithConfig
    runs: dict[str, RunHandle] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def _store(self) -> Iterator[StateStore]:
        """A store scoped to one operation.

        ``ThreadingHTTPServer`` serves each request on its own thread and SQLite connections
        are bound to the thread that created them, so a single shared store would fail the
        moment two requests overlapped. Opening per operation is both correct and cheap; the
        durable state is the file, not the handle.
        """
        store = open_store(self._state_dir())
        try:
            yield store
        finally:
            store.close()

    def start(self, project_name: str, request: str, profile: str | None = None) -> RunHandle:
        """Create a workspace and run the autonomous loop in the background.

        ``profile`` selects a configured model for this run only. It is resolved through the
        same provider factory the CLI uses and injected into the orchestrator, so choosing a
        model changes nothing about the agents, the gateway, verification or merge -- which is
        the property the whole abstraction exists to preserve. An unknown or unavailable
        profile is refused rather than quietly replaced by the default.

        Raises:
            EdithError: The workspace could not be created, or the profile was refused.
        """
        from edith.models.registry import build_provider  # noqa: PLC0415
        from edith.orchestrator import (  # noqa: PLC0415 - heavy import
            Orchestrator,
            create_execution,
        )
        from edith.schemas.common import new_id  # noqa: PLC0415
        from edith.workspaces import WorkspaceManager  # noqa: PLC0415

        chosen = profile or self.config.models.default_profile
        if chosen not in self.config.models.profiles:
            raise ConfigurationError(
                f"unknown model profile {chosen!r}; "
                f"available: {sorted(self.config.models.profiles)}",
                details={"profile": chosen},
            )
        params = self.config.models.profiles[chosen]

        # Built before the workspace so an unavailable model fails the request rather than
        # leaving an orphaned project directory behind.
        provider = build_provider(self.config, chosen)
        health = provider.health_check()
        if not health.configured_model_present:
            provider.close()
            raise ConfigurationError(
                f"model {params.model_name!r} is not available in the local runtime",
                details={
                    "profile": chosen,
                    "detail": health.detail,
                    "remediation": health.remediation,
                },
            )

        manager = WorkspaceManager(self.config)
        workspace = manager.create(project_name, new_id("proj"))
        with self._store() as store:
            _, execution = create_execution(store, workspace, request)

        handle = RunHandle(
            execution_id=execution.execution_id,
            project_id=workspace.project_id,
            project_name=project_name,
            request=request,
            profile=chosen,
            model_name=params.model_name,
        )

        def _run() -> None:
            worker_store = open_store(self._state_dir())
            orchestrator = Orchestrator(
                self.config, worker_store, workspace, provider=provider
            )
            try:
                result = orchestrator.run(execution)
                handle.verdict = str(result.verdict)
                handle.changed_files = tuple(result.changed_files)
                handle.model_calls = result.model_calls
                handle.duration_seconds = round(result.duration_seconds, 1)
            except EdithError as exc:
                handle.error = exc.message
                handle.failure_category = exc.category.value
                logger.warning("ui.execution_failed", error=exc.message)
            except Exception as exc:  # noqa: BLE001 - a UI run must not kill the server
                handle.error = f"{type(exc).__name__}: {exc}"
                handle.failure_category = "UNKNOWN"
                logger.warning("ui.execution_crashed", error=traceback.format_exc(limit=3))
            finally:
                handle.finished = True
                orchestrator.close()
                worker_store.close()

        thread = threading.Thread(target=_run, name=f"edith-run-{handle.execution_id}")
        thread.daemon = True
        handle.thread = thread
        with self._lock:
            self.runs[handle.execution_id] = handle
        thread.start()
        return handle

    def _state_dir(self) -> Path:
        state_dir = self.config.system.state_dir
        return state_dir if state_dir.is_absolute() else Path.cwd() / state_dir

    # -- Projections over durable state --------------------------------------------

    def artifacts(self, project_id: str) -> list[dict[str, Any]]:
        """Product artifacts for a project, with the authority each actually carries.

        Status and authority are reported exactly as stored. A draft must never be shown as
        though it were approved -- the whole point of M4's authority model is that a model
        recommendation and a human decision are different things, and a panel that blurred
        them would undo it.
        """
        from edith.product.store import open_artifacts  # noqa: PLC0415

        try:
            with open_artifacts(self._state_dir()) as store:
                found = store.current(project_id)
        except EdithError:
            return []
        return [
            {
                "artifact_id": item.artifact_id,
                "kind": str(item.kind),
                "version": item.version,
                "title": item.title,
                "status": str(item.status),
                "authority": str(item.authority),
                "author": item.author,
                "validation": (
                    str(item.validation.state) if item.validation else "NOT VALIDATED"
                ),
                "depends_on": [str(ref) for ref in item.depends_on],
                "supersedes": item.supersedes or "",
                "approved": str(item.status).upper() == "APPROVED",
            }
            for item in found
        ]

    def snapshot(self, execution_id: str) -> dict[str, Any]:
        """Everything the engine recorded about one execution.

        Read from the store rather than from memory, so it is the same picture the CLI would
        show and it survives a restart of this server.
        """
        with self._store() as store:
            execution = store.get_execution(execution_id)
            if execution is None:
                return {"error": "unknown execution"}

            tasks = store.load_tasks(execution_id)
            dependencies = store.task_dependencies(execution_id)
            runs = store.agent_runs(execution_id)
            verifications = store.verifications(execution_id)
            failures = store.failures(execution_id)
            tools = store.tool_executions(execution_id)
        handle = self.runs.get(execution_id)

        return {
            "execution": {
                "execution_id": execution.execution_id,
                "state": str(execution.state),
                "request": execution.request,
                "attempts": execution.attempts,
                "summary": execution.result_summary,
                "finished_at": str(execution.finished_at) if execution.finished_at else None,
            },
            "handle": handle.as_dict() if handle else None,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "agent": task.agent,
                    "status": str(task.status),
                    "attempts": task.attempts,
                    "failure_reason": task.failure_reason,
                    "depends_on": list(dependencies.get(task.task_id, ())),
                }
                for task in tasks
            ],
            "agent_runs": [
                {
                    "run_id": run.run_id,
                    "task_id": run.task_id,
                    "agent": run.agent,
                    "status": str(run.status),
                    "attempt": run.attempt,
                    "model": run.model,
                    "duration_seconds": run.duration_seconds,
                    "error": run.error,
                    "failure_category": (
                        run.failure_category.value if run.failure_category else ""
                    ),
                    "created_at": str(run.created_at),
                }
                for run in runs
            ],
            "verifications": [
                {
                    "task_id": record.task_id,
                    "kind": record.kind,
                    "command": record.command,
                    "passed": record.passed,
                    "exit_code": record.exit_code,
                    "tests_passed": record.tests_passed,
                    "tests_failed": record.tests_failed,
                    "duration_seconds": record.duration_seconds,
                    "created_at": str(record.created_at),
                }
                for record in verifications
            ],
            "failures": [
                {
                    "task_id": record.task_id,
                    "category": record.category.value,
                    "action": str(record.action),
                    "message": record.message,
                    "attempt": record.attempt,
                    "created_at": str(record.created_at),
                }
                for record in failures
            ],
            # Denials are the security events worth surfacing: a refusal the gateway made.
            "security_events": [
                {
                    "tool": record.tool,
                    "run_id": record.run_id,
                    "category": (
                        record.failure_category.value if record.failure_category else ""
                    ),
                    "error": record.error,
                    "created_at": str(record.created_at),
                }
                for record in tools
                if not record.ok and record.error
            ][-30:],
            "tool_calls": len(tools),
        }


def describe_models(config: EdithConfig) -> list[dict[str, Any]]:
    """Every configured model profile, and whether the runtime actually has it.

    Availability is asked of the provider rather than assumed. A profile that is configured but
    not pulled is reported unavailable with the reason -- never silently swapped for another
    model, because a run whose model quietly changed is a run whose results mean nothing.

    Being pulled is necessary but not sufficient. A profile whose weights exceed this machine's
    VRAM and RAM together cannot load at any speed, so offering it as a choice would hand the
    user a run that fails minutes later for a reason the screen already knew. Those are reported
    unavailable too, with the arithmetic that says so.
    """
    from edith.models.registry import build_provider  # noqa: PLC0415 - heavy import

    installed: set[str] = set()
    probe_error = ""
    try:
        provider = build_provider(config)
        installed = set(provider.list_models())
        provider.close()
    except Exception as exc:  # noqa: BLE001 - an unreachable runtime is a UI state
        probe_error = f"{type(exc).__name__}: {exc}"

    snap = snapshot()
    entries: list[dict[str, Any]] = []
    for name, params in config.models.profiles.items():
        fit = classify_fit(params.estimated_vram_mb, snap)
        loadable = fit is not ModelFit.EXCEEDS_MACHINE
        pulled = params.model_name in installed
        available = pulled and loadable and not probe_error

        if probe_error:
            reason = probe_error
        elif not pulled:
            reason = "not pulled into the local runtime"
        elif not loadable:
            capacity = (snap.free_vram_mb or 0) + snap.ram_available_mb
            reason = (
                f"needs ~{params.estimated_vram_mb} MB; this machine has {capacity} MB "
                "of VRAM and free RAM combined"
            )
        else:
            reason = ""

        entries.append(
            {
                "profile": name,
                "model_name": params.model_name,
                "default": name == config.models.default_profile,
                "available": available,
                "fit": fit.value,
                "reason": reason,
            }
        )
    return entries


def describe_config(config: EdithConfig) -> dict[str, Any]:
    """The settings the UI may show, with experimental features flagged as such."""
    orchestration = config.orchestration
    return {
        "provider": config.models.provider,
        "default_profile": config.models.default_profile,
        "max_repair_attempts": orchestration.max_repair_attempts,
        "max_total_agent_runs": orchestration.max_total_agent_runs,
        "experimental": {
            "model_quality_review": {
                "value": orchestration.model_quality_review,
                "experimental": True,
                "evidence": "0 findings in 36 runs, +41% runtime (M6.1, M6.2, M7)",
            },
            "requirement_derived_testing": {
                "value": False,
                "experimental": True,
                "evidence": "blocked 32/36 runs ungated; no false-PASS gain when gated (M8, M9)",
            },
            "memory": {
                "value": False,
                "experimental": True,
                "evidence": "`always` scored 0/12; `none` was best (M3.2)",
            },
        },
        "supported": {
            # Honest capability reporting: the UI must not offer controls the engine lacks.
            "pause": False,
            "cancel": False,
            "retry": False,
        },
    }


def describe_agents() -> list[dict[str, Any]]:
    """The agents that actually exist, with the permissions they actually hold.

    Read from each agent's declared identity, so the panel cannot claim a capability the
    gateway would refuse, and cannot list a role that was never built.
    """
    from edith.engineering.agents import ENGINEERING_AGENTS  # noqa: PLC0415
    from edith.quality.agents import (  # noqa: PLC0415
        CodeReviewAgent,
        JudgeAgent,
        SecurityAgent,
        TestingAgent,
    )

    seen: dict[str, dict[str, Any]] = {}

    def add(identity: Any, kind: str) -> None:
        permissions = identity.permissions
        seen[identity.name] = {
            "name": identity.name,
            "kind": kind,
            "description": identity.description,
            "tools": sorted(permissions.allowed_tools),
            "write_paths": list(permissions.allowed_write_paths),
            "read_paths": list(permissions.allowed_read_paths),
            "can_write": bool(permissions.allowed_write_paths),
            "can_execute": "shell.run" in permissions.allowed_tools,
        }

    for engineering_agent in ENGINEERING_AGENTS.values():
        add(engineering_agent.identity, "engineering")
    for quality_agent in (TestingAgent, CodeReviewAgent, SecurityAgent, JudgeAgent):
        add(quality_agent.identity, "quality")

    try:
        from edith.agents.architect import ArchitectAgent  # noqa: PLC0415
        from edith.agents.product_manager import ProductManagerAgent  # noqa: PLC0415
        from edith.agents.ux_designer import UXDesignerAgent  # noqa: PLC0415

        for product_agent in (ProductManagerAgent, UXDesignerAgent, ArchitectAgent):
            add(product_agent.identity, "product")
    except ImportError:  # pragma: no cover - product layer is optional at import time
        pass

    return sorted(seen.values(), key=lambda item: (item["kind"], item["name"]))


class Handler(BaseHTTPRequestHandler):
    """Routes the small JSON API and serves the static page."""

    manager: ExecutionManager
    config: EdithConfig

    server_version = "edith-ui"

    def log_message(self, format: str, *args: Any) -> None:
        """Route request logging through structured logging instead of stderr."""
        logger.debug("ui.request", message=format % args)

    # -- helpers -------------------------------------------------------------------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # A local control panel has no business being framed or sniffed.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative: str) -> None:
        name = relative.lstrip("/") or "index.html"
        target = (STATIC_ROOT / name).resolve()
        # The static root is the only thing this server may read from disk.
        if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- routes --------------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json(
                {
                    "online": True,
                    "models": describe_models(self.config),
                    "config": describe_config(self.config),
                    "workspaces_root": str(self.config.orchestration.workspaces_root),
                }
            )
        elif path == "/api/agents":
            self._send_json({"agents": describe_agents()})
        elif path == "/api/runs":
            self._send_json(
                {"runs": [handle.as_dict() for handle in self.manager.runs.values()]}
            )
        elif path.startswith("/api/artifacts/"):
            project_id = path.rsplit("/", 1)[-1]
            self._send_json({"artifacts": self.manager.artifacts(project_id)})
        elif path.startswith("/api/run/"):
            execution_id = path.rsplit("/", 1)[-1]
            self._send_json(self.manager.snapshot(execution_id))
        elif path.startswith("/api/"):
            self._send_json({"error": "unknown endpoint"}, status=404)
        else:
            self._send_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            self._send_json({"error": "unknown endpoint"}, status=404)
            return

        payload = self._read_json()
        name = str(payload.get("project") or "").strip()
        request = str(payload.get("request") or "").strip()
        profile = str(payload.get("profile") or "").strip() or None
        if not name or not request:
            self._send_json(
                {"error": "both a project name and a request are required"}, status=400
            )
            return

        try:
            handle = self.manager.start(name, request, profile)
        except EdithError as exc:
            # The engine refused. Surface the refusal and its category rather than a
            # generic failure, because the category is what says whether it is recoverable.
            self._send_json(
                {"error": exc.message, "category": exc.category.value}, status=400
            )
            return
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the browser
            logger.warning("ui.start_failed", error=traceback.format_exc(limit=3))
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        self._send_json(handle.as_dict(), status=202)


def serve(
    config: EdithConfig,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Build the control-surface server. Loopback unless deliberately overridden."""
    handler = type(
        "BoundHandler",
        (Handler,),
        {"manager": ExecutionManager(config=config), "config": config},
    )
    return ThreadingHTTPServer((host, port), handler)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ExecutionManager",
    "Handler",
    "RunHandle",
    "describe_agents",
    "describe_config",
    "describe_models",
    "serve",
]
