"""The ``edith`` command-line interface.

M0 ships the commands the kernel can genuinely support: ``doctor``, ``config``, ``agents``,
``run``, ``selftest``, and ``version``. Commands for milestones that do not exist yet
(``project``, ``memory``, ``task``) are deliberately absent rather than stubbed -- a command
that prints "not implemented" is worse than a clear "no such command".

Exit codes: ``0`` success, ``1`` operational failure, ``2`` configuration/usage error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from edith import __version__
from edith.agents.registry import AgentRegistry, build_default_registry
from edith.benchmarks import BENCHMARKS, get_benchmark, run_benchmark
from edith.config.loader import load_config
from edith.config.schema import EdithConfig
from edith.diagnostics.doctor import CheckStatus, DoctorReport, run_doctor
from edith.environment.install import UnsafeDependencyError
from edith.environment.provision import inspect_project, provision
from edith.errors import EdithError
from edith.experiments import (
    ABLATION_ARMS,
    BUDGET_ARMS,
    run_budget_comparison,
    run_memory_experiment,
    run_strategy_comparison,
)
from edith.memory.retrieval import MemoryRetriever, RetrievalRequest
from edith.memory.schema import MemoryProposal, MemoryScope, MemorySource, MemoryType
from edith.memory.store import open_memory
from edith.models.registry import build_provider
from edith.observability.logging import configure_logging, get_logger
from edith.orchestrator import Orchestrator, create_execution
from edith.product.architecture import (
    ImplementationPlanDocument,
    SystemArchitectureDocument,
)
from edith.product.artifacts import ArtifactKind
from edith.product.coverage import analyse_coverage
from edith.product.prd import PRDDocument
from edith.product.service import ProductService, StageResult, run_pipeline
from edith.product.store import open_artifacts
from edith.product.ux import UXSpecDocument
from edith.research.agent import ResearchOutput, build_report, build_source_block
from edith.research.provider import (
    DuckDuckGoProvider,
    OfflineProvider,
    ResearchCache,
    ResearchProvider,
)
from edith.schemas.agent import AgentRequest
from edith.schemas.common import new_id
from edith.state.store import open_store
from edith.tools.gateway import ToolGateway
from edith.tools.permissions import UNRESTRICTED
from edith.tools.registry import build_default_registry as build_default_tool_registry
from edith.tools.schemas import ToolCall
from edith.workspaces import WorkspaceManager

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2

app = typer.Typer(
    name="edith",
    help="Edith - local-first autonomous product development agent platform.",
    no_args_is_help=True,
    add_completion=False,
)

logger = get_logger(__name__)

_STATUS_MARK = {CheckStatus.OK: "[ OK ]", CheckStatus.WARN: "[WARN]", CheckStatus.FAIL: "[FAIL]"}

ConfigDirOption = Annotated[
    Path | None,
    typer.Option("--config-dir", "-c", help="Directory containing the YAML config files."),
]
ProfileOption = Annotated[
    str | None,
    typer.Option("--profile", "-p", help="Model profile from models.yaml."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable JSON instead of human text."),
]


def _echo(message: str = "") -> None:
    """Write a line to stdout. Logs go to stderr; command output goes here."""
    typer.echo(message)


def _load(config_dir: Path | None, *, quiet: bool = False) -> EdithConfig:
    """Load configuration and configure logging, exiting cleanly on failure."""
    try:
        config = load_config(config_dir)
    except EdithError as exc:
        typer.secho(f"configuration error: {exc.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc
    logging_config = config.system.logging
    if quiet:
        logging_config = logging_config.model_copy(update={"level": "ERROR"})
    configure_logging(logging_config)
    return config


def _render_report(report: DoctorReport, *, as_json: bool) -> None:
    """Print a doctor report as text or JSON."""
    if as_json:
        payload: dict[str, Any] = {
            "healthy": report.healthy,
            "checks": [
                {
                    "name": r.name,
                    "status": str(r.status),
                    "detail": r.detail,
                    "remediation": r.remediation,
                }
                for r in report.results
            ],
        }
        if report.resources is not None:
            snap = report.resources
            payload["resources"] = {
                "platform": snap.platform,
                "python_version": snap.python_version,
                "cpu_logical": snap.cpu_logical,
                "ram_total_mb": snap.ram_total_mb,
                "ram_available_mb": snap.ram_available_mb,
                "disk_free_mb": snap.disk_free_mb,
                "gpus": [
                    {"name": g.name, "total_mb": g.total_mb, "free_mb": g.free_mb}
                    for g in snap.gpus
                ],
            }
        _echo(json.dumps(payload, indent=2))
        return

    _echo("Edith environment diagnostics")
    _echo("=" * 60)
    for result in report.results:
        _echo(f"{_STATUS_MARK[result.status]} {result.name:<16} {result.detail}")
        if result.remediation and result.status is not CheckStatus.OK:
            _echo(f"{'':7}{'':<16} -> {result.remediation}")
    _echo("=" * 60)
    if report.healthy:
        warnings = len(report.warnings)
        suffix = f" ({warnings} warning{'s' if warnings != 1 else ''})" if warnings else ""
        _echo(f"Result: HEALTHY{suffix}")
    else:
        _echo(f"Result: UNHEALTHY ({len(report.failed)} failed, {len(report.warnings)} warnings)")


@app.command()
def version() -> None:
    """Print the Edith version."""
    _echo(f"edith {__version__}")


@app.command()
def doctor(
    config_dir: ConfigDirOption = None,
    profile: ProfileOption = None,
    as_json: JsonOption = False,
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip the live model runtime probe.")
    ] = False,
) -> None:
    """Diagnose the environment and report actionable problems."""
    config = _load(config_dir, quiet=as_json)
    report = run_doctor(config, profile=profile, include_provider=not offline)
    _render_report(report, as_json=as_json)
    raise typer.Exit(report.exit_code())


@app.command(name="config")
def show_config(
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show the fully-resolved configuration after file and environment merging."""
    config = _load(config_dir, quiet=True)
    if as_json:
        _echo(config.model_dump_json(indent=2))
        return
    models = config.models
    _echo(f"config_dir:          {config.config_dir}")
    _echo(f"project_name:        {config.system.project_name}")
    _echo(f"state_dir:           {config.system.state_dir}")
    _echo(f"log_level:           {config.system.logging.level}")
    _echo(f"provider:            {models.provider}")
    _echo(f"endpoint:            {models.ollama.host}")
    _echo(f"max_concurrent_inf:  {config.system.resources.max_concurrent_inferences}")
    _echo(f"default_profile:     {models.default_profile}")
    _echo("profiles:")
    for name, params in sorted(models.profiles.items()):
        marker = "*" if name == models.default_profile else " "
        _echo(
            f"  {marker} {name:<10} {params.model_name}  "
            f"ctx={params.context_length} temp={params.temperature} "
            f"~{params.estimated_vram_mb} MB VRAM"
        )


@app.command(name="agents")
def list_agents(
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """List registered agents, their capabilities, and their declared permissions."""
    config = _load(config_dir, quiet=True)
    registry = build_default_registry(config)
    identities = registry.identities()
    if as_json:
        _echo(json.dumps([i.model_dump(mode="json") for i in identities], indent=2))
        return
    if not identities:
        _echo("No agents registered.")
        return
    for identity in identities:
        settings = config.agents.for_agent(identity.name)
        capabilities = ", ".join(sorted(identity.capabilities)) or "none"
        scope = "read-only" if identity.permissions.read_only else ", ".join(
            identity.permissions.allowed_write_paths
        )
        _echo(f"{identity.name}  v{identity.version}")
        _echo(f"  {identity.description}")
        _echo(f"  capabilities: {capabilities}")
        _echo(f"  write scope:  {scope}")
        _echo(f"  profile:      {identity.model_profile or settings.model_profile}")
        _echo("")


@app.command(name="tools")
def list_tools(
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """List the tools registered in the gateway and their risk surface."""
    _load(config_dir, quiet=True)
    specs = build_default_tool_registry().specs()
    if as_json:
        _echo(json.dumps([s.model_dump(mode="json") for s in specs], indent=2))
        return
    _echo(f"{'TOOL':<20} {'ACCESS':<12} {'PROCESS':<8} DESCRIPTION")
    _echo("-" * 100)
    for spec in specs:
        access = "read+write" if spec.writes else "read"
        _echo(
            f"{spec.name:<20} {access:<12} {'yes' if spec.spawns_process else 'no':<8} "
            f"{spec.description}"
        )


@app.command(name="tool")
def run_tool(
    tool: Annotated[str, typer.Argument(help="Registered tool name, e.g. filesystem.read.")],
    arguments: Annotated[
        str | None,
        typer.Option("--args", help="JSON object matching the tool's input schema."),
    ] = None,
    config_dir: ConfigDirOption = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root to operate on."),
    ] = None,
) -> None:
    """Invoke a single tool directly.

    Runs with unrestricted permissions because the caller is a human operator who already
    has shell access; restricting them would be theatre. Agents never get this path -- they
    receive a gateway scoped to their declared identity.
    """
    config = _load(config_dir)
    if workspace is not None:
        config = config.model_copy(
            update={"tools": config.tools.model_copy(update={"workspace_root": workspace})}
        )

    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        typer.secho(f"--args is not valid JSON: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc
    if not isinstance(parsed, dict):
        typer.secho("--args must be a JSON object", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR)

    try:
        gateway = ToolGateway(config, UNRESTRICTED, agent="cli")
    except EdithError as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc

    result = gateway.execute(ToolCall(tool=tool, arguments=parsed, agent="cli"))
    _echo(result.model_dump_json(indent=2))
    raise typer.Exit(EXIT_OK if result.ok else EXIT_FAILURE)


@app.command(name="run")
def run_agent(
    agent: Annotated[str, typer.Argument(help="Registered agent name.")],
    payload: Annotated[
        str | None,
        typer.Option("--payload", help="JSON object matching the agent's input schema."),
    ] = None,
    payload_file: Annotated[
        Path | None,
        typer.Option("--payload-file", help="Read the JSON payload from a file."),
    ] = None,
    config_dir: ConfigDirOption = None,
    profile: ProfileOption = None,
) -> None:
    """Invoke a single agent with a structured payload and print its structured response."""
    config = _load(config_dir)
    if payload and payload_file:
        typer.secho("use --payload or --payload-file, not both", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR)

    raw = payload
    if payload_file is not None:
        try:
            raw = payload_file.read_text(encoding="utf-8")
        except OSError as exc:
            typer.secho(f"cannot read {payload_file}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR) from exc

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        typer.secho(f"payload is not valid JSON: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc
    if not isinstance(parsed, dict):
        typer.secho("payload must be a JSON object", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR)

    registry = build_default_registry(config)
    try:
        instance = registry.get(agent)
    except EdithError as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc

    try:
        response = instance.execute(
            AgentRequest(payload=parsed, model_profile=profile)
        )
    finally:
        registry.close()

    _echo(response.model_dump_json(indent=2))
    raise typer.Exit(EXIT_OK if response.ok else EXIT_FAILURE)


@app.command()
def selftest(
    config_dir: ConfigDirOption = None,
    profile: ProfileOption = None,
    statement: Annotated[
        str,
        typer.Option("--statement", help="Statement fed to the kernel self-test agent."),
    ] = "Edith runs local models on constrained hardware with a zero dollar API budget.",
) -> None:
    """Prove the M0 kernel end to end against the live local model.

    Exercises config -> registry -> provider -> constrained decoding -> schema validation.
    Exits non-zero if any stage fails, so it is usable as a gate in scripts.
    """
    config = _load(config_dir)
    registry: AgentRegistry = build_default_registry(config)
    try:
        agent = registry.get("echo")
        health = agent.health_check()
        _echo(f"provider health: {health.provider_state or 'n/a'} - {health.detail}")
        if not health.healthy:
            typer.secho(
                "self-test aborted: model provider is not healthy. Run `edith doctor`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(EXIT_FAILURE)

        response = agent.execute(AgentRequest(payload={"statement": statement}))
    finally:
        registry.close()

    _echo("")
    _echo(response.model_dump_json(indent=2))
    _echo("")
    if response.ok:
        typer.secho(
            f"M0 SELF-TEST PASS - validated {agent.output_schema.__name__} from "
            f"{response.model} in {response.duration_seconds:.2f}s",
            fg=typer.colors.GREEN,
        )
        raise typer.Exit(EXIT_OK)
    typer.secho(
        f"M0 SELF-TEST FAIL - {response.failure_category}: {response.error}",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(EXIT_FAILURE)


@app.command(name="benchmark")
def run_benchmark_command(
    benchmark_id: Annotated[
        str | None, typer.Argument(help="Benchmark id. Omit with --all to run every one.")
    ] = None,
    config_dir: ConfigDirOption = None,
    run_all: Annotated[bool, typer.Option("--all", help="Run every benchmark.")] = False,
    show_list: Annotated[bool, typer.Option("--list", help="List benchmarks and exit.")] = False,
    workspace_root: Annotated[
        Path | None,
        typer.Option("--workspace-root", help="Where benchmark workspaces are created."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Run a benchmark against the real local model and check the result independently."""
    if show_list:
        for benchmark in BENCHMARKS:
            _echo(f"{benchmark.benchmark_id:<12} {benchmark.description}")
        raise typer.Exit(EXIT_OK)

    config = _load(config_dir, quiet=as_json)
    manager = WorkspaceManager(config)
    root = (workspace_root or manager.root / "_benchmarks").resolve()
    root.mkdir(parents=True, exist_ok=True)

    if run_all:
        selected = list(BENCHMARKS)
    elif benchmark_id:
        try:
            selected = [get_benchmark(benchmark_id)]
        except KeyError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR) from exc
    else:
        typer.secho("name a benchmark or pass --all", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR)

    results = []
    for benchmark in selected:
        if not as_json:
            _echo(f"running benchmark '{benchmark.benchmark_id}' ...")
        results.append(run_benchmark(benchmark, config, root))

    if as_json:
        _echo(
            json.dumps(
                [
                    {
                        "benchmark": item.benchmark_id,
                        "passed": item.passed,
                        "reason": item.reason,
                        "baseline_failed": item.baseline_failed,
                        "final_verification_passed": item.final_verification_passed,
                        "protected_files_intact": item.protected_files_intact,
                        "duration_seconds": round(item.duration_seconds, 1),
                        "workspace": item.workspace,
                        "metrics": item.metrics.as_dict(),
                    }
                    for item in results
                ],
                indent=2,
            )
        )
    else:
        _echo("")
        for item in results:
            _echo(item.summary())
            metrics = item.metrics
            _echo(
                f"{'':7}tasks={metrics.tasks_succeeded}/{metrics.tasks_total} "
                f"model_calls={metrics.model_calls} repairs={metrics.repairs} "
                f"env_failures={metrics.environment_failures} "
                f"{metrics.duration_seconds:.0f}s"
            )
            if metrics.false_positive:
                typer.secho(
                    f"{'':7}FALSE POSITIVE: Edith reported success but an independent "
                    f"check disagreed",
                    fg=typer.colors.RED,
                )

    raise typer.Exit(EXIT_OK if all(item.passed for item in results) else EXIT_FAILURE)


@app.command(name="execute")
def execute_request(
    request: Annotated[str, typer.Argument(help="What you want Edith to do.")],
    project: Annotated[
        str, typer.Option("--project", "-P", help="Project workspace name.")
    ],
    config_dir: ConfigDirOption = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Use this existing directory as the workspace."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Run the autonomous loop against a project workspace."""
    config = _load(config_dir, quiet=as_json)
    manager = WorkspaceManager(config)

    try:
        project_workspace = (
            manager.adopt(workspace, new_id("proj"), project)
            if workspace is not None
            else manager.create(project, new_id("proj"))
        )
    except EdithError as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc

    state_dir = config.system.state_dir
    if not state_dir.is_absolute():
        state_dir = Path.cwd() / state_dir
    store = open_store(state_dir)
    orchestrator = Orchestrator(config, store, project_workspace)
    try:
        _, execution = create_execution(store, project_workspace, request)
        result = orchestrator.run(execution)
    finally:
        orchestrator.close()
        store.close()

    if as_json:
        _echo(
            json.dumps(
                {
                    "execution_id": result.execution_id,
                    "state": str(result.state),
                    "verdict": str(result.verdict),
                    "summary": result.summary,
                    "tasks_total": result.tasks_total,
                    "tasks_succeeded": result.tasks_succeeded,
                    "changed_files": result.changed_files,
                    "model_calls": result.model_calls,
                    "duration_seconds": round(result.duration_seconds, 1),
                },
                indent=2,
            )
        )
    else:
        _echo(f"execution:   {result.execution_id}")
        _echo(f"state:       {result.state}")
        _echo(f"verdict:     {result.verdict}")
        _echo(f"tasks:       {result.tasks_succeeded}/{result.tasks_total} succeeded")
        _echo(f"model calls: {result.model_calls}")
        _echo(f"changed:     {', '.join(result.changed_files) or '(nothing)'}")
        _echo(f"duration:    {result.duration_seconds:.1f}s")
        _echo(f"summary:     {result.summary}")

    raise typer.Exit(EXIT_OK if result.succeeded else EXIT_FAILURE)


@app.command(name="memory")
def memory_command(
    action: Annotated[
        str, typer.Argument(help="list | search | add | inspect | forget")
    ] = "list",
    query: Annotated[str | None, typer.Argument(help="Search text, or a memory id.")] = None,
    config_dir: ConfigDirOption = None,
    project: Annotated[
        str | None, typer.Option("--project", "-P", help="Project scope.")
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Title, for add.")] = None,
    content: Annotated[str | None, typer.Option("--content", help="Body, for add.")] = None,
    source_ref: Annotated[
        str | None, typer.Option("--source", help="Provenance reference, for add.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Inspect, search, and curate persistent memory."""
    config = _load(config_dir, quiet=True)
    state_dir = config.system.state_dir
    if not state_dir.is_absolute():
        state_dir = Path.cwd() / state_dir
    store = open_memory(state_dir)

    try:
        if action == "list":
            records = store.visible_to(project) if project else store.all_records()
            if as_json:
                _echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
            elif not records:
                _echo("No memories stored.")
            else:
                for entry in records:
                    _echo(
                        f"{entry.memory_id}  [{entry.type}/{entry.scope}] {entry.title}\n"
                        f"{'':22}{entry.provenance}  status={entry.status}"
                    )

        elif action == "search":
            if not query:
                typer.secho("search needs a query", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_CONFIG_ERROR)
            bundle = MemoryRetriever(store).retrieve(
                RetrievalRequest(query=query, project_id=project)
            )
            if as_json:
                _echo(bundle.model_dump_json(indent=2))
            else:
                _echo(bundle.render())
                _echo("")
                for line in bundle.rationale:
                    _echo(f"  why: {line}")

        elif action == "add":
            if not (title and content and source_ref):
                typer.secho(
                    "add needs --title, --content and --source (provenance is mandatory)",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(EXIT_CONFIG_ERROR)
            record, outcome = store.propose(
                MemoryProposal(
                    type=MemoryType.PROJECT if project else MemoryType.ENGINEERING,
                    scope=MemoryScope.PROJECT if project else MemoryScope.GLOBAL,
                    project_id=project,
                    title=title,
                    content=content,
                    source=MemorySource.USER,
                    source_reference=source_ref,
                )
            )
            if record is None:
                typer.secho(f"refused: {outcome.reason}", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_FAILURE)
            _echo(f"stored {record.memory_id}")

        elif action == "inspect":
            if not query:
                typer.secho("inspect needs a memory id", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_CONFIG_ERROR)
            chain = store.history(query)
            if not chain:
                typer.secho("no such memory", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_FAILURE)
            for depth, entry in enumerate(chain):
                prefix = "current: " if depth == 0 else f"replaced ({depth}): "
                _echo(f"{prefix}{entry.title}")
                _echo(f"  {entry.content}")
                _echo(f"  {entry.provenance}  status={entry.status}")

        elif action == "forget":
            if not query:
                typer.secho("forget needs a memory id", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_CONFIG_ERROR)
            _echo("deleted" if store.delete(query) else "no such memory")

        else:
            typer.secho(f"unknown action {action!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR)
    finally:
        store.close()


@app.command(name="research")
def research_command(
    action: Annotated[str, typer.Argument(help="search | fetch | question | cache")],
    target: Annotated[str | None, typer.Argument(help="Query, URL, or question.")] = None,
    config_dir: ConfigDirOption = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Force the offline provider.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Search results to return.")] = 5,
    as_json: JsonOption = False,
) -> None:
    """Search, fetch, and synthesise external sources.

    Research is the one internet-dependent capability. With ``--offline``, or when the
    network is unreachable, it reports RESEARCH_UNAVAILABLE rather than inventing an answer.
    """
    config = _load(config_dir, quiet=as_json)
    state_dir = config.system.state_dir
    if not state_dir.is_absolute():
        state_dir = Path.cwd() / state_dir
    cache = ResearchCache(state_dir / "research-cache")
    provider: ResearchProvider = (
        OfflineProvider("--offline was requested") if offline else DuckDuckGoProvider(cache)
    )

    try:
        if action == "cache":
            _echo(f"cleared {cache.clear()} cached source(s)")
            return

        if not target:
            typer.secho(f"{action} needs an argument", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR)

        if action == "search":
            hits = provider.search(target, limit=limit)
            if as_json:
                _echo(json.dumps([hit.model_dump(mode="json") for hit in hits], indent=2))
            elif not hits:
                _echo("RESEARCH_UNAVAILABLE: no results (offline, or the search failed)")
                raise typer.Exit(EXIT_FAILURE)
            else:
                for hit in hits:
                    _echo(f"{hit.rank + 1}. {hit.title}\n   {hit.url}")

        elif action == "fetch":
            fetched = provider.fetch(target)
            if as_json:
                _echo(fetched.model_dump_json(indent=2))
            else:
                _echo(f"url:       {fetched.url}")
                _echo(f"status:    {fetched.status}")
                _echo(f"authority: {fetched.tier}")
                if fetched.usable:
                    _echo(f"excerpt:   {fetched.excerpt[:500]}")
                else:
                    _echo(f"error:     {fetched.error}")
            if not fetched.usable:
                raise typer.Exit(EXIT_FAILURE)

        elif action == "question":
            hits = provider.search(target, limit=limit)
            sources = [provider.fetch(hit.url) for hit in hits]
            usable = [item for item in sources if item.usable]
            if not usable:
                report = build_report(
                    target, [target], sources, None,
                    unavailable_reason="no source could be retrieved",
                )
                _echo(report.render())
                raise typer.Exit(EXIT_FAILURE)

            registry = build_default_registry(config)
            agent = registry.get("researcher")
            response = agent.execute(
                AgentRequest(
                    payload={"question": target, "sources": build_source_block(usable)}
                )
            )
            synthesis = ResearchOutput.model_validate(response.output) if response.ok else None
            report = build_report(target, [target], sources, synthesis)
            registry.close()
            _echo(report.model_dump_json(indent=2) if as_json else report.render())

        else:
            typer.secho(f"unknown action {action!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR)
    finally:
        if isinstance(provider, DuckDuckGoProvider):
            provider.close()


@app.command(name="environment")
def environment_command(
    config_dir: ConfigDirOption = None,
    project: Annotated[
        Path | None, typer.Option("--project", help="Project to inspect. Defaults to CWD.")
    ] = None,
    write: Annotated[
        bool,
        typer.Option(
            "--write",
            help="Write requirements.txt and scripts/install.* through the tool gateway.",
        ),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Inspect a project's environment and optionally generate install artifacts.

    Inspection is read-only and deterministic. ``--write`` generates the manifest and
    install scripts; nothing is ever installed by this command, and every write goes
    through the M1 tool gateway so the path policy and the audit log apply.
    """
    config = _load(config_dir, quiet=as_json)
    root = (project or Path.cwd()).resolve()

    report = inspect_project(root)

    written: list[str] = []
    denied: list[str] = []
    if write:
        try:
            scoped = config.model_copy(
                update={"tools": config.tools.model_copy(update={"workspace_root": root})}
            )
            gateway = ToolGateway(scoped, UNRESTRICTED, agent="cli")
            _, outcome = provision(gateway, report.spec)
        except (EdithError, UnsafeDependencyError, ValueError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_FAILURE) from exc
        written, denied = outcome.written, outcome.denied

    if as_json:
        payload = report.model_dump(mode="json")
        payload["written"] = written
        payload["denied"] = denied
        _echo(json.dumps(payload, indent=2))
    else:
        _echo(report.summary())
        for dependency in report.spec.dependencies:
            _echo(f"  {dependency.name:<28} {dependency.status:<14} {dependency.origin}")
        for note in report.notes:
            typer.secho(f"  ! {note}", fg=typer.colors.YELLOW)
        for path in written:
            _echo(f"  wrote {path}")
        for path in denied:
            typer.secho(f"  DENIED {path}", fg=typer.colors.RED, err=True)

    if denied or not report.runtime.usable:
        raise typer.Exit(EXIT_FAILURE)


def _product_state_dir(config: EdithConfig) -> Path:
    """Where product artifacts live. Beside execution state, in its own database."""
    return (WorkspaceManager(config).root / "_product").resolve()


@app.command(name="product")
def product_command(
    action: Annotated[
        str, typer.Argument(help="create | plan | status | agents | approve")
    ],
    project: Annotated[
        str, typer.Option("--project", help="Project id the artifacts belong to.")
    ] = "default",
    idea: Annotated[
        str, typer.Option("--idea", help="The product idea, for `create` and `plan`.")
    ] = "",
    constraints: Annotated[
        str, typer.Option("--constraints", help="Binding project constraints.")
    ] = "",
    artifact_id: Annotated[
        str, typer.Option("--artifact", help="Artifact id, for `approve`.")
    ] = "",
    monolithic: Annotated[
        bool,
        typer.Option(
            "--monolithic",
            help=(
                "Generate UX and architecture in one large call each. Measured at 0/5 on "
                "the 3B model; kept only so the M4.1 comparison stays reproducible."
            ),
        ),
    ] = False,
    complete_gaps: Annotated[
        bool,
        typer.Option(
            "--complete-gaps",
            help=(
                "After generation, run one targeted call per uncovered requirement and "
                "merge the result. See experiment 0002."
            ),
        ),
    ] = False,
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Drive the product-development pipeline: PRD, UX specification, architecture.

    ``create`` produces a PRD. ``plan`` runs the whole pipeline through to an implementation
    plan. Nothing here executes the plan -- M4 produces it, and the autonomous loop is a
    separate, deliberate step.
    """
    config = _load(config_dir, quiet=as_json)

    with open_artifacts(_product_state_dir(config)) as store:
        provider = build_provider(config) if action in {"create", "plan"} else None
        service = ProductService(
            config, store, provider=provider, targeted_completion=complete_gaps
        )
        try:
            if action == "agents":
                _emit(list(service.available_agents()), as_json)
                return

            if action == "status":
                _emit(service.status(project), as_json)
                return

            if action == "approve":
                if not artifact_id:
                    typer.secho("--artifact is required", fg=typer.colors.RED, err=True)
                    raise typer.Exit(EXIT_CONFIG_ERROR)
                approved = service.approve_artifact(artifact_id)
                _emit(
                    {
                        "artifact_id": approved.artifact_id,
                        "version": approved.version,
                        "status": approved.status.value,
                    },
                    as_json,
                )
                return

            if action not in {"create", "plan"}:
                typer.secho(f"unknown action {action!r}", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_CONFIG_ERROR)

            if not idea:
                typer.secho("--idea is required", fg=typer.colors.RED, err=True)
                raise typer.Exit(EXIT_CONFIG_ERROR)

            if action == "create":
                outcome = service.create_prd(project, idea, constraints=constraints)
                _render_stage(outcome, as_json)
                raise typer.Exit(EXIT_OK if outcome.ok else EXIT_FAILURE)

            result = run_pipeline(
                service,
                project,
                idea,
                constraints=constraints,
                decomposed=not monolithic,
            )
            if as_json:
                _echo(
                    json.dumps(
                        {
                            "project_id": result.project_id,
                            "ok": result.ok,
                            "blocked": result.blocked,
                            "stages": [
                                {
                                    "stage": stage.stage,
                                    "ok": stage.ok,
                                    "error": stage.error,
                                    "artifact_id": (
                                        stage.artifact.artifact_id if stage.artifact else None
                                    ),
                                    "summary": stage.summary(),
                                }
                                for stage in result.stages
                            ],
                            "metrics": result.total_metrics(),
                        },
                        indent=2,
                    )
                )
            else:
                _echo(result.summary())
                _echo("")
                metrics = result.total_metrics()
                _echo(
                    f"context: {metrics['input_chars']} chars in, "
                    f"{metrics['artifact_chars']} chars of artifact, "
                    f"{metrics['model_calls']} model call(s), "
                    f"{metrics['duration_seconds']}s"
                )
            raise typer.Exit(EXIT_OK if result.ok else EXIT_FAILURE)
        finally:
            if provider is not None:
                provider.close()


@app.command(name="coverage")
def coverage_command(
    project: Annotated[str, typer.Option("--project", help="Project id.")] = "default",
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show which requirements each artifact actually addresses, and the evidence.

    Every state is computed from element references. A requirement is covered because some
    flow or component names it, never because a model said so.
    """
    config = _load(config_dir, quiet=as_json)
    with open_artifacts(_product_state_dir(config)) as store:
        prd_artifact = store.latest(project, ArtifactKind.PRD)
        if prd_artifact is None:
            typer.secho(f"project {project!r} has no PRD", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_FAILURE)

        prd = PRDDocument.model_validate(prd_artifact.body)
        ux_artifact = store.latest(project, ArtifactKind.UX_SPEC)
        architecture_artifact = store.latest(project, ArtifactKind.SYSTEM_ARCHITECTURE)
        plan_artifact = store.latest(project, ArtifactKind.IMPLEMENTATION_PLAN)

        matrix = analyse_coverage(
            prd,
            ux=(
                UXSpecDocument.model_validate(ux_artifact.body) if ux_artifact else None
            ),
            architecture=(
                SystemArchitectureDocument.model_validate(architecture_artifact.body)
                if architecture_artifact
                else None
            ),
            plan=(
                ImplementationPlanDocument.model_validate(plan_artifact.body)
                if plan_artifact
                else None
            ),
        )

        if as_json:
            _echo(
                json.dumps(
                    {
                        "project_id": project,
                        "ux_coverage": matrix.coverage(ArtifactKind.UX_SPEC),
                        "architecture_coverage": matrix.coverage(
                            ArtifactKind.SYSTEM_ARCHITECTURE
                        ),
                        "entries": [
                            {
                                "requirement_id": entry.requirement_id,
                                "criticality": entry.criticality.value,
                                "ux": entry.ux.value,
                                "architecture": entry.architecture.value,
                                "plan": entry.plan.value,
                                "evidence": [
                                    item.element_id
                                    for item in (
                                        *entry.ux_evidence,
                                        *entry.architecture_evidence,
                                    )
                                ],
                            }
                            for entry in matrix.entries
                        ],
                        "gaps": [gap.render() for gap in matrix.gaps],
                        "blocking_gaps": len(matrix.blocking_gaps),
                    },
                    indent=2,
                )
            )
            return

        _echo(matrix.render())
        _echo("")
        _echo(
            f"UX coverage: {matrix.coverage(ArtifactKind.UX_SPEC):.0%}  "
            f"architecture: {matrix.coverage(ArtifactKind.SYSTEM_ARCHITECTURE):.0%}"
        )
        if matrix.blocking_gaps:
            typer.secho(
                f"{len(matrix.blocking_gaps)} blocking coverage gap(s)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(EXIT_FAILURE)


@app.command(name="requirements")
def requirements_command(
    project: Annotated[str, typer.Option("--project", help="Project id.")] = "default",
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Inspect the project's requirements and their traceability."""
    config = _load(config_dir, quiet=as_json)
    with open_artifacts(_product_state_dir(config)) as store:
        artifact = store.latest(project, ArtifactKind.PRD)
        if artifact is None:
            typer.secho(f"project {project!r} has no PRD", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_FAILURE)

        prd = PRDDocument.model_validate(artifact.body)
        if as_json:
            _echo(json.dumps(prd.model_dump(mode="json"), indent=2))
            return

        _echo(f"{prd.product_name} ({artifact.status.value} v{artifact.version})")
        _echo("")
        verified = {
            requirement_id
            for criterion in prd.acceptance_criteria
            for requirement_id in criterion.verifies
        }
        for requirement in prd.requirements:
            mark = "ok" if requirement.requirement_id in verified else "NO AC"
            _echo(
                f"  {requirement.requirement_id} [{requirement.priority}] "
                f"{requirement.title}  ({mark})"
            )
        if prd.open_questions:
            _echo("")
            _echo("Open questions:")
            for question in prd.open_questions:
                _echo(f"  {question.question_id}: {question.question}")


@app.command(name="ux")
def ux_command(
    project: Annotated[str, typer.Option("--project", help="Project id.")] = "default",
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Inspect the project's UX specification."""
    config = _load(config_dir, quiet=as_json)
    with open_artifacts(_product_state_dir(config)) as store:
        artifact = store.latest(project, ArtifactKind.UX_SPEC)
        if artifact is None:
            typer.secho(
                f"project {project!r} has no UX specification", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(EXIT_FAILURE)

        spec = UXSpecDocument.model_validate(artifact.body)
        if as_json:
            _echo(json.dumps(spec.model_dump(mode="json"), indent=2))
            return
        _echo(spec.render())
        missing = spec.screens_missing_states()
        if missing:
            _echo("")
            for screen_id, states in missing.items():
                typer.secho(
                    f"  {screen_id} does not specify "
                    f"{', '.join(state.value for state in states)}",
                    fg=typer.colors.YELLOW,
                )


@app.command(name="architecture")
def architecture_command(
    action: Annotated[
        str, typer.Argument(help="inspect | decisions | plan")
    ] = "inspect",
    project: Annotated[str, typer.Option("--project", help="Project id.")] = "default",
    config_dir: ConfigDirOption = None,
    as_json: JsonOption = False,
) -> None:
    """Inspect the project's architecture, its decisions, or its implementation plan."""
    config = _load(config_dir, quiet=as_json)
    with open_artifacts(_product_state_dir(config)) as store:
        if action == "plan":
            artifact = store.latest(project, ArtifactKind.IMPLEMENTATION_PLAN)
            if artifact is None:
                typer.secho(
                    f"project {project!r} has no implementation plan",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(EXIT_FAILURE)
            plan = ImplementationPlanDocument.model_validate(artifact.body)
            _emit(plan.model_dump(mode="json") if as_json else plan.render(), as_json)
            return

        artifact = store.latest(project, ArtifactKind.SYSTEM_ARCHITECTURE)
        if artifact is None:
            typer.secho(
                f"project {project!r} has no architecture", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(EXIT_FAILURE)

        architecture = SystemArchitectureDocument.model_validate(artifact.body)
        if action == "decisions":
            if as_json:
                _echo(
                    json.dumps(
                        [item.model_dump(mode="json") for item in architecture.decisions],
                        indent=2,
                    )
                )
                return
            for decision in architecture.decisions:
                _echo(f"{decision.decision_id} {decision.title} [{decision.confidence}]")
                _echo(f"  context:      {decision.context}")
                _echo(f"  decision:     {decision.decision}")
                _echo(f"  alternatives: {', '.join(decision.alternatives)}")
                _echo(f"  rationale:    {decision.rationale}")
                _echo(f"  consequences: {'; '.join(decision.consequences)}")
                _echo("")
            return

        if action != "inspect":
            typer.secho(f"unknown action {action!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_CONFIG_ERROR)

        _emit(
            architecture.model_dump(mode="json") if as_json else architecture.render(),
            as_json,
        )


def _emit(payload: object, as_json: bool) -> None:
    """Print a payload as JSON or as text."""
    if as_json:
        _echo(json.dumps(payload, indent=2, default=str))
    elif isinstance(payload, str):
        _echo(payload)
    else:
        _echo(json.dumps(payload, indent=2, default=str))


def _render_stage(outcome: StageResult, as_json: bool) -> None:
    """Render one pipeline stage's result."""
    if as_json:
        _echo(
            json.dumps(
                {
                    "stage": outcome.stage,
                    "ok": outcome.ok,
                    "error": outcome.error,
                    "artifact_id": outcome.artifact.artifact_id if outcome.artifact else None,
                    "summary": outcome.summary(),
                    "metrics": outcome.metrics.as_dict() if outcome.metrics else {},
                },
                indent=2,
            )
        )
        return
    _echo(outcome.summary())
    for review in outcome.reviews:
        for finding in review.findings:
            _echo(f"  {finding.render()}")
    for contradiction in outcome.contradictions:
        typer.secho(f"  {contradiction.render()}", fg=typer.colors.YELLOW)


@app.command(name="budget")
def budget_command(
    config_dir: ConfigDirOption = None,
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="Benchmark to measure.")
    ] = "multi_repair",
    runs: Annotated[int, typer.Option("--runs", help="Iterations per arm.")] = 3,
    ablation: Annotated[
        bool,
        typer.Option("--ablation", help="Compare budget sizes instead of the A/B/C arms."),
    ] = False,
    workspace_root: Annotated[
        Path | None, typer.Option("--workspace-root", help="Where scratch workspaces go.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Measure whether an execution memory budget keeps memory from becoming harmful."""
    config = _load(config_dir, quiet=as_json)
    root = (workspace_root or WorkspaceManager(config).root / "_budget").resolve()

    arms = run_budget_comparison(
        config,
        root,
        benchmark_id=benchmark,
        runs=runs,
        arms=ABLATION_ARMS if ablation else BUDGET_ARMS,
    )

    if as_json:
        _echo(json.dumps({name: arm.as_dict() for name, arm in arms.items()}, indent=2))
        return

    _echo(
        f"{'arm':<24} {'pass':<8} {'mem_chars':<11} {'peak':<8} "
        f"{'retr':<6} {'exhaust':<8} secs"
    )
    _echo("-" * 78)
    for name, arm in arms.items():
        _echo(
            f"{name:<24} {arm.successes}/{arm.runs:<6} {arm.mean_memory_chars:<11.0f} "
            f"{arm.peak_memory_chars:<8} {arm.mean_memory_retrievals:<6.1f} "
            f"{arm.budget_exhaustions:<8} {arm.mean_duration:.0f}"
        )
    _echo("")
    for name, arm in arms.items():
        if arm.false_positives:
            typer.secho(
                f"{name}: {arm.false_positives} false positive(s)",
                fg=typer.colors.RED,
                err=True,
            )


@app.command(name="strategies")
def strategies_command(
    config_dir: ConfigDirOption = None,
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="Benchmark to measure.")
    ] = "multi_repair",
    runs: Annotated[int, typer.Option("--runs", help="Iterations per strategy.")] = 3,
    workspace_root: Annotated[
        Path | None, typer.Option("--workspace-root", help="Where scratch workspaces go.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Compare every memory strategy on the same benchmark."""
    config = _load(config_dir, quiet=as_json)
    root = (workspace_root or WorkspaceManager(config).root / "_strategies").resolve()

    arms = run_strategy_comparison(config, root, benchmark_id=benchmark, runs=runs)

    if as_json:
        _echo(json.dumps({name: arm.as_dict() for name, arm in arms.items()}, indent=2))
        return

    _echo(f"{'strategy':<18} {'pass':<7} {'calls':<7} {'repairs':<8} {'mem_chars':<10} secs")
    _echo("-" * 62)
    for name, arm in arms.items():
        _echo(
            f"{name:<18} {arm.successes}/{arm.runs:<5} {arm.mean_model_calls:<7.1f} "
            f"{arm.mean_repairs:<8.1f} {arm.mean_memory_chars:<10.0f} {arm.mean_duration:.0f}"
        )
    best = max(arms.values(), key=lambda arm: (arm.success_rate, -arm.mean_model_calls))
    _echo("")
    _echo(f"best: {best.name} ({best.successes}/{best.runs})")


@app.command(name="experiment")
def experiment_command(
    config_dir: ConfigDirOption = None,
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="Benchmark to measure.")
    ] = "multi_repair",
    runs: Annotated[int, typer.Option("--runs", help="Iterations per arm.")] = 3,
    workspace_root: Annotated[
        Path | None, typer.Option("--workspace-root", help="Where scratch workspaces go.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Measure whether engineering memory improves the local model."""
    config = _load(config_dir, quiet=as_json)
    root = (workspace_root or WorkspaceManager(config).root / "_experiment").resolve()

    result = run_memory_experiment(config, root, benchmark_id=benchmark, runs=runs)

    if as_json:
        _echo(json.dumps(result.as_dict(), indent=2))
    else:
        for arm in (result.baseline, result.memory):
            _echo(
                f"{arm.name:<9} {arm.successes}/{arm.runs} passed  "
                f"model_calls={arm.mean_model_calls:.1f} repairs={arm.mean_repairs:.1f} "
                f"mem_chars={arm.mean_memory_chars:.0f} {arm.mean_duration:.0f}s"
            )
        _echo("")
        _echo(result.verdict())


@app.command(name="ui")
def launch_ui(
    config_dir: ConfigDirOption = None,
    host: Annotated[
        str, typer.Option("--host", help="Interface to bind. Loopback by default.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8765,
) -> None:
    """Serve the local control surface.

    A client over the existing engine: it starts executions through the same orchestrator the
    ``execute`` command uses and reads state from the same durable store. It holds no authority
    of its own, so nothing it can do is outside what the gateway already permits.

    Binds to loopback. Exposing an autonomous code executor to a network is a deliberate act,
    never a default, so ``--host`` warns when it is used to widen that.
    """
    from edith.ui.server import serve  # noqa: PLC0415 - keeps the CLI import light

    config = _load(config_dir)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.secho(
            f"warning: binding to {host} exposes Edith beyond this machine",
            fg=typer.colors.YELLOW,
            err=True,
        )

    try:
        server = serve(config, host=host, port=port)
    except OSError as exc:
        typer.secho(f"could not bind {host}:{port}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc

    _echo(f"edith ui  http://{host}:{port}")
    _echo("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _echo("")
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    """Console-script entry point."""
    try:
        app()
    except SystemExit as exc:
        return int(exc.code or 0)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
