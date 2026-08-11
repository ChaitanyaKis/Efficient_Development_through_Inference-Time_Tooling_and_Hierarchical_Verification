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
from edith.config.loader import load_config
from edith.config.schema import EdithConfig
from edith.diagnostics.doctor import CheckStatus, DoctorReport, run_doctor
from edith.errors import EdithError
from edith.observability.logging import configure_logging, get_logger
from edith.schemas.agent import AgentRequest

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


def main() -> int:
    """Console-script entry point."""
    try:
        app()
    except SystemExit as exc:
        return int(exc.code or 0)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
