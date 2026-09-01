"""Typed configuration schema.

Every tunable value in Edith is declared here and loaded from ``config/*.yaml`` with
environment-variable overrides. Source code must never hard-code a model name, timeout,
context length, host, or path -- it reads them from :class:`EdithConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json"]


class StrictModel(BaseModel):
    """Base for config models: unknown keys are an error, not a silent typo."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(StrictModel):
    """Structured logging configuration."""

    level: LogLevel = "INFO"
    format: LogFormat = "console"
    file_enabled: bool = True
    file_path: Path = Path(".edith/logs/edith.jsonl")
    redact_keys: tuple[str, ...] = (
        "api_key",
        "apikey",
        "authorization",
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "private_key",
        "access_key",
        "session",
    )


class ResourceConfig(StrictModel):
    """Resource-awareness thresholds used by doctor and (later) the scheduler."""

    max_concurrent_inferences: int = Field(default=1, ge=1, le=8)
    min_free_vram_mb: int = Field(default=2048, ge=0)
    min_free_ram_mb: int = Field(default=2048, ge=0)
    min_free_disk_mb: int = Field(default=10_240, ge=0)

    @model_validator(mode="after")
    def _warn_on_parallelism(self) -> ResourceConfig:
        # Not an error -- overridable -- but the default must stay sequential on 6 GB VRAM.
        return self


class RetryConfig(StrictModel):
    """Bounded retry policy. Autonomous loops must never retry indefinitely."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    max_backoff_seconds: float = Field(default=8.0, ge=0.0, le=300.0)


class ModelParams(StrictModel):
    """Inference parameters for one named model profile."""

    model_name: str = Field(min_length=1)
    context_length: int = Field(default=8192, ge=512, le=131_072)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=2048, ge=1, le=32_768)
    seed: int | None = None
    stop: tuple[str, ...] = ()
    supports_tools: bool = False
    keep_alive: str = "5m"
    estimated_vram_mb: int = Field(default=0, ge=0)


class OllamaProviderConfig(StrictModel):
    """Connection settings for a local Ollama runtime.

    ``host`` must resolve to a loopback address unless ``allow_remote`` is explicitly set;
    this enforces the local-first invariant and prevents a config typo from shipping
    prompts to a third party.
    """

    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)
    allow_remote: bool = False

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("host must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _enforce_local(self) -> OllamaProviderConfig:
        if self.allow_remote:
            return self
        local_prefixes = (
            "http://127.0.0.1",
            "http://localhost",
            "http://[::1]",
            "https://127.0.0.1",
            "https://localhost",
        )
        if not self.host.startswith(local_prefixes):
            raise ValueError(
                f"host {self.host!r} is not loopback; set allow_remote: true to permit it. "
                "Edith is local-first and must not send prompts off-machine by accident."
            )
        return self


class ModelsConfig(StrictModel):
    """Model provider selection and the profiles agents may reference by role."""

    provider: str = "ollama"
    ollama: OllamaProviderConfig = OllamaProviderConfig()
    retry: RetryConfig = RetryConfig()
    default_profile: str = "default"
    profiles: dict[str, ModelParams]

    @model_validator(mode="after")
    def _default_profile_exists(self) -> ModelsConfig:
        if self.default_profile not in self.profiles:
            raise ValueError(
                f"default_profile {self.default_profile!r} is not defined in profiles "
                f"({sorted(self.profiles)})"
            )
        return self

    def profile(self, name: str | None = None) -> ModelParams:
        """Return a profile by name, falling back to the configured default."""
        key = name or self.default_profile
        try:
            return self.profiles[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown model profile {key!r}; available: {sorted(self.profiles)}"
            ) from exc


class AgentDefaults(StrictModel):
    """Defaults applied to every agent unless the agent overrides them."""

    model_profile: str = "default"
    max_attempts: int = Field(default=2, ge=1, le=10)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class AgentsConfig(StrictModel):
    """Agent-layer configuration.

    ``overrides`` maps agent name -> partial settings. Agent *implementations* are
    registered in code; this file only tunes them.
    """

    defaults: AgentDefaults = AgentDefaults()
    overrides: dict[str, AgentDefaults] = Field(default_factory=dict)

    def for_agent(self, name: str) -> AgentDefaults:
        """Return effective settings for ``name``."""
        return self.overrides.get(name, self.defaults)


class PathPolicyConfig(StrictModel):
    """Filesystem safety policy applied before any authorization decision.

    ``protected_patterns`` are denied to *every* agent regardless of its granted write
    scope. They are a floor, not a default: an agent cannot be configured around them.
    """

    #: Glob patterns (matched against the workspace-relative POSIX path and each of its
    #: parent segments) that no agent may read or write.
    protected_patterns: tuple[str, ...] = (
        ".git/**",
        ".git",
        "**/.git",
        "**/.git/**",
        ".env",
        ".env.*",
        "**/.env",
        "**/.env.*",
        "secrets/**",
        "credentials/**",
        "**/id_rsa",
        "**/id_ed25519",
        "*.pem",
        "**/*.pem",
        "*.key",
        "**/*.key",
        "*.pfx",
        "**/*.pfx",
        ".ssh/**",
        ".aws/**",
        "**/.npmrc",
        "**/.pypirc",
    )
    #: Refuse to read or write a single file larger than this.
    max_file_bytes: int = Field(default=2_097_152, ge=1024)
    #: Cap on entries returned by a directory or content search.
    max_search_results: int = Field(default=200, ge=1)
    #: Follow symlinks that resolve back inside the workspace. Off by default on the
    #: assumption that a symlink in an agent-managed tree is more likely an escape attempt
    #: than a legitimate need.
    allow_symlinks: bool = False


class ShellPolicyConfig(StrictModel):
    """Shell execution policy.

    ``shell.run`` takes an argv list and never invokes a system shell, so there is no
    metacharacter injection surface. The allowlist constrains *which program* may run.
    """

    #: Executables an agent may invoke, by bare name. Empty means shell.run is disabled.
    allowed_executables: tuple[str, ...] = (
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "git",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "cargo",
        "go",
        "dotnet",
        "java",
        "mvn",
        "gradle",
        "make",
    )
    #: Environment variables passed through to the child. Everything else is stripped, so
    #: an API key in the parent environment cannot leak into an agent-invoked process.
    env_passthrough: tuple[str, ...] = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "LANG",
        "LC_ALL",
    )
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    #: Truncate captured stdout/stderr beyond this many bytes each.
    max_output_bytes: int = Field(default=131_072, ge=1024)


class GitPolicyConfig(StrictModel):
    """Git tool policy."""

    timeout_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)
    max_output_bytes: int = Field(default=262_144, ge=1024)
    #: Cap on commits returned by git.log.
    max_log_entries: int = Field(default=100, ge=1)
    #: Branch names an agent may create must start with one of these prefixes, keeping
    #: agent work off shared branches (CLAUDE.md: agents must never destroy unrelated work).
    branch_prefixes: tuple[str, ...] = ("agent/",)
    #: Branches an agent may never delete or force-update.
    protected_branches: tuple[str, ...] = ("main", "master", "develop", "release")
    #: Directory (workspace-relative) holding agent worktrees. Kept inside the workspace so
    #: the path policy still applies, and gitignored so worktrees never pollute the index.
    worktree_dir: str = ".edith/worktrees"
    #: Identity recorded on agent commits. Explicit so commits are attributable and so a
    #: fresh worktree without a configured global identity does not fail to commit.
    committer_name: str = "Edith Agent"
    committer_email: str = "edith-agent@localhost"


class ToolsConfig(StrictModel):
    """Tool gateway configuration."""

    #: Workspace root the tools operate on. Relative values resolve against the process
    #: working directory. Every path an agent supplies is confined to this tree.
    workspace_root: Path = Path(".")
    #: Default wall-clock budget for a single tool call.
    default_timeout_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)
    paths: PathPolicyConfig = PathPolicyConfig()
    shell: ShellPolicyConfig = ShellPolicyConfig()
    git: GitPolicyConfig = GitPolicyConfig()


class ContextConfig(StrictModel):
    """Budget for the Context Engine.

    The whole point of the engine is to *not* send the repository to the model. On a 3B
    model with an 8k window these ceilings are what keep the prompt inside it.
    """

    #: Maximum files included in a bundle.
    max_files: int = Field(default=8, ge=1, le=100)
    #: Total character budget across all included content.
    max_total_chars: int = Field(default=12_000, ge=500)
    #: Per-file character budget; longer files are excerpted.
    max_file_chars: int = Field(default=4_000, ge=200)
    #: Include matching test files for the code under change.
    include_tests: bool = True
    #: Include architecture/README documents when they look relevant.
    include_docs: bool = True


class VerificationProfile(StrictModel):
    """Commands used to verify a project.

    Commands are argv lists, never strings, and run through the M1 ``shell.run`` allowlist.
    A planner or model can select a *profile key*, never author a command line.
    """

    tests: tuple[str, ...] = ()
    lint: tuple[str, ...] = ()
    typecheck: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    def command_for(self, kind: str) -> tuple[str, ...]:
        """Return the argv for a verification kind, empty when not configured."""
        return {
            "tests": self.tests,
            "lint": self.lint,
            "typecheck": self.typecheck,
            "build": self.build,
        }.get(kind, ())


class MemoryBudgetConfig(StrictModel):
    """The memory allowance for one whole execution.

    M3.1 measured that per-prompt limits do not bound total cost: a repair loop retrieves
    again after every failure, so ~1,200 characters per prompt became ~14,000 across an
    execution. These ceilings are charged against the execution, which is the unit that
    actually competes for the model's 8,192-token window.
    """

    #: When false the governor still runs — strategy, relevance, and duplicate suppression
    #: all still apply — but no execution-wide ceiling is enforced. This is the "without
    #: budget" arm of the M3.2 experiment, not a supported production setting.
    enabled: bool = True
    max_total_chars: int = Field(default=2400, ge=0)
    max_retrievals: int = Field(default=3, ge=0)
    max_total_memories: int = Field(default=4, ge=0)
    max_chars_per_retrieval: int = Field(default=1200, ge=0)
    max_memories_per_retrieval: int = Field(default=2, ge=0)


class MemoryConfig(StrictModel):
    """Memory retrieval settings for the autonomous loop.

    ``enabled`` exists so the memory experiment has a real control arm: the same code path
    runs with retrieval switched off, rather than a separate untested branch.
    """

    enabled: bool = True
    #: How memory is integrated into the loop.
    #:
    #: The default is set by measurement, not intuition. All five strategies were compared
    #: on ``multi_repair`` with a 3B model over 6 runs each (ADR 0006 §4): no arm beat the
    #: no-memory control, and ``always`` scored 0/6 -- 0/12 counting the M3 batches. Until a
    #: strategy demonstrates a benefit, the default is the one that spends no context on an
    #: unproven one. Re-run ``edith strategies`` to revisit this on other models or tasks;
    #: the retrieval machinery is fully built and one config line away.
    strategy: Literal[
        "none", "always", "failure_triggered", "debugger_only", "high_relevance"
    ] = "none"
    #: Memories offered to one agent invocation.
    max_memories: int = Field(default=4, ge=1, le=20)
    #: Character budget, kept small: on an 8k window memory competes with the code itself.
    max_chars: int = Field(default=1200, ge=100)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Let global engineering lessons reach every project.
    include_global: bool = True
    #: The execution-wide allowance. Enforced by the Memory Governor, which every
    #: autonomous injection passes through.
    budget: MemoryBudgetConfig = MemoryBudgetConfig()


class OrchestrationConfig(StrictModel):
    """The autonomous loop's limits and workspace layout."""

    #: Root under which project workspaces live. Kept outside the Edith repository so an
    #: autonomous task cannot casually modify the kernel that is running it.
    workspaces_root: Path = Path("../Edith_Workspaces")
    #: Attempts per task before it is failed. Bounded loops are non-negotiable.
    max_task_attempts: int = Field(default=3, ge=1, le=10)
    #: Debugger invocations per task.
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    #: Whether the M6.1 model reviewers run after the deterministic quality gates.
    #:
    #: Off by default, and that default is a measurement rather than caution: M6.1's A/B found
    #: model review costs roughly four seconds per file and adds nothing on code the AST
    #: scanners already judge correctly. It catches semantic defects no scanner can see, so it
    #: is worth having available -- but enabling it by default would slow every task for a
    #: benefit that has not been demonstrated on real generated code. Same discipline as M3.2's
    #: memory strategy, which is also off until evidence says otherwise.
    model_quality_review: bool = False
    #: Floor on total agent invocations in one execution, a backstop against any loop the
    #: per-task budgets fail to bound. The effective ceiling is this or
    #: ``agent_runs_per_task`` times the number of planned tasks, whichever is larger.
    #:
    #: A flat number conflated two different things. This exists to stop a runaway loop, but
    #: applied unscaled it also caps how much work may be asked for: a six-function library
    #: fans out to seven tasks, spent the flat 40 on the first five, and was cut off with two
    #: tasks never attempted. Bounding per task keeps the loop guard while letting a larger
    #: request cost proportionally more.
    max_total_agent_runs: int = Field(default=40, ge=1, le=500)
    #: Agent invocations one task may cost before the run-level backstop trips. Measured at
    #: roughly seven on a fan-out task that exhausts its repair budget, so eight leaves room
    #: without letting a pathological task run away.
    agent_runs_per_task: int = Field(default=8, ge=1, le=50)
    #: Ceiling on repair attempts across a whole run, on top of the per-task budget.
    #:
    #: Fan-out turns one request into many tasks, so the per-task budget alone no longer
    #: bounds the run: eight functions at two repairs each is sixteen, and one pathological
    #: function can spend an afternoon while the rest wait. This caps the total. Failures that
    #: never enter repair -- environment, dependency, timeout, security -- do not count
    #: against it, because they never consumed it.
    max_total_repairs: int = Field(default=12, ge=0, le=100)
    #: Create an isolated git branch for each execution.
    use_branch_isolation: bool = True
    branch_prefix: str = "agent/exec"
    context: ContextConfig = ContextConfig()
    memory: MemoryConfig = MemoryConfig()
    #: Verification profiles by language/toolchain key.
    verification_profiles: dict[str, VerificationProfile] = Field(default_factory=dict)
    #: Profile used when a project does not name one.
    default_verification_profile: str = "python"

    def profile(self, name: str | None = None) -> VerificationProfile:
        """Return a verification profile by name, falling back to the default."""
        key = name or self.default_verification_profile
        return self.verification_profiles.get(key, VerificationProfile())


class SystemConfig(StrictModel):
    """Top-level system settings."""

    project_name: str = "edith"
    state_dir: Path = Path(".edith")
    logging: LoggingConfig = LoggingConfig()
    resources: ResourceConfig = ResourceConfig()


class EdithConfig(StrictModel):
    """The fully-resolved configuration object passed through dependency injection."""

    system: SystemConfig = SystemConfig()
    models: ModelsConfig
    agents: AgentsConfig = AgentsConfig()
    tools: ToolsConfig = ToolsConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()

    #: Absolute directory the config was loaded from; ``None`` when built in-memory.
    config_dir: Path | None = None
