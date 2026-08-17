"""Provider-neutral dependency and environment types.

The requirement this exists to serve is concrete: a generated application must not leave
its user staring at ``ModuleNotFoundError``. Getting there needs three things a coding loop
alone does not provide — knowing what the project actually requires, being able to install
it reproducibly, and being able to tell "the package is missing" apart from "the code is
wrong".

Nothing here is Python-specific. Python is implemented first because Edith is written in it,
but :class:`Ecosystem` is an open enumeration and every field below is expressed in terms
that a Node, Rust, or Go implementation could fill in without reshaping the model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from edith.schemas.common import EdithModel


class Ecosystem(StrEnum):
    """A language/package-manager world."""

    PYTHON = "python"
    NODE = "node"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    DOTNET = "dotnet"
    UNKNOWN = "unknown"


class DependencyKind(StrEnum):
    """What role a dependency plays.

    Kept distinct because they install differently and fail differently: a missing runtime
    dependency breaks the application, a missing dev dependency breaks the build, and a
    missing test dependency breaks verification without saying anything about the code.
    """

    RUNTIME = "runtime"
    DEVELOPMENT = "development"
    TEST = "test"
    OPTIONAL = "optional"
    BUILD = "build"


class DependencyOrigin(StrEnum):
    """How Edith learned that a dependency exists.

    Ordered by trustworthiness. A declared manifest entry is a statement of intent; an
    import found by parsing source is evidence of actual use; a model's suggestion is
    neither, and is marked as such so it can never be installed unreviewed.
    """

    #: Declared in pyproject.toml, package.json, Cargo.toml, ...
    MANIFEST = "manifest"
    #: Pinned in a lock file.
    LOCKFILE = "lockfile"
    #: Found by parsing the source for imports.
    SOURCE_IMPORT = "source_import"
    #: Observed installed in the environment.
    INSTALLED = "installed"
    #: Proposed by a model. Never installed without review.
    MODEL_SUGGESTION = "model_suggestion"


class DependencyStatus(StrEnum):
    """Whether a dependency is actually usable right now."""

    #: Declared and importable at the required version.
    SATISFIED = "satisfied"
    #: Required but absent from the environment.
    MISSING = "missing"
    #: Present but at a version the project does not ask for.
    VERSION_MISMATCH = "version_mismatch"
    #: Installed but never referenced by the source.
    UNUSED = "unused"
    #: Imported by the source but declared nowhere.
    UNDECLARED = "undeclared"
    UNKNOWN = "unknown"


class SecurityStatus(StrEnum):
    """What is known about a dependency's safety.

    Populated by a future Dependency Agent. ``UNREVIEWED`` is the honest default: claiming
    a package is clean because nothing checked it would be worse than admitting ignorance.
    """

    UNREVIEWED = "unreviewed"
    OK = "ok"
    VULNERABLE = "vulnerable"
    UNTRUSTED_SOURCE = "untrusted_source"
    YANKED = "yanked"


class Dependency(EdithModel):
    """One package the project needs."""

    name: str = Field(min_length=1, max_length=200)
    ecosystem: Ecosystem = Ecosystem.PYTHON
    #: The constraint the project asks for, e.g. ``>=2.0,<3``. Empty means unpinned.
    requested_version: str = Field(default="", max_length=100)
    #: What is actually present in the environment.
    resolved_version: str = Field(default="", max_length=100)
    kind: DependencyKind = DependencyKind.RUNTIME
    origin: DependencyOrigin = DependencyOrigin.MANIFEST
    status: DependencyStatus = DependencyStatus.UNKNOWN
    security: SecurityStatus = SecurityStatus.UNREVIEWED
    #: Why the project needs this, e.g. "imported by src/api.py".
    reason: str = Field(default="", max_length=500)
    #: The import name when it differs from the distribution name (``yaml`` vs ``PyYAML``).
    import_name: str = Field(default="", max_length=200)

    @property
    def satisfied(self) -> bool:
        """Whether this dependency is usable as-is."""
        return self.status is DependencyStatus.SATISFIED

    @property
    def blocks_execution(self) -> bool:
        """Whether this dependency being unsatisfied stops the project running."""
        return self.status in {
            DependencyStatus.MISSING,
            DependencyStatus.VERSION_MISMATCH,
        } and self.kind in {DependencyKind.RUNTIME, DependencyKind.BUILD}

    def pinned(self) -> str:
        """The manifest line for this dependency.

        Prefers the resolved version: a manifest that records what actually worked is
        reproducible, whereas one that records only a loose constraint is a wish.
        """
        if self.resolved_version:
            return f"{self.name}=={self.resolved_version}"
        if self.requested_version:
            return f"{self.name}{self.requested_version}"
        return self.name


class RuntimeInfo(EdithModel):
    """The interpreter or runtime a project will execute on."""

    ecosystem: Ecosystem = Ecosystem.PYTHON
    #: Absolute path to the executable actually detected.
    executable: str = ""
    version: str = ""
    #: Path to a project-local environment, when one exists.
    virtual_env: str = ""
    #: Whether ``executable`` belongs to that environment rather than a global install.
    is_project_local: bool = False
    package_manager: str = ""
    package_manager_version: str = ""
    available: bool = False
    detail: str = ""

    @property
    def usable(self) -> bool:
        """Whether this runtime can execute the project."""
        return self.available and bool(self.executable)


class EnvironmentSpec(EdithModel):
    """Everything needed to reproduce a project's environment.

    This is the artifact a future Dependency Agent produces and an installer consumes. It
    is deliberately declarative: it says what must be true, not how a particular shell
    should achieve it, so one spec can generate a ``.bat``, a ``.ps1`` and a ``.sh``.
    """

    ecosystem: Ecosystem = Ecosystem.PYTHON
    #: Minimum runtime version, e.g. ``3.11``.
    runtime_version: str = Field(default="", max_length=50)
    package_manager: str = Field(default="pip", max_length=50)
    dependencies: list[Dependency] = Field(default_factory=list)
    #: Non-secret environment variables the project needs. Secrets are never stored here.
    environment_variables: dict[str, str] = Field(default_factory=dict)
    #: Project-local environment directory, relative to the project root.
    venv_path: str = Field(default=".venv", max_length=200)
    #: Modules that must import successfully for the install to count as done.
    verification_imports: list[str] = Field(default_factory=list)
    #: Commands proving the install worked, as argv lists for the M1 shell allowlist.
    verification_commands: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_secret_looking_variables(self) -> EnvironmentSpec:
        """Refuse to carry anything credential-shaped.

        An environment spec is written to disk and committed. A token that reaches it is a
        leaked token, so the type refuses to hold one rather than trusting every caller to
        remember.
        """
        suspicious = ("secret", "token", "password", "passwd", "api_key", "apikey")
        for key in self.environment_variables:
            lowered = key.lower()
            if any(marker in lowered for marker in suspicious):
                raise ValueError(
                    f"environment variable {key!r} looks like a credential and must not be "
                    "stored in an environment spec"
                )
        return self

    @property
    def runtime_dependencies(self) -> list[Dependency]:
        """Dependencies needed to run the application."""
        return [
            dependency
            for dependency in self.dependencies
            if dependency.kind in {DependencyKind.RUNTIME, DependencyKind.BUILD}
        ]

    @property
    def unsatisfied(self) -> list[Dependency]:
        """Dependencies that are not currently usable."""
        return [
            dependency for dependency in self.dependencies if not dependency.satisfied
        ]

    def manifest_lines(self) -> list[str]:
        """Pinned requirement lines, sorted for a stable diff."""
        return sorted({dependency.pinned() for dependency in self.dependencies})


class EnvironmentReport(EdithModel):
    """The result of inspecting a project's environment."""

    project_root: str = ""
    runtime: RuntimeInfo = RuntimeInfo()
    spec: EnvironmentSpec = EnvironmentSpec()
    #: Imported by the source but declared nowhere.
    undeclared: list[Dependency] = Field(default_factory=list)
    #: Declared but not installed.
    missing: list[Dependency] = Field(default_factory=list)
    manifests_found: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Whether the project can run right now."""
        return self.runtime.usable and not self.missing and not self.undeclared

    def summary(self) -> str:
        """A short, human-readable statement of the environment's state."""
        if not self.runtime.usable:
            return f"runtime unavailable: {self.runtime.detail or 'not detected'}"
        parts = [
            f"{self.runtime.ecosystem} {self.runtime.version}",
            "project-local venv" if self.runtime.is_project_local else "global interpreter",
            f"{len(self.spec.dependencies)} dependency(ies)",
        ]
        if self.missing:
            parts.append(f"{len(self.missing)} MISSING")
        if self.undeclared:
            parts.append(f"{len(self.undeclared)} UNDECLARED")
        return ", ".join(parts)
