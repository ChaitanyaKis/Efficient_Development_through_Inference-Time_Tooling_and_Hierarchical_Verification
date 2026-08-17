"""The Dependency Agent's deterministic half.

M5 item 6 asks for dependency discovery, manifest reconciliation, install artifacts, and
import verification. M3.1 already built all of it. This module is the workflow that joins
them, and it is almost entirely model-free by design:

    generated code -> discovery -> reconciliation -> provisioning -> import verification

Every step is a parser, a comparison, or a template. A model is not asked which packages a
project needs, because the AST already knows: it lists exactly what the source imports, and
guessing would be strictly worse than reading.

The security posture is inherited rather than re-argued. Install artifacts are *generated
text*; they reach disk through the M1 gateway; `assert_safe` refuses alternate package
sources and shell metacharacters at generation time; and nothing here executes an installer.
Running one is a separate, explicitly-approved `shell.run` call, which is what keeps
dependency installation an execution boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from edith.environment.install import UnsafeDependencyError
from edith.environment.provision import ArtifactWriteResult, inspect_project, provision
from edith.environment.schema import (
    Dependency,
    DependencyKind,
    DependencyOrigin,
    DependencyStatus,
    EnvironmentReport,
    EnvironmentSpec,
)
from edith.errors import FailureCategory
from edith.observability.logging import get_logger
from edith.tools.gateway import ToolGateway

logger = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """What discovery found that the manifests did not say.

    The interesting output is not the dependency list -- it is the *disagreement* between
    what the code imports and what the project declares. An import nothing declares breaks on
    a fresh machine; a declaration nothing imports is dead weight someone still installs.
    """

    #: Imported by the source, declared by no manifest.
    undeclared: tuple[Dependency, ...] = ()
    #: Declared but not installed in the detected environment.
    missing: tuple[Dependency, ...] = ()
    #: Declared and installed and never imported.
    unused: tuple[Dependency, ...] = ()
    spec: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    report: EnvironmentReport = field(default_factory=EnvironmentReport)

    @property
    def reconciled(self) -> bool:
        """Whether the manifests now describe what the code actually needs."""
        return not self.undeclared and not self.missing

    @property
    def blocking(self) -> bool:
        """Whether anything here stops the project running.

        An undeclared import is blocking: the code will fail on any machine but this one.
        An unused declaration is not -- it is waste, not breakage.
        """
        return bool(self.undeclared) or any(item.blocks_execution for item in self.missing)

    def summary(self) -> str:
        """A one-line description."""
        if self.reconciled:
            return f"{len(self.spec.dependencies)} dependency(ies), manifests reconciled"
        parts = []
        if self.undeclared:
            parts.append(f"{len(self.undeclared)} undeclared")
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.unused:
            parts.append(f"{len(self.unused)} unused")
        return ", ".join(parts)


def reconcile(project_root: Path) -> ReconciliationResult:
    """Compare what the code imports against what the manifests declare.

    Deterministic: imports come from the AST, declarations from the manifests, and installed
    versions from the interpreter. No model is involved and none could improve the answer.
    """
    report = inspect_project(project_root)
    spec = report.spec

    unused = tuple(
        dependency
        for dependency in spec.dependencies
        if dependency.origin is DependencyOrigin.MANIFEST
        and dependency.status is DependencyStatus.SATISFIED
        and not dependency.import_name
        and dependency.kind is DependencyKind.RUNTIME
    )

    result = ReconciliationResult(
        undeclared=tuple(report.undeclared),
        missing=tuple(report.missing),
        unused=unused,
        spec=spec,
        report=report,
    )
    logger.info(
        "dependency.reconciled",
        project=str(project_root),
        undeclared=len(result.undeclared),
        missing=len(result.missing),
        unused=len(result.unused),
        reconciled=result.reconciled,
    )
    return result


def declare_undeclared(spec: EnvironmentSpec) -> EnvironmentSpec:
    """Promote source-discovered imports into declared dependencies.

    The manifest reconciliation step. A package the code imports becomes a package the
    project declares, so a fresh checkout installs it. Only *discovered* imports are promoted
    -- nothing is added because a model suggested it, which is the rule that keeps a
    hallucinated package out of a generated installer.
    """
    promoted = tuple(
        dependency.model_copy(update={"origin": DependencyOrigin.MANIFEST})
        if dependency.status is DependencyStatus.UNDECLARED
        and dependency.origin is DependencyOrigin.SOURCE_IMPORT
        else dependency
        for dependency in spec.dependencies
    )
    return spec.model_copy(update={"dependencies": list(promoted)})


@dataclass
class ProvisionOutcome:
    """The result of generating and writing install artifacts."""

    reconciliation: ReconciliationResult
    written: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    error: str = ""
    failure_category: FailureCategory | None = None

    @property
    def ok(self) -> bool:
        """Whether reproducible install artifacts now exist."""
        return bool(self.written) and not self.denied and not self.error


def provision_environment(
    gateway: ToolGateway, project_root: Path, *, promote: bool = True
) -> ProvisionOutcome:
    """Reconcile manifests and write reproducible install artifacts.

    Args:
        gateway: A gateway scoped to the dependency role. Every write goes through it, so a
            manifest outside the role's scope is refused by the policy layer rather than by
            this function remembering to check.
        project_root: The project to provision.
        promote: Whether to declare imports the manifests missed.

    Returns:
        What was written and what was refused. Never raises: a refused write is a policy
        decision the caller must be able to render, not an exception to catch.
    """
    reconciliation = reconcile(project_root)
    spec = (
        declare_undeclared(reconciliation.spec) if promote else reconciliation.spec
    )

    try:
        _, write_result = provision(gateway, spec)
    except UnsafeDependencyError as exc:
        # A dependency that could smuggle an alternate source or a shell metacharacter.
        # Generation fails before anything reaches disk.
        logger.warning("dependency.unsafe", error=str(exc))
        return ProvisionOutcome(
            reconciliation=reconciliation,
            error=str(exc),
            failure_category=FailureCategory.SECURITY_FAILURE,
        )
    except ValueError as exc:
        return ProvisionOutcome(
            reconciliation=reconciliation,
            error=str(exc),
            failure_category=FailureCategory.CONFIGURATION_ERROR,
        )

    outcome = ProvisionOutcome(
        reconciliation=reconciliation,
        written=tuple(write_result.written),
        denied=tuple(write_result.denied),
        error="; ".join(f"{path}: {reason}" for path, reason in write_result.errors.items()),
        failure_category=(
            FailureCategory.SECURITY_FAILURE if write_result.denied else None
        ),
    )
    logger.info(
        "dependency.provisioned",
        written=len(outcome.written),
        denied=len(outcome.denied),
        reconciled=reconciliation.reconciled,
    )
    return outcome


def verify_imports(report: EnvironmentReport) -> tuple[bool, str]:
    """Whether every runtime dependency is actually importable.

    Checked against what the interpreter reports as installed, not against what the manifest
    claims. "pip reported success" and "the application can start" are different statements,
    and only the second matters to a user.
    """
    unresolved = [
        dependency.name
        for dependency in report.spec.runtime_dependencies
        if dependency.status
        in {DependencyStatus.MISSING, DependencyStatus.UNDECLARED}
    ]
    if unresolved:
        return (
            False,
            f"{len(unresolved)} runtime dependency(ies) are not importable: "
            f"{', '.join(sorted(unresolved)[:5])}",
        )
    return (True, "every runtime dependency is importable")


def artifact_paths(outcome: ArtifactWriteResult | ProvisionOutcome) -> tuple[str, ...]:
    """The install artifacts a provisioning run produced."""
    if isinstance(outcome, ProvisionOutcome):
        return outcome.written
    return tuple(outcome.written)
