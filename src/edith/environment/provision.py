"""Inspecting a project's environment, and writing install artifacts through the gateway.

The generation in :mod:`edith.environment.install` produces *text*. This module is the only
place that puts that text on disk, and it does so through the M1 tool gateway rather than
through :mod:`pathlib`.

That is not ceremony. An installer is executable content, and the path it lands on may have
been influenced by a model. Routing the write through ``filesystem.write`` means the path
policy, the write scope, and the audit log all apply to it exactly as they apply to source
code. A denied write is reported as a denial; nothing here falls back to writing directly.

Nothing in this module installs anything. Running a generated script is a separate,
explicitly-approved ``shell.run`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from edith.observability.logging import get_logger
from edith.tools.gateway import ToolGateway
from edith.tools.schemas import ToolCall

from .install import InstallArtifacts, generate
from .python_env import detect_runtime, discover
from .schema import (
    Dependency,
    DependencyStatus,
    EnvironmentReport,
    EnvironmentSpec,
)

logger = get_logger(__name__)


def inspect_project(project_root: Path) -> EnvironmentReport:
    """Determine what a project requires and whether it can run right now.

    Entirely deterministic: manifests are parsed, imports are read from the AST, and the
    interpreter is asked what it has installed. No model is consulted, because a model asked
    "what does this project depend on?" will produce a plausible list rather than a true one.
    """
    runtime = detect_runtime(project_root)
    spec = discover(project_root, runtime)

    undeclared: list[Dependency] = []
    missing: list[Dependency] = []
    for dependency in spec.dependencies:
        if dependency.status is DependencyStatus.UNDECLARED:
            undeclared.append(dependency)
        elif dependency.status in {
            DependencyStatus.MISSING,
            DependencyStatus.VERSION_MISMATCH,
        }:
            missing.append(dependency)

    notes: list[str] = []
    if not runtime.usable:
        notes.append(f"no usable interpreter was detected: {runtime.detail}")
    elif not runtime.is_project_local:
        notes.append(
            "no project-local virtual environment exists, so what works here may not "
            "work on another machine; run the generated install script to create one"
        )
    for dependency in undeclared:
        notes.append(
            f"{dependency.name} is imported but declared in no manifest, so a fresh "
            f"checkout will fail with ModuleNotFoundError"
        )

    manifests = [
        name
        for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt")
        if (project_root / name).is_file()
    ]

    report = EnvironmentReport(
        project_root=str(project_root),
        runtime=runtime,
        spec=spec,
        undeclared=undeclared,
        missing=missing,
        manifests_found=manifests,
        notes=notes,
    )
    logger.info(
        "environment.inspected",
        project=str(project_root),
        runtime_available=runtime.usable,
        dependencies=len(spec.dependencies),
        undeclared=len(undeclared),
        missing=len(missing),
    )
    return report


@dataclass(frozen=True)
class ArtifactWriteResult:
    """What reached disk, and what was refused."""

    written: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every artifact was written."""
        return not self.denied and not self.errors


def write_artifacts(
    gateway: ToolGateway,
    artifacts: InstallArtifacts,
    *,
    overwrite: bool = True,
) -> ArtifactWriteResult:
    """Write install artifacts through ``filesystem.write``.

    Args:
        gateway: A permission-scoped gateway. The calling agent must hold
            ``filesystem.write`` and have the target paths inside its write scope.
        artifacts: Rendered text from :func:`edith.environment.install.generate`.
        overwrite: Whether to replace existing artifacts. True by default because these
            files are generated and meant to be regenerated, never hand-edited.

    Returns:
        Which paths were written and which were refused. A refusal is returned, never
        raised past and never worked around: the gateway declining a path is the policy
        layer doing its job.
    """
    written: list[str] = []
    denied: list[str] = []
    errors: dict[str, str] = {}

    for relative, content in artifacts.as_files().items():
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={
                    "path": relative,
                    "content": content,
                    "overwrite": overwrite,
                    "create_parents": True,
                },
            )
        )
        if result.ok:
            written.append(relative)
        elif result.denied:
            denied.append(relative)
            logger.warning(
                "environment.artifact_denied", path=relative, error=result.error
            )
        else:
            errors[relative] = result.error or "the write failed"

    logger.info(
        "environment.artifacts_written",
        written=len(written),
        denied=len(denied),
        errors=len(errors),
    )
    return ArtifactWriteResult(written=written, denied=denied, errors=errors)


def provision(
    gateway: ToolGateway,
    spec: EnvironmentSpec,
    *,
    manifest_name: str = "requirements.txt",
    overwrite: bool = True,
) -> tuple[InstallArtifacts, ArtifactWriteResult]:
    """Generate install artifacts for a spec and write them through the gateway.

    Raises:
        UnsafeDependencyError: A dependency could not be safely written. Generation fails
            before anything reaches disk, so a rejected manifest never produces a partial
            set of scripts.
    """
    artifacts = generate(spec, manifest_name=manifest_name)
    return (artifacts, write_artifacts(gateway, artifacts, overwrite=overwrite))
