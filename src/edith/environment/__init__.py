"""Environment and dependency foundation.

Establishes what a future Dependency Agent needs: a provider-neutral model of what a
project requires, deterministic discovery of what it actually uses, classification that
distinguishes a missing package from a broken test, and reproducible installation
artifacts.
"""

from .classify import FailureDiagnosis, classify_failure, missing_modules
from .install import (
    InstallArtifacts,
    UnsafeDependencyError,
    assert_safe,
    generate,
    render_manifest,
    render_posix_shell,
    render_powershell,
    render_windows_batch,
)
from .provision import (
    ArtifactWriteResult,
    inspect_project,
    provision,
    write_artifacts,
)
from .python_env import (
    IMPORT_TO_DISTRIBUTION,
    detect_runtime,
    discover,
    distribution_for,
    find_project_venv,
    find_source_imports,
    installed_distributions,
    local_module_names,
    parse_pyproject,
    parse_requirements,
    stdlib_modules,
)
from .schema import (
    Dependency,
    DependencyKind,
    DependencyOrigin,
    DependencyStatus,
    Ecosystem,
    EnvironmentReport,
    EnvironmentSpec,
    RuntimeInfo,
    SecurityStatus,
)

__all__ = [
    "IMPORT_TO_DISTRIBUTION",
    "ArtifactWriteResult",
    "Dependency",
    "DependencyKind",
    "DependencyOrigin",
    "DependencyStatus",
    "Ecosystem",
    "EnvironmentReport",
    "EnvironmentSpec",
    "FailureDiagnosis",
    "InstallArtifacts",
    "RuntimeInfo",
    "SecurityStatus",
    "UnsafeDependencyError",
    "assert_safe",
    "classify_failure",
    "detect_runtime",
    "discover",
    "distribution_for",
    "find_project_venv",
    "find_source_imports",
    "generate",
    "inspect_project",
    "installed_distributions",
    "local_module_names",
    "missing_modules",
    "parse_pyproject",
    "parse_requirements",
    "provision",
    "render_manifest",
    "render_posix_shell",
    "render_powershell",
    "render_windows_batch",
    "stdlib_modules",
    "write_artifacts",
]
