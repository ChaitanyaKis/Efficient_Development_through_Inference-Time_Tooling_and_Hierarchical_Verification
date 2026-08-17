"""Generating reproducible installation artifacts.

The requirement is a user-visible one: a generated application must not hand its user a
``ModuleNotFoundError``. That needs a manifest recording exactly what worked, plus a script
that reproduces it on a machine that has never run the project.

Every generated script obeys the same rules, because each one is the difference between
"works on Edith's machine" and "works":

1. Find a supported interpreter, and fail loudly if there isn't one.
2. Create or reuse a **project-local** environment -- never install system-wide.
3. Install pinned dependencies from the manifest, never from an ad-hoc argument list.
4. Verify the imports actually import.
5. Exit non-zero on any failure, with a diagnostic saying what to do.

The scripts are *generated text*, not executed here. Running them goes through the M1 tool
gateway like any other command, so installation stays behind the same execution boundary as
everything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from edith.observability.logging import get_logger

from .schema import Dependency, DependencyKind, Ecosystem, EnvironmentSpec

logger = get_logger(__name__)

#: A package name pip will accept. Anything else is rejected rather than escaped: a name
#: that needs escaping to be safe is a name that should not be in a manifest.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")

#: A version specifier we are willing to write.
_SAFE_SPECIFIER_RE = re.compile(r"^[<>=!~^ ,.0-9A-Za-z*+-]{0,100}$")

#: Package sources other than the default index require explicit operator approval, so any
#: manifest entry carrying one is refused. Applied to package *names*, where none of these
#: characters is ever legitimate.
_FORBIDDEN_FRAGMENTS: tuple[str, ...] = (
    "--index-url",
    "--extra-index-url",
    "--find-links",
    "--trusted-host",
    "--pre",
    "git+",
    "http://",
    "https://",
    "file:",
    ";",
    "&&",
    "||",
    "|",
    "`",
    "$(",
    ">",
    "<",
    "\n",
    "\r",
)

#: The same list, minus the comparison operators. ``>=2.31,<3`` is what an ordinary pinned
#: dependency looks like, so treating ``<`` and ``>`` as redirection here would refuse every
#: real manifest. The specifier is not left unguarded: :data:`_SAFE_SPECIFIER_RE` is an
#: allowlist that admits no scheme, path separator, or shell character at all.
_SPECIFIER_FORBIDDEN_FRAGMENTS: tuple[str, ...] = tuple(
    fragment for fragment in _FORBIDDEN_FRAGMENTS if fragment not in {"<", ">"}
)


class UnsafeDependencyError(ValueError):
    """A dependency entry is not safe to write into an installation artifact."""


def assert_safe(dependency: Dependency) -> None:
    """Reject a dependency that could smuggle a command or an alternate source.

    An installer is executable text assembled from names that may have originated with a
    model. Validating them here means a malformed or hostile name becomes a refusal at
    generation time rather than a surprise at install time.

    Raises:
        UnsafeDependencyError: The name or version specifier is not acceptable.
    """
    if not _SAFE_NAME_RE.match(dependency.name):
        raise UnsafeDependencyError(
            f"dependency name {dependency.name!r} is not a plain package name"
        )

    _reject_fragments(dependency.name, _FORBIDDEN_FRAGMENTS, dependency.name)

    for specifier in (dependency.requested_version, dependency.resolved_version):
        if not specifier:
            continue
        if not _SAFE_SPECIFIER_RE.match(specifier):
            raise UnsafeDependencyError(
                f"version specifier {specifier!r} for {dependency.name!r} is not acceptable"
            )
        _reject_fragments(specifier, _SPECIFIER_FORBIDDEN_FRAGMENTS, dependency.name)


def _reject_fragments(text: str, fragments: tuple[str, ...], subject: str) -> None:
    """Raise when ``text`` contains any forbidden fragment."""
    lowered = text.lower()
    for fragment in fragments:
        if fragment in lowered:
            raise UnsafeDependencyError(
                f"dependency {subject!r} carries a forbidden fragment {fragment!r}; "
                "alternate package sources and shell metacharacters are not permitted"
            )


@dataclass(frozen=True)
class InstallArtifacts:
    """The files that make an environment reproducible."""

    manifest_name: str
    manifest: str
    windows_batch: str
    windows_powershell: str
    posix_shell: str

    def as_files(self) -> dict[str, str]:
        """Map of repository-relative path -> content, ready to write through the gateway."""
        return {
            self.manifest_name: self.manifest,
            "scripts/install.bat": self.windows_batch,
            "scripts/install.ps1": self.windows_powershell,
            "scripts/install.sh": self.posix_shell,
        }


def render_manifest(spec: EnvironmentSpec) -> str:
    """Render a pinned ``requirements.txt``.

    Pinned to resolved versions wherever known. A manifest that records a loose constraint
    reproduces a *range* of environments; one that records what actually worked reproduces
    the environment that actually worked.
    """
    lines = [
        "# Generated by Edith. Pinned to the versions verified at generation time.",
        "# Regenerate rather than editing by hand.",
        "",
    ]
    runtime = [d for d in spec.dependencies if d.kind is not DependencyKind.TEST]
    tests = [d for d in spec.dependencies if d.kind is DependencyKind.TEST]

    for dependency in sorted(runtime, key=lambda item: item.name.lower()):
        assert_safe(dependency)
        lines.append(dependency.pinned())

    if tests:
        lines.extend(["", "# Test dependencies"])
        for dependency in sorted(tests, key=lambda item: item.name.lower()):
            assert_safe(dependency)
            lines.append(dependency.pinned())

    return "\n".join(lines) + "\n"


def _import_checks(spec: EnvironmentSpec) -> list[str]:
    """Import names worth verifying after an install."""
    return sorted({name for name in spec.verification_imports if name.isidentifier()})


def render_windows_batch(spec: EnvironmentSpec, *, manifest: str = "requirements.txt") -> str:
    """Render ``scripts/install.bat``.

    Windows-first because that is the target machine. ``py -3`` is preferred over ``python``:
    on Windows ``python`` is frequently a Microsoft Store alias that exits without
    installing anything, which produces the exact confusing failure this script exists to
    prevent.
    """
    minimum = spec.runtime_version or "3.11"
    venv = spec.venv_path or ".venv"
    imports = _import_checks(spec)
    import_line = ", ".join(imports)

    lines = [
        "@echo off",
        "REM Generated by Edith. Creates a PROJECT-LOCAL environment; installs nothing",
        "REM system-wide. Exits non-zero on any failure.",
        "setlocal enabledelayedexpansion",
        "cd /d \"%~dp0..\"",
        "",
        f"echo [1/5] Locating Python {minimum} or newer...",
        "set PYTHON_CMD=",
        "REM `py` is the Windows launcher and resolves real installations; bare `python`",
        "REM may be a Store alias that silently does nothing.",
        "py -3 --version >nul 2>&1 && set PYTHON_CMD=py -3",
        "if not defined PYTHON_CMD (",
        "    python --version >nul 2>&1 && set PYTHON_CMD=python",
        ")",
        "if not defined PYTHON_CMD (",
        "    echo ERROR: No Python installation was found.",
        f"    echo Install Python {minimum} or newer from https://www.python.org/downloads/",
        "    echo and ensure it is added to PATH, then re-run this script.",
        "    exit /b 1",
        ")",
        "",
        f"echo [2/5] Creating the project-local environment in {venv}...",
        f"if not exist \"{venv}\" (",
        f"    %PYTHON_CMD% -m venv \"{venv}\"",
        "    if errorlevel 1 (",
        "        echo ERROR: Could not create the virtual environment.",
        "        echo Check available disk space and write permissions in this folder.",
        "        exit /b 1",
        "    )",
        ")",
        f"set VENV_PY={venv}\\Scripts\\python.exe",
        "if not exist \"%VENV_PY%\" (",
        "    echo ERROR: The virtual environment is incomplete: %VENV_PY% is missing.",
        f"    echo Delete the {venv} folder and re-run this script.",
        "    exit /b 1",
        ")",
        "",
        "echo [3/5] Upgrading pip...",
        "\"%VENV_PY%\" -m pip install --upgrade pip --disable-pip-version-check",
        "if errorlevel 1 (",
        "    echo ERROR: pip could not be upgraded. Check your network connection.",
        "    exit /b 1",
        ")",
        "",
        f"echo [4/5] Installing pinned dependencies from {manifest}...",
        f"if not exist \"{manifest}\" (",
        f"    echo ERROR: {manifest} is missing; there is nothing to install.",
        "    exit /b 1",
        ")",
        f"\"%VENV_PY%\" -m pip install -r \"{manifest}\" --disable-pip-version-check",
        "if errorlevel 1 (",
        "    echo ERROR: Dependency installation failed.",
        f"    echo Review the output above and check {manifest} for unavailable versions.",
        "    exit /b 1",
        ")",
        "",
    ]

    if imports:
        lines.extend(
            [
                "echo [5/5] Verifying imports...",
                f"\"%VENV_PY%\" -c \"import {import_line}\"",
                "if errorlevel 1 (",
                "    echo ERROR: Dependencies installed but could not be imported.",
                "    echo The environment is incomplete; re-run after deleting "
                f"{venv}.",
                "    exit /b 1",
                ")",
                "",
            ]
        )
    else:
        lines.extend(["echo [5/5] No imports to verify.", ""])

    lines.extend(
        [
            "echo.",
            "echo Environment ready.",
            f"echo Activate it with: {venv}\\Scripts\\activate",
            "exit /b 0",
        ]
    )
    return "\r\n".join(lines) + "\r\n"


def render_powershell(spec: EnvironmentSpec, *, manifest: str = "requirements.txt") -> str:
    """Render ``scripts/install.ps1``."""
    minimum = spec.runtime_version or "3.11"
    venv = spec.venv_path or ".venv"
    imports = _import_checks(spec)

    lines = [
        "#requires -Version 5.1",
        "# Generated by Edith. Creates a PROJECT-LOCAL environment; installs nothing",
        "# system-wide. Throws and exits non-zero on any failure.",
        "$ErrorActionPreference = 'Stop'",
        "Set-Location (Join-Path $PSScriptRoot '..')",
        "",
        f"Write-Host '[1/5] Locating Python {minimum} or newer...'",
        "$pythonCmd = $null",
        "if (Get-Command py -ErrorAction SilentlyContinue) { $pythonCmd = 'py' }",
        "elseif (Get-Command python -ErrorAction SilentlyContinue) { $pythonCmd = 'python' }",
        "if (-not $pythonCmd) {",
        "    Write-Error 'No Python installation was found. Install Python "
        f"{minimum} or newer from https://www.python.org/downloads/ and re-run.'",
        "    exit 1",
        "}",
        "",
        f"Write-Host '[2/5] Creating the project-local environment in {venv}...'",
        f"if (-not (Test-Path '{venv}')) {{",
        f"    if ($pythonCmd -eq 'py') {{ & py -3 -m venv '{venv}' }} "
        f"else {{ & python -m venv '{venv}' }}",
        "}",
        f"$venvPython = Join-Path '{venv}' 'Scripts/python.exe'",
        "if (-not (Test-Path $venvPython)) {",
        f"    Write-Error \"The virtual environment is incomplete. Delete {venv} and re-run.\"",
        "    exit 1",
        "}",
        "",
        "Write-Host '[3/5] Upgrading pip...'",
        "& $venvPython -m pip install --upgrade pip --disable-pip-version-check",
        "if ($LASTEXITCODE -ne 0) { Write-Error 'pip could not be upgraded.'; exit 1 }",
        "",
        f"Write-Host '[4/5] Installing pinned dependencies from {manifest}...'",
        f"if (-not (Test-Path '{manifest}')) {{",
        f"    Write-Error '{manifest} is missing; there is nothing to install.'",
        "    exit 1",
        "}",
        f"& $venvPython -m pip install -r '{manifest}' --disable-pip-version-check",
        "if ($LASTEXITCODE -ne 0) {",
        "    Write-Error 'Dependency installation failed. Review the output above.'",
        "    exit 1",
        "}",
        "",
    ]

    if imports:
        lines.extend(
            [
                "Write-Host '[5/5] Verifying imports...'",
                f"& $venvPython -c \"import {', '.join(imports)}\"",
                "if ($LASTEXITCODE -ne 0) {",
                "    Write-Error 'Dependencies installed but could not be imported.'",
                "    exit 1",
                "}",
                "",
            ]
        )
    else:
        lines.append("Write-Host '[5/5] No imports to verify.'")

    lines.extend(
        [
            "Write-Host ''",
            "Write-Host 'Environment ready.'",
            f"Write-Host 'Activate it with: {venv}\\Scripts\\Activate.ps1'",
            "exit 0",
        ]
    )
    return "\r\n".join(lines) + "\r\n"


def render_posix_shell(spec: EnvironmentSpec, *, manifest: str = "requirements.txt") -> str:
    """Render ``scripts/install.sh``."""
    minimum = spec.runtime_version or "3.11"
    venv = spec.venv_path or ".venv"
    imports = _import_checks(spec)

    lines = [
        "#!/usr/bin/env bash",
        "# Generated by Edith. Creates a PROJECT-LOCAL environment; installs nothing",
        "# system-wide. Exits non-zero on any failure.",
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        "",
        f'echo "[1/5] Locating Python {minimum} or newer..."',
        'PYTHON_CMD=""',
        'if command -v python3 >/dev/null 2>&1; then PYTHON_CMD=python3;',
        'elif command -v python >/dev/null 2>&1; then PYTHON_CMD=python; fi',
        'if [ -z "$PYTHON_CMD" ]; then',
        f'    echo "ERROR: No Python installation was found. Install Python {minimum}+." >&2',
        "    exit 1",
        "fi",
        "",
        f'echo "[2/5] Creating the project-local environment in {venv}..."',
        f'if [ ! -d "{venv}" ]; then "$PYTHON_CMD" -m venv "{venv}"; fi',
        f'VENV_PY="{venv}/bin/python"',
        'if [ ! -x "$VENV_PY" ]; then',
        f'    echo "ERROR: The virtual environment is incomplete. Delete {venv} and re-run." >&2',
        "    exit 1",
        "fi",
        "",
        'echo "[3/5] Upgrading pip..."',
        '"$VENV_PY" -m pip install --upgrade pip --disable-pip-version-check',
        "",
        f'echo "[4/5] Installing pinned dependencies from {manifest}..."',
        f'if [ ! -f "{manifest}" ]; then',
        f'    echo "ERROR: {manifest} is missing; there is nothing to install." >&2',
        "    exit 1",
        "fi",
        f'"$VENV_PY" -m pip install -r "{manifest}" --disable-pip-version-check',
        "",
    ]

    if imports:
        lines.extend(
            [
                'echo "[5/5] Verifying imports..."',
                f'"$VENV_PY" -c "import {", ".join(imports)}"',
                "",
            ]
        )
    else:
        lines.append('echo "[5/5] No imports to verify."')

    lines.extend(
        [
            'echo ""',
            'echo "Environment ready."',
            f'echo "Activate it with: source {venv}/bin/activate"',
            "exit 0",
        ]
    )
    return "\n".join(lines) + "\n"


def generate(spec: EnvironmentSpec, *, manifest_name: str = "requirements.txt") -> InstallArtifacts:
    """Generate every installation artifact for a spec.

    Raises:
        UnsafeDependencyError: A dependency could not be safely written.
        ValueError: The ecosystem is not yet supported.
    """
    if spec.ecosystem is not Ecosystem.PYTHON:
        raise ValueError(
            f"installation artifacts are only implemented for Python, not {spec.ecosystem}"
        )

    artifacts = InstallArtifacts(
        manifest_name=manifest_name,
        manifest=render_manifest(spec),
        windows_batch=render_windows_batch(spec, manifest=manifest_name),
        windows_powershell=render_powershell(spec, manifest=manifest_name),
        posix_shell=render_posix_shell(spec, manifest=manifest_name),
    )
    logger.info(
        "environment.artifacts_generated",
        dependencies=len(spec.dependencies),
        imports_verified=len(_import_checks(spec)),
    )
    return artifacts
