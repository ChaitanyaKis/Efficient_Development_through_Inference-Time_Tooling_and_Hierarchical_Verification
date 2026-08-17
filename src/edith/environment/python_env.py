"""Python environment detection and deterministic dependency discovery.

Two jobs:

**Detection.** Which interpreter will actually run this project? Not ``python`` from PATH --
M2 learned that lesson when PATH's interpreter had no pytest and every run was reported as a
failing test. A project-local ``.venv`` is preferred, because that is what a reproducible
install produces.

**Discovery.** What does the project actually require? Determined by parsing imports and
reading manifests, not by asking a model. An LLM guessing at dependencies is how you end up
installing a plausible-sounding package that nobody asked for.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from edith.observability.logging import get_logger

from .schema import (
    Dependency,
    DependencyKind,
    DependencyOrigin,
    DependencyStatus,
    Ecosystem,
    EnvironmentSpec,
    RuntimeInfo,
)

logger = get_logger(__name__)

_PROBE_TIMEOUT_SECONDS = 20.0

#: Import name -> distribution name, where they differ. Guessing this mapping is a classic
#: source of wrong installs (``pip install yaml`` installs something unrelated).
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "jwt": "PyJWT",
    "serial": "pyserial",
    "OpenSSL": "pyOpenSSL",
    "attr": "attrs",
    "google": "google-api-python-client",
    "pkg_resources": "setuptools",
    "win32api": "pywin32",
    "win32com": "pywin32",
}

#: Directories never scanned for project imports.
_SKIP_DIRECTORIES = frozenset(
    {
        ".git", ".venv", "venv", "env", "__pycache__", "node_modules", "build", "dist",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs", "site-packages",
        ".edith",
    }
)

#: Requirement line: name, optional extras, optional version specifier.
_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([<>=!~][^;#]*)?"
)


def stdlib_modules() -> frozenset[str]:
    """Top-level modules shipped with this Python.

    Uses ``sys.stdlib_module_names`` rather than a hand-maintained list, so it stays
    correct across versions without anyone remembering to update it.
    """
    return frozenset(sys.stdlib_module_names)


def distribution_for(import_name: str) -> str:
    """Map a top-level import name to its distribution name."""
    return IMPORT_TO_DISTRIBUTION.get(import_name, import_name)


def _run(argv: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a probe command, returning ``(exit_code, stdout, stderr)``.

    Used only for *inspecting* an interpreter Edith already located. Anything that installs
    or modifies goes through the M1 tool gateway instead.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, absolute executable
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (-1, "", f"{type(exc).__name__}: {exc}")
    return (completed.returncode, completed.stdout, completed.stderr)


def find_project_venv(project_root: Path) -> Path | None:
    """Return the project-local virtual environment's interpreter, if one exists."""
    for name in (".venv", "venv", "env"):
        candidate = project_root / name
        for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
            executable = candidate / relative
            if executable.is_file():
                return executable
    return None


def detect_runtime(project_root: Path) -> RuntimeInfo:
    """Detect the interpreter that should run this project.

    Preference order, and the reason for it:

    1. A project-local virtual environment -- the only reproducible answer.
    2. The interpreter running Edith -- known to exist and known to have its dependencies.

    ``python`` from PATH is deliberately *not* consulted. It is the least predictable option
    on Windows, where it may be a Store alias, a different major version, or absent.
    """
    executable = find_project_venv(project_root)
    is_local = executable is not None
    if executable is None:
        executable = Path(sys.executable)

    code, out, err = _run([str(executable), "--version"])
    if code != 0:
        return RuntimeInfo(
            ecosystem=Ecosystem.PYTHON,
            executable=str(executable),
            available=False,
            detail=(err or out).strip()[:200] or "interpreter did not respond",
        )

    version = (out or err).strip().removeprefix("Python").strip()
    manager_code, manager_out, _ = _run([str(executable), "-m", "pip", "--version"])
    manager_version = ""
    if manager_code == 0:
        manager_version = manager_out.strip().split(" ")[1] if " " in manager_out else ""

    venv_path = ""
    if is_local:
        # Scripts/python.exe -> .venv
        venv_path = str(executable.parent.parent)
    else:
        base = sysconfig.get_config_var("base") or ""
        prefix = sys.prefix
        if base and prefix != base:
            venv_path = prefix

    return RuntimeInfo(
        ecosystem=Ecosystem.PYTHON,
        executable=str(executable),
        version=version,
        virtual_env=venv_path,
        is_project_local=is_local,
        package_manager="pip" if manager_code == 0 else "",
        package_manager_version=manager_version,
        available=True,
        detail="project-local virtual environment"
        if is_local
        else "the interpreter running Edith (no project venv found)",
    )


def installed_distributions(executable: str) -> dict[str, str]:
    """Return ``{normalized name: version}`` for packages installed in an interpreter."""
    code, out, _ = _run(
        [
            executable,
            "-c",
            "import json,importlib.metadata as m;"
            "print(json.dumps({d.metadata['Name']: d.version "
            "for d in m.distributions() if d.metadata['Name']}))",
        ]
    )
    if code != 0:
        return {}
    try:
        import json  # noqa: PLC0415 - only needed on this path

        raw = json.loads(out)
    except (ValueError, TypeError):
        return {}
    return {str(name).lower().replace("_", "-"): str(version) for name, version in raw.items()}


def find_source_imports(project_root: Path, *, limit: int = 400) -> dict[str, list[str]]:
    """Return ``{top-level import: [files that import it]}`` for a project.

    AST-based, not regex: a commented-out or string-literal import is not a dependency, and
    a regex cannot tell the difference.
    """
    found: dict[str, list[str]] = {}
    scanned = 0

    for path in sorted(project_root.rglob("*.py")):
        if scanned >= limit:
            break
        if any(part in _SKIP_DIRECTORIES for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        scanned += 1

        relative = path.relative_to(project_root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], []).append(relative)
            # `from . import x` is internal, never a dependency.
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.setdefault(node.module.split(".")[0], []).append(relative)
    return found


def local_module_names(project_root: Path) -> set[str]:
    """Modules and packages defined by the project itself.

    Without this, a project's own ``calculator.py`` looks like a missing third-party
    package, and the installer would try to fetch it from an index.
    """
    names: set[str] = set()
    for path in project_root.iterdir():
        if path.is_dir() and (path / "__init__.py").is_file():
            names.add(path.name)
        elif path.is_file() and path.suffix == ".py":
            names.add(path.stem)
    for source_dir in ("src", "lib"):
        candidate = project_root / source_dir
        if candidate.is_dir():
            for path in candidate.iterdir():
                if path.is_dir() and (path / "__init__.py").is_file():
                    names.add(path.name)
                elif path.is_file() and path.suffix == ".py":
                    names.add(path.stem)
    return names


def parse_requirements(text: str) -> list[tuple[str, str]]:
    """Parse ``requirements.txt`` content into ``(name, specifier)`` pairs."""
    parsed: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        # Matched against full URL schemes, not a bare "http" prefix: `httpx` and
        # `httpcore` are ordinary packages, and dropping them silently would leave every
        # generated manifest missing a dependency the project actually imports.
        if not line or line.startswith(("#", "-", "git+", "http://", "https://", "file:")):
            continue
        match = _REQUIREMENT_RE.match(line)
        if match:
            parsed.append((match.group(1), (match.group(3) or "").strip()))
    return parsed


def parse_pyproject(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """Parse a ``pyproject.toml`` into runtime deps, dev deps, and requires-python."""
    try:
        import tomllib  # noqa: PLC0415 - stdlib since 3.11, imported where used
    except ImportError:  # pragma: no cover - unreachable on supported versions
        return ([], [], "")

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return ([], [], "")

    project = data.get("project", {})
    runtime = [
        pair
        for entry in project.get("dependencies", [])
        for pair in parse_requirements(str(entry))
    ]
    development: list[tuple[str, str]] = []
    for group in (project.get("optional-dependencies") or {}).values():
        for entry in group:
            development.extend(parse_requirements(str(entry)))

    return (runtime, development, str(project.get("requires-python", "")))


def discover(project_root: Path, runtime: RuntimeInfo | None = None) -> EnvironmentSpec:
    """Build an :class:`EnvironmentSpec` from a project's manifests and source.

    Declared dependencies come from manifests; actual usage comes from imports. Both are
    recorded, because the interesting cases are exactly where they disagree: an import with
    no declaration will break on another machine, and a declaration with no import is dead
    weight.
    """
    info = runtime or detect_runtime(project_root)
    installed = installed_distributions(info.executable) if info.usable else {}

    dependencies: dict[str, Dependency] = {}
    manifests: list[str] = []
    requires_python = ""

    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        manifests.append("pyproject.toml")
        runtime_deps, dev_deps, requires_python = parse_pyproject(
            pyproject.read_text(encoding="utf-8")
        )
        for name, specifier in runtime_deps:
            dependencies[name.lower()] = Dependency(
                name=name,
                requested_version=specifier,
                kind=DependencyKind.RUNTIME,
                origin=DependencyOrigin.MANIFEST,
                reason="declared in pyproject.toml",
            )
        for name, specifier in dev_deps:
            dependencies.setdefault(
                name.lower(),
                Dependency(
                    name=name,
                    requested_version=specifier,
                    kind=DependencyKind.DEVELOPMENT,
                    origin=DependencyOrigin.MANIFEST,
                    reason="declared as an optional dependency",
                ),
            )

    for filename in ("requirements.txt", "requirements-dev.txt", "requirements/base.txt"):
        candidate = project_root / filename
        if not candidate.is_file():
            continue
        manifests.append(filename)
        kind = DependencyKind.DEVELOPMENT if "dev" in filename else DependencyKind.RUNTIME
        for name, specifier in parse_requirements(candidate.read_text(encoding="utf-8")):
            dependencies.setdefault(
                name.lower(),
                Dependency(
                    name=name,
                    requested_version=specifier,
                    kind=kind,
                    origin=DependencyOrigin.MANIFEST,
                    reason=f"declared in {filename}",
                ),
            )

    # Imports that no manifest declares. These are the ones that break on a fresh machine.
    standard = stdlib_modules()
    local = local_module_names(project_root)
    for import_name, files in find_source_imports(project_root).items():
        if import_name in standard or import_name in local or import_name.startswith("_"):
            continue
        distribution = distribution_for(import_name)
        key = distribution.lower()
        if key in dependencies:
            dependencies[key].import_name = import_name
            continue
        dependencies[key] = Dependency(
            name=distribution,
            import_name=import_name,
            kind=DependencyKind.TEST if _looks_like_test(files) else DependencyKind.RUNTIME,
            origin=DependencyOrigin.SOURCE_IMPORT,
            status=DependencyStatus.UNDECLARED,
            reason=f"imported by {files[0]}",
        )

    for dependency in dependencies.values():
        _apply_installed_status(dependency, installed)

    spec = EnvironmentSpec(
        ecosystem=Ecosystem.PYTHON,
        runtime_version=requires_python or _major_minor(info.version),
        package_manager=info.package_manager or "pip",
        dependencies=sorted(dependencies.values(), key=lambda item: item.name.lower()),
        verification_imports=sorted(
            {
                dependency.import_name or dependency.name
                for dependency in dependencies.values()
                if dependency.kind is DependencyKind.RUNTIME
            }
        ),
    )
    logger.info(
        "environment.discovered",
        project=str(project_root),
        manifests=manifests,
        dependencies=len(spec.dependencies),
    )
    return spec


def _looks_like_test(files: list[str]) -> bool:
    """Whether every file importing something is a test file."""
    return all(
        Path(item).name.startswith("test_") or "/tests/" in f"/{item}" for item in files
    )


def _apply_installed_status(dependency: Dependency, installed: dict[str, str]) -> None:
    """Set a dependency's status from what is actually installed."""
    key = dependency.name.lower().replace("_", "-")
    version = installed.get(key)
    if version is None:
        dependency.status = (
            DependencyStatus.UNDECLARED
            if dependency.origin is DependencyOrigin.SOURCE_IMPORT
            else DependencyStatus.MISSING
        )
        return

    dependency.resolved_version = version
    if dependency.origin is DependencyOrigin.SOURCE_IMPORT:
        # Imported and installed, but nothing declares it: it works here and nowhere else.
        dependency.status = DependencyStatus.UNDECLARED
    else:
        dependency.status = DependencyStatus.SATISFIED


def _major_minor(version: str) -> str:
    """Reduce a full version to ``major.minor``."""
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version
