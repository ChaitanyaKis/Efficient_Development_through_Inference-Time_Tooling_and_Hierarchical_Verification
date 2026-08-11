"""``edith doctor``: environment diagnostics with actionable remediation.

Each check returns a :class:`CheckResult`. Checks never raise -- an exploding diagnostic is
worse than the problem it was meant to find -- and every failure carries a concrete next
step rather than only a symptom.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from edith.config.schema import EdithConfig
from edith.models.base import ModelProvider
from edith.models.registry import available_providers, build_provider
from edith.observability.logging import get_logger
from edith.system.resources import ResourceSnapshot, fits_in_vram, snapshot

logger = get_logger(__name__)

_TOOL_TIMEOUT_S = 10.0
MIN_PYTHON = (3, 11)


class CheckStatus(StrEnum):
    """Outcome of a single diagnostic check."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """The result of one diagnostic check."""

    name: str
    status: CheckStatus
    detail: str
    remediation: str | None = None

    @property
    def ok(self) -> bool:
        """True when the check did not fail (warnings are tolerated)."""
        return self.status is not CheckStatus.FAIL


@dataclass
class DoctorReport:
    """The full diagnostic report."""

    results: list[CheckResult] = field(default_factory=list)
    resources: ResourceSnapshot | None = None

    @property
    def failed(self) -> list[CheckResult]:
        """Checks that failed."""
        return [r for r in self.results if r.status is CheckStatus.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        """Checks that warned."""
        return [r for r in self.results if r.status is CheckStatus.WARN]

    @property
    def healthy(self) -> bool:
        """True when no check failed."""
        return not self.failed

    def exit_code(self) -> int:
        """Process exit code: 0 when healthy, 1 otherwise."""
        return 0 if self.healthy else 1


def _version_of(executable: str, *args: str) -> str | None:
    """Return the first line of ``executable``'s version output, or ``None``."""
    resolved = shutil.which(executable)
    if resolved is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, resolved absolute path
            [resolved, *args],
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if output else ""


def check_python(snap: ResourceSnapshot) -> CheckResult:
    """Verify the interpreter meets the minimum supported version."""
    parts = tuple(int(p) for p in snap.python_version.split(".")[:2])
    if parts >= MIN_PYTHON:
        return CheckResult("python", CheckStatus.OK, f"Python {snap.python_version}")
    return CheckResult(
        "python",
        CheckStatus.FAIL,
        f"Python {snap.python_version} is below the minimum "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and recreate the virtualenv.",
    )


def check_git() -> CheckResult:
    """Verify Git is installed. Git is core architecture, not an optional extra."""
    version = _version_of("git", "--version")
    if version:
        return CheckResult("git", CheckStatus.OK, version)
    return CheckResult(
        "git",
        CheckStatus.FAIL,
        "git not found on PATH",
        "Install Git (https://git-scm.com/download/win). Edith uses Git for recoverability.",
    )


def check_ram(config: EdithConfig, snap: ResourceSnapshot) -> CheckResult:
    """Verify enough free system RAM for inference plus the toolchain."""
    threshold = config.system.resources.min_free_ram_mb
    detail = f"{snap.ram_available_mb} MB free of {snap.ram_total_mb} MB total"
    if snap.ram_available_mb >= threshold:
        return CheckResult("ram", CheckStatus.OK, detail)
    return CheckResult(
        "ram",
        CheckStatus.WARN,
        f"{detail} (below the {threshold} MB threshold)",
        "Close memory-heavy applications before running inference; "
        "Windows will otherwise page the model to disk.",
    )


def check_disk(config: EdithConfig, snap: ResourceSnapshot) -> CheckResult:
    """Verify enough free disk for model weights and project state."""
    threshold = config.system.resources.min_free_disk_mb
    detail = f"{snap.disk_free_mb} MB free of {snap.disk_total_mb} MB total"
    if snap.disk_free_mb >= threshold:
        return CheckResult("disk", CheckStatus.OK, detail)
    return CheckResult(
        "disk",
        CheckStatus.WARN,
        f"{detail} (below the {threshold} MB threshold)",
        "Free disk space; model weights and Edith state both live on this volume.",
    )


def check_gpu(config: EdithConfig, snap: ResourceSnapshot) -> CheckResult:
    """Report GPU presence and free VRAM.

    A missing GPU is a warning, not a failure: Ollama falls back to CPU, slowly but
    correctly.
    """
    if not snap.has_gpu:
        return CheckResult(
            "gpu",
            CheckStatus.WARN,
            snap.gpu_probe_error or "no NVIDIA GPU detected",
            "Inference will run on CPU and will be substantially slower.",
        )
    gpu = snap.gpus[0]
    detail = f"{gpu.name}: {gpu.free_mb} MB free of {gpu.total_mb} MB (driver {gpu.driver_version})"
    threshold = config.system.resources.min_free_vram_mb
    if gpu.free_mb >= threshold:
        return CheckResult("gpu", CheckStatus.OK, detail)
    return CheckResult(
        "gpu",
        CheckStatus.WARN,
        f"{detail} - below the {threshold} MB threshold",
        "Close GPU-using applications, or select a smaller model profile.",
    )


def check_model_fit(
    config: EdithConfig, snap: ResourceSnapshot, profile: str | None
) -> CheckResult:
    """Verify the selected model profile's estimated footprint fits in free VRAM."""
    try:
        params = config.models.profile(profile)
    except KeyError as exc:
        return CheckResult(
            "model_fit",
            CheckStatus.FAIL,
            str(exc),
            "Correct `default_profile` or the --profile argument in config/models.yaml.",
        )

    name = params.model_name
    if params.estimated_vram_mb <= 0:
        return CheckResult(
            "model_fit",
            CheckStatus.WARN,
            f"{name}: no estimated_vram_mb recorded",
            "Set `estimated_vram_mb` for this profile so fit can be verified before load.",
        )
    free = snap.free_vram_mb
    if free is None:
        return CheckResult(
            "model_fit",
            CheckStatus.WARN,
            f"{name}: needs ~{params.estimated_vram_mb} MB VRAM; no GPU detected",
            "The model will run on CPU using system RAM instead.",
        )
    if fits_in_vram(params.estimated_vram_mb, snap):
        return CheckResult(
            "model_fit",
            CheckStatus.OK,
            f"{name}: ~{params.estimated_vram_mb} MB estimated, {free} MB free VRAM",
        )
    return CheckResult(
        "model_fit",
        CheckStatus.WARN,
        f"{name}: ~{params.estimated_vram_mb} MB estimated exceeds {free} MB free VRAM",
        "Select a smaller profile (e.g. `fast`) or reduce `context_length`. "
        "Running anyway will spill layers to system RAM and be very slow.",
    )


def check_provider_config(config: EdithConfig) -> CheckResult:
    """Verify the configured provider key is registered."""
    provider = config.models.provider
    if provider.strip().lower() in available_providers():
        return CheckResult("provider_config", CheckStatus.OK, f"provider {provider!r} registered")
    return CheckResult(
        "provider_config",
        CheckStatus.FAIL,
        f"provider {provider!r} is not registered; available: {list(available_providers())}",
        "Correct `provider` in config/models.yaml.",
    )


def check_provider_health(config: EdithConfig, profile: str | None) -> CheckResult:
    """Verify the model runtime is reachable and the configured model is pulled."""
    provider: ModelProvider | None = None
    try:
        provider = build_provider(config, profile)
        health = provider.health_check()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never crash
        return CheckResult(
            "model_provider",
            CheckStatus.FAIL,
            f"{type(exc).__name__}: {exc}",
            "Check config/models.yaml and that the model runtime is installed.",
        )
    finally:
        if provider is not None:
            provider.close()

    status = {
        "HEALTHY": CheckStatus.OK,
        "DEGRADED": CheckStatus.FAIL,
        "UNAVAILABLE": CheckStatus.FAIL,
    }[str(health.state)]
    return CheckResult("model_provider", status, health.detail, health.remediation)


def check_config_dir(config: EdithConfig) -> CheckResult:
    """Verify the configuration directory that was actually loaded."""
    if config.config_dir is None:
        return CheckResult(
            "config",
            CheckStatus.WARN,
            "configuration was built in memory, not loaded from disk",
            "Run from the repository root or set EDITH_CONFIG_DIR.",
        )
    return CheckResult("config", CheckStatus.OK, f"loaded from {config.config_dir}")


def check_state_dir(config: EdithConfig, base_dir: Path | None = None) -> CheckResult:
    """Verify the state directory is creatable and writable.

    Project state must survive restart (CLAUDE.md invariant 10); an unwritable state
    directory makes that impossible and is worth catching before a long run, not during it.
    """
    path = config.system.state_dir
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult(
            "state_dir",
            CheckStatus.FAIL,
            f"{path} is not writable: {exc}",
            "Choose a writable `state_dir` in config/system.yaml.",
        )
    return CheckResult("state_dir", CheckStatus.OK, f"{path} is writable")


def check_test_framework() -> CheckResult:
    """Verify pytest is importable. Edith must be able to test itself."""
    try:
        import pytest  # noqa: PLC0415 - intentionally probed at call time
    except ImportError:
        return CheckResult(
            "test_framework",
            CheckStatus.WARN,
            "pytest is not installed",
            "Install dev dependencies: pip install -e .[dev]",
        )
    return CheckResult("test_framework", CheckStatus.OK, f"pytest {pytest.__version__}")


def run_doctor(
    config: EdithConfig,
    *,
    profile: str | None = None,
    include_provider: bool = True,
    base_dir: Path | None = None,
    snapshot_fn: Callable[[], ResourceSnapshot] = snapshot,
) -> DoctorReport:
    """Run every diagnostic check and return a consolidated report.

    Args:
        config: Resolved configuration.
        profile: Model profile to diagnose; ``None`` uses the default.
        include_provider: Skip the live provider probe when ``False`` (offline runs).
        base_dir: Root for resolving relative paths.
        snapshot_fn: Resource probe, injected for tests.
    """
    snap = snapshot_fn()
    results = [
        check_python(snap),
        check_git(),
        check_config_dir(config),
        check_state_dir(config, base_dir),
        check_ram(config, snap),
        check_disk(config, snap),
        check_gpu(config, snap),
        check_provider_config(config),
        check_model_fit(config, snap, profile),
        check_test_framework(),
    ]
    if include_provider:
        results.append(check_provider_health(config, profile))
    return DoctorReport(results=results, resources=snap)
