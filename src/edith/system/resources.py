"""Hardware resource probing.

The target machine has 6 GB VRAM and frequently under 5 GB free RAM. Edith must know that
before it loads a model, not after Windows starts swapping. Every probe degrades to
``None`` rather than raising -- an unavailable reading is a diagnostic, not a crash.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import psutil

from edith.observability.logging import get_logger

logger = get_logger(__name__)

#: nvidia-smi must never hang the CLI.
_NVIDIA_SMI_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GPUInfo:
    """A single detected NVIDIA GPU."""

    name: str
    total_mb: int
    free_mb: int
    driver_version: str = ""


@dataclass(frozen=True)
class ResourceSnapshot:
    """A point-in-time view of machine resources."""

    platform: str
    python_version: str
    cpu_logical: int
    cpu_physical: int | None
    ram_total_mb: int
    ram_available_mb: int
    disk_total_mb: int
    disk_free_mb: int
    gpus: tuple[GPUInfo, ...] = ()
    gpu_probe_error: str | None = None

    @property
    def has_gpu(self) -> bool:
        """True when at least one NVIDIA GPU was detected."""
        return bool(self.gpus)

    @property
    def free_vram_mb(self) -> int | None:
        """Free VRAM on the largest-free GPU, or ``None`` when no GPU was detected."""
        if not self.gpus:
            return None
        return max(gpu.free_mb for gpu in self.gpus)


def probe_gpus() -> tuple[tuple[GPUInfo, ...], str | None]:
    """Query NVIDIA GPUs via ``nvidia-smi``.

    Returns:
        ``(gpus, error)``. ``error`` is a human-readable reason when probing failed; both
        are empty/None on a machine that simply has no NVIDIA GPU.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return (), "nvidia-smi not found on PATH (no NVIDIA GPU, or driver not installed)"
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, resolved absolute path
            [
                executable,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (), f"nvidia-smi failed: {type(exc).__name__}"

    if completed.returncode != 0:
        return (), f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()[:200]}"

    gpus: list[GPUInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                GPUInfo(
                    name=parts[0],
                    total_mb=int(float(parts[1])),
                    free_mb=int(float(parts[2])),
                    driver_version=parts[3] if len(parts) > 3 else "",
                )
            )
        except ValueError:
            logger.debug("gpu.parse_failed", line=line)
    return tuple(gpus), None


def snapshot(disk_path: Path | None = None) -> ResourceSnapshot:
    """Capture current machine resources.

    Args:
        disk_path: Path whose filesystem is measured. Defaults to the current directory.
    """
    memory = psutil.virtual_memory()
    usage = psutil.disk_usage(str(disk_path or Path.cwd()))
    gpus, gpu_error = probe_gpus()
    return ResourceSnapshot(
        platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        cpu_physical=psutil.cpu_count(logical=False),
        ram_total_mb=memory.total // (1024 * 1024),
        ram_available_mb=memory.available // (1024 * 1024),
        disk_total_mb=usage.total // (1024 * 1024),
        disk_free_mb=usage.free // (1024 * 1024),
        gpus=gpus,
        gpu_probe_error=gpu_error,
    )


def fits_in_vram(estimated_vram_mb: int, snap: ResourceSnapshot, *, headroom_mb: int = 512) -> bool:
    """Whether a model's estimated VRAM footprint fits in currently free VRAM.

    Args:
        estimated_vram_mb: Weights + KV cache + compute buffers + CUDA context.
        snap: Current resource reading.
        headroom_mb: Reserve left for the desktop compositor and fragmentation.

    Returns:
        ``True`` when it fits, or when the estimate is unknown (0) / no GPU was detected --
        the caller should treat those as "cannot determine", not "will not fit".
    """
    free = snap.free_vram_mb
    if free is None or estimated_vram_mb <= 0:
        return True
    return estimated_vram_mb + headroom_mb <= free


class ModelFit(StrEnum):
    """Where a model's weights would have to live on this machine.

    ``fits_in_vram`` answers one bit -- does it fit in VRAM -- which is the wrong shape for
    the advice a caller has to give. A 7B that misses VRAM by a few hundred MB spills into
    system RAM and runs slowly; a 27B exceeds VRAM *and* RAM together and does not load at
    all. Telling someone the second case will be "very slow" sends them off to wait for a
    run that was never going to start.
    """

    FITS_VRAM = "fits_vram"
    SPILLS_TO_RAM = "spills_to_ram"
    EXCEEDS_MACHINE = "exceeds_machine"
    UNKNOWN = "unknown"


def classify_fit(
    estimated_vram_mb: int, snap: ResourceSnapshot, *, headroom_mb: int = 512
) -> ModelFit:
    """Classify where a model of this size would have to run.

    The ceiling for :attr:`ModelFit.EXCEEDS_MACHINE` is free VRAM plus *available* RAM: what
    the machine can actually offer right now. Total RAM would be the wrong number -- on a 16 GB
    Windows box the OS and open applications routinely hold 10 GB, so a model measured against
    the total looks loadable and then thrashes the pagefile instead of starting.

    This makes the answer a reading rather than a property, and it can change between calls as
    memory frees up. That is the same basis :func:`check_ram` and :func:`check_disk` already
    report on, and the honest claim here is "cannot load now", not "cannot ever load".

    Args:
        estimated_vram_mb: Weights + KV cache + compute buffers + CUDA context.
        snap: Current resource reading.
        headroom_mb: Reserve left for the desktop compositor and fragmentation.

    Returns:
        :attr:`ModelFit.UNKNOWN` when the estimate is missing or no GPU was detected -- the
        caller must treat that as "cannot determine", never as a refusal.
    """
    free_vram = snap.free_vram_mb
    if estimated_vram_mb <= 0 or free_vram is None:
        return ModelFit.UNKNOWN
    if estimated_vram_mb + headroom_mb <= free_vram:
        return ModelFit.FITS_VRAM
    if estimated_vram_mb > free_vram + snap.ram_available_mb:
        return ModelFit.EXCEEDS_MACHINE
    return ModelFit.SPILLS_TO_RAM
