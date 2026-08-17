"""Resource probing, VRAM fit checks, and the doctor's diagnostics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from edith.config.schema import EdithConfig, ModelParams, ModelsConfig
from edith.diagnostics.doctor import (
    CheckStatus,
    DoctorReport,
    check_disk,
    check_gpu,
    check_model_fit,
    check_provider_config,
    check_provider_health,
    check_python,
    check_ram,
    check_state_dir,
    run_doctor,
)
from edith.models.registry import register_provider
from edith.system.resources import GPUInfo, ResourceSnapshot, fits_in_vram, probe_gpus, snapshot

from .fakes import FakeProvider


def make_snapshot(**overrides: object) -> ResourceSnapshot:
    """Build a snapshot resembling the target machine, with selective overrides."""
    defaults: dict = {
        "platform": "Windows 11",
        "python_version": "3.13.14",
        "cpu_logical": 12,
        "cpu_physical": 6,
        "ram_total_mb": 16144,
        "ram_available_mb": 4700,
        "disk_total_mb": 487000,
        "disk_free_mb": 197000,
        "gpus": (GPUInfo(name="NVIDIA GeForce RTX 2060", total_mb=6144, free_mb=5955),),
        "gpu_probe_error": None,
    }
    return ResourceSnapshot(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestSnapshot:
    def test_real_probe_returns_sane_values(self) -> None:
        """Runs against the actual machine - the numbers must be plausible, not fabricated."""
        snap = snapshot()
        assert snap.ram_total_mb > 0
        assert 0 <= snap.ram_available_mb <= snap.ram_total_mb
        assert snap.disk_free_mb >= 0
        assert snap.cpu_logical >= 1
        assert snap.python_version.count(".") >= 1

    def test_disk_path_is_honoured(self, tmp_path: Path) -> None:
        assert snapshot(tmp_path).disk_total_mb > 0

    def test_free_vram_picks_the_best_gpu(self) -> None:
        snap = make_snapshot(
            gpus=(
                GPUInfo(name="a", total_mb=6144, free_mb=1000),
                GPUInfo(name="b", total_mb=6144, free_mb=5000),
            )
        )
        assert snap.free_vram_mb == 5000

    def test_no_gpu_reports_none(self) -> None:
        snap = make_snapshot(gpus=())
        assert snap.free_vram_mb is None and not snap.has_gpu


class TestProbeGpus:
    def test_missing_binary_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("edith.system.resources.shutil.which", lambda _: None)
        gpus, error = probe_gpus()
        assert gpus == () and error is not None and "nvidia-smi not found" in error

    def test_parses_csv_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("edith.system.resources.shutil.which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            "edith.system.resources.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(
                a, 0, stdout="NVIDIA GeForce RTX 2060, 6144, 5955, 610.47\n", stderr=""
            ),
        )
        gpus, error = probe_gpus()
        assert error is None and len(gpus) == 1
        assert gpus[0].total_mb == 6144 and gpus[0].free_mb == 5955

    def test_nonzero_exit_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("edith.system.resources.shutil.which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            "edith.system.resources.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a, 9, stdout="", stderr="driver error"),
        )
        gpus, error = probe_gpus()
        assert gpus == () and error is not None and "exited 9" in error

    def test_timeout_does_not_hang_the_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        monkeypatch.setattr("edith.system.resources.shutil.which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr("edith.system.resources.subprocess.run", explode)
        gpus, error = probe_gpus()
        assert gpus == () and error is not None

    def test_malformed_line_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("edith.system.resources.shutil.which", lambda _: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            "edith.system.resources.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(
                a, 0, stdout="garbage\nGPU, 6144, 5955, 1.0\n", stderr=""
            ),
        )
        gpus, _ = probe_gpus()
        assert len(gpus) == 1


class TestFitsInVram:
    def test_the_chosen_3b_profile_fits_the_rtx_2060(self) -> None:
        """The documented model choice must hold against the real hardware numbers."""
        assert fits_in_vram(2816, make_snapshot())

    def test_the_7b_profile_does_not_fit(self) -> None:
        """This is why the 7B profile is opt-in only."""
        assert not fits_in_vram(5800, make_snapshot())

    def test_headroom_is_reserved(self) -> None:
        snap = make_snapshot(gpus=(GPUInfo(name="g", total_mb=6144, free_mb=3000),))
        assert not fits_in_vram(2800, snap, headroom_mb=512)
        assert fits_in_vram(2400, snap, headroom_mb=512)

    def test_unknown_estimate_is_not_a_refusal(self) -> None:
        assert fits_in_vram(0, make_snapshot())

    def test_no_gpu_is_not_a_refusal(self) -> None:
        assert fits_in_vram(5800, make_snapshot(gpus=()))


class TestIndividualChecks:
    def test_python_ok(self) -> None:
        assert check_python(make_snapshot()).status is CheckStatus.OK

    def test_old_python_fails(self) -> None:
        result = check_python(make_snapshot(python_version="3.9.1"))
        assert result.status is CheckStatus.FAIL and result.remediation

    def test_low_ram_warns_with_advice(self, config: EdithConfig) -> None:
        result = check_ram(config, make_snapshot(ram_available_mb=500))
        assert result.status is CheckStatus.WARN and result.remediation

    def test_sufficient_ram_ok(self, config: EdithConfig) -> None:
        assert check_ram(config, make_snapshot()).status is CheckStatus.OK

    def test_low_disk_warns(self, config: EdithConfig) -> None:
        assert check_disk(config, make_snapshot(disk_free_mb=100)).status is CheckStatus.WARN

    def test_gpu_ok(self, config: EdithConfig) -> None:
        assert check_gpu(config, make_snapshot()).status is CheckStatus.OK

    def test_absent_gpu_warns_but_does_not_fail(self, config: EdithConfig) -> None:
        """CPU inference is slow but correct; that is a warning, not a failure."""
        result = check_gpu(config, make_snapshot(gpus=(), gpu_probe_error="no GPU"))
        assert result.status is CheckStatus.WARN

    def test_busy_gpu_warns(self, config: EdithConfig) -> None:
        snap = make_snapshot(gpus=(GPUInfo(name="g", total_mb=6144, free_mb=100),))
        assert check_gpu(config, snap).status is CheckStatus.WARN

    def test_model_fit_ok(self, config: EdithConfig) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(
                profiles={"default": ModelParams(model_name="m", estimated_vram_mb=2000)}
            )
        )
        assert check_model_fit(cfg, make_snapshot(), None).status is CheckStatus.OK

    def test_model_too_large_warns_with_advice(self) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(
                profiles={"default": ModelParams(model_name="big", estimated_vram_mb=5800)}
            )
        )
        result = check_model_fit(cfg, make_snapshot(), None)
        assert result.status is CheckStatus.WARN
        assert result.remediation is not None and "smaller profile" in result.remediation

    def test_unknown_profile_fails(self, config: EdithConfig) -> None:
        assert check_model_fit(config, make_snapshot(), "ghost").status is CheckStatus.FAIL

    def test_missing_estimate_warns(self) -> None:
        cfg = EdithConfig(models=ModelsConfig(profiles={"default": ModelParams(model_name="m")}))
        assert check_model_fit(cfg, make_snapshot(), None).status is CheckStatus.WARN

    def test_provider_config_ok(self, config: EdithConfig) -> None:
        assert check_provider_config(config).status is CheckStatus.OK

    def test_unknown_provider_fails(self, model_params: ModelParams) -> None:
        cfg = EdithConfig(
            models=ModelsConfig(provider="mystery", profiles={"default": model_params})
        )
        assert check_provider_config(cfg).status is CheckStatus.FAIL

    def test_state_dir_writable(self, tmp_path: Path, config: EdithConfig) -> None:
        result = check_state_dir(config, tmp_path)
        assert result.status is CheckStatus.OK
        assert not (tmp_path / ".edith" / ".write_probe").exists()  # probe cleaned up

    def test_unwritable_state_dir_fails(self, tmp_path: Path, config: EdithConfig) -> None:
        """State must survive restart; an unwritable dir makes that impossible."""
        blocker = tmp_path / ".edith"
        blocker.write_text("I am a file, not a directory", encoding="utf-8")
        assert check_state_dir(config, tmp_path).status is CheckStatus.FAIL


class TestProviderHealthCheck:
    def test_healthy_provider(self, model_params: ModelParams) -> None:
        register_provider(
            "fake_healthy", lambda cfg, params: FakeProvider(params), replace=True
        )
        cfg = EdithConfig(
            models=ModelsConfig(provider="fake_healthy", profiles={"default": model_params})
        )
        assert check_provider_health(cfg, None).status is CheckStatus.OK

    def test_unhealthy_provider_fails(self, model_params: ModelParams) -> None:
        from edith.schemas.model import HealthState, ProviderHealth

        register_provider(
            "fake_down",
            lambda cfg, params: FakeProvider(
                params,
                health=ProviderHealth(
                    provider="fake",
                    state=HealthState.UNAVAILABLE,
                    detail="not running",
                    remediation="start it",
                ),
            ),
            replace=True,
        )
        cfg = EdithConfig(
            models=ModelsConfig(provider="fake_down", profiles={"default": model_params})
        )
        result = check_provider_health(cfg, None)
        assert result.status is CheckStatus.FAIL and result.remediation == "start it"

    def test_construction_failure_is_reported_not_raised(
        self, model_params: ModelParams
    ) -> None:
        def explode(cfg: EdithConfig, params: ModelParams) -> FakeProvider:
            raise RuntimeError("cannot build")

        register_provider("fake_broken", explode, replace=True)
        cfg = EdithConfig(
            models=ModelsConfig(provider="fake_broken", profiles={"default": model_params})
        )
        assert check_provider_health(cfg, None).status is CheckStatus.FAIL


class TestRunDoctor:
    def test_offline_run_skips_the_provider_probe(self, config: EdithConfig) -> None:
        report = run_doctor(
            config, include_provider=False, snapshot_fn=lambda: make_snapshot()
        )
        assert "model_provider" not in {r.name for r in report.results}

    def test_all_expected_checks_present(self, config: EdithConfig, tmp_path: Path) -> None:
        report = run_doctor(
            config,
            include_provider=False,
            base_dir=tmp_path,
            snapshot_fn=lambda: make_snapshot(),
        )
        expected = {
            "python", "git", "config", "state_dir", "ram", "disk",
            "gpu", "provider_config", "model_fit", "test_framework",
        }
        assert expected <= {r.name for r in report.results}

    def test_warnings_do_not_make_the_report_unhealthy(
        self, config: EdithConfig, tmp_path: Path
    ) -> None:
        report = run_doctor(
            config,
            include_provider=False,
            base_dir=tmp_path,
            snapshot_fn=lambda: make_snapshot(ram_available_mb=100, gpus=()),
        )
        assert report.warnings and report.healthy and report.exit_code() == 0

    def test_failure_sets_a_nonzero_exit_code(
        self, config: EdithConfig, tmp_path: Path
    ) -> None:
        report = run_doctor(
            config,
            include_provider=False,
            base_dir=tmp_path,
            snapshot_fn=lambda: make_snapshot(python_version="3.8.0"),
        )
        assert not report.healthy and report.exit_code() == 1

    def test_every_failure_carries_remediation(
        self, config: EdithConfig, tmp_path: Path
    ) -> None:
        """A diagnostic that only says 'broken' is not actionable."""
        report = run_doctor(
            config,
            include_provider=False,
            base_dir=tmp_path,
            snapshot_fn=lambda: make_snapshot(python_version="3.8.0", ram_available_mb=10),
        )
        for result in report.failed + report.warnings:
            assert result.remediation, f"{result.name} has no remediation"

    def test_empty_report_is_healthy(self) -> None:
        assert DoctorReport().healthy
