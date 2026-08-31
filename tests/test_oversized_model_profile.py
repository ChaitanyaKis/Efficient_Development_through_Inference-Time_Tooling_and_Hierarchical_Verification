"""A model too large for the machine is refused, not offered and failed later.

Adding a 27B profile exposed two places that only ever asked "does it fit in VRAM". That
question has three answers, not two, and collapsing them misleads in opposite directions:

**The doctor called it a warning.** A 7B that misses VRAM by a few hundred MB spills into
system RAM and runs slowly, which is a real tradeoff worth a WARN. A 27B exceeds VRAM and
free RAM *together* and does not start at all -- reporting that as "will be very slow" sends
someone off to wait for a run that was never going to begin.

**The UI called it selectable.** Availability was "is it pulled", so a model present on disk
but too large to load would appear in the picker, be chosen, and fail minutes into a run for
a reason the screen already had in hand.

The profile is kept in the list rather than omitted, for the reason ``large`` is: a tradeoff
that is written down and refused by a check stays visible, where one that is left out gets
rediscovered by whoever tries it next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from edith.config.loader import load_config
from edith.config.schema import EdithConfig, ModelParams, ModelsConfig
from edith.diagnostics.doctor import CheckStatus, check_model_fit
from edith.system.resources import GPUInfo, ModelFit, ResourceSnapshot, classify_fit

#: The target machine: 6 GB card, 16 GB RAM, and most of the RAM already spoken for.
TARGET = ResourceSnapshot(
    platform="Windows 11",
    python_version="3.13.14",
    cpu_logical=12,
    cpu_physical=6,
    ram_total_mb=16144,
    ram_available_mb=5600,
    disk_total_mb=487000,
    disk_free_mb=197000,
    gpus=(GPUInfo(name="NVIDIA GeForce RTX 2060", total_mb=6144, free_mb=5955),),
)


class TestFitHasThreeAnswers:
    def test_a_3b_fits_in_vram(self) -> None:
        assert classify_fit(2816, TARGET) is ModelFit.FITS_VRAM

    def test_a_7b_spills_into_system_ram(self) -> None:
        """Slow, but it starts -- the case the existing WARN is right about."""
        assert classify_fit(5800, TARGET) is ModelFit.SPILLS_TO_RAM

    def test_a_27b_exceeds_the_machine(self) -> None:
        """5955 MB VRAM + 5600 MB free RAM cannot hold 17700 MB at any speed."""
        assert classify_fit(17700, TARGET) is ModelFit.EXCEEDS_MACHINE

    def test_the_ceiling_is_free_ram_not_total_ram(self) -> None:
        """Judging against total RAM would call a 17.7 GB model loadable on a 16 GB box."""
        assert TARGET.free_vram_mb + TARGET.ram_total_mb > 17700
        assert classify_fit(17700, TARGET) is ModelFit.EXCEEDS_MACHINE

    def test_freeing_ram_can_change_the_answer(self) -> None:
        """It is a reading, not a property, and must not claim to be permanent."""
        roomy = ResourceSnapshot(**{**TARGET.__dict__, "ram_available_mb": 14000})
        assert classify_fit(17700, roomy) is ModelFit.SPILLS_TO_RAM

    def test_an_unrecorded_estimate_is_unknown_not_refused(self) -> None:
        assert classify_fit(0, TARGET) is ModelFit.UNKNOWN

    def test_no_gpu_is_unknown_not_refused(self) -> None:
        headless = ResourceSnapshot(**{**TARGET.__dict__, "gpus": ()})
        assert classify_fit(17700, headless) is ModelFit.UNKNOWN


def config_with(estimated_vram_mb: int) -> EdithConfig:
    params = ModelParams(model_name="oversized:27b", estimated_vram_mb=estimated_vram_mb)
    return EdithConfig(
        models=ModelsConfig(default_profile="p", profiles={"p": params})
    )


class TestTheDoctorRefusesRatherThanWarns:
    def test_an_impossible_model_fails(self) -> None:
        result = check_model_fit(config_with(17700), TARGET, None)
        assert result.status is CheckStatus.FAIL

    def test_the_failure_shows_the_arithmetic(self) -> None:
        """A bare refusal invites a retry; the numbers say why retrying will not help."""
        result = check_model_fit(config_with(17700), TARGET, None)
        assert "17700" in result.detail
        assert "5955" in result.detail and "5600" in result.detail

    def test_it_does_not_advise_waiting_for_a_slow_run(self) -> None:
        """The old advice for this case. It cannot load, so slowness is not the problem."""
        result = check_model_fit(config_with(17700), TARGET, None)
        assert "very slow" not in (result.remediation or "")
        assert "cannot load" in (result.remediation or "")

    def test_a_merely_oversized_model_still_only_warns(self) -> None:
        """The 7B tradeoff is real and must not be escalated into a refusal."""
        result = check_model_fit(config_with(5800), TARGET, None)
        assert result.status is CheckStatus.WARN

    def test_a_fitting_model_passes(self) -> None:
        assert check_model_fit(config_with(2816), TARGET, None).status is CheckStatus.OK


class TestTheProfileIsConfiguredHonestly:
    @pytest.fixture
    def profiles(self) -> dict[str, Any]:
        raw = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
        return dict(raw["profiles"])

    def test_the_gemma_profile_is_present(self, profiles: dict[str, Any]) -> None:
        assert "gemma" in profiles
        assert "27b" in profiles["gemma"]["model_name"]

    def test_it_records_a_footprint_so_fit_can_be_checked(
        self, profiles: dict[str, Any]
    ) -> None:
        """A profile with no estimate is un-checkable, and would load blind."""
        assert profiles["gemma"]["estimated_vram_mb"] > 0

    def test_it_is_not_the_default(self) -> None:
        """Defaulting to a model that cannot load would break every run on this machine."""
        config = load_config(None)
        assert config.models.default_profile != "gemma"
        assert config.models.profiles[config.models.default_profile].estimated_vram_mb < 6000

    def test_every_profile_records_a_footprint(self, profiles: dict[str, Any]) -> None:
        for name, entry in profiles.items():
            assert entry.get("estimated_vram_mb", 0) > 0, f"{name} has no estimate"


class TestTheUiWillNotOfferAModelThatCannotLoad:
    """Being pulled is necessary but not sufficient: it must also fit."""

    def describe(self, monkeypatch: pytest.MonkeyPatch, *, installed: tuple[str, ...]) -> Any:
        from edith.ui import server

        class Stub:
            def list_models(self) -> tuple[str, ...]:
                return installed

            def close(self) -> None:
                return None

        monkeypatch.setattr("edith.models.registry.build_provider", lambda config: Stub())
        monkeypatch.setattr(server, "snapshot", lambda: TARGET)

        config = EdithConfig(
            models=ModelsConfig(
                default_profile="small",
                profiles={
                    "small": ModelParams(model_name="small:3b", estimated_vram_mb=2816),
                    "huge": ModelParams(model_name="huge:27b", estimated_vram_mb=17700),
                },
            )
        )
        return {entry["profile"]: entry for entry in server.describe_models(config)}

    def test_a_pulled_but_oversized_model_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the old "is it pulled" check got wrong."""
        entries = self.describe(monkeypatch, installed=("small:3b", "huge:27b"))
        assert entries["huge"]["available"] is False

    def test_the_reason_is_the_capacity_not_a_missing_download(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Telling someone to pull a model they already have wastes 17 GB and their time."""
        entries = self.describe(monkeypatch, installed=("small:3b", "huge:27b"))
        reason = entries["huge"]["reason"]
        assert "not pulled" not in reason
        assert "17700" in reason and "11555" in reason

    def test_a_model_that_fits_and_is_pulled_stays_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = self.describe(monkeypatch, installed=("small:3b", "huge:27b"))
        assert entries["small"]["available"] is True
        assert entries["small"]["reason"] == ""

    def test_not_pulled_is_reported_before_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pulling is the first thing to fix, so it is the first thing reported."""
        entries = self.describe(monkeypatch, installed=())
        assert entries["small"]["reason"] == "not pulled into the local runtime"

    def test_the_fit_is_reported_alongside_availability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = self.describe(monkeypatch, installed=("small:3b", "huge:27b"))
        assert entries["small"]["fit"] == ModelFit.FITS_VRAM.value
        assert entries["huge"]["fit"] == ModelFit.EXCEEDS_MACHINE.value
