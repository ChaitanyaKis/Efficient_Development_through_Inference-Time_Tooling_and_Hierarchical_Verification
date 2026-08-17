"""Benchmark harness behaviour, plus the live end-to-end runs behind a marker.

The harness tests are hermetic and always run: they are what guarantees a benchmark cannot
report success for the wrong reason. The live runs call the real model and are slow, so
they sit behind ``-m benchmark``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from edith.benchmarks import (
    BENCHMARKS,
    Benchmark,
    check_protected_files,
    fixtures_root,
    get_benchmark,
    prepare_workspace,
    run_benchmark,
    run_verification,
)
from edith.config.loader import load_config

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


class TestSuiteDefinition:
    def test_every_benchmark_has_a_fixture(self) -> None:
        for benchmark in BENCHMARKS:
            assert (fixtures_root() / benchmark.fixture).is_dir(), benchmark.benchmark_id

    def test_ids_are_unique(self) -> None:
        ids = [benchmark.benchmark_id for benchmark in BENCHMARKS]
        assert len(set(ids)) == len(ids)

    def test_lookup(self) -> None:
        assert get_benchmark("feature").fixture == "calculator"
        with pytest.raises(KeyError, match="unknown benchmark"):
            get_benchmark("nope")

    def test_every_benchmark_protects_its_tests(self) -> None:
        """A benchmark whose tests can be edited measures nothing."""
        for benchmark in BENCHMARKS:
            assert benchmark.protected_files, benchmark.benchmark_id

    def test_fixtures_carry_no_instructions_to_the_agent(self) -> None:
        """Regression: a "do not fix this, it is deliberate" note in a fixture was read and
        obeyed by the model, which then declined to repair the bug for three attempts."""
        for benchmark in BENCHMARKS:
            for path in (fixtures_root() / benchmark.fixture).rglob("*.py"):
                text = path.read_text(encoding="utf-8").lower()
                assert "do not fix" not in text, path
                assert "deliberately seeded" not in text, path


class TestHarnessIntegrity:
    def test_prepare_creates_a_git_repository(self, tmp_path: Path) -> None:
        workspace = prepare_workspace(get_benchmark("feature"), tmp_path / "ws")
        assert (workspace / ".git").is_dir()
        assert (workspace / "calculator.py").is_file()

    def test_prepare_is_repeatable(self, tmp_path: Path) -> None:
        """Each run must start from an identical state, including after a previous run."""
        benchmark = get_benchmark("feature")
        first = prepare_workspace(benchmark, tmp_path / "ws")
        (first / "calculator.py").write_text("garbage\n", encoding="utf-8")
        second = prepare_workspace(benchmark, tmp_path / "ws")
        assert "def add" in (second / "calculator.py").read_text(encoding="utf-8")

    def test_fixture_starts_failing(self, tmp_path: Path) -> None:
        """Every benchmark must begin red, or it proves nothing."""
        for benchmark in BENCHMARKS:
            workspace = prepare_workspace(benchmark, tmp_path / benchmark.benchmark_id)
            assert not run_verification(workspace, benchmark.verify_argv), (
                f"{benchmark.benchmark_id} passes before Edith runs"
            )

    def test_protected_file_tampering_is_detected(self, tmp_path: Path) -> None:
        """The check that catches "make the tests pass by editing the tests"."""
        benchmark = get_benchmark("feature")
        workspace = prepare_workspace(benchmark, tmp_path / "ws")
        assert check_protected_files(benchmark, workspace)[0]

        (workspace / "test_calculator.py").write_text("# gone\n", encoding="utf-8")
        intact, tampered = check_protected_files(benchmark, workspace)
        assert not intact
        assert "test_calculator.py (modified)" in tampered

    def test_protected_file_deletion_is_detected(self, tmp_path: Path) -> None:
        benchmark = get_benchmark("feature")
        workspace = prepare_workspace(benchmark, tmp_path / "ws")
        (workspace / "test_calculator.py").unlink()
        intact, tampered = check_protected_files(benchmark, workspace)
        assert not intact and "deleted" in tampered[0]

    def test_a_broken_fixture_is_reported(self, tmp_path: Path) -> None:
        """A fixture that already passes would make its benchmark meaningless."""
        broken = Benchmark(
            benchmark_id="already_green",
            fixture="calculator",
            request="do nothing",
            description="a fixture that already passes",
            verify_argv=("python", "-c", "pass"),
        )
        config = load_config()
        result = run_benchmark(broken, config, tmp_path / "root")
        assert not result.passed
        assert "already passes" in result.reason


@pytest.mark.benchmark
class TestLiveBenchmarks:
    """End-to-end runs against the real local model. Slow; run with `-m benchmark`."""

    @pytest.mark.parametrize("benchmark_id", ["feature", "repair"])
    def test_benchmark_passes(self, benchmark_id: str, tmp_path: Path) -> None:
        result = run_benchmark(get_benchmark(benchmark_id), load_config(), tmp_path / "bench")
        assert result.baseline_failed, "the fixture must start red"
        assert result.passed, result.reason
        assert result.final_verification_passed
        assert result.protected_files_intact
