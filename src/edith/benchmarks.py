"""The benchmark harness.

Benchmarks are how M2 is measured rather than asserted. Each one copies a fixture
repository into a scratch workspace, runs the real orchestrator against it with the real
local model, and then checks the *workspace itself* -- not any agent's report.

The independent check is the point. Three things must hold, and Edith's own opinion counts
for none of them:

1. the execution reached RELEASE with a PASS verdict,
2. the benchmark's verification command passes when re-run afterwards,
3. every protected file is byte-identical to the fixture.

Rule 3 catches the classic cheat: making the tests pass by editing the tests.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edith.config.schema import EdithConfig
from edith.errors import FailureCategory
from edith.observability.logging import get_logger
from edith.orchestrator import ExecutionResult, Orchestrator, create_execution
from edith.schemas.common import Verdict, new_id
from edith.state.store import StateStore, open_store
from edith.workspaces import ProjectWorkspace

logger = get_logger(__name__)

BENCHMARK_TIMEOUT_SECONDS = 900.0


def fixtures_root() -> Path:
    """Return the directory holding benchmark fixture repositories."""
    return Path(__file__).resolve().parents[2] / "benchmarks" / "fixtures"


@dataclass(frozen=True)
class Benchmark:
    """A deterministic benchmark scenario."""

    benchmark_id: str
    fixture: str
    request: str
    description: str
    #: Files that must be byte-identical after the run. Editing a test to make it pass is
    #: not a repair, and this is what detects that.
    protected_files: tuple[str, ...] = ()
    #: Command re-run independently after the execution to confirm the claimed result.
    verify_argv: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    #: Whether the fixture is expected to fail verification before Edith touches it.
    starts_failing: bool = True


@dataclass
class BenchmarkMetrics:
    """Measurements from one run.

    Recorded whether the run passed or failed. A benchmark suite that only reports
    successes cannot tell you whether the system is improving.
    """

    model_calls: int = 0
    agent_runs: int = 0
    repairs: int = 0
    tasks_total: int = 0
    tasks_succeeded: int = 0
    duration_seconds: float = 0.0
    #: The benchmark reached PASS but an independent check disagreed. The single most
    #: important metric in the suite: it counts the times Edith believed a lie.
    false_positive: bool = False
    #: Verification failed for an environment reason rather than a code defect.
    environment_failures: int = 0
    tool_failures: int = 0
    test_tampering_detected: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable form for reporting."""
        return {
            "model_calls": self.model_calls,
            "agent_runs": self.agent_runs,
            "repairs": self.repairs,
            "tasks_total": self.tasks_total,
            "tasks_succeeded": self.tasks_succeeded,
            "duration_seconds": round(self.duration_seconds, 1),
            "false_positive": self.false_positive,
            "environment_failures": self.environment_failures,
            "tool_failures": self.tool_failures,
            "test_tampering_detected": self.test_tampering_detected,
        }


@dataclass
class BenchmarkResult:
    """The outcome of one benchmark, independently checked."""

    benchmark_id: str
    passed: bool
    reason: str
    execution: ExecutionResult | None = None
    baseline_failed: bool | None = None
    final_verification_passed: bool = False
    protected_files_intact: bool = True
    tampered_files: list[str] = field(default_factory=list)
    workspace: str = ""
    duration_seconds: float = 0.0
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    def summary(self) -> str:
        """One-line human summary."""
        status = "PASS" if self.passed else "FAIL"
        marker = " [FALSE POSITIVE]" if self.metrics.false_positive else ""
        return f"[{status}]{marker} {self.benchmark_id}: {self.reason}"


#: The shipped benchmark suite.
BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        benchmark_id="feature",
        fixture="calculator",
        request=(
            "The calculator module is missing a multiply function. "
            "Add a multiply(a, b) function to calculator.py that returns the product "
            "of its two arguments, so that the existing test_multiply test passes."
        ),
        description="Implement a missing function so an existing failing test passes.",
        protected_files=("test_calculator.py",),
    ),
    Benchmark(
        benchmark_id="multi_repair",
        fixture="inventory",
        request=(
            "The inventory module has several bugs and its tests are failing. "
            "Fix inventory.py so that all the tests in test_inventory.py pass. "
            "Do not change the tests."
        ),
        description=(
            "Repair three independent defects. Cannot be satisfied by a single edit, so "
            "reaching PASS requires the detect/diagnose/repair loop to iterate."
        ),
        protected_files=("test_inventory.py",),
    ),
    Benchmark(
        benchmark_id="repair",
        fixture="calculator_bug",
        request=(
            "The subtract function in calculator.py is returning the wrong result and "
            "its test is failing. Find the bug and fix it so all tests pass."
        ),
        description="Detect, diagnose, and repair a seeded defect.",
        protected_files=("test_calculator.py",),
    ),
)


def get_benchmark(benchmark_id: str) -> Benchmark:
    """Return a benchmark by id."""
    for benchmark in BENCHMARKS:
        if benchmark.benchmark_id == benchmark_id:
            return benchmark
    known = ", ".join(item.benchmark_id for item in BENCHMARKS)
    raise KeyError(f"unknown benchmark {benchmark_id!r}; available: {known}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _force_remove(function: Any, path: str, _excinfo: Any) -> None:
    """``shutil.rmtree`` error handler that clears the read-only bit and retries.

    Git marks objects and pack files read-only. On Windows that makes ``os.unlink`` fail
    with ``Access is denied``, so a plain ``rmtree`` cannot delete a repository it created.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        logger.warning("benchmark.cleanup_failed", path=path)


def remove_tree(path: Path) -> None:
    """Delete a directory tree, including read-only git objects."""
    if path.exists():
        shutil.rmtree(path, onexc=_force_remove)


def _git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run git directly for harness setup.

    Deliberately not an agent path: the harness sets the stage before Edith runs, and must
    not depend on the machinery it is about to audit.
    """
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git is required to prepare a benchmark workspace")
    return subprocess.run(  # noqa: S603 - fixed argv, absolute resolved executable
        [executable, *arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )


def prepare_workspace(benchmark: Benchmark, destination: Path) -> Path:
    """Copy a fixture into ``destination`` and initialize a git repository there.

    Hooks are pointed at an empty directory for this repo only: a developer machine may have
    a global commit gate installed, and inheriting it would make benchmark results depend on
    unrelated tooling. The global configuration is never modified.
    """
    source = fixtures_root() / benchmark.fixture
    if not source.is_dir():
        raise FileNotFoundError(f"benchmark fixture not found: {source}")

    remove_tree(destination)
    shutil.copytree(source, destination)

    _git(destination, "init", "-b", "main")
    hooks = destination / ".git" / "empty-hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    _git(destination, "config", "core.hooksPath", str(hooks))
    _git(destination, "config", "user.email", "benchmark@localhost")
    _git(destination, "config", "user.name", "Edith Benchmark")
    _git(destination, "config", "commit.gpgsign", "false")
    _git(destination, "add", "-A")
    _git(destination, "commit", "-m", "benchmark fixture baseline")
    return destination


def run_verification(workspace: Path, argv: tuple[str, ...], timeout: float = 120.0) -> bool:
    """Re-run the benchmark's own check directly, outside Edith.

    Deliberately does not go through the tool gateway: this is the harness auditing Edith,
    so it must not depend on the machinery being audited.
    """
    # Matches the interpreter Edith itself runs under, for the same reason shell.run
    # does: the project's test runner lives in that environment, not on PATH.
    resolved = (
        sys.executable
        if argv[0] in {"python", "python3"} and sys.executable
        else shutil.which(argv[0])
    )
    if resolved is None:
        logger.warning("benchmark.verify_unavailable", executable=argv[0])
        return False
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, absolute resolved executable
            [resolved, *argv[1:]],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("benchmark.verify_failed", error=str(exc))
        return False
    return completed.returncode == 0


def check_protected_files(
    benchmark: Benchmark, workspace: Path
) -> tuple[bool, list[str]]:
    """Verify protected files were not modified."""
    source = fixtures_root() / benchmark.fixture
    tampered: list[str] = []
    for relative in benchmark.protected_files:
        original = source / relative
        current = workspace / relative
        if not current.is_file():
            tampered.append(f"{relative} (deleted)")
        elif _digest(original) != _digest(current):
            tampered.append(f"{relative} (modified)")
    return (not tampered, tampered)


def run_benchmark(
    benchmark: Benchmark,
    config: EdithConfig,
    workspace_root: Path,
    *,
    store: StateStore | None = None,
) -> BenchmarkResult:
    """Run one benchmark end to end and check it independently."""
    started = time.monotonic()
    workspace_path = prepare_workspace(benchmark, workspace_root / benchmark.benchmark_id)

    baseline_passed = run_verification(workspace_path, benchmark.verify_argv)
    if benchmark.starts_failing and baseline_passed:
        return BenchmarkResult(
            benchmark_id=benchmark.benchmark_id,
            passed=False,
            reason="fixture already passes its own verification; the benchmark is broken",
            baseline_failed=False,
            workspace=str(workspace_path),
            duration_seconds=time.monotonic() - started,
        )

    # Make the files that *define correctness* unwritable for this run, using the same M1
    # protected-path mechanism that guards .env and .git. Detecting tampering afterwards is
    # necessary but not sufficient: a run that rewrites the failing assertion to match the
    # bug has already wasted its budget and produced a worthless result. Observed for real
    # here -- the agent changed `assert subtract(5, 3) == 2` to `== 8` and declared victory.
    protected_config = config.model_copy(
        update={
            "tools": config.tools.model_copy(
                update={
                    "paths": config.tools.paths.model_copy(
                        update={
                            "protected_patterns": (
                                *config.tools.paths.protected_patterns,
                                *benchmark.protected_files,
                            )
                        }
                    )
                }
            )
        }
    )

    owns_store = store is None
    active_store = store or open_store(workspace_root / ".edith-benchmark-state")
    workspace = ProjectWorkspace(
        project_id=new_id("proj"),
        name=f"benchmark-{benchmark.benchmark_id}",
        root=workspace_path,
    )

    orchestrator = Orchestrator(protected_config, active_store, workspace)
    try:
        _, execution = create_execution(active_store, workspace, benchmark.request)
        result = orchestrator.run(execution)
    finally:
        orchestrator.close()

    final_passed = run_verification(workspace_path, benchmark.verify_argv)
    intact, tampered = check_protected_files(benchmark, workspace_path)

    if not intact:
        reason = f"protected files were modified: {', '.join(tampered)}"
        passed = False
    elif not final_passed:
        reason = "verification still fails in the resulting workspace"
        passed = False
    elif result.verdict is not Verdict.PASS:
        reason = f"verification passes but Edith reported {result.verdict}: {result.summary}"
        passed = False
    else:
        reason = (
            f"completed in {result.tasks_total} task(s) with "
            f"{result.repairs_attempted} repair(s)"
        )
        passed = True

    # A false positive is the case that matters most: Edith said PASS and an independent
    # check disagreed. It is counted separately from an ordinary failure, because the two
    # mean completely different things about how much the system can be trusted.
    metrics = BenchmarkMetrics(
        model_calls=result.model_calls,
        agent_runs=result.agent_runs,
        repairs=result.repairs_attempted,
        tasks_total=result.tasks_total,
        tasks_succeeded=result.tasks_succeeded,
        duration_seconds=time.monotonic() - started,
        false_positive=result.verdict is Verdict.PASS and (not final_passed or not intact),
        test_tampering_detected="integrity" in result.summary.lower(),
    )
    for record in active_store.verifications(result.execution_id):
        if not record.passed and record.exit_code == -1:
            metrics.environment_failures += 1
    for failure in active_store.failures(result.execution_id):
        if failure.category is FailureCategory.TOOL_ERROR:
            metrics.tool_failures += 1

    outcome = BenchmarkResult(
        benchmark_id=benchmark.benchmark_id,
        passed=passed,
        reason=reason,
        execution=result,
        baseline_failed=not baseline_passed,
        final_verification_passed=final_passed,
        protected_files_intact=intact,
        tampered_files=tampered,
        workspace=str(workspace_path),
        duration_seconds=time.monotonic() - started,
        metrics=metrics,
    )
    if owns_store:
        active_store.close()

    logger.info(
        "benchmark.finished",
        benchmark=benchmark.benchmark_id,
        passed=passed,
        reason=reason,
        duration_seconds=round(outcome.duration_seconds, 1),
    )
    return outcome
