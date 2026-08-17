"""The memory experiment.

M3 asks a question that unit tests cannot answer: **does engineering memory actually make a
weak local model better at repairing code?** The honest answer might be no, and this harness
is built so that a null result is reported rather than hidden.

Design:

- Both arms run the *same* benchmark, the *same* code path, and the *same* model. The only
  difference is whether the orchestrator was handed a memory store.
- The memory arm is seeded with lessons drawn from real observed M2 failures, each carrying
  provenance. Nothing is invented to flatter the experiment, and nothing states the answer:
  a memory saying "return sorted(...)" would be handing over the fix, which measures nothing.
- Each arm runs N times, because a 3B model is not deterministic and a single run of each
  proves nothing either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edith.benchmarks import (
    BenchmarkResult,
    check_protected_files,
    get_benchmark,
    prepare_workspace,
    remove_tree,
    run_verification,
)
from edith.config.schema import EdithConfig, MemoryBudgetConfig
from edith.memory.governor import (
    BUDGET_PRESETS,
    DEFAULT_LIMITS,
    MemoryBudgetLimits,
)
from edith.memory.schema import (
    MemoryProposal,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from edith.memory.store import MemoryStore, open_memory
from edith.memory.strategy import MemoryStrategy
from edith.observability.logging import get_logger
from edith.orchestrator import Orchestrator, create_execution
from edith.schemas.common import Verdict, new_id
from edith.state.store import open_store
from edith.workspaces import ProjectWorkspace

logger = get_logger(__name__)


#: Lessons seeded into the memory arm.
#:
#: Every one is a *generalisation of an observed failure*, phrased as guidance about a class
#: of mistake rather than as the answer to this benchmark. The provenance is real: these
#: behaviours were seen in the M2 and M2.1 benchmark runs recorded in the ADRs.
SEEDED_LESSONS: tuple[dict[str, Any], ...] = (
    {
        "title": "Rewriting a file often drops unrelated functions",
        "content": (
            "When asked to change one function, a small model rewriting the whole file "
            "frequently omits other functions that were already there. Always reproduce "
            "every existing definition, or edit only the one function that needs changing."
        ),
        "tags": ("editing", "regression", "small-model"),
        "importance": 85,
    },
    {
        "title": "A test asserting order requires sorting, not just filtering",
        "content": (
            "When a test compares a returned list to a specific ordered list, filtering the "
            "input is not enough - the result must also be ordered. Check whether the "
            "expected value implies an ordering as well as a selection."
        ),
        "tags": ("collections", "ordering", "assertions"),
        "importance": 80,
    },
    {
        "title": "Boundary tests distinguish < from <=",
        "content": (
            "A test that exercises a value exactly equal to a threshold is testing "
            "inclusivity. If it expects the boundary value to be included, the comparison "
            "must be <= or >=, not < or >."
        ),
        "tags": ("boundaries", "comparison", "off-by-one"),
        "importance": 80,
    },
    {
        "title": "Fix every failing test, not only the first",
        "content": (
            "When a test run reports several failures, addressing one leaves the run red. "
            "Read the whole failure list and account for each one before finishing."
        ),
        "tags": ("multi-defect", "verification"),
        "importance": 75,
    },
    {
        "title": "Rounding up to a whole batch needs ceiling division",
        "content": (
            "When a quantity must be raised to the next multiple of a batch size, integer "
            "division truncates. Use ceiling division so a partial batch rounds up."
        ),
        "tags": ("arithmetic", "rounding"),
        "importance": 70,
    },
)


@dataclass
class ArmResult:
    """Aggregated measurements for one experiment arm."""

    name: str
    runs: int = 0
    successes: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of runs that passed the independent check."""
        return self.successes / self.runs if self.runs else 0.0

    def _mean(self, attribute: str) -> float:
        values = [
            getattr(result.metrics, attribute)
            for result in self.results
            if result.execution is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_model_calls(self) -> float:
        """Average model invocations per run."""
        return self._mean("model_calls")

    @property
    def mean_repairs(self) -> float:
        """Average debugger invocations per run."""
        return self._mean("repairs")

    @property
    def mean_duration(self) -> float:
        """Average wall-clock seconds per run."""
        return self._mean("duration_seconds")

    @property
    def mean_memory_chars(self) -> float:
        """Average characters of memory injected per run."""
        values = [
            result.execution.memory_chars
            for result in self.results
            if result.execution is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_memory_retrievals(self) -> float:
        """Average injections per run. The number a per-prompt limit cannot bound."""
        values = [
            result.execution.memory_retrievals
            for result in self.results
            if result.execution is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def peak_memory_chars(self) -> int:
        """The worst single run's memory cost.

        The mean hides exactly the case that matters: one execution that ran away. A budget
        that works must flatten this, not just the average.
        """
        values = [
            result.execution.memory_chars
            for result in self.results
            if result.execution is not None
        ]
        return max(values) if values else 0

    @property
    def budget_exhaustions(self) -> int:
        """Total requests refused across the arm because the allowance was spent."""
        return sum(
            result.execution.budget_exhaustions
            for result in self.results
            if result.execution is not None
        )

    @property
    def false_positives(self) -> int:
        """Runs Edith called a success that an independent check refuted."""
        return sum(1 for result in self.results if result.metrics.false_positive)

    def failure_reasons(self) -> dict[str, int]:
        """Failure reasons and how often each occurred."""
        counts: dict[str, int] = {}
        for result in self.results:
            if result.passed:
                continue
            counts[result.reason] = counts.get(result.reason, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary."""
        return {
            "arm": self.name,
            "runs": self.runs,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 3),
            "mean_model_calls": round(self.mean_model_calls, 1),
            "mean_repairs": round(self.mean_repairs, 1),
            "mean_duration_seconds": round(self.mean_duration, 1),
            "mean_memory_chars": round(self.mean_memory_chars, 1),
            "peak_memory_chars": self.peak_memory_chars,
            "mean_memory_retrievals": round(self.mean_memory_retrievals, 2),
            "budget_exhaustions": self.budget_exhaustions,
            "false_positives": self.false_positives,
            "failure_reasons": self.failure_reasons(),
        }


@dataclass
class ExperimentResult:
    """Both arms plus the comparison between them."""

    benchmark_id: str
    baseline: ArmResult
    memory: ArmResult

    @property
    def success_rate_delta(self) -> float:
        """Memory arm success rate minus baseline."""
        return self.memory.success_rate - self.baseline.success_rate

    @property
    def improved(self) -> bool:
        """Whether memory measurably helped."""
        return self.success_rate_delta > 0

    def verdict(self) -> str:
        """A plain-language statement of what was measured.

        Deliberately blunt about small samples: with a handful of runs on a stochastic
        model, "no measurable difference" is usually the honest reading.
        """
        delta = self.success_rate_delta
        if abs(delta) < 1e-9:
            return (
                f"No measurable difference: both arms succeeded "
                f"{self.baseline.successes}/{self.baseline.runs}."
            )
        direction = "improved" if delta > 0 else "reduced"
        return (
            f"Memory {direction} the success rate by {abs(delta) * 100:.0f} points "
            f"({self.baseline.successes}/{self.baseline.runs} -> "
            f"{self.memory.successes}/{self.memory.runs}). With this sample size that is "
            f"suggestive, not conclusive."
        )

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary of the whole experiment."""
        return {
            "benchmark": self.benchmark_id,
            "baseline": self.baseline.as_dict(),
            "memory": self.memory.as_dict(),
            "success_rate_delta": round(self.success_rate_delta, 3),
            "verdict": self.verdict(),
        }


def seed_lessons(store: MemoryStore, project_id: str) -> int:
    """Seed the engineering lessons used by the memory arm.

    Stored as ``GLOBAL`` ``ENGINEERING`` memories, which is what they are: generalisations
    that apply beyond one repository. Their source is ``TEST_RESULT`` because each was
    derived from an observed failing benchmark run, not from a model's opinion.
    """
    stored = 0
    for lesson in SEEDED_LESSONS:
        record, outcome = store.propose(
            MemoryProposal(
                type=MemoryType.ENGINEERING,
                scope=MemoryScope.GLOBAL,
                project_id=None,
                title=str(lesson["title"]),
                content=str(lesson["content"]),
                tags=tuple(lesson["tags"]),
                importance=int(lesson["importance"]),
                source=MemorySource.TEST_RESULT,
                source_reference="observed in M2/M2.1 benchmark runs; see docs/adr/0004",
            )
        )
        if record is not None:
            stored += 1
        else:
            logger.warning("experiment.seed_rejected", reason=outcome.reason)
    return stored


def configure_strategy(config: EdithConfig, strategy: MemoryStrategy) -> EdithConfig:
    """Return a config with one memory strategy selected.

    Every arm runs the *same* code path with a different policy, rather than a separate
    branch per arm -- otherwise the experiment would be comparing implementations rather
    than strategies.
    """
    return config.model_copy(
        update={
            "orchestration": config.orchestration.model_copy(
                update={
                    "memory": config.orchestration.memory.model_copy(
                        update={"strategy": str(strategy), "enabled": True}
                    )
                }
            )
        }
    )


def configure_budget(
    config: EdithConfig, limits: MemoryBudgetLimits | None, *, enabled: bool = True
) -> EdithConfig:
    """Return a config with one execution memory budget selected.

    ``enabled=False`` is the unbudgeted arm: the governor still runs, so strategy,
    relevance, and duplicate suppression all still apply, and the arms differ by the
    execution ceiling alone rather than by two different code paths.
    """
    limits = limits or DEFAULT_LIMITS
    budget = MemoryBudgetConfig(
        enabled=enabled,
        max_total_chars=limits.max_total_chars,
        max_retrievals=limits.max_retrievals,
        max_total_memories=limits.max_total_memories,
        max_chars_per_retrieval=limits.max_chars_per_retrieval,
        max_memories_per_retrieval=limits.max_memories_per_retrieval,
    )
    return config.model_copy(
        update={
            "orchestration": config.orchestration.model_copy(
                update={
                    "memory": config.orchestration.memory.model_copy(
                        update={"budget": budget}
                    )
                }
            )
        }
    )


#: The M3.2 arms. A is the control, B reproduces the M3.1 unbounded behaviour, and C adds
#: the execution budget to B while changing nothing else -- so any difference between B and
#: C is the budget and not the strategy.
BUDGET_ARMS: tuple[tuple[str, MemoryStrategy, MemoryBudgetLimits | None, bool], ...] = (
    ("A_no_memory", MemoryStrategy.NONE, None, True),
    ("B_debugger_unbudgeted", MemoryStrategy.DEBUGGER_ONLY, None, False),
    ("C_debugger_budgeted", MemoryStrategy.DEBUGGER_ONLY, BUDGET_PRESETS["medium"], True),
)

#: The ablation: the same strategy at three allowances, to expose the tradeoff between
#: memory's benefit and its context cost rather than assuming where the knee is.
ABLATION_ARMS: tuple[tuple[str, MemoryStrategy, MemoryBudgetLimits | None, bool], ...] = (
    ("budget_small", MemoryStrategy.DEBUGGER_ONLY, BUDGET_PRESETS["small"], True),
    ("budget_medium", MemoryStrategy.DEBUGGER_ONLY, BUDGET_PRESETS["medium"], True),
    ("budget_large", MemoryStrategy.DEBUGGER_ONLY, BUDGET_PRESETS["large"], True),
)


def run_budget_comparison(
    config: EdithConfig,
    workspace_root: Path,
    *,
    benchmark_id: str = "multi_repair",
    runs: int = 3,
    arms: tuple[tuple[str, MemoryStrategy, MemoryBudgetLimits | None, bool], ...] = (
        BUDGET_ARMS
    ),
) -> dict[str, ArmResult]:
    """Measure whether an execution-level budget keeps memory from becoming harmful.

    All arms share one seeded memory store and one benchmark, so the only variable is the
    memory policy. The control arm keeps a live store attached with a strategy that never
    retrieves, which keeps every arm on the identical code path.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    memory_dir = workspace_root / ".memory-budget"
    remove_tree(memory_dir)

    results: dict[str, ArmResult] = {}
    with open_memory(memory_dir) as store:
        seeded = seed_lessons(store, "experiment")
        logger.info("experiment.seeded", lessons=seeded)

        for name, strategy, limits, budget_enabled in arms:
            arm = ArmResult(name=name)
            arm_config = configure_budget(
                configure_strategy(config, strategy), limits, enabled=budget_enabled
            )
            for index in range(runs):
                outcome = _run_once(
                    benchmark_id,
                    arm_config,
                    workspace_root,
                    memory=store,
                    label=f"{name}-{index}",
                )
                arm.runs += 1
                arm.successes += int(outcome.passed)
                arm.results.append(outcome)
                logger.info(
                    "experiment.run",
                    arm=name,
                    index=index,
                    passed=outcome.passed,
                    reason=outcome.reason,
                    memory_chars=(
                        outcome.execution.memory_chars if outcome.execution else 0
                    ),
                    budget_exhaustions=(
                        outcome.execution.budget_exhaustions if outcome.execution else 0
                    ),
                )
            results[name] = arm
    return results


def _run_once(
    benchmark_id: str,
    config: EdithConfig,
    workspace_root: Path,
    *,
    memory: MemoryStore | None,
    label: str,
) -> BenchmarkResult:
    """Run one benchmark iteration in one arm."""
    benchmark = get_benchmark(benchmark_id)
    started = time.monotonic()
    workspace_path = prepare_workspace(
        benchmark, workspace_root / f"{benchmark_id}-{label}"
    )

    state_dir = workspace_root / f".state-{label}"
    remove_tree(state_dir)
    store = open_store(state_dir)
    workspace = ProjectWorkspace(
        project_id=new_id("proj"),
        name=f"experiment-{benchmark_id}-{label}",
        root=workspace_path,
    )

    orchestrator = Orchestrator(config, store, workspace, memory=memory)
    try:
        _, execution = create_execution(store, workspace, benchmark.request)
        result = orchestrator.run(execution)
    finally:
        orchestrator.close()

    final_passed = run_verification(workspace_path, benchmark.verify_argv)
    intact, tampered = check_protected_files(benchmark, workspace_path)
    store.close()

    passed = final_passed and intact and result.verdict is Verdict.PASS
    reason = (
        "completed"
        if passed
        else (
            f"protected files modified: {tampered}"
            if not intact
            else "verification still fails"
            if not final_passed
            else f"Edith reported {result.verdict}"
        )
    )

    outcome = BenchmarkResult(
        benchmark_id=benchmark_id,
        passed=passed,
        reason=reason,
        execution=result,
        final_verification_passed=final_passed,
        protected_files_intact=intact,
        tampered_files=tampered,
        workspace=str(workspace_path),
        duration_seconds=time.monotonic() - started,
    )
    outcome.metrics.model_calls = result.model_calls
    outcome.metrics.agent_runs = result.agent_runs
    outcome.metrics.repairs = result.repairs_attempted
    outcome.metrics.tasks_total = result.tasks_total
    outcome.metrics.tasks_succeeded = result.tasks_succeeded
    outcome.metrics.duration_seconds = outcome.duration_seconds
    outcome.metrics.false_positive = result.verdict is Verdict.PASS and not (
        final_passed and intact
    )
    return outcome


def run_strategy_comparison(
    config: EdithConfig,
    workspace_root: Path,
    *,
    benchmark_id: str = "multi_repair",
    runs: int = 3,
    strategies: tuple[MemoryStrategy, ...] = (
        MemoryStrategy.NONE,
        MemoryStrategy.ALWAYS,
        MemoryStrategy.FAILURE_TRIGGERED,
        MemoryStrategy.DEBUGGER_ONLY,
        MemoryStrategy.HIGH_RELEVANCE,
    ),
) -> dict[str, ArmResult]:
    """Run every memory strategy against the same benchmark and return their results.

    All arms share one seeded memory store and one benchmark, so the only variable is the
    retrieval policy. ``NONE`` is run with a live store attached but a policy that never
    retrieves, which keeps even the control arm on the identical code path.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    memory_dir = workspace_root / ".memory-strategies"
    remove_tree(memory_dir)

    results: dict[str, ArmResult] = {}
    with open_memory(memory_dir) as store:
        seeded = seed_lessons(store, "experiment")
        logger.info("experiment.seeded", lessons=seeded)

        for strategy in strategies:
            arm = ArmResult(name=str(strategy))
            arm_config = configure_strategy(config, strategy)
            for index in range(runs):
                outcome = _run_once(
                    benchmark_id,
                    arm_config,
                    workspace_root,
                    memory=store,
                    label=f"{strategy}-{index}",
                )
                arm.runs += 1
                arm.successes += int(outcome.passed)
                arm.results.append(outcome)
                logger.info(
                    "experiment.run",
                    arm=str(strategy),
                    index=index,
                    passed=outcome.passed,
                    memory_chars=(
                        outcome.execution.memory_chars if outcome.execution else 0
                    ),
                )
            results[str(strategy)] = arm
            logger.info(
                "experiment.arm_finished",
                arm=str(strategy),
                successes=f"{arm.successes}/{arm.runs}",
            )
    return results


def run_memory_experiment(
    config: EdithConfig,
    workspace_root: Path,
    *,
    benchmark_id: str = "multi_repair",
    runs: int = 3,
) -> ExperimentResult:
    """Run both arms of the memory experiment and compare them.

    Args:
        config: Resolved configuration.
        workspace_root: Where the scratch workspaces are created.
        benchmark_id: Which benchmark to measure. Defaults to the multi-defect repair, the
            only shipped scenario the 3B model does not already solve reliably -- measuring
            a benchmark that always passes could not show an improvement.
        runs: Iterations per arm.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    baseline = ArmResult(name="baseline")
    memory_arm = ArmResult(name="memory")

    logger.info("experiment.start", benchmark=benchmark_id, runs=runs)

    for index in range(runs):
        outcome = _run_once(
            benchmark_id, config, workspace_root, memory=None, label=f"baseline-{index}"
        )
        baseline.runs += 1
        baseline.successes += int(outcome.passed)
        baseline.results.append(outcome)
        logger.info(
            "experiment.run", arm="baseline", index=index, passed=outcome.passed
        )

    memory_dir = workspace_root / ".memory-experiment"
    remove_tree(memory_dir)
    with open_memory(memory_dir) as store:
        seeded = seed_lessons(store, "experiment")
        logger.info("experiment.seeded", lessons=seeded)

        for index in range(runs):
            outcome = _run_once(
                benchmark_id, config, workspace_root, memory=store, label=f"memory-{index}"
            )
            memory_arm.runs += 1
            memory_arm.successes += int(outcome.passed)
            memory_arm.results.append(outcome)
            logger.info(
                "experiment.run", arm="memory", index=index, passed=outcome.passed
            )

    experiment = ExperimentResult(
        benchmark_id=benchmark_id, baseline=baseline, memory=memory_arm
    )
    logger.info(
        "experiment.finished",
        baseline=f"{baseline.successes}/{baseline.runs}",
        memory=f"{memory_arm.successes}/{memory_arm.runs}",
        delta=round(experiment.success_rate_delta, 3),
    )
    return experiment
