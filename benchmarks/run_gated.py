"""M9 A/B: does a scaffold gate make requirement-derived tests usable?

M8 ran the same generator without a gate: false PASSes went to zero, but only because 32 of 36
runs were rejected. The tests were mechanically valid and semantically wrong.

Arm B here adds one deterministic step. Every generated test is executed against a
human-authored known-correct implementation; a test the scaffold does not satisfy is asserting
the wrong thing and is discarded before it can block a coder. The gate is per test, so one bad
assertion no longer condemns its siblings.

Three controls keep the measurement honest:

- the scaffold is never shown to the generator or the coder;
- the *mutant* is never shown to the gate, only to the strength probe afterwards;
- acceptance stays a separate pytest process against the merged tree.

Usage::

    python benchmarks/run_gated.py [trials]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_semantic import acceptance, build_config, plan_for, prepare
from benchmarks.semantic import TASKS, BenchmarkTask, Category

from edith.config.loader import load_config
from edith.engineering.executor import EngineeringExecutor
from edith.models.ollama import OllamaProvider
from edith.quality.testgate import GateOutcome, gate_tests
from edith.quality.testgen import (
    GENERATED_TEST_DIR,
    GeneratedTest,
    TestGeneratorAgent,
    generate_tests,
    module_for,
)
from edith.workspaces import ProjectWorkspace


def write_generated(root: Path, tests: tuple[GeneratedTest, ...]) -> int:
    """Write retained tests into the workspace and commit, so the worktree inherits them."""
    if not tests:
        return 0
    directory = root / GENERATED_TEST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    for index, test in enumerate(tests):
        (directory / f"test_gen_{index}.py").write_text(test.source, encoding="utf-8")
    for argv in (["git", "add", "-A"], ["git", "commit", "-qm", "requirement-derived tests"]):
        subprocess.run(argv, cwd=str(root), capture_output=True, text=True)  # noqa: S603
    return len(tests)


def run_suite(implementation: str, tests: tuple[GeneratedTest, ...], task: BenchmarkTask) -> bool:
    """Run the retained tests against one implementation. True when they all pass."""
    root = Path(tempfile.mkdtemp(prefix="m9-probe-"))
    target = root / task.path
    target.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in Path(task.path).parent.parts:
        current = current / part
        (current / "__init__.py").write_text("", encoding="utf-8")
    target.write_text(implementation, encoding="utf-8")
    directory = root / GENERATED_TEST_DIR
    directory.mkdir(parents=True)
    for index, test in enumerate(tests):
        (directory / f"test_gen_{index}.py").write_text(test.source, encoding="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [sys.executable, "-m", "pytest", GENERATED_TEST_DIR, "-q", "-p", "no:cacheprovider"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def run_one(task: BenchmarkTask, gated: bool, provider: Any) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="m9-"))
    prepare(root)
    config = build_config(root, False)  # model_quality_review off in BOTH arms

    generated: tuple[GeneratedTest, ...] = ()
    retained: tuple[GeneratedTest, ...] = ()
    discarded = contradicts = did_not_execute = 0
    gen_seconds = 0.0
    if gated:
        started = time.monotonic()
        generated = generate_tests(
            TestGeneratorAgent(provider=provider),
            requirement_id=task.task_id,
            requirement=task.requirement,
            module=module_for(task.path),
            known_requirements=frozenset({task.task_id}),
        )
        verdict = gate_tests(
            generated, scaffold=task.scaffold, module_path=task.path
        )
        retained = verdict.retained
        discarded = len(verdict.discarded)
        contradicts = verdict.count(GateOutcome.CONTRADICTS_SCAFFOLD)
        did_not_execute = verdict.count(GateOutcome.DID_NOT_EXECUTE)
        gen_seconds = time.monotonic() - started
        write_generated(root, retained)

    executor = EngineeringExecutor(
        config, ProjectWorkspace(project_id="m9", name="n", root=root), provider=provider
    )
    started = time.monotonic()
    try:
        report = executor.execute(plan_for(task), verify=True)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the benchmark
        return {
            "task": task.task_id, "category": task.category.value,
            "error": f"{type(exc).__name__}: {exc}"[:160], "completed": False,
            "accepted": False, "false_pass": False, "repairs": 0, "generated": 0,
            "retained": 0, "discarded": 0, "contradicts": 0, "did_not_execute": 0,
            "correct_pass": 0, "incorrect_fail": 0, "seconds": 0.0,
        }
    elapsed = time.monotonic() - started + gen_seconds
    item = report.executions[0]
    accepted = acceptance(root, task) if item.ok else False

    # Strength control, run only on what survived the gate. The mutant never reached the gate.
    correct_pass = incorrect_fail = 0
    if retained:
        correct_pass = int(run_suite(task.scaffold, retained, task))
        incorrect_fail = int(not run_suite(task.mutant, retained, task))

    return {
        "task": task.task_id, "category": task.category.value,
        "completed": item.ok, "accepted": accepted,
        "false_pass": bool(item.ok and not accepted),
        "repairs": item.repair_attempts,
        "generated": len(generated), "retained": len(retained),
        "discarded": discarded, "contradicts": contradicts,
        "did_not_execute": did_not_execute,
        "correct_pass": correct_pass, "incorrect_fail": incorrect_fail,
        "seconds": round(elapsed, 1),
    }


def summarise(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "arm": arm, "runs": len(rows),
        "completed": sum(r["completed"] for r in rows),
        "accepted": sum(r["accepted"] for r in rows),
        "false_pass": sum(r["false_pass"] for r in rows),
        "repairs": sum(r["repairs"] for r in rows),
        "generated": sum(r["generated"] for r in rows),
        "retained": sum(r["retained"] for r in rows),
        "discarded": sum(r["discarded"] for r in rows),
        "contradicts": sum(r["contradicts"] for r in rows),
        "did_not_execute": sum(r["did_not_execute"] for r in rows),
        "correct_pass": sum(r["correct_pass"] for r in rows),
        "incorrect_fail": sum(r["incorrect_fail"] for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
    }


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = load_config(None)
    provider = OllamaProvider(
        base.models.ollama, base.models.profiles[base.models.default_profile]
    )
    results: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for trial in range(trials):
        for arm, gated in (("A", False), ("B", True)):
            for task in TASKS:
                row = run_one(task, gated, provider)
                row["arm"] = arm
                row["trial"] = trial
                results[arm].append(row)
                print(
                    f"  {arm}{trial} {row['task']}: acc={row['accepted']} "
                    f"fp={row['false_pass']} gen={row['generated']} "
                    f"kept={row['retained']} cp={row['correct_pass']} "
                    f"if={row['incorrect_fail']} {row['seconds']}s",
                    flush=True,
                )

    print("\n=== M9 RESULTS ===")
    for arm in ("A", "B"):
        print(json.dumps(summarise(results[arm], arm)))
    print("\n=== BY CATEGORY ===")
    for category in Category:
        parts = [category.value]
        for arm in ("A", "B"):
            rows = [r for r in results[arm] if r["category"] == category.value]
            parts.append(
                f"{arm}: acc={sum(r['accepted'] for r in rows)}/{len(rows)}"
                f" fp={sum(r['false_pass'] for r in rows)}"
                f" gen={sum(r['generated'] for r in rows)}"
                f" kept={sum(r['retained'] for r in rows)}"
                f" cp={sum(r['correct_pass'] for r in rows)}"
                f" if={sum(r['incorrect_fail'] for r in rows)}"
            )
        print("  " + "\n    ".join(parts))


if __name__ == "__main__":
    main()
