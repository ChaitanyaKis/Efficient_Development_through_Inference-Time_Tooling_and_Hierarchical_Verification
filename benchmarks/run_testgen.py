"""M8 A/B: do requirement-derived tests reduce false PASSes?

A false PASS is the thing being measured: EDITH's own verification reports COMPLETED, and the
hand-written acceptance test then rejects the merged implementation. M7 measured three per arm.

Arm A is M7's pipeline unchanged. Arm B generates tests from the requirement *before* the coder
runs, validates them, writes the valid ones into ``tests/generated/``, and lets the task's own
verification pick them up. ``model_quality_review`` is off in both arms so it cannot confound.

Test strength is measured separately and deterministically: each generated test is run against a
known-correct and a known-incorrect implementation of the same function. A test that passes both
is worse than useless -- it adds confidence without information, which is exactly how a false
PASS happens. The incorrect implementation is never shown to the generator.

Usage::

    python benchmarks/run_testgen.py [trials]
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

from benchmarks.run_semantic import (
    acceptance,
    build_config,
    plan_for,
    prepare,
)
from benchmarks.semantic import TASKS, BenchmarkTask, Category

from edith.config.loader import load_config
from edith.engineering.executor import EngineeringExecutor
from edith.models.ollama import OllamaProvider
from edith.quality.testgen import (
    GENERATED_TEST_DIR,
    GeneratedTest,
    TestGeneratorAgent,
    generate_tests,
    module_for,
)
from edith.workspaces import ProjectWorkspace

#: Known-wrong implementations, used only for the strength probe. The generator never sees them.
MUTANTS: dict[str, tuple[str, str]] = {
    "SEM-001": (
        "def running_total(values):\n    return list(values)\n",
        "def running_total(values):\n    out = []\n    total = 0\n"
        "    for v in values:\n        total += v\n        out.append(total)\n    return out\n",
    ),
    "SEM-003": (
        "def apply_discount(price, percent):\n    return round(price + price * percent / 100, 2)\n",
        "def apply_discount(price, percent):\n    return round(price - price * percent / 100, 2)\n",
    ),
    "EDGE-001": (
        "def clamp(value, low, high):\n    return max(low, min(high - 1, value))\n",
        "def clamp(value, low, high):\n    return max(low, min(high, value))\n",
    ),
    "BIZ-001": (
        "def shipping_cost(weight_kg):\n    return 0.0 if weight_kg > 10 else 5.0\n",
        "def shipping_cost(weight_kg):\n    return 0.0 if weight_kg >= 10 else 5.0\n",
    ),
}


def write_generated(root: Path, tests: tuple[GeneratedTest, ...]) -> int:
    """Write the validated tests into the task workspace. Invalid ones are never written."""
    directory = root / GENERATED_TEST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    written = 0
    for index, test in enumerate(tests):
        if not test.authoritative:
            continue
        (directory / f"test_gen_{index}.py").write_text(test.source, encoding="utf-8")
        written += 1
    if written:
        # Commit them, or the task worktree branches from HEAD without them and the
        # generated suite never runs during verification -- which is the whole point.
        for argv in (["git", "add", "-A"], ["git", "commit", "-qm", "requirement-derived tests"]):
            subprocess.run(
                argv, cwd=str(root), capture_output=True, text=True
            )
    return written


def strength(task: BenchmarkTask, tests: tuple[GeneratedTest, ...]) -> dict[str, int]:
    """Run the generated tests against a known-correct and a known-incorrect implementation."""
    result = {"probed": 0, "killed_mutant": 0, "passed_correct": 0}
    if task.task_id not in MUTANTS:
        return result
    wrong, right = MUTANTS[task.task_id]
    valid = [test for test in tests if test.authoritative]
    if not valid:
        return result
    result["probed"] = 1
    for implementation, key in ((right, "passed_correct"), (wrong, "killed_mutant")):
        root = Path(tempfile.mkdtemp(prefix="m8-probe-"))
        (root / "src" / "backend").mkdir(parents=True)
        for package in ("src", "src/backend"):
            (root / package / "__init__.py").write_text("", encoding="utf-8")
        (root / task.path).write_text(implementation, encoding="utf-8")
        directory = root / GENERATED_TEST_DIR
        directory.mkdir(parents=True)
        for index, test in enumerate(valid):
            (directory / f"test_gen_{index}.py").write_text(test.source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", GENERATED_TEST_DIR, "-q"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        passed = completed.returncode == 0
        if key == "passed_correct" and passed:
            result["passed_correct"] = 1
        # A useful test FAILS the wrong implementation.
        if key == "killed_mutant" and not passed:
            result["killed_mutant"] = 1
    return result


def run_one(
    task: BenchmarkTask, use_testgen: bool, provider: Any
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="m8-"))
    prepare(root)
    config = build_config(root, False)  # model_quality_review off in BOTH arms

    generated: tuple[GeneratedTest, ...] = ()
    written = 0
    gen_seconds = 0.0
    if use_testgen:
        started = time.monotonic()
        generated = generate_tests(
            TestGeneratorAgent(provider=provider),
            requirement_id=task.task_id,
            requirement=task.requirement,
            module=module_for(task.path),
            known_requirements=frozenset({task.task_id}),
        )
        gen_seconds = time.monotonic() - started
        written = write_generated(root, generated)

    executor = EngineeringExecutor(
        config, ProjectWorkspace(project_id="m8", name="n", root=root), provider=provider
    )
    started = time.monotonic()
    try:
        report = executor.execute(plan_for(task), verify=True)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the benchmark
        return {
            "task": task.task_id, "category": task.category.value,
            "error": f"{type(exc).__name__}: {exc}"[:160], "completed": False,
            "accepted": False, "false_pass": False, "repairs": 0, "generated": 0,
            "valid": 0, "written": 0, "seconds": 0.0, "probed": 0,
            "killed_mutant": 0, "passed_correct": 0,
        }
    elapsed = time.monotonic() - started + gen_seconds
    item = report.executions[0]
    accepted = acceptance(root, task) if item.ok else False
    probe = strength(task, generated)
    return {
        "task": task.task_id, "category": task.category.value,
        "completed": item.ok, "accepted": accepted,
        # The metric M8 exists to move: verification said done, acceptance disagreed.
        "false_pass": bool(item.ok and not accepted),
        "repairs": item.repair_attempts,
        "generated": len(generated),
        "valid": sum(1 for test in generated if test.valid),
        "written": written,
        "seconds": round(elapsed, 1),
        **probe,
    }


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = load_config(None)
    provider = OllamaProvider(
        base.models.ollama, base.models.profiles[base.models.default_profile]
    )
    results: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for trial in range(trials):
        for arm, use_testgen in (("A", False), ("B", True)):
            for task in TASKS:
                row = run_one(task, use_testgen, provider)
                row["arm"] = arm
                row["trial"] = trial
                results[arm].append(row)
                print(
                    f"  {arm}{trial} {row['task']}: acc={row['accepted']} "
                    f"fp={row['false_pass']} gen={row['generated']}/{row['valid']} "
                    f"kill={row['killed_mutant']} {row['seconds']}s",
                    flush=True,
                )

    print("\n=== M8 RESULTS ===")
    for arm in ("A", "B"):
        rows = results[arm]
        print(
            json.dumps(
                {
                    "arm": arm, "runs": len(rows),
                    "completed": sum(r["completed"] for r in rows),
                    "accepted": sum(r["accepted"] for r in rows),
                    "false_pass": sum(r["false_pass"] for r in rows),
                    "repairs": sum(r["repairs"] for r in rows),
                    "generated": sum(r["generated"] for r in rows),
                    "valid": sum(r["valid"] for r in rows),
                    "written": sum(r["written"] for r in rows),
                    "probed": sum(r["probed"] for r in rows),
                    "killed_mutant": sum(r["killed_mutant"] for r in rows),
                    "passed_correct": sum(r["passed_correct"] for r in rows),
                    "seconds": round(sum(r["seconds"] for r in rows), 1),
                }
            )
        )
    print("\n=== BY CATEGORY (accepted / runs, false passes) ===")
    for category in Category:
        parts = [category.value]
        for arm in ("A", "B"):
            rows = [r for r in results[arm] if r["category"] == category.value]
            parts.append(
                f"{arm}={sum(r['accepted'] for r in rows)}/{len(rows)}"
                f" fp={sum(r['false_pass'] for r in rows)}"
            )
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
