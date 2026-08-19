"""M10 A/B: does stating the boundary explicitly stop the model implementing the wrong one?

Three milestones of test generation could not catch BIZ-003, because the tests inherited the
same misreading as the code. M10 attacks the requirement instead.

Arm B runs the deterministic detector over the requirement *before* implementation. When it
finds an EXPLICIT threshold, the operator and its neighbouring cases are appended to the task
description the coder receives. When it finds an ambiguous one, the task is blocked rather than
guessed at -- which is a false BLOCK by construction, and is counted as one.

Everything else is held identical: same model, same coder, same retries, same verification,
same acceptance tests, memory off, model review off, requirement-derived testing off.

Usage::

    python benchmarks/run_boundary.py [trials]
"""

from __future__ import annotations

import json
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
from edith.requirements.boundaries import (
    BoundaryStatus,
    detect_boundaries,
    render_for_plan,
    unresolved,
)
from edith.workspaces import ProjectWorkspace


def augment(task: BenchmarkTask) -> tuple[str, dict[str, Any]]:
    """Return the description the coder should receive, plus what the detector found."""
    found = detect_boundaries(task.requirement, requirement_id=task.task_id)
    explicit = [item for item in found if item.status is BoundaryStatus.EXPLICIT]
    blocked = unresolved(found)
    stats = {
        "detected": len(found),
        "explicit": len(explicit),
        "clarification": len(blocked),
    }
    if not explicit:
        return (task.requirement, stats)
    rendered = render_for_plan(tuple(explicit))
    return (
        f"{task.requirement}\n\n"
        "The following boundary conditions are authoritative and were derived from the "
        "requirement text. Implement them exactly as stated:\n"
        f"{rendered}",
        stats,
    )


def run_one(task: BenchmarkTask, boundary_aware: bool, provider: Any) -> dict[str, Any]:
    stats = {"detected": 0, "explicit": 0, "clarification": 0}
    description = task.requirement
    if boundary_aware:
        description, stats = augment(task)

    row: dict[str, Any] = {
        "task": task.task_id, "category": task.category.value,
        "completed": False, "accepted": False, "false_pass": False,
        "false_block": False, "repairs": 0, "seconds": 0.0, **stats,
    }

    # An unresolved boundary blocks implementation. That is the designed behaviour and it is
    # counted honestly as a false block when the task would otherwise have been accepted.
    if boundary_aware and stats["clarification"] and not stats["explicit"]:
        row["false_block"] = True
        return row

    root = Path(tempfile.mkdtemp(prefix="m10-"))
    prepare(root)
    executor = EngineeringExecutor(
        build_config(root, False),
        ProjectWorkspace(project_id="m10", name="n", root=root),
        provider=provider,
    )
    started = time.monotonic()
    try:
        report = executor.execute(plan_for(task, description), verify=True)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the benchmark
        row["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return row
    elapsed = time.monotonic() - started
    item = report.executions[0]
    # Acceptance always uses the ORIGINAL task: the augmentation changes what the coder is
    # told, never what correctness means.
    accepted = acceptance(root, task) if item.ok else False
    row.update(
        {
            "completed": item.ok,
            "accepted": accepted,
            "false_pass": bool(item.ok and not accepted),
            "repairs": item.repair_attempts,
            "seconds": round(elapsed, 1),
        }
    )
    return row


def summarise(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "arm": arm, "runs": len(rows),
        "completed": sum(r["completed"] for r in rows),
        "accepted": sum(r["accepted"] for r in rows),
        "false_pass": sum(r["false_pass"] for r in rows),
        "false_block": sum(r["false_block"] for r in rows),
        "detected": sum(r["detected"] for r in rows),
        "explicit": sum(r["explicit"] for r in rows),
        "clarification": sum(r["clarification"] for r in rows),
        "repairs": sum(r["repairs"] for r in rows),
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
        for arm, aware in (("A", False), ("B", True)):
            for task in TASKS:
                row = run_one(task, aware, provider)
                row["arm"] = arm
                row["trial"] = trial
                results[arm].append(row)
                print(
                    f"  {arm}{trial} {row['task']}: acc={row['accepted']} "
                    f"fp={row['false_pass']} fb={row['false_block']} "
                    f"bnd={row['explicit']}/{row['detected']} {row['seconds']}s",
                    flush=True,
                )

    print("\n=== M10 RESULTS ===")
    for arm in ("A", "B"):
        print(json.dumps(summarise(results[arm], arm)))
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
    print("\n=== PER TASK false passes ===")
    for arm in ("A", "B"):
        counts: dict[str, int] = {}
        for row in results[arm]:
            if row["false_pass"]:
                counts[row["task"]] = counts.get(row["task"], 0) + 1
        print(f"  {arm}: {counts or 'none'}")


if __name__ == "__main__":
    main()
