"""M13: the independent end-to-end benchmark.

Runs :mod:`benchmarks.production` -- twelve tasks written after M10-M12 and never used to tune
them -- through EDITH at production defaults.

Arm A is production defaults exactly as shipped. Arm B adds M10 boundary analysis, the one
optional feature with measured support, so its contribution is visible rather than assumed.

Acceptance is a separate pytest process against the merged tree, after the task is done. It is
never shown to the coder and nothing in the quality pipeline can reach it.

Usage::

    python benchmarks/run_production.py [trials]
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

from benchmarks.production import TASKS, Area, ProductionTask

from edith.config.loader import load_config
from edith.config.schema import ShellPolicyConfig, VerificationProfile
from edith.engineering.executor import EngineeringExecutor
from edith.models.ollama import OllamaProvider
from edith.product.architecture import ImplementationPlanDocument
from edith.requirements.boundaries import (
    BoundaryStatus,
    detect_boundaries,
    render_for_plan,
)
from edith.workspaces import ProjectWorkspace


def prepare(root: Path) -> None:
    for relative in ("src/backend", "tests"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for package in ("src", "src/backend"):
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_placeholder.py").write_text(
        "def test_p():\n    assert True\n", encoding="utf-8"
    )
    hooks = root / ".hk"
    hooks.mkdir(exist_ok=True)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@e.i"],
        ["git", "config", "user.name", "T"],
        ["git", "config", "core.hooksPath", str(hooks)],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "base"],
    ):
        completed = subprocess.run(argv, cwd=str(root), capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{argv} failed: {completed.stderr}")


def build_config(root: Path) -> Any:
    """Production defaults, with only the workspace and verification command bound."""
    base = load_config(None)
    return base.model_copy(
        update={
            "tools": base.tools.model_copy(
                update={
                    "workspace_root": root,
                    "shell": ShellPolicyConfig(
                        allowed_executables=(Path(sys.executable).stem, "python")
                    ),
                }
            ),
            "orchestration": base.orchestration.model_copy(
                update={
                    "workspaces_root": root.parent,
                    "verification_profiles": {
                        "python": VerificationProfile(
                            tests=("python", "-m", "pytest", "-q")
                        )
                    },
                }
            ),
        }
    )


def plan_for(task: ProductionTask, description: str, index: int) -> ImplementationPlanDocument:
    return ImplementationPlanDocument.model_validate(
        {
            "product_name": "m13",
            "goal": "implement the module",
            "tasks": [
                {
                    "task_id": f"TASK-{index + 1:03d}",
                    "title": task.requirement[:80],
                    "description": description,
                    "agent": "backend",
                    "paths": list(task.paths),
                    "verification": ["tests"],
                    "acceptance_criteria": ["the module implements the described behaviour"],
                    "depends_on": [],
                }
            ],
        }
    )


def acceptance(root: Path, task: ProductionTask) -> tuple[bool, str]:
    name = task.task_id.lower().replace("-", "_")
    (root / "tests" / f"test_acc_{name}.py").write_text(task.acceptance, encoding="utf-8")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"tests/test_acc_{name}.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (False, f"{type(exc).__name__}: {exc}")
    return (completed.returncode == 0, (completed.stdout + completed.stderr)[-800:])


def describe(task: ProductionTask, boundary_aware: bool) -> tuple[str, int]:
    """The description the coder receives, and how many boundaries were stated."""
    if not boundary_aware:
        return (task.requirement, 0)
    found = detect_boundaries(task.requirement, requirement_id=task.task_id)
    explicit = [item for item in found if item.status is BoundaryStatus.EXPLICIT]
    if not explicit:
        return (task.requirement, 0)
    return (
        f"{task.requirement}\n\n"
        "The following boundary conditions are authoritative and were derived from the "
        "requirement text. Implement them exactly as stated:\n"
        f"{render_for_plan(tuple(explicit))}",
        len(explicit),
    )


def run_one(
    task: ProductionTask, index: int, boundary_aware: bool, provider: Any
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="m13-"))
    prepare(root)
    description, boundaries = describe(task, boundary_aware)
    executor = EngineeringExecutor(
        build_config(root),
        ProjectWorkspace(project_id="m13", name="n", root=root),
        provider=provider,
    )
    started = time.monotonic()
    row: dict[str, Any] = {
        "task": task.task_id, "area": task.area.value, "ambiguous": task.ambiguous,
        "boundaries": boundaries, "completed": False, "accepted": False,
        "false_pass": False, "repairs": 0, "model_calls": 0, "security_failure": False,
        "merge_failure": False, "files_present": 0, "seconds": 0.0,
    }
    try:
        report = executor.execute(plan_for(task, description, index), verify=True)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the benchmark
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    elapsed = time.monotonic() - started
    item = report.executions[0]
    accepted, output = (False, "task did not complete")
    if item.ok:
        accepted, output = acceptance(root, task)
    category = item.failure_category.value if item.failure_category else ""
    row.update(
        {
            "completed": item.ok,
            "accepted": accepted,
            "false_pass": bool(item.ok and not accepted),
            "repairs": item.repair_attempts,
            "model_calls": item.model_calls,
            "security_failure": category == "SECURITY_FAILURE",
            "merge_failure": "merge" in (item.detail or "").lower(),
            "files_present": sum(1 for p in task.paths if (root / p).is_file()),
            "seconds": round(elapsed, 1),
            "outcome": str(item.outcome),
            "evidence": output[-300:],
        }
    )
    return row


def summarise(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    real = [r for r in rows if not r["ambiguous"]]
    times = [r["seconds"] for r in rows if r["seconds"]]
    return {
        "arm": arm,
        "runs": len(rows),
        "completed": sum(r["completed"] for r in rows),
        "accepted": sum(r["accepted"] for r in rows),
        # Ambiguous tasks are excluded from the headline false-pass figure: a defensible
        # reading of an under-specified requirement is not a defect.
        "false_pass_strict": sum(r["false_pass"] for r in real),
        "false_pass_incl_ambiguous": sum(r["false_pass"] for r in rows),
        "repairs": sum(r["repairs"] for r in rows),
        "model_calls": sum(r["model_calls"] for r in rows),
        "security_failures": sum(r["security_failure"] for r in rows),
        "merge_failures": sum(r["merge_failure"] for r in rows),
        "boundaries_stated": sum(r["boundaries"] for r in rows),
        "seconds": round(sum(times), 1),
        "worst_seconds": max(times) if times else 0.0,
        "mean_seconds": round(sum(times) / len(times), 1) if times else 0.0,
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
            for index, task in enumerate(TASKS):
                row = run_one(task, index, aware, provider)
                row["arm"] = arm
                row["trial"] = trial
                results[arm].append(row)
                print(
                    f"  {arm}{trial} {row['task']} [{row['area']}]: "
                    f"acc={row['accepted']} fp={row['false_pass']} "
                    f"rep={row['repairs']} {row['seconds']}s",
                    flush=True,
                )

    print("\n=== M13 PRODUCTION BENCHMARK ===")
    for arm in ("A", "B"):
        print(json.dumps(summarise(results[arm], arm)))
    print("\n=== BY AREA (accepted / runs) ===")
    for area in Area:
        parts = [f"{area.value:<15}"]
        for arm in ("A", "B"):
            rows = [r for r in results[arm] if r["area"] == area.value]
            if rows:
                parts.append(
                    f"{arm}={sum(r['accepted'] for r in rows)}/{len(rows)}"
                    f" fp={sum(r['false_pass'] for r in rows)}"
                )
        print("  " + "  ".join(parts))
    print("\n=== PER TASK failures ===")
    for arm in ("A", "B"):
        failed: dict[str, int] = {}
        for row in results[arm]:
            if not row["accepted"]:
                failed[row["task"]] = failed.get(row["task"], 0) + 1
        print(f"  {arm}: {failed or 'none'}")
    Path(tempfile.gettempdir(), "m13_raw.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
