"""M7 A/B runner: does model quality review improve acceptance on semantic defects?

Both arms receive identical tasks, identical model, identical retry budget and identical
acceptance tests. Only ``orchestration.model_quality_review`` differs.

Acceptance is decided by a separate pytest process running the hand-written tests in
:mod:`benchmarks.semantic` against the *merged main tree* -- not the worktree, and not
anything the Judge said. M6.2 found a merge that moved ledger state without moving files, so
``merged_ok`` re-checks the filesystem rather than trusting the MERGED state.

Usage::

    python benchmarks/run_semantic.py [trials]
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

from benchmarks.semantic import TASKS, BenchmarkTask, Category

from edith.config.loader import load_config
from edith.config.schema import ShellPolicyConfig, VerificationProfile
from edith.engineering.executor import EngineeringExecutor
from edith.models.ollama import OllamaProvider
from edith.product.architecture import ImplementationPlanDocument
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
        ["git", "commit", "-qm", "b"],
    ):
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            argv, cwd=str(root), capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{argv} failed: {completed.stderr}")


def build_config(root: Path, model_review: bool) -> Any:
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
                    "model_quality_review": model_review,
                    "verification_profiles": {
                        "python": VerificationProfile(
                            tests=("python", "-m", "pytest", "-q")
                        )
                    },
                }
            ),
        }
    )


def plan_id(task: BenchmarkTask) -> str:
    """The plan schema requires ``TASK-NNN``; the benchmark id is kept for reporting."""
    return f"TASK-{TASKS.index(task) + 1:03d}"


def plan_for(task: BenchmarkTask) -> ImplementationPlanDocument:
    return ImplementationPlanDocument.model_validate(
        {
            "product_name": "m7",
            "goal": "implement the module",
            "tasks": [
                {
                    "task_id": plan_id(task),
                    "title": task.requirement[:80],
                    "description": task.requirement,
                    "agent": "backend",
                    "paths": [task.path],
                    "verification": ["tests"],
                    "acceptance_criteria": ["the module implements the described behaviour"],
                    "depends_on": [],
                }
            ],
        }
    )


def acceptance(root: Path, task: BenchmarkTask, suffix: str = "") -> bool:
    """Run the hand-written acceptance test against the merged main tree."""
    name = task.task_id.lower().replace("-", "_") + suffix
    (root / "tests" / f"test_acc_{name}.py").write_text(task.acceptance, encoding="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [str(Path(sys.executable)), "-m", "pytest", f"tests/test_acc_{name}.py", "-q"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def run_one(task: BenchmarkTask, model_review: bool, provider: Any) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="m7-"))
    prepare(root)
    executor = EngineeringExecutor(
        build_config(root, model_review),
        ProjectWorkspace(project_id="m7", name="n", root=root),
        provider=provider,
    )
    started = time.monotonic()
    try:
        report = executor.execute(plan_for(task), verify=True)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the benchmark
        return {
            "task": task.task_id, "category": task.category.value,
            "error": f"{type(exc).__name__}: {exc}"[:160], "accepted": False,
            "completed": False, "repairs": 0, "model_calls": 0, "det": 0, "mod": 0,
            "seconds": 0.0, "merged_ok": False, "false_block": False,
        }
    elapsed = time.monotonic() - started
    item = report.executions[0]
    quality = item.quality_report
    findings = quality.findings if quality else ()
    accepted = acceptance(root, task) if item.ok else False
    # Do not trust the MERGED ledger state: check the file is really in main (M6.2).
    merged_ok = (root / task.path).is_file() if item.ok else True

    false_block = False
    if not item.ok and "quality review rejected" in (item.detail or ""):
        worktree = root / ".edith" / "worktrees" / f"task-{plan_id(task).lower()}" / task.path
        if worktree.is_file():
            (root / task.path).parent.mkdir(parents=True, exist_ok=True)
            (root / task.path).write_bytes(worktree.read_bytes())
            false_block = acceptance(root, task, "_fb")

    return {
        "task": task.task_id, "category": task.category.value, "accepted": accepted,
        "completed": item.ok, "repairs": item.repair_attempts,
        "model_calls": item.model_calls,
        "det": len([f for f in findings if f.origin.value == "DETERMINISTIC"]),
        "mod": len([f for f in findings if f.origin.value == "MODEL"]),
        "seconds": round(elapsed, 1), "merged_ok": merged_ok,
        "false_block": false_block, "outcome": str(item.outcome),
    }


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = load_config(None)
    provider = OllamaProvider(
        base.models.ollama, base.models.profiles[base.models.default_profile]
    )
    results: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for trial in range(trials):
        for arm, review in (("A", False), ("B", True)):
            for task in TASKS:
                row = run_one(task, review, provider)
                row["arm"] = arm
                row["trial"] = trial
                results[arm].append(row)
                print(
                    f"  {arm}{trial} {row['task']}: acc={row['accepted']} "
                    f"rep={row['repairs']} det={row['det']} mod={row['mod']} "
                    f"merge={row['merged_ok']} {row['seconds']}s",
                    flush=True,
                )

    print("\n=== M7 RESULTS ===")
    for arm in ("A", "B"):
        rows = results[arm]
        print(
            json.dumps(
                {
                    "arm": arm, "runs": len(rows),
                    "completed": sum(r["completed"] for r in rows),
                    "accepted": sum(r["accepted"] for r in rows),
                    "repairs": sum(r["repairs"] for r in rows),
                    "model_calls": sum(r["model_calls"] for r in rows),
                    "det_findings": sum(r["det"] for r in rows),
                    "model_findings": sum(r["mod"] for r in rows),
                    "false_block": sum(r["false_block"] for r in rows),
                    "merge_ok": sum(r["merged_ok"] for r in rows),
                    "seconds": round(sum(r["seconds"] for r in rows), 1),
                }
            )
        )
    print("\n=== BY CATEGORY (accepted / runs) ===")
    for category in Category:
        parts = [category.value]
        for arm in ("A", "B"):
            rows = [r for r in results[arm] if r["category"] == category.value]
            parts.append(f"{arm}={sum(r['accepted'] for r in rows)}/{len(rows)}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
