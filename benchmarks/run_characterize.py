"""M11: reproduce the remaining semantic failures and preserve the evidence.

M10 closed the boundary defect. What survives is SEM-002 and SEM-003, and this milestone is
characterisation only -- no mechanism is added, because four milestones of evidence say the
expensive mistake is building a fix for a cause nobody has looked at.

So this harness collects artifacts rather than verdicts. For every run it preserves the exact
description handed to the coder, the implementation that came back, the acceptance output, and
the same for every repair attempt. Successes are preserved too: an explanation built only from
failures is a story, and comparing a passing implementation of the same task against a failing
one is the only way to see what actually differs.

Classification happens afterwards, by hand, from these artifacts. Nothing here reads the
acceptance test before the implementation exists, and nothing feeds acceptance back into the
coder.

Usage::

    python benchmarks/run_characterize.py [trials] [task-id-prefix]
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

from benchmarks.run_boundary import augment
from benchmarks.run_semantic import build_config, plan_for, prepare
from benchmarks.semantic import TASKS, BenchmarkTask

from edith.config.loader import load_config
from edith.engineering.executor import EngineeringExecutor
from edith.models.ollama import OllamaProvider
from edith.workspaces import ProjectWorkspace

#: Where captured artifacts land. Kept out of the repository.
CAPTURE_ROOT = Path(tempfile.gettempdir()) / "m11-capture"


def acceptance_detail(root: Path, task: BenchmarkTask) -> tuple[bool, str]:
    """Run the hand-written acceptance test and keep its output verbatim.

    Called only after the implementation exists and has been merged, so nothing about the
    acceptance test can influence what was written.
    """
    name = task.task_id.lower().replace("-", "_")
    (root / "tests" / f"test_acc_{name}.py").write_text(task.acceptance, encoding="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
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
    return (completed.returncode == 0, (completed.stdout + completed.stderr)[-2500:])


def run_one(task: BenchmarkTask, trial: int, provider: Any) -> dict[str, Any]:
    """One run, with every artifact preserved whether it passes or fails."""
    root = Path(tempfile.mkdtemp(prefix="m11-"))
    prepare(root)
    # M10 boundary analysis stays ON, as the milestone specifies.
    description, boundary_stats = augment(task)

    executor = EngineeringExecutor(
        build_config(root, False),
        ProjectWorkspace(project_id="m11", name="n", root=root),
        provider=provider,
    )
    started = time.monotonic()
    try:
        report = executor.execute(plan_for(task, description), verify=True)
    except Exception as exc:  # noqa: BLE001 - one run must not abort the study
        return {
            "task": task.task_id, "trial": trial,
            "error": f"{type(exc).__name__}: {exc}"[:200], "accepted": False,
        }
    elapsed = time.monotonic() - started
    item = report.executions[0]

    implementation = ""
    merged = root / task.path
    if merged.is_file():
        implementation = merged.read_text(encoding="utf-8")

    accepted, output = (False, "task did not complete")
    if item.ok:
        accepted, output = acceptance_detail(root, task)

    record = {
        "task": task.task_id,
        "category": task.category.value,
        "trial": trial,
        "requirement": task.requirement,
        "description_given_to_coder": description,
        "boundary": boundary_stats,
        "implementation": implementation,
        "scaffold": task.scaffold,
        "acceptance_output": output,
        "accepted": accepted,
        "completed": item.ok,
        "outcome": str(item.outcome),
        "verifier_detail": (item.detail or "")[:1500],
        "repair_attempts": item.repair_attempts,
        "model_calls": item.model_calls,
        "seconds": round(elapsed, 1),
    }

    CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    label = "pass" if accepted else "fail"
    destination = CAPTURE_ROOT / f"{task.task_id}-t{trial}-{label}.json"
    destination.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    prefix = sys.argv[2] if len(sys.argv) > 2 else "SEM"
    tasks = [task for task in TASKS if task.task_id.startswith(prefix)]

    base = load_config(None)
    provider = OllamaProvider(
        base.models.ollama, base.models.profiles[base.models.default_profile]
    )
    records: list[dict[str, Any]] = []
    for trial in range(trials):
        for task in tasks:
            record = run_one(task, trial, provider)
            records.append(record)
            print(
                f"  t{trial} {record['task']}: accepted={record.get('accepted')} "
                f"repairs={record.get('repair_attempts', 0)} "
                f"{record.get('seconds', 0)}s",
                flush=True,
            )

    print("\n=== M11 CAPTURE ===")
    for task in tasks:
        rows = [r for r in records if r["task"] == task.task_id]
        passed = sum(1 for r in rows if r.get("accepted"))
        print(f"  {task.task_id}: {passed}/{len(rows)} accepted")
    print(f"  artifacts: {CAPTURE_ROOT}")
    print(f"  total runs: {len(records)}")
    failures = [r for r in records if not r.get("accepted")]
    print(f"  failures captured: {len(failures)}")


if __name__ == "__main__":
    main()
