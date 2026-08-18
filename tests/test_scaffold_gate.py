"""M9: the scaffold gate, and the oracles it depends on.

M8 produced test suites that were 98% mechanically valid, killed 12/12 mutants, and passed a
known-correct implementation only 4 times in 12. They were not discriminating; they were
failing nearly everything, and turned loose they blocked 32 of 36 runs.

The gate is the cheapest available correction: a test asserting something a correct
implementation does not satisfy is asserting the wrong thing, and that is decidable by
execution with no model involved.

Two properties are load-bearing and tested here rather than assumed.

**The gate is generic.** It receives the scaffold as an argument and contains no task
identifier, requirement text, or expected value. A gate that recognised individual benchmark
tasks would be measuring its own hard-coding.

**The oracles are correct.** Every scaffold must pass its task's hand-written acceptance test,
and every mutant must fail it. A wrong scaffold would make the gate discard good tests and
invert the whole experiment, so each is verified by execution, per task.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.semantic import TASKS, BenchmarkTask

from edith.quality.testgate import (
    GateOutcome,
    SuiteVerdict,
    gate_tests,
)
from edith.quality.testgen import GeneratedTest, InvalidReason, module_for

# Aliased: pytest tries to collect any module-level name starting with "Test".
from edith.quality.testgen import TestProvenance as Provenance

MODULE = "src.backend.calc"
PATH = "src/backend/calc.py"
CORRECT = "def add(a, b):\n    return a + b\n"


def make(source: str, *, valid: bool = True) -> GeneratedTest:
    return GeneratedTest(
        name="t",
        requirement_id="REQ-001",
        module=MODULE,
        source=source,
        provenance=(
            Provenance.REQUIREMENT_DERIVED_TEST
            if valid
            else Provenance.MODEL_ADVISORY_TEST
        ),
        valid=valid,
        reason=None if valid else InvalidReason.VACUOUS,
    )


def case(body: str) -> str:
    return f"from {MODULE} import add\n\n\ndef test_case():\n    {body}\n"


class TestTheGateIsGeneric:
    """No task may be recognised by name, requirement, or expected value."""

    def test_the_module_contains_no_task_identifier(self) -> None:
        source = Path("src/edith/quality/testgate.py").read_text(encoding="utf-8")
        for task in TASKS:
            assert task.task_id not in source

    def test_the_module_does_not_branch_on_content(self) -> None:
        source = Path("src/edith/quality/testgate.py").read_text(encoding="utf-8")
        for marker in ("requirement ==", "task_id ==", "expected ==", "SEM-", "BIZ-"):
            assert marker not in source

    def test_the_scaffold_arrives_as_an_argument(self) -> None:
        tree = ast.parse(Path("src/edith/quality/testgate.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "gate_tests"
        )
        names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        assert "scaffold" in names


class TestTheOraclesAreCorrect:
    """A wrong scaffold would make the gate discard good tests and invert the result."""

    def run_acceptance(
        self, task: BenchmarkTask, implementation: str, tmp_path: Path
    ) -> bool:
        root = tmp_path / task.task_id.replace("-", "_") / str(abs(hash(implementation)) % 997)
        target = root / task.path
        target.parent.mkdir(parents=True, exist_ok=True)
        current = root
        for part in Path(task.path).parent.parts:
            current = current / part
            (current / "__init__.py").write_text("", encoding="utf-8")
        target.write_text(implementation, encoding="utf-8")
        (root / "tests").mkdir(parents=True, exist_ok=True)
        (root / "tests" / "test_acc.py").write_text(task.acceptance, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_acc.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return completed.returncode == 0

    @pytest.mark.parametrize("task", TASKS, ids=lambda task: task.task_id)
    def test_the_scaffold_passes_its_acceptance_test(
        self, task: BenchmarkTask, tmp_path: Path
    ) -> None:
        assert self.run_acceptance(task, task.scaffold, tmp_path), (
            f"{task.task_id}: the scaffold does not satisfy its own requirement"
        )

    @pytest.mark.parametrize("task", TASKS, ids=lambda task: task.task_id)
    def test_the_mutant_fails_its_acceptance_test(
        self, task: BenchmarkTask, tmp_path: Path
    ) -> None:
        assert not self.run_acceptance(task, task.mutant, tmp_path), (
            f"{task.task_id}: the mutant is not actually wrong"
        )

    def test_every_task_carries_both_oracles(self) -> None:
        for task in TASKS:
            assert task.scaffold.strip()
            assert task.mutant.strip()
            assert task.scaffold != task.mutant


class TestTheGateDecides:
    def test_a_test_the_scaffold_satisfies_is_retained(self) -> None:
        verdict = gate_tests(
            (make(case("assert add(2, 3) == 5")),), scaffold=CORRECT, module_path=PATH
        )
        assert verdict.usable
        assert verdict.count(GateOutcome.RETAINED) == 1

    def test_a_test_the_scaffold_contradicts_is_discarded(self) -> None:
        """The M8 failure mode: a valid test asserting the wrong expectation."""
        verdict = gate_tests(
            (make(case("assert add(2, 3) == 6")),), scaffold=CORRECT, module_path=PATH
        )
        assert not verdict.usable
        assert verdict.count(GateOutcome.CONTRADICTS_SCAFFOLD) == 1

    def test_a_test_that_cannot_run_is_distinguished_from_one_that_disagrees(self) -> None:
        """A collection error says nothing about the requirement; a failure does."""
        broken = make(
            f"from {MODULE} import missing\n\n\ndef test_case():\n    assert missing() == 1\n"
        )
        verdict = gate_tests((broken,), scaffold=CORRECT, module_path=PATH)
        assert verdict.count(GateOutcome.DID_NOT_EXECUTE) == 1
        assert verdict.count(GateOutcome.CONTRADICTS_SCAFFOLD) == 0

    def test_gating_is_per_test_not_per_suite(self) -> None:
        """M8 condemned three good tests for one bad one. The survivors are kept."""
        verdict = gate_tests(
            (
                make(case("assert add(2, 3) == 5")),
                make(case("assert add(2, 3) == 99")),
                make(case("assert add(0, 0) == 0")),
            ),
            scaffold=CORRECT,
            module_path=PATH,
        )
        assert len(verdict.retained) == 2
        assert verdict.count(GateOutcome.CONTRADICTS_SCAFFOLD) == 1

    def test_an_already_invalid_test_is_carried_through_not_rerun(self) -> None:
        """The two rejection stages stay separately measurable."""
        verdict = gate_tests((make("", valid=False),), scaffold=CORRECT, module_path=PATH)
        assert verdict.count(GateOutcome.ALREADY_INVALID) == 1
        assert not verdict.usable

    def test_an_empty_verdict_is_not_usable(self) -> None:
        assert not SuiteVerdict().usable

    def test_a_discarded_test_is_never_authoritative(self) -> None:
        verdict = gate_tests(
            (make(case("assert add(1, 1) == 3")),), scaffold=CORRECT, module_path=PATH
        )
        assert all(not item.authoritative for item in verdict.gated)


class TestTheProbeIsIsolated:
    def test_the_gate_never_sees_the_mutant(self) -> None:
        tree = ast.parse(Path("src/edith/quality/testgate.py").read_text(encoding="utf-8"))
        names = {
            arg.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for arg in node.args.args + node.args.kwonlyargs
        }
        assert "mutant" not in names
        assert "incorrect" not in names

    def test_the_gate_never_reads_acceptance_tests(self) -> None:
        source = Path("src/edith/quality/testgate.py").read_text(encoding="utf-8")
        assert "task.acceptance" not in source
        assert "acceptance=" not in source

    def test_the_gate_uses_no_model(self) -> None:
        """The scaffold is the only correctness oracle available to it."""
        source = Path("src/edith/quality/testgate.py").read_text(encoding="utf-8")
        for marker in ("Judge", "provider=", "structured_generate", "run_model_review"):
            assert marker not in source

    def test_module_naming_matches_the_generator(self) -> None:
        assert module_for(PATH) == MODULE
