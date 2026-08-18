"""M8: requirement-derived tests, and the rules that stop them defining their own correctness.

The generator writes tests before the coder runs, from the requirement alone. That ordering is
the whole point, so it is asserted structurally rather than trusted: :class:`TestGenInput` has
no field through which an implementation could arrive, and a test proves it.

The failure this milestone exists to attack is M7's: EDITH reported COMPLETED three times per
arm on code that independent tests rejected, because the only tests it had were the coder's,
and a coder that misreads a requirement writes tests agreeing with its misreading.

The most dangerous outcome for M8 is not an invalid generated test -- those are caught. It is a
*valid* generated test that passes a wrong implementation, because that adds confidence without
adding information. :class:`TestStrength` measures exactly that, by running generated tests
against a known-correct and a known-incorrect implementation.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from edith.quality.principals import TESTGEN
from edith.quality.testgen import (
    GENERATED_TEST_DIR,
    GeneratedTest,
    InvalidReason,
    ModelTestCase,
    generate_tests,
    module_for,
    render_test,
    validate_test,
)

# Aliased on import: pytest tries to collect any module-level name starting with "Test",
# and warns because these are real classes with constructors rather than test suites.
from edith.quality.testgen import TestCaseSet as CaseSet
from edith.quality.testgen import TestGeneratorAgent as GeneratorAgent
from edith.quality.testgen import TestGenInput as GenInput
from edith.quality.testgen import TestProvenance as Provenance
from edith.schemas.agent import AgentPermissions
from edith.tools.schemas import ToolCall

from .tool_fixtures import build_gateway

KNOWN = frozenset({"REQ-001"})
MODULE = "src.backend.calc"


def rendered(body: str, *, name: str = "adds") -> str:
    return render_test(
        ModelTestCase(name=name, intent="checks addition", body=body),
        requirement_id="REQ-001",
        module=MODULE,
    )


def validate(body: str) -> InvalidReason | None:
    return validate_test(
        rendered(body), requirement_id="REQ-001", module=MODULE, known_requirements=KNOWN
    )


class TestTheGeneratorCannotSeeTheImplementation:
    """The anti-circularity rule, enforced by the type rather than by convention."""

    def test_the_input_has_no_field_for_source_code(self) -> None:
        fields = set(GenInput.model_fields)
        assert fields == {"requirement_id", "requirement", "acceptance_criteria", "module"}
        assert not fields & {"source", "diff", "implementation", "code", "files"}

    def test_generate_tests_accepts_no_implementation_argument(self) -> None:
        """A future caller cannot pass source in without changing the signature first."""
        source = Path("src/edith/quality/testgen.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "generate_tests"
        )
        names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        assert not names & {"source", "diff", "implementation", "code", "changed_files"}

    def test_the_prompt_tells_the_model_it_has_not_seen_the_code(self) -> None:
        from edith.quality.testgen import TESTGEN_PROMPT

        assert "NOT seen the implementation" in TESTGEN_PROMPT


class TestProvenanceIsSystemOwned:
    def test_the_model_cannot_declare_a_test_authoritative(self) -> None:
        """ModelTestCase has no provenance field, so the abuse is unrepresentable."""
        assert set(ModelTestCase.model_fields) == {"name", "intent", "body"}

    def test_human_acceptance_tests_are_authoritative(self) -> None:
        assert Provenance.HUMAN_ACCEPTANCE_TEST.authoritative

    def test_validated_requirement_derived_tests_are_authoritative(self) -> None:
        assert Provenance.REQUIREMENT_DERIVED_TEST.authoritative

    def test_advisory_tests_are_not(self) -> None:
        assert not Provenance.MODEL_ADVISORY_TEST.authoritative

    def test_an_invalid_test_is_never_authoritative(self) -> None:
        item = GeneratedTest(
            name="x",
            requirement_id="REQ-001",
            module=MODULE,
            source="",
            provenance=Provenance.REQUIREMENT_DERIVED_TEST,
            valid=False,
            reason=InvalidReason.VACUOUS,
        )
        assert not item.authoritative

    def test_the_rendered_header_records_provenance(self) -> None:
        assert Provenance.REQUIREMENT_DERIVED_TEST.value in rendered("assert add(1, 1) == 2")
        assert "REQ-001" in rendered("assert add(1, 1) == 2")


class TestValidation:
    def test_a_good_test_validates(self) -> None:
        assert validate("assert add(2, 3) == 5") is None

    def test_a_vacuous_test_is_refused(self) -> None:
        """The exact shape that would make a generated suite pass any implementation."""
        assert validate("assert True") is InvalidReason.VACUOUS

    def test_a_test_with_no_assertion_is_refused(self) -> None:
        assert validate("add(1, 2)") is InvalidReason.NO_ASSERTION

    def test_broken_syntax_is_refused(self) -> None:
        assert validate("assert add(1,") is InvalidReason.SYNTAX

    def test_an_unknown_requirement_is_refused(self) -> None:
        reason = validate_test(
            rendered("assert add(1, 1) == 2"),
            requirement_id="REQ-999",
            module=MODULE,
            known_requirements=KNOWN,
        )
        assert reason is InvalidReason.UNRESOLVED_REQUIREMENT

    def test_a_test_that_writes_is_refused(self) -> None:
        """A test that mutates state is not a test, and no permission check would see it."""
        assert validate("open('x', 'w')\nassert add(1, 1) == 2") is InvalidReason.WRITES_SOURCE

    def test_a_test_that_imports_the_quality_layer_is_refused(self) -> None:
        source = rendered("assert add(1, 1) == 2").replace(
            "import pytest", "import pytest\nfrom edith.quality.artifacts import QualityVerdict"
        )
        reason = validate_test(
            source, requirement_id="REQ-001", module=MODULE, known_requirements=KNOWN
        )
        assert reason is InvalidReason.IMPORTS_QUALITY_LAYER

    def test_a_test_that_does_not_import_the_target_is_refused(self) -> None:
        reason = validate_test(
            "import pytest\n\n\ndef test_x():\n    assert 1 == 1\n",
            requirement_id="REQ-001",
            module=MODULE,
            known_requirements=KNOWN,
        )
        assert reason is InvalidReason.DOES_NOT_IMPORT_TARGET

    def test_a_traversing_path_is_refused(self) -> None:
        reason = validate("assert add(1, 1) == 2  # ../../etc")
        assert reason is InvalidReason.ESCAPES_TEST_WORKSPACE

    def test_a_test_with_no_test_function_is_refused(self) -> None:
        reason = validate_test(
            f"from {MODULE} import add\nassert add(1, 1) == 2\n",
            requirement_id="REQ-001",
            module=MODULE,
            known_requirements=KNOWN,
        )
        assert reason is InvalidReason.NO_TEST_FUNCTION


class TestTheGeneratorPrincipal:
    def test_it_cannot_write_implementation(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "calc.py").write_text("VALUE = 1\n", encoding="utf-8")
        gateway = build_gateway(tmp_path, TESTGEN)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/calc.py", "content": "VALUE = 2\n"},
            )
        )
        assert not result.ok
        assert (tmp_path / "src" / "calc.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_it_cannot_overwrite_a_human_acceptance_test(self, tmp_path: Path) -> None:
        """The leakage that would make the whole experiment meaningless."""
        (tmp_path / "tests").mkdir()
        original = "def test_acc():\n    assert add(2, 3) == 5\n"
        (tmp_path / "tests" / "test_acceptance.py").write_text(original, encoding="utf-8")
        gateway = build_gateway(tmp_path, TESTGEN)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "tests/test_acceptance.py", "content": "def test_acc(): pass\n"},
            )
        )
        assert not result.ok
        assert (tmp_path / "tests" / "test_acceptance.py").read_text(encoding="utf-8") == original

    def test_it_may_write_inside_the_generated_directory(self, tmp_path: Path) -> None:
        (tmp_path / GENERATED_TEST_DIR).mkdir(parents=True)
        gateway = build_gateway(tmp_path, TESTGEN)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={
                    "path": f"{GENERATED_TEST_DIR}/test_req_001.py",
                    "content": "def test_x():\n    assert 1 == 1\n",
                },
            )
        )
        assert result.ok, result.error

    def test_it_holds_no_shell(self) -> None:
        """It cannot run the suite it wrote and report on its own work."""
        assert "shell.run" not in TESTGEN.allowed_tools

    def test_it_holds_no_git(self) -> None:
        assert not {tool for tool in TESTGEN.allowed_tools if tool.startswith("git.")}


class TestGenerationFailsSafe:
    def test_a_generator_that_raises_yields_no_tests(self) -> None:
        """No tests is a measurable outcome; an exception would abort an unrelated task."""

        class _Broken(GeneratorAgent):
            def execute(self, request: object) -> object:  # type: ignore[override]
                raise RuntimeError("ollama is down")

        assert generate_tests(
            _Broken(), requirement_id="REQ-001", requirement="add two numbers", module=MODULE
        ) == ()

    def test_an_invalid_case_is_downgraded_not_dropped(self) -> None:
        """It is recorded as advisory, so the failure rate stays measurable."""

        class _Vacuous(GeneratorAgent):
            def execute(self, request: object) -> object:  # type: ignore[override]
                class _Response:
                    ok = True
                    output = CaseSet(
                        cases=[ModelTestCase(name="x", intent="i", body="assert True")]
                    ).model_dump()

                return _Response()

        results = generate_tests(
            _Vacuous(), requirement_id="REQ-001", requirement="add", module=MODULE
        )
        assert len(results) == 1
        assert not results[0].valid
        assert results[0].provenance is Provenance.MODEL_ADVISORY_TEST
        assert results[0].reason is InvalidReason.VACUOUS


class TestStrength:
    """Does a generated test distinguish a correct implementation from a wrong one?

    A valid test that passes both is the M8 blind spot: it adds confidence without adding
    information, which is precisely how M7's false PASSes happened. Measured by execution, not
    by inspection, and the incorrect implementation is never shown to the generator.
    """

    CORRECT = "def add(a, b):\n    return a + b\n"
    INCORRECT = "def add(a, b):\n    return a - b\n"

    def run_against(self, tmp_path: Path, implementation: str, test_source: str) -> bool:
        root = tmp_path / f"probe{abs(hash(implementation)) % 1000}"
        (root / "src" / "backend").mkdir(parents=True)
        for package in ("src", "src/backend"):
            (root / package / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "backend" / "calc.py").write_text(implementation, encoding="utf-8")
        (root / GENERATED_TEST_DIR).mkdir(parents=True)
        (root / GENERATED_TEST_DIR / "test_gen.py").write_text(test_source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", f"{GENERATED_TEST_DIR}/test_gen.py", "-q"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return completed.returncode == 0

    def test_a_strong_test_passes_correct_and_fails_incorrect(self, tmp_path: Path) -> None:
        source = rendered("assert add(2, 3) == 5")
        assert self.run_against(tmp_path, self.CORRECT, source)
        assert not self.run_against(tmp_path, self.INCORRECT, source)

    def test_a_weak_test_passes_both_and_is_therefore_useless(self, tmp_path: Path) -> None:
        """Not a bug in the runner -- the point of measuring strength by execution."""
        source = rendered("assert add(0, 0) == 0")
        assert self.run_against(tmp_path, self.CORRECT, source)
        assert self.run_against(tmp_path, self.INCORRECT, source)

    def test_a_vacuous_test_never_reaches_execution(self) -> None:
        """Validation refuses it first, so strength never has to catch this case."""
        assert validate("assert True") is InvalidReason.VACUOUS


class TestModuleNaming:
    def test_a_path_becomes_a_dotted_module(self) -> None:
        assert module_for("src/backend/calc.py") == "src.backend.calc"

    def test_windows_separators_are_handled(self) -> None:
        assert module_for("src\\backend\\calc.py") == "src.backend.calc"


class TestNoLeakageOfHiddenAcceptance:
    def test_the_benchmark_acceptance_is_never_passed_to_the_generator(self) -> None:
        """The generator receives the requirement; the assertions stay hidden."""
        from benchmarks.semantic import TASKS

        task = TASKS[0]
        payload = GenInput(
            requirement_id=task.task_id,
            requirement=task.requirement,
            module=module_for(task.path),
        )
        rendered_payload = payload.model_dump_json()
        for line in task.acceptance.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert "):
                assert stripped not in rendered_payload

    def test_an_unscoped_principal_cannot_write_generated_tests(self, tmp_path: Path) -> None:
        gateway = build_gateway(tmp_path, AgentPermissions(allowed_tools=frozenset()))
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": f"{GENERATED_TEST_DIR}/test_x.py", "content": "x"},
            )
        )
        assert not result.ok


def test_generated_tests_do_not_participate_when_invalid() -> None:
    """A green invalid suite must never count towards the verification contract."""
    invalid = GeneratedTest(
        name="x",
        requirement_id="REQ-001",
        module=MODULE,
        source="",
        provenance=Provenance.MODEL_ADVISORY_TEST,
        valid=False,
    )
    assert not invalid.authoritative
    with pytest.raises(AssertionError):
        assert invalid.authoritative
