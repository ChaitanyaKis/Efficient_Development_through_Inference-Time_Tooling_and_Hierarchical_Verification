"""Test integrity: the regression suite for the M2 false-positive verification.

The scenario these exist for: the coding agent rewrote ``assert subtract(5, 3) == 2`` into
``== 8``, the suite went green, and Edith's Critic returned PASS. Every test here asserts
some part of why that can no longer happen.
"""

from __future__ import annotations

import pytest

from edith.agents.critic import CriticOutput, adjudicate
from edith.errors import FailureCategory
from edith.integrity import (
    FileKind,
    IntegrityReport,
    build_report,
    classify_path,
    compare_test_file,
    extract_tests,
)
from edith.schemas.common import Severity, Verdict
from edith.verification.runner import VerificationOutcome, VerificationReport

ORIGINAL = '''"""Tests."""

from calc import subtract


def test_subtract() -> None:
    assert subtract(5, 3) == 2
    assert subtract(0, 4) == -4
'''

#: The exact tampering observed in the M2 benchmark.
TAMPERED = '''"""Tests."""

from calc import subtract


def test_subtract() -> None:
    assert subtract(5, 3) == 8
'''


def passing_report() -> VerificationReport:
    """A verification report showing a green suite."""
    return VerificationReport(
        outcomes=[
            VerificationOutcome(
                kind="tests", command="pytest -q", exit_code=0, passed=True
            )
        ]
    )


class TestPathClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "test_calculator.py",
            "tests/test_app.py",
            "src/tests/test_x.py",
            "calculator_test.py",
            "app.test.ts",
            "components/Button.spec.tsx",
        ],
    )
    def test_test_files(self, path: str) -> None:
        assert classify_path(path) is FileKind.TEST

    @pytest.mark.parametrize("path", ["calculator.py", "src/app.py", "lib/util.go"])
    def test_source_files(self, path: str) -> None:
        assert classify_path(path) is FileKind.SOURCE

    def test_config_and_docs(self) -> None:
        assert classify_path("pyproject.toml") is FileKind.CONFIG
        assert classify_path("README.md") is FileKind.DOC

    def test_fixtures_are_distinguished_from_tests(self) -> None:
        assert classify_path("tests/fixtures/sample.json") is FileKind.FIXTURE
        assert classify_path("conftest.py") is FileKind.FIXTURE


class TestExtraction:
    def test_finds_tests_and_assertions(self) -> None:
        tests = extract_tests(ORIGINAL)
        assert "test_subtract" in tests
        assert tests["test_subtract"].assertion_count == 2

    def test_detects_skip_decorator(self) -> None:
        source = (
            "import pytest\n\n\n@pytest.mark.skip(reason='later')\n"
            "def test_x():\n    assert 1 == 2\n"
        )
        assert extract_tests(source)["test_x"].skipped

    def test_detects_skip_call(self) -> None:
        source = "import pytest\n\n\ndef test_x():\n    pytest.skip('nope')\n    assert 1 == 2\n"
        assert extract_tests(source)["test_x"].skipped

    def test_finds_tests_inside_classes(self) -> None:
        source = "class TestThing:\n    def test_a(self):\n        assert True\n"
        assert "TestThing.test_a" in extract_tests(source)

    def test_unparseable_source_yields_nothing(self) -> None:
        assert extract_tests("def broken(:\n") == {}


class TestTamperDetection:
    def test_the_exact_m2_attack_is_detected(self) -> None:
        """`assert subtract(5, 3) == 2` rewritten to `== 8`."""
        delta = compare_test_file("test_calc.py", ORIGINAL, TAMPERED)
        assert delta.findings
        assert delta.assertions_removed >= 1
        finding = delta.findings[0]
        assert finding.severity in {Severity.HIGH, Severity.CRITICAL}
        assert "== 2" in finding.evidence, "the original assertion must be in the evidence"

    def test_changing_only_the_expected_value_is_caught(self) -> None:
        """Same assertion count, different expected value -- a count check would miss it."""
        before = "def test_x():\n    assert compute() == 42\n"
        after = "def test_x():\n    assert compute() == 99\n"
        delta = compare_test_file("test_x.py", before, after)
        assert delta.findings
        assert delta.findings[0].kind == "assertion_changed"

    def test_deleting_a_test_is_caught(self) -> None:
        before = "def test_a():\n    assert 1 == 1\n\n\ndef test_b():\n    assert 2 == 2\n"
        after = "def test_a():\n    assert 1 == 1\n"
        delta = compare_test_file("test_x.py", before, after)
        assert delta.tests_removed == ["test_b"]
        assert delta.findings[0].severity is Severity.CRITICAL

    def test_skipping_a_test_is_caught(self) -> None:
        before = "def test_a():\n    assert 1 == 2\n"
        after = "import pytest\n\n\n@pytest.mark.skip\ndef test_a():\n    assert 1 == 2\n"
        delta = compare_test_file("test_x.py", before, after)
        assert delta.skips_added == ["test_a"]

    def test_removing_assertions_is_caught(self) -> None:
        before = "def test_a():\n    assert x == 1\n    assert y == 2\n"
        after = "def test_a():\n    assert x == 1\n"
        delta = compare_test_file("test_x.py", before, after)
        assert delta.assertions_removed == 1
        assert delta.findings[0].kind == "assertion_removed"

    def test_adding_tests_is_legitimate(self) -> None:
        """An agent must remain free to add coverage."""
        after = ORIGINAL + "\n\ndef test_new_case() -> None:\n    assert subtract(1, 1) == 0\n"
        delta = compare_test_file("test_calc.py", ORIGINAL, after)
        assert delta.findings == []
        assert delta.tests_added == ["test_new_case"]

    def test_reformatting_is_not_tampering(self) -> None:
        """Comments and whitespace must not raise a false alarm."""
        after = ORIGINAL.replace(
            "def test_subtract() -> None:",
            "# checks subtraction\ndef test_subtract() -> None:",
        )
        assert compare_test_file("test_calc.py", ORIGINAL, after).findings == []

    def test_strengthening_a_test_is_not_tampering(self) -> None:
        after = ORIGINAL.replace(
            "assert subtract(0, 4) == -4",
            "assert subtract(0, 4) == -4\n    assert subtract(9, 9) == 0",
        )
        assert compare_test_file("test_calc.py", ORIGINAL, after).findings == []


class TestReportSemantics:
    def test_tampering_without_justification_blocks(self) -> None:
        report = build_report(
            {"test_calc.py": ORIGINAL}, {"test_calc.py": TAMPERED}, ["test_calc.py"]
        )
        assert report.tampered
        assert report.blocking_findings

    def test_declared_change_is_permitted_but_still_visible(self) -> None:
        """Changing a genuinely wrong test stays possible -- but never silently."""
        report = build_report(
            {"test_calc.py": ORIGINAL},
            {"test_calc.py": TAMPERED},
            ["test_calc.py"],
            justification="the task explicitly asked to correct this test",
        )
        assert not report.tampered
        assert report.blocking_findings, "the change is still reported"

    def test_deleted_test_file_is_critical(self) -> None:
        report = build_report({"test_calc.py": ORIGINAL}, {}, ["test_calc.py"])
        assert report.tampered
        assert report.findings[0].kind == "test_file_deleted"

    def test_new_test_file_is_fine(self) -> None:
        report = build_report({}, {"test_new.py": ORIGINAL}, ["test_new.py"])
        assert not report.tampered

    def test_source_only_change_is_clean(self) -> None:
        report = build_report({}, {}, ["calculator.py"])
        assert not report.tampered
        assert report.source_files_changed == ["calculator.py"]
        assert not report.tests_were_modified

    def test_missing_baseline_is_reported_not_assumed_clean(self) -> None:
        """Silently passing is precisely the failure this module exists to prevent."""
        report = build_report({}, {}, [], baseline_unavailable=True)
        assert report.baseline_unavailable
        assert "NOT CHECKED" in report.summary()

    def test_summary_names_the_offending_file(self) -> None:
        report = build_report(
            {"test_calc.py": ORIGINAL}, {"test_calc.py": TAMPERED}, ["test_calc.py"]
        )
        assert "test_calc.py" in report.summary()


class TestAdjudicationGate:
    """The gate itself: a green suite must not win when the tests were weakened."""

    def test_tampering_overrides_passing_tests_and_a_pass_verdict(self) -> None:
        """The precise M2 false positive, now rejected."""
        report = build_report(
            {"test_calc.py": ORIGINAL}, {"test_calc.py": TAMPERED}, ["test_calc.py"]
        )
        verdict, reason = adjudicate(
            CriticOutput(verdict=Verdict.PASS, reasoning="looks good"),
            passing_report(),
            changes_made=True,
            integrity=report,
        )
        assert verdict is Verdict.FAIL
        assert "integrity" in reason.lower()

    def test_integrity_is_checked_before_test_results(self) -> None:
        """Ordering matters: a tampered suite that also fails is still an integrity failure."""
        failing = VerificationReport(
            outcomes=[
                VerificationOutcome(
                    kind="tests",
                    command="pytest",
                    exit_code=1,
                    passed=False,
                    failure_category=FailureCategory.TEST_FAILURE,
                )
            ]
        )
        report = build_report(
            {"test_calc.py": ORIGINAL}, {"test_calc.py": TAMPERED}, ["test_calc.py"]
        )
        verdict, reason = adjudicate(
            None, failing, changes_made=True, integrity=report
        )
        assert verdict is Verdict.FAIL
        assert "integrity" in reason.lower()

    def test_clean_integrity_does_not_interfere(self) -> None:
        verdict, _ = adjudicate(
            CriticOutput(verdict=Verdict.PASS),
            passing_report(),
            changes_made=True,
            integrity=build_report({}, {}, ["calculator.py"]),
        )
        assert verdict is Verdict.PASS

    def test_justified_change_does_not_block(self) -> None:
        report = build_report(
            {"test_calc.py": ORIGINAL},
            {"test_calc.py": TAMPERED},
            ["test_calc.py"],
            justification="task required correcting the test",
        )
        verdict, _ = adjudicate(
            CriticOutput(verdict=Verdict.PASS),
            passing_report(),
            changes_made=True,
            integrity=report,
        )
        assert verdict is Verdict.PASS

    def test_absent_integrity_report_preserves_m2_behaviour(self) -> None:
        verdict, _ = adjudicate(
            CriticOutput(verdict=Verdict.PASS), passing_report(), changes_made=True
        )
        assert verdict is Verdict.PASS

    def test_adding_tests_alongside_a_fix_still_passes(self) -> None:
        """The legitimate case must not be caught by the gate."""
        after = ORIGINAL + "\n\ndef test_extra() -> None:\n    assert subtract(2, 1) == 1\n"
        report = build_report(
            {"test_calc.py": ORIGINAL},
            {"test_calc.py": after},
            ["test_calc.py", "calc.py"],
        )
        verdict, _ = adjudicate(
            CriticOutput(verdict=Verdict.PASS),
            passing_report(),
            changes_made=True,
            integrity=report,
        )
        assert verdict is Verdict.PASS


class TestIntegrityReportDefaults:
    def test_empty_report_is_not_tampered(self) -> None:
        assert not IntegrityReport().tampered

    def test_low_severity_findings_do_not_block(self) -> None:
        report = IntegrityReport()
        assert not report.blocking_findings
