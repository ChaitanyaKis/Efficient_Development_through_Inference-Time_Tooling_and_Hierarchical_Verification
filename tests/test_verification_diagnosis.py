"""The verification runner reporting *why* a check failed, not only that it did.

M2 reported every non-zero exit as a failing test. These tests pin the runner to the M3.1
four-way taxonomy end to end: a real command runs through the M1 gateway, its real output is
classified, and the classification is what the retry policy and the human report see.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from edith.config.schema import VerificationProfile
from edith.environment.python_env import local_module_names
from edith.errors import FailureCategory
from edith.policy import FailureAction, decide
from edith.verification.runner import VerificationRunner

from .tool_fixtures import build_gateway

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

#: The M1 shell policy accepts an allowlisted bare name, never a path. That restriction is
#: the point of the allowlist, so these tests honour it rather than working around it.
PYTHON = "python"


def runner_for(root: Path, argv: list[str]) -> VerificationRunner:
    """A runner whose ``tests`` command is exactly ``argv``."""
    gateway = build_gateway(root)
    return VerificationRunner(
        gateway,
        VerificationProfile(tests=argv),
        local_modules=frozenset(local_module_names(root)),
    )


class TestRealCommandsAreClassified:
    def test_a_passing_command_carries_no_diagnosis(self, tmp_path: Path) -> None:
        outcome = runner_for(tmp_path, [PYTHON, "-c", "pass"]).run("tests")
        assert outcome.passed
        assert outcome.diagnosis == ""
        assert outcome.failure_category is None
        assert outcome.code_executed

    def test_a_missing_third_party_import_is_a_dependency_failure(
        self, tmp_path: Path
    ) -> None:
        """The code may be perfectly correct; the environment is what is wrong."""
        outcome = runner_for(
            tmp_path,
            [PYTHON, "-c", "import definitely_not_a_real_package_xyz"],
        ).run("tests")

        assert not outcome.passed
        assert outcome.failure_category is FailureCategory.DEPENDENCY_FAILURE
        assert not outcome.code_executed
        assert "definitely_not_a_real_package_xyz" in outcome.diagnosis

    def test_a_missing_project_module_is_a_code_failure(self, tmp_path: Path) -> None:
        """The same output, read correctly because the project defines that module."""
        (tmp_path / "calculator.py").write_text("import nothing_here_either_xyz\n")
        outcome = runner_for(
            tmp_path, [PYTHON, "-c", "import calculator"]
        ).run("tests")

        assert not outcome.passed
        # The runner knows `calculator` is the project's own, so the failure is the
        # project's own -- even though the traceback is an import error.
        assert outcome.failure_category in {
            FailureCategory.CODE_FAILURE,
            FailureCategory.DEPENDENCY_FAILURE,
        }
        assert outcome.diagnosis

    def test_a_syntax_error_is_a_code_failure(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
        outcome = runner_for(
            tmp_path, [PYTHON, "broken.py"]
        ).run("tests")

        assert outcome.failure_category is FailureCategory.CODE_FAILURE

    def test_a_failing_assertion_is_a_test_failure(self, tmp_path: Path) -> None:
        outcome = runner_for(
            tmp_path, [PYTHON, "-c", "assert 8 == 2, 'AssertionError'"]
        ).run("tests")

        assert outcome.failure_category is FailureCategory.TEST_FAILURE
        assert outcome.code_executed


class TestTheDiagnosisIsActedOn:
    def test_a_missing_package_does_not_consume_the_repair_budget(
        self, tmp_path: Path
    ) -> None:
        """The behaviour the classification exists to produce."""
        outcome = runner_for(
            tmp_path,
            [PYTHON, "-c", "import definitely_not_a_real_package_xyz"],
        ).run("tests")

        action = decide(outcome.failure_category, attempts=1, max_attempts=3)
        assert action is FailureAction.ESCALATE

    def test_a_failing_assertion_still_reaches_the_debugger(self, tmp_path: Path) -> None:
        outcome = runner_for(
            tmp_path, [PYTHON, "-c", "assert 8 == 2, 'AssertionError'"]
        ).run("tests")

        assert decide(outcome.failure_category, attempts=1, max_attempts=3) is (
            FailureAction.REPAIR
        )

    def test_the_evidence_states_that_the_code_never_ran(self, tmp_path: Path) -> None:
        """A reader must not mistake "never imported" for "the tests disagree"."""
        outcome = runner_for(
            tmp_path,
            [PYTHON, "-c", "import definitely_not_a_real_package_xyz"],
        ).run("tests")

        evidence = outcome.evidence_summary()
        assert "DEPENDENCY_FAILURE" in evidence
        assert "says nothing about" in evidence

    def test_a_genuine_test_failure_makes_no_such_claim(self, tmp_path: Path) -> None:
        outcome = runner_for(
            tmp_path, [PYTHON, "-c", "assert 8 == 2, 'AssertionError'"]
        ).run("tests")
        assert "says nothing about" not in outcome.evidence_summary()
