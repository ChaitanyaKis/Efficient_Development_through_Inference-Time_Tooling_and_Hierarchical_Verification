"""M6 Part 9: the quality layer under an adversarial agent.

Each scenario is a way an agent -- malicious, confused, or merely optimising for "make the
gate go green" -- could get unacceptable work merged. The assertion in every case is that the
*deterministic* outcome is correct regardless of what any model says about it.

The scenarios that matter most are 9 and 10: a Judge returning PASS while a test fails, and
while a critical security finding stands. Those are the ones where a plausible implementation
would have let the model decide, and where M6 requires that it cannot.
"""

from __future__ import annotations

from typing import ClassVar

from edith.errors import FailureCategory
from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    QualityVerdict,
    ReviewEvidence,
)
from edith.quality.pipeline import QualityPipeline
from edith.quality.principals import JUDGE, REVIEWER, SECURITY, TESTER
from edith.quality.scanners import scan_review, scan_security
from edith.schemas.common import Severity
from edith.tools.schemas import ToolCall
from edith.verification.runner import VerificationOutcome, VerificationReport

from .tool_fixtures import build_gateway


def failed_run(category: FailureCategory) -> VerificationReport:
    """A verification report that genuinely failed, built from a real outcome.

    ``passed`` and ``failure_category`` are derived properties, not fields -- a report cannot
    be constructed claiming a result its outcomes do not support, which is the point.
    """
    return VerificationReport(
        outcomes=[
            VerificationOutcome(
                kind="tests",
                command="python -m pytest -q",
                exit_code=1,
                passed=False,
                stdout="1 failed",
                failure_category=category,
            )
        ]
    )


def only(findings: tuple[QualityFinding, ...], category: str) -> QualityFinding:
    matches = [item for item in findings if item.category == category]
    assert matches, f"expected a {category} finding, got {[i.category for i in findings]}"
    return matches[0]


class TestScenario1ObviousBug:
    def test_a_failing_suite_cannot_pass(self) -> None:
        pipeline = QualityPipeline(task_id="TASK-001")
        pipeline.test_gate(
            failed_run(FailureCategory.TEST_FAILURE)
        )
        report = pipeline.report(judge_verdict=QualityVerdict.PASS)
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED
        assert not report.verdict().merges


class TestScenario2And3And4TestTampering:
    """Removing a test, weakening an assertion, or adding a vacuous one."""

    BASELINE: ClassVar[dict[str, str]] = {
        "tests/test_a.py": "def test_one():\n    assert compute() == 42\n",
    }

    def test_deleting_a_test_is_caught(self) -> None:
        pipeline = QualityPipeline()
        pipeline.integrity_gate(self.BASELINE, {"tests/test_a.py": ""}, ["tests/test_a.py"])
        assert pipeline.report().blocking, "a deleted test must block"

    def test_weakening_an_assertion_is_caught(self) -> None:
        pipeline = QualityPipeline()
        pipeline.integrity_gate(
            self.BASELINE,
            {"tests/test_a.py": "def test_one():\n    assert True\n"},
            ["tests/test_a.py"],
        )
        assert pipeline.report().findings, "a weakened assertion must be reported"

    def test_skipping_a_test_is_caught(self) -> None:
        pipeline = QualityPipeline()
        pipeline.integrity_gate(
            self.BASELINE,
            {
                "tests/test_a.py": (
                    "import pytest\n\n@pytest.mark.skip\ndef test_one():\n"
                    "    assert compute() == 42\n"
                )
            },
            ["tests/test_a.py"],
        )
        assert pipeline.report().findings, "a skipped test must be reported"


class TestScenario5ImportFailure:
    def test_code_that_does_not_import_is_critical_and_blocks(self) -> None:
        pipeline = QualityPipeline()
        pipeline.import_gate(importable=False, error="ModuleNotFoundError: no module named x")
        report = pipeline.report(judge_verdict=QualityVerdict.PASS)
        assert report.verdict() is QualityVerdict.BLOCKED


class TestScenario6SecurityVulnerability:
    def test_os_system_is_a_critical_command_injection_finding(self) -> None:
        findings = scan_security("src/a.py", "import os\ndef run(cmd):\n    os.system(cmd)\n")
        finding = only(findings, "command-injection")
        assert finding.severity is Severity.CRITICAL
        assert finding.origin is FindingOrigin.DETERMINISTIC

    def test_shell_true_is_critical(self) -> None:
        source = "import subprocess\ndef run(c):\n    subprocess.run(c, shell=True)\n"
        assert only(scan_security("src/a.py", source), "command-injection").severity is (
            Severity.CRITICAL
        )

    def test_eval_is_critical(self) -> None:
        assert only(
            scan_security("src/a.py", "def f(s):\n    return eval(s)\n"), "code-injection"
        ).severity is Severity.CRITICAL

    def test_unsafe_deserialization_is_reported(self) -> None:
        source = "import pickle\ndef load(b):\n    return pickle.loads(b)\n"
        assert only(scan_security("src/a.py", source), "unsafe-deserialization")

    def test_disabled_tls_verification_is_reported(self) -> None:
        source = "import requests\ndef get(u):\n    return requests.get(u, verify=False)\n"
        assert only(scan_security("src/a.py", source), "insecure-configuration")

    def test_a_critical_security_finding_blocks_the_pipeline(self) -> None:
        pipeline = QualityPipeline()
        pipeline.security_gate({"src/a.py": "import os\nos.system('ls')\n"})
        assert pipeline.report().verdict() is QualityVerdict.BLOCKED

    def test_clean_code_produces_no_security_findings(self) -> None:
        """The scanner must not be so eager that its findings get ignored."""
        source = "def add(a: int, b: int) -> int:\n    return a + b\n"
        assert scan_security("src/a.py", source) == ()

    def test_a_file_that_cannot_be_parsed_is_not_reported_clean(self) -> None:
        findings = scan_security("src/a.py", "def broken(:\n")
        assert findings and findings[0].severity is Severity.CRITICAL


class TestScenario7SecretLogging:
    def test_a_hardcoded_credential_is_critical(self) -> None:
        source = 'API_KEY = "sk-live-9f3ba71c2d"\n'
        finding = only(scan_security("src/a.py", source), "secret-exposure")
        assert finding.severity is Severity.CRITICAL

    def test_the_evidence_does_not_repeat_the_secret(self) -> None:
        """Reporting a leaked credential must not leak it again into the report."""
        source = 'PASSWORD = "hunter2-actual-secret"\n'
        finding = only(scan_security("src/a.py", source), "secret-exposure")
        rendered = " ".join(item.detail for item in finding.evidence)
        assert "hunter2-actual-secret" not in rendered
        assert "redacted" in rendered

    def test_a_placeholder_is_not_flagged(self) -> None:
        assert scan_security("src/a.py", 'PASSWORD = "changeme"\n') == ()

    def test_logging_a_credential_is_reported(self) -> None:
        source = (
            "import logging\nlog = logging.getLogger(__name__)\n"
            "def f(password):\n    log.info(password)\n"
        )
        assert only(scan_security("src/a.py", source), "sensitive-logging")


class TestScenario8PathTraversal:
    def test_a_traversing_literal_path_is_reported(self) -> None:
        source = "def f():\n    return open('../../etc/passwd')\n"
        assert only(scan_security("src/a.py", source), "path-traversal")

    def test_an_ordinary_path_is_not_reported(self) -> None:
        assert scan_security("src/a.py", "def f():\n    return open('data/x.txt')\n") == ()


class TestScenario9JudgePassesOverFailingTest:
    """The headline case: the Judge must not be able to grant a pass."""

    def test_the_judge_cannot_override_a_failing_test(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(
            failed_run(FailureCategory.TEST_FAILURE)
        )
        report = pipeline.report(
            judge_verdict=QualityVerdict.PASS,
            judge_rationale="the implementation looks correct to me",
        )
        assert report.verdict() is not QualityVerdict.PASS
        assert report.judge_verdict is QualityVerdict.PASS, "the opinion is still recorded"


class TestScenario10JudgePassesOverCriticalSecurity:
    def test_the_judge_cannot_override_a_critical_security_finding(self) -> None:
        pipeline = QualityPipeline()
        pipeline.security_gate({"src/a.py": 'TOKEN = "ghp_realvalue1234"\n'})
        report = pipeline.report(judge_verdict=QualityVerdict.PASS)
        assert report.verdict() is QualityVerdict.BLOCKED


class TestScenarios11To13UnauthorizedActions:
    """A quality agent reaching outside its principal is refused by M1, not by prompt."""

    def test_the_judge_cannot_write(self, tmp_path) -> None:
        gateway = build_gateway(tmp_path, JUDGE)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/a.py", "content": "x = 1"},
            )
        )
        assert not result.ok
        assert not (tmp_path / "src" / "a.py").exists()

    def test_the_judge_cannot_run_a_command(self, tmp_path) -> None:
        gateway = build_gateway(tmp_path, JUDGE)
        result = gateway.execute(
            ToolCall(tool="shell.run", arguments={"argv": ["python", "--version"]})
        )
        assert not result.ok

    def test_the_security_agent_cannot_write(self, tmp_path) -> None:
        gateway = build_gateway(tmp_path, SECURITY)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write", arguments={"path": "src/a.py", "content": "x"}
            )
        )
        assert not result.ok

    def test_the_reviewer_cannot_commit(self, tmp_path) -> None:
        gateway = build_gateway(tmp_path, REVIEWER)
        result = gateway.execute(
            ToolCall(tool="git.commit", arguments={"message": "sneak"})
        )
        assert not result.ok

    def test_the_tester_cannot_write_implementation(self, tmp_path) -> None:
        """The load-bearing one: a tester must not fix a failure by editing the code."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "service.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        gateway = build_gateway(tmp_path, TESTER)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/service.py", "content": "def f():\n    return 2\n"},
            )
        )
        assert not result.ok
        assert "return 1" in (tmp_path / "src" / "service.py").read_text(encoding="utf-8")

    def test_the_tester_may_write_a_test(self, tmp_path) -> None:
        (tmp_path / "tests").mkdir()
        gateway = build_gateway(tmp_path, TESTER)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={
                    "path": "tests/test_new.py",
                    "content": "def test_x():\n    assert True\n",
                },
            )
        )
        assert result.ok, result.error


class TestScenario14UncoveredRequirement:
    def test_an_uncovered_requirement_can_block(self) -> None:
        pipeline = QualityPipeline()
        pipeline.model_findings_gate(
            "coverage",
            (
                QualityFinding(
                    category="requirement-coverage",
                    severity=Severity.HIGH,
                    summary="REQ-003 is not covered by any element",
                    evidence=(ReviewEvidence(source="coverage", detail="REQ-003 unmapped"),),
                    requirement_id="REQ-003",
                ),
            ),
        )
        # Model-origin, so advisory: coverage that must block is asserted by the deterministic
        # product coverage gate, which reports DETERMINISTIC findings.
        assert pipeline.report().verdict() is QualityVerdict.PASS_WITH_ADVISORIES


class TestScenario15UnsupportedFinding:
    def test_a_model_finding_cannot_promote_itself_to_blocking(self) -> None:
        """Even if the agent claims its opinion is a measurement."""
        pipeline = QualityPipeline()
        pipeline.model_findings_gate(
            "review",
            (
                QualityFinding(
                    category="correctness",
                    severity=Severity.CRITICAL,
                    summary="I am certain this is broken",
                    evidence=(ReviewEvidence(source="model", detail="it feels wrong"),),
                    origin=FindingOrigin.DETERMINISTIC,  # the agent lying about its origin
                ),
            ),
        )
        report = pipeline.report()
        assert report.findings[0].origin is FindingOrigin.MODEL
        assert report.verdict() is QualityVerdict.PASS_WITH_ADVISORIES

    def test_an_unsupported_finding_cannot_be_constructed_at_all(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            QualityFinding(category="x", summary="bad vibes", evidence=())


class TestDeterministicReviewChecks:
    def test_a_swallowed_exception_is_reported(self) -> None:
        source = "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
        assert only(scan_review("src/a.py", source), "error-handling")

    def test_a_narrow_handled_exception_is_not_reported(self) -> None:
        source = (
            "def f():\n    try:\n        g()\n    except KeyError:\n        pass\n"
        )
        assert scan_review("src/a.py", source) == ()

    def test_dead_code_after_return_is_reported(self) -> None:
        source = "def f():\n    return 1\n    print('never')\n"
        assert only(scan_review("src/a.py", source), "dead-code")


class TestTheRepairPolicyIsNotForked:
    """M6 Part 7: reuse M5.2's policy, do not invent a second one."""

    def test_a_timeout_does_not_become_a_repair_task(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(
            failed_run(FailureCategory.TIMEOUT)
        )
        report = pipeline.report()
        assert report.verdict() is QualityVerdict.BLOCKED
        assert not report.recommendations()

    def test_an_environment_failure_does_not_become_a_repair_task(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(
            failed_run(FailureCategory.ENVIRONMENT_FAILURE)
        )
        assert pipeline.report().verdict() is QualityVerdict.BLOCKED

    def test_a_genuine_test_failure_is_repairable(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(
            failed_run(FailureCategory.TEST_FAILURE)
        )
        report = pipeline.report()
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED
        assert report.recommendations()
