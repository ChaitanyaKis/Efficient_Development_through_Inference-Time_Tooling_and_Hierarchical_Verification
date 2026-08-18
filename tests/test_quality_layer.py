"""M6: the quality layer's two load-bearing invariants.

**Deterministic evidence outranks model judgement.** The adjudicator takes the Judge's verdict
as input and can only lower the outcome, never raise it. A model saying PASS over a failing
test or a critical security finding is the exact failure M6 exists to prevent, so it is tested
directly rather than inferred from the code reading correctly.

**Principals cannot become each other.** Each new quality agent needs *some* access, and the
natural way to grant it is to copy a neighbouring permission set and add to it. Do that twice
and the reviewer holds the coder's writes while every individual grant still looks defensible.
The relationships are therefore asserted, not trusted.
"""

from __future__ import annotations

import pytest

from edith.errors import FailureCategory
from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    QualityReport,
    QualityVerdict,
    ReviewEvidence,
)
from edith.quality.principals import (
    IMPLEMENTATION_SCOPE,
    JUDGE,
    QUALITY_PERMISSIONS,
    REVIEWER,
    SECURITY,
    TEST_SCOPE,
    TESTER,
    TESTGEN,
    VERIFIER,
    Principal,
    may_execute,
    may_mutate_git,
    may_write,
)
from edith.schemas.common import Severity

EVIDENCE = (ReviewEvidence(source="pytest", detail="1 failed, 0 passed"),)


def finding(
    *,
    severity: Severity = Severity.HIGH,
    origin: FindingOrigin = FindingOrigin.DETERMINISTIC,
    repairable: bool = False,
    category: str = "correctness",
) -> QualityFinding:
    return QualityFinding(
        category=category,
        severity=severity,
        summary="something is wrong",
        evidence=EVIDENCE,
        origin=origin,
        repairable=repairable,
    )


class TestAFindingRequiresEvidence:
    def test_a_finding_without_evidence_cannot_be_constructed(self) -> None:
        """"This code looks bad" must not be representable."""
        with pytest.raises(ValueError, match="evidence"):
            QualityFinding(category="style", summary="looks bad", evidence=())

    def test_blank_evidence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            ReviewEvidence(source="   ", detail="   ")

    def test_evidence_may_anchor_to_a_file_and_line(self) -> None:
        item = ReviewEvidence(
            source="bandit", detail="hardcoded password", file="src/a.py", line=12
        )
        assert item.line == 12

    def test_a_zero_line_is_refused(self) -> None:
        """Lines are 1-indexed; 0 would silently mean "the whole file"."""
        with pytest.raises(ValueError):
            ReviewEvidence(source="s", detail="d", line=0)


class TestTheJudgeCannotGrantAPass:
    """M6 item 8: no model-generated verdict authorizes a merge."""

    def test_a_critical_deterministic_finding_blocks_over_a_judge_pass(self) -> None:
        report = QualityReport(
            findings=(finding(severity=Severity.CRITICAL),),
            judge_verdict=QualityVerdict.PASS,
        )
        assert report.verdict() is QualityVerdict.BLOCKED
        assert not report.verdict().merges

    def test_a_critical_finding_blocks_even_when_marked_repairable(self) -> None:
        """Severity wins: a critical defect is not merged because someone might fix it."""
        report = QualityReport(
            findings=(finding(severity=Severity.CRITICAL, repairable=True),),
            judge_verdict=QualityVerdict.PASS,
        )
        assert report.verdict() is QualityVerdict.BLOCKED

    def test_a_failing_test_blocks_over_a_judge_pass(self) -> None:
        report = QualityReport(
            findings=(finding(category="tests", repairable=True),),
            judge_verdict=QualityVerdict.PASS,
        )
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED

    def test_the_judge_may_still_withhold_a_pass(self) -> None:
        """It can lower the verdict. That direction is safe."""
        report = QualityReport(judge_verdict=QualityVerdict.REPAIR_REQUIRED)
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED

    def test_a_clean_report_with_no_judge_passes(self) -> None:
        assert QualityReport().verdict() is QualityVerdict.PASS

    def test_a_model_opinion_alone_does_not_block(self) -> None:
        """Otherwise an unaided model verdict becomes a merge gate by the back door."""
        report = QualityReport(
            findings=(finding(severity=Severity.HIGH, origin=FindingOrigin.MODEL),)
        )
        assert report.verdict() is QualityVerdict.PASS_WITH_ADVISORIES
        assert report.verdict().merges

    def test_low_severity_findings_are_advisory(self) -> None:
        report = QualityReport(findings=(finding(severity=Severity.LOW),))
        assert report.verdict() is QualityVerdict.PASS_WITH_ADVISORIES


class TestRepairabilityDecidesBetweenRepairAndBlock:
    def test_all_repairable_blocking_findings_request_repair(self) -> None:
        report = QualityReport(
            findings=(
                finding(category="tests", repairable=True),
                finding(category="imports", repairable=True),
            )
        )
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED

    def test_one_unrepairable_finding_blocks_the_whole_set(self) -> None:
        """M5.2's lesson: a set containing an unfixable failure is not a repair task."""
        report = QualityReport(
            findings=(
                finding(category="tests", repairable=True),
                finding(category="environment", repairable=False),
            )
        )
        assert report.verdict() is QualityVerdict.BLOCKED

    def test_recommendations_cover_only_repairable_findings(self) -> None:
        report = QualityReport(
            findings=(
                finding(category="tests", repairable=True),
                finding(category="secrets", severity=Severity.HIGH, repairable=False),
            )
        )
        categories = {item.finding_category for item in report.recommendations()}
        assert categories == {"tests"}

    def test_recommendations_carry_the_evidence(self) -> None:
        """Repair without the failure output does not work; M2.1 established that."""
        report = QualityReport(findings=(finding(category="tests", repairable=True),))
        assert report.recommendations()[0].evidence == EVIDENCE


class TestFindingOriginIsHonest:
    def test_a_deterministic_finding_normalises_to_full_confidence(self) -> None:
        normalised = finding().model_copy(update={"confidence": 0.3}).normalise()
        assert normalised.confidence == 1.0

    def test_a_model_finding_keeps_its_stated_confidence(self) -> None:
        item = finding(origin=FindingOrigin.MODEL).model_copy(
            update={"confidence": 0.4}
        )
        assert item.normalise().confidence == 0.4

    def test_a_finding_can_carry_its_taxonomy_category(self) -> None:
        item = finding().model_copy(
            update={"failure_category": FailureCategory.TEST_FAILURE}
        )
        assert item.failure_category is FailureCategory.TEST_FAILURE


class TestPrincipalIsolation:
    """M6 item 8/Part 8: no principal may become a permission superset of another."""

    def test_the_judge_cannot_write(self) -> None:
        assert not may_write(JUDGE)

    def test_the_judge_cannot_execute(self) -> None:
        """A judge that can run a command can change what it is judging."""
        assert not may_execute(JUDGE)

    def test_the_judge_cannot_mutate_git(self) -> None:
        assert not may_mutate_git(JUDGE)

    def test_the_security_agent_cannot_write(self) -> None:
        assert not may_write(SECURITY)

    def test_the_security_agent_cannot_execute(self) -> None:
        assert not may_execute(SECURITY)

    def test_the_reviewer_cannot_write_or_execute(self) -> None:
        assert not may_write(REVIEWER)
        assert not may_execute(REVIEWER)

    def test_the_tester_may_write_only_tests(self) -> None:
        assert may_write(TESTER)
        assert TESTER.allowed_write_paths == TEST_SCOPE

    def test_the_tester_cannot_reach_implementation_paths(self) -> None:
        """The point of the split: a tester cannot make a suite green by editing the code."""
        assert not set(TESTER.allowed_write_paths) & set(IMPLEMENTATION_SCOPE)

    def test_no_quality_principal_can_mutate_git(self) -> None:
        for name, permissions in QUALITY_PERMISSIONS.items():
            assert not may_mutate_git(permissions), f"{name} may mutate git"

    def test_only_the_tester_and_verifier_may_execute(self) -> None:
        executors = {
            name for name, perms in QUALITY_PERMISSIONS.items() if may_execute(perms)
        }
        assert executors == {Principal.TESTER, Principal.VERIFIER}

    def test_only_test_writing_principals_may_write(self) -> None:
        """Reviewers, security and the judge write nothing. Only the two test roles do."""
        writers = {
            name for name, perms in QUALITY_PERMISSIONS.items() if may_write(perms)
        }
        assert writers == {Principal.TESTER, Principal.TESTGEN}

    def test_the_generator_is_a_strict_reduction_of_the_tester(self) -> None:
        """M8 narrows the test writer rather than adding capability.

        TESTGEN drops shell entirely and confines writes to ``tests/generated/**``, so it
        cannot run the suite it wrote, and a generated file can never land on a hand-written
        acceptance test. Recorded as an explicit relationship because the generic superset
        check below cannot tell a deliberate reduction from a role quietly gaining reach.
        """
        from fnmatch import fnmatch

        assert TESTGEN.allowed_tools < TESTER.allowed_tools
        assert not may_execute(TESTGEN)

        def allows(permissions: object, path: str) -> bool:
            return any(
                fnmatch(path, pattern)
                for pattern in permissions.allowed_write_paths  # type: ignore[attr-defined]
            )

        # Everything the generator may write, the tester may write too.
        assert allows(TESTGEN, "tests/generated/test_req.py")
        assert allows(TESTER, "tests/generated/test_req.py")
        # But the reverse fails: a hand-written acceptance test is out of the generator's reach.
        assert allows(TESTER, "tests/test_acceptance.py")
        assert not allows(TESTGEN, "tests/test_acceptance.py")

    def test_no_principal_is_a_superset_of_another(self) -> None:
        """The invariant that stops permissions drifting together over time.

        One pair is exempt and asserted separately: TESTGEN is a documented strict reduction
        of TESTER, proved by
        :meth:`test_the_generator_is_a_strict_reduction_of_the_tester`. Every other pair must
        remain incomparable, which is what stops a reviewer acquiring a writer's reach.
        """
        reductions = {(Principal.TESTER, Principal.TESTGEN)}
        for name, permissions in QUALITY_PERMISSIONS.items():
            for other_name, other in QUALITY_PERMISSIONS.items():
                if name == other_name or (name, other_name) in reductions:
                    continue
                assert not (
                    permissions.allowed_tools > other.allowed_tools
                    and permissions.allowed_write_paths
                    and other.allowed_write_paths
                ), f"{name} is a superset of {other_name}"

    def test_the_verifier_matches_the_executors_existing_principal(self) -> None:
        """M6 must not fork a second definition of the verifier; M5.2 proved that one."""
        from edith.engineering.executor import VERIFIER_PERMISSIONS

        assert VERIFIER.allowed_tools == VERIFIER_PERMISSIONS.allowed_tools
