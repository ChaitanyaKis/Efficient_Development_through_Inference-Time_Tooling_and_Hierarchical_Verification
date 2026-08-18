"""M6.1 Part 9: the model quality agents under adversarial conditions.

M6 proved the deterministic layer cannot be talked out of a finding. These tests prove the
model agents cannot talk their way *into* one -- that a reviewer's confidence, its claimed
origin, and its cited evidence are all subject to the system rather than the model.

The sharpest case is 14: a model that quotes a line which does not exist in the file it is
reviewing. Nothing downstream can detect that, because a fabricated citation reads exactly like
a real one. So it is caught at the boundary, by checking the quote against the source.
"""

from __future__ import annotations

from edith.errors import FailureCategory
from edith.quality.agents import (
    CodeReviewAgent,
    JudgeAgent,
    JudgeOutput,
    ModelFinding,
    ReviewOutput,
    SecurityAgent,
    render_findings,
    to_findings,
)
from edith.quality.agents import (
    TestingAgent as _TestingAgent,
)
from edith.quality.artifacts import FindingOrigin, QualityVerdict
from edith.quality.pipeline import QualityPipeline, run_model_review
from edith.quality.principals import (
    JUDGE,
    REVIEWER,
    SECURITY,
    TESTER,
    may_execute,
    may_mutate_git,
    may_write,
)
from edith.schemas.common import Severity
from edith.tools.schemas import ToolCall
from edith.verification.runner import VerificationOutcome, VerificationReport

from .tool_fixtures import build_gateway

SOURCE = "def add(a, b):\n    return a - b\n"


def failed_run(category: FailureCategory = FailureCategory.TEST_FAILURE) -> VerificationReport:
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


class _StubAgent:
    """An agent returning a fixed response, standing in for a model."""

    def __init__(self, output: object, *, raises: bool = False) -> None:
        self._output = output
        self._raises = raises
        self.calls = 0

    def execute(self, request: object) -> object:
        self.calls += 1
        if self._raises:
            raise RuntimeError("model produced garbage")

        class _Response:
            output = (
                self._output.model_dump()
                if hasattr(self._output, "model_dump")
                else self._output
            )

        return _Response()


class TestAgentsDeclareTheRightPrincipals:
    """Items 1-6: the agents hold exactly the permissions their role allows."""

    def test_the_testing_agent_writes_only_tests(self) -> None:
        assert _TestingAgent.identity.permissions == TESTER
        assert may_write(_TestingAgent.identity.permissions)

    def test_the_code_reviewer_cannot_write(self) -> None:
        assert CodeReviewAgent.identity.permissions == REVIEWER
        assert not may_write(CodeReviewAgent.identity.permissions)

    def test_the_security_agent_cannot_write_or_execute(self) -> None:
        assert SecurityAgent.identity.permissions == SECURITY
        assert not may_write(SecurityAgent.identity.permissions)
        assert not may_execute(SecurityAgent.identity.permissions)

    def test_the_judge_holds_nothing_but_read(self) -> None:
        assert JudgeAgent.identity.permissions == JUDGE
        assert not may_write(JUDGE)
        assert not may_execute(JUDGE)
        assert not may_mutate_git(JUDGE)

    def test_the_testing_agent_cannot_edit_implementation_through_the_gateway(
        self, tmp_path
    ) -> None:
        """Enforced by M1, not by the prompt: the real gateway refuses the write."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "s.py").write_text("original\n", encoding="utf-8")
        gateway = build_gateway(tmp_path, _TestingAgent.identity.permissions)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/s.py", "content": "tampered"},
            )
        )
        assert not result.ok
        assert (tmp_path / "src" / "s.py").read_text(encoding="utf-8") == "original\n"

    def test_the_security_agent_cannot_write_a_vulnerability_then_approve_it(
        self, tmp_path
    ) -> None:
        """Item 10, and the reason SECURITY holds no write scope at all."""
        (tmp_path / "src").mkdir()
        gateway = build_gateway(tmp_path, SecurityAgent.identity.permissions)
        result = gateway.execute(
            ToolCall(
                tool="filesystem.write",
                arguments={"path": "src/backdoor.py", "content": "import os"},
            )
        )
        assert not result.ok
        assert not (tmp_path / "src" / "backdoor.py").exists()


class TestModelFindingsAreSystemOwned:
    def test_a_model_finding_is_always_recorded_as_model_origin(self) -> None:
        """Item 7: the model does not get to say its opinion was a measurement."""
        output = ReviewOutput(
            findings=[
                ModelFinding(
                    category="correctness",
                    severity=Severity.CRITICAL,
                    description="broken",
                    quoted_line="return a - b",
                )
            ]
        )
        findings = to_findings(output, path="src/a.py", source=SOURCE, agent="reviewer")
        assert findings[0].origin is FindingOrigin.MODEL
        assert not findings[0].blocking

    def test_hallucinated_evidence_is_discarded(self) -> None:
        """Item 14: a quote that is not in the file means the finding is unsupported."""
        output = ReviewOutput(
            findings=[
                ModelFinding(
                    category="security",
                    severity=Severity.CRITICAL,
                    description="there is an eval here",
                    quoted_line="eval(user_input)",
                )
            ]
        )
        assert to_findings(output, path="src/a.py", source=SOURCE, agent="sec") == ()

    def test_a_finding_without_a_quote_is_kept_but_advisory(self) -> None:
        output = ReviewOutput(
            findings=[ModelFinding(category="style", description="hard to follow")]
        )
        findings = to_findings(output, path="src/a.py", source=SOURCE, agent="reviewer")
        assert len(findings) == 1
        assert not findings[0].blocking

    def test_the_model_cannot_set_confidence_or_identity(self) -> None:
        """ModelFinding has no such fields, so the abuse is unrepresentable."""
        fields = set(ModelFinding.model_fields)
        assert fields == {"category", "severity", "description", "quoted_line"}

    def test_findings_are_capped(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReviewOutput(
                findings=[
                    ModelFinding(category="c", description="d") for _ in range(20)
                ]
            )


class TestTheReviewGateFailsSafe:
    def test_a_model_that_raises_produces_no_findings(self) -> None:
        """Item 12: a malformed model response must not abort or pass the pipeline."""
        pipeline = QualityPipeline()
        gate = run_model_review(
            pipeline, _StubAgent(None, raises=True), name="review", sources={"a.py": SOURCE}
        )
        assert gate.findings == ()
        assert pipeline.report().verdict() is QualityVerdict.PASS

    def test_model_review_is_skipped_once_deterministically_blocked(self) -> None:
        """No model call can change a verdict that is already BLOCKED."""
        pipeline = QualityPipeline()
        pipeline.import_gate(importable=False, error="boom")
        stub = _StubAgent(ReviewOutput(findings=[]))
        gate = run_model_review(pipeline, stub, name="review", sources={"a.py": SOURCE})
        assert stub.calls == 0
        assert not gate.ran

    def test_a_clean_review_leaves_the_deterministic_verdict_alone(self) -> None:
        """Item 15: the deterministic result does not depend on the model agents."""
        pipeline = QualityPipeline()
        pipeline.test_gate(failed_run())
        deterministic = pipeline.report().verdict()
        run_model_review(
            pipeline, _StubAgent(ReviewOutput(findings=[])), name="review", sources={}
        )
        assert pipeline.report().verdict() is deterministic

    def test_the_reviewer_is_called_once_per_file(self) -> None:
        stub = _StubAgent(ReviewOutput(findings=[]))
        run_model_review(
            QualityPipeline(),
            stub,
            name="review",
            sources={"a.py": SOURCE, "b.py": SOURCE},
        )
        assert stub.calls == 2

    def test_empty_files_are_not_sent_to_the_model(self) -> None:
        stub = _StubAgent(ReviewOutput(findings=[]))
        run_model_review(QualityPipeline(), stub, name="review", sources={"a.py": "   "})
        assert stub.calls == 0


class TestTheJudgeRemainsNonAuthoritative:
    def test_judge_pass_cannot_override_a_failing_test(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(failed_run())
        report = pipeline.report(
            judge_verdict=QualityVerdict.PASS, judge_rationale="looks right to me"
        )
        assert report.verdict() is QualityVerdict.REPAIR_REQUIRED

    def test_judge_pass_cannot_override_critical_security(self) -> None:
        pipeline = QualityPipeline()
        pipeline.security_gate({"src/a.py": "import os\nos.system('rm -rf /')\n"})
        report = pipeline.report(judge_verdict=QualityVerdict.PASS)
        assert report.verdict() is QualityVerdict.BLOCKED

    def test_judge_disagreement_is_preserved_as_evidence(self) -> None:
        pipeline = QualityPipeline()
        pipeline.test_gate(failed_run())
        report = pipeline.report(
            judge_verdict=QualityVerdict.PASS, judge_rationale="I disagree with the suite"
        )
        assert report.judge_verdict is QualityVerdict.PASS
        assert "disagree" in report.judge_rationale

    def test_the_judge_output_defaults_to_the_safe_verdict(self) -> None:
        """A Judge that says nothing must not thereby approve anything."""
        assert JudgeOutput().verdict is QualityVerdict.FAILED


class TestJudgePromptStaysSmall:
    """M3.2: this model degrades as its prompt grows, so the summary is capped."""

    def test_findings_are_summarised_and_capped(self) -> None:
        from edith.quality.scanners import scan_security

        findings = scan_security("src/a.py", "import os\n" + "os.system('x')\n" * 20)
        rendered = render_findings(findings, limit=3)
        assert rendered.count("\n") <= 3
        assert "more)" in rendered

    def test_no_findings_renders_compactly(self) -> None:
        assert render_findings(()) == "none"
