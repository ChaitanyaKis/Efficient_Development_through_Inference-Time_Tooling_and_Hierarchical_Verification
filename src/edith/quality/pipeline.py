"""The quality pipeline: gates in order, deterministic adjudication at the end.

The order matters and is not arbitrary. Each gate is cheaper and more certain than the one
after it, so the pipeline stops paying for inference the moment deterministic evidence has
already settled the question:

    import gate -> tests -> test integrity -> security -> review -> judge -> adjudicate

The last step is a pure function (:func:`~edith.quality.artifacts.adjudicate`), and it is the
only thing that produces a verdict. No agent in this module returns PASS. They contribute
findings; the adjudicator decides. That is M6 item 8, and it is what makes an adversarial or
merely over-confident model harmless: a Judge insisting on PASS over a failing test changes
nothing, because its verdict is an input.

Two further properties are structural rather than conventional:

**Short-circuit on blocking evidence.** Once a CRITICAL deterministic finding exists the
verdict cannot become PASS, so running a model reviewer afterwards buys nothing and costs a
model call on 6GB of VRAM. :meth:`QualityPipeline.run` stops.

**Test integrity is checked, not assumed.** M6 item 5: tests must not be trusted blindly. A
suite that went green because an assertion was weakened or a test deleted is a FAIL, and M2.1's
AST comparison already decides that -- so it is reused here rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from edith.integrity import IntegrityReport, build_report
from edith.observability.logging import get_logger
from edith.quality.artifacts import (
    FindingOrigin,
    QualityFinding,
    QualityReport,
    QualityVerdict,
    ReviewEvidence,
)
from edith.quality.scanners import scan_review, scan_security
from edith.schemas.agent import AgentRequest
from edith.schemas.common import Severity
from edith.verification.runner import VerificationReport

logger = get_logger(__name__)


@dataclass
class GateResult:
    """What one gate contributed."""

    name: str
    findings: tuple[QualityFinding, ...] = ()
    #: False when the gate could not run at all, as distinct from running and finding nothing.
    ran: bool = True

    @property
    def blocking(self) -> bool:
        return any(item.blocking for item in self.findings)


@dataclass
class QualityPipeline:
    """Runs the quality gates for one task and adjudicates the result.

    Constructed with whatever evidence the caller already has. The verification report comes
    from M5's runner; the sources are read from the task workspace. Nothing here executes a
    process itself -- the pipeline consumes evidence rather than gathering privilege.
    """

    task_id: str = ""
    gates: list[GateResult] = field(default_factory=list)

    def _record(self, result: GateResult) -> GateResult:
        self.gates.append(result)
        logger.info(
            "quality.gate",
            task_id=self.task_id,
            gate=result.name,
            findings=len(result.findings),
            blocking=result.blocking,
            ran=result.ran,
        )
        return result

    # -- Gates ---------------------------------------------------------------------

    def import_gate(self, *, importable: bool, error: str) -> GateResult:
        """M5's import gate, expressed as a finding.

        For Python, importability *is* the build. A module that does not import has not been
        implemented, whatever the tests say about the rest of the tree.
        """
        if importable:
            return self._record(GateResult(name="import"))
        return self._record(
            GateResult(
                name="import",
                findings=(
                    QualityFinding(
                        category="build",
                        severity=Severity.CRITICAL,
                        summary="the generated code does not import",
                        evidence=(
                            ReviewEvidence(source="import", detail=error or "import failed"),
                        ),
                        origin=FindingOrigin.DETERMINISTIC,
                        repairable=True,
                        confidence=1.0,
                    ),
                ),
            )
        )

    def test_gate(self, report: VerificationReport | None) -> GateResult:
        """Turn the verification report into findings.

        A report that could not run is *not* a passing report, and is not repairable either --
        that is M5.2's distinction, preserved here rather than re-derived.
        """
        if report is None:
            return self._record(GateResult(name="tests", ran=False))
        if report.passed:
            return self._record(GateResult(name="tests"))

        category = report.failure_category
        # Only a genuine test/code failure is the agent's to fix. Everything else -- timeout,
        # environment, dependency -- describes the machine, and must not consume repair budget.
        from edith.engineering.executor import (  # noqa: PLC0415 - avoids a cycle
            REPAIRABLE_FAILURES,
        )

        repairable = category is not None and category in REPAIRABLE_FAILURES
        return self._record(
            GateResult(
                name="tests",
                findings=(
                    QualityFinding(
                        category="tests",
                        severity=Severity.HIGH,
                        summary=(
                            "verification failed "
                            f"({category.value if category else 'unknown'})"
                        ),
                        evidence=(
                            ReviewEvidence(
                                source="verification", detail=report.evidence(2000) or "no output"
                            ),
                        ),
                        origin=FindingOrigin.DETERMINISTIC,
                        failure_category=category,
                        repairable=repairable,
                        confidence=1.0,
                    ),
                ),
            )
        )

    def integrity_gate(
        self,
        baseline: dict[str, str],
        current: dict[str, str],
        changed_paths: list[str],
        *,
        baseline_unavailable: bool = False,
    ) -> GateResult:
        """M6 item 5: a green suite that was made green by weakening it is not a pass."""
        report: IntegrityReport = build_report(
            baseline, current, changed_paths, baseline_unavailable=baseline_unavailable
        )
        if not report.findings:
            return self._record(GateResult(name="test-integrity"))
        findings = tuple(
            QualityFinding(
                category="test-integrity",
                severity=item.severity,
                summary=item.detail,
                evidence=(
                    ReviewEvidence(
                        source="integrity:ast",
                        detail=item.evidence or item.detail,
                        file=item.path,
                    ),
                ),
                affected_files=(item.path,) if item.path else (),
                origin=FindingOrigin.DETERMINISTIC,
                repairable=True,
                confidence=1.0,
            )
            for item in report.findings
        )
        return self._record(GateResult(name="test-integrity", findings=findings))

    def security_gate(self, sources: dict[str, str]) -> GateResult:
        """Deterministic security scan over the changed sources."""
        findings: list[QualityFinding] = []
        for path, content in sorted(sources.items()):
            findings.extend(scan_security(path, content))
        return self._record(GateResult(name="security", findings=tuple(findings)))

    def review_gate(self, sources: dict[str, str]) -> GateResult:
        """Deterministic review checks. The model's review is a separate, advisory gate."""
        findings: list[QualityFinding] = []
        for path, content in sorted(sources.items()):
            findings.extend(scan_review(path, content))
        return self._record(GateResult(name="review", findings=tuple(findings)))

    def model_findings_gate(
        self, name: str, findings: tuple[QualityFinding, ...]
    ) -> GateResult:
        """Accept a model agent's findings, forcing them to be advisory.

        A model reviewer cannot promote its own opinion into a merge gate: the origin is
        overwritten to ``MODEL`` here regardless of what the agent claimed. If a model wants a
        finding to block, it has to be corroborated by a deterministic check, which is exactly
        the property M6 asks for.
        """
        coerced = tuple(
            item.model_copy(update={"origin": FindingOrigin.MODEL}) for item in findings
        )
        return self._record(GateResult(name=name, findings=coerced))

    # -- Adjudication --------------------------------------------------------------

    def report(
        self,
        *,
        judge_verdict: QualityVerdict | None = None,
        judge_rationale: str = "",
    ) -> QualityReport:
        """Collect every gate's findings into one adjudicated report."""
        findings = tuple(
            item.normalise() for gate in self.gates for item in gate.findings
        )
        return QualityReport(
            task_id=self.task_id,
            findings=findings,
            judge_verdict=judge_verdict,
            judge_rationale=judge_rationale,
        )

    @property
    def blocked(self) -> bool:
        """Whether deterministic evidence already prevents a pass.

        Used to short-circuit: there is no point spending a model call once the verdict cannot
        become PASS.
        """
        return any(gate.blocking for gate in self.gates)


def read_sources(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    """Read the changed files for scanning, skipping what cannot be read as text.

    Unreadable files are omitted rather than guessed at. The scanners report a parse failure as
    CRITICAL, so a file that is present but broken still surfaces; a file that is absent is the
    diff's business, not the scanner's.
    """
    sources: dict[str, str] = {}
    for relative in paths:
        candidate = root / relative
        try:
            sources[relative] = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return sources


def run_model_review(
    pipeline: QualityPipeline,
    agent: object,
    *,
    name: str,
    sources: dict[str, str],
    task: str = "",
) -> GateResult:
    """Run one model reviewer over the changed sources and record its findings.

    Returns without calling the model at all when deterministic evidence has already blocked.
    That is not only a cost decision: a reviewer's opinion on code that is already rejected
    cannot change the verdict, so the call would buy nothing on a 6GB card.

    A malformed or failing model response degrades to *no findings*, never to an exception and
    never to a pass. M6.1 item 12: the model failing must be safe.
    """
    from edith.quality.agents import ReviewInput, ReviewOutput, to_findings  # noqa: PLC0415

    if pipeline.blocked:
        logger.info(
            "quality.skipped",
            gate=name,
            reason="already blocked by deterministic evidence",
        )
        return pipeline._record(GateResult(name=name, ran=False))

    collected: list[QualityFinding] = []
    for path, source in sorted(sources.items()):
        if not source.strip():
            continue
        try:
            request = AgentRequest(
                payload=ReviewInput(path=path, source=source, task=task).model_dump()
            )
            response = agent.execute(request)  # type: ignore[attr-defined]
            output = ReviewOutput.model_validate(response.output)
        except Exception as exc:  # noqa: BLE001 - a reviewer must never abort the pipeline
            logger.warning(
                "quality.review_failed", gate=name, path=path, error=f"{type(exc).__name__}: {exc}"
            )
            continue
        collected.extend(
            to_findings(
                output, path=path, source=source, agent=name, task_id=pipeline.task_id
            )
        )
    return pipeline.model_findings_gate(name, tuple(collected))
